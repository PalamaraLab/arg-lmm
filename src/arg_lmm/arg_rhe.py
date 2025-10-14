import logging
import time

import arg_needle_lib
from .third_party.chiscore.liu import liu_sf
import click
import numpy as np
import pandas as pd
import scipy.stats


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def rsvd(
    arg, rank, rng, n_oversamples=None, n_subspace_iters=None, alpha=-1
):
    """Randomized SVD (p. 227 of Halko et al).

    :param arg:              ARG-needle arg object.
    :param rank:             Desired rank approximation.
    :param n_oversamples:    Oversampling parameter for Gaussian random samples.
    :param n_subspace_iters: Number of power iterations.
    :param alpha:            Normalizing factor in gentoype matrix.
    :return:                 Singular values as in truncated SVD.
    """
    if n_oversamples is None:
        # This is the default used in the paper.
        n_samples = 2 * rank
    else:
        n_samples = rank + n_oversamples

    # Stage A.
    Q = find_range(arg, n_samples, rng, n_subspace_iters, alpha)

    # Stage B.
    B = arg_needle_lib.ARG_by_matrix_multiply_muts(
        arg, Q.T, standardize=True, alpha=alpha, diploid=True
    )
    S = np.linalg.svd(B, compute_uv=False)

    return S[:rank]


# ------------------------------------------------------------------------------


def find_range(arg, n_samples, rng, n_subspace_iters=None, alpha=-1):
    """Algorithm 4.1: Randomized range finder (p. 240 of Halko et al).

    Given a matrix A and a number of samples, computes an orthonormal matrix
    that approximates the range of A.

    :param arg:              ARG-needle arg object.
    :param n_samples:        Number of Gaussian random samples.
    :param n_subspace_iters: Number of subspace iterations.
    :alpha:                  Normalizing factor in gentoype matrix.
    :return:                 Orthonormal basis for approximate range of A.
    """
    m, n = arg.num_samples() // 2, arg.num_mutations()
    O = rng.normal(size=(n, n_samples))
    Y = arg_needle_lib.ARG_by_matrix_multiply_samples(
        arg, O, standardize=True, alpha=alpha, diploid=True
    )

    if n_subspace_iters:
        return subspace_iter(arg, Y, n_subspace_iters, alpha)
    else:
        return ortho_basis(Y)


# ------------------------------------------------------------------------------


def subspace_iter(arg, Y0, n_iters, alpha=-1):
    """Algorithm 4.4: Randomized subspace iteration (p. 244 of Halko et al).

    Uses a numerically stable subspace iteration algorithm to down-weight
    smaller singular values.

    :param arg:     ARG-needle arg object.
    :param Y0:      Initial approximate range of arg genotype.
    :param n_iters: Number of subspace iterations.
    :alpha:         Normalizing factor in gentoype matrix.
    :return:        Orthonormalized approximate range of A after power
                    iterations.
    """
    Q = ortho_basis(Y0)
    for _ in range(n_iters):
        Z = arg_needle_lib.ARG_by_matrix_multiply_muts(
            arg, Q.T, standardize=True, alpha=alpha, diploid=True
        ).T
        Q = ortho_basis(
            arg_needle_lib.ARG_by_matrix_multiply_samples(
                arg, Z, standardize=True, alpha=alpha, diploid=True
            )
        )
    return Q


# ------------------------------------------------------------------------------


def ortho_basis(M):
    """Computes an orthonormal basis for a matrix.

    :param M: (m x n) matrix.
    :return:  An orthonormal basis for M.
    """
    Q, _ = np.linalg.qr(M)
    return Q


def cumulant_est(
    arg,
    rng,
    alpha=-1,
    nVectors=200,
    diploid=True,
    debug=True,
):
    """
    Estimates the first 2 cumulants of quadratic form y^t K y,
    where y is assumed to be standard normal, and K is the random ARG-GRM.
    """

    N = arg.num_samples() // 2 if diploid else arg.num_samples()
    M = arg.num_mutations()

    Z = rng.normal(0, 1, size=(N, nVectors))
    results = np.zeros(2)

    U = arg_needle_lib.ARG_by_matrix_multiply_muts(
        arg, Z.T, standardize=True, alpha=alpha, diploid=diploid
    ).T
    trace_K = np.linalg.norm(U) ** 2 / nVectors

    #  normalizing factor to make trace K = N
    factor = N / trace_K
    results[0] = factor

    U = arg_needle_lib.ARG_by_matrix_multiply_samples(
        arg, U, standardize=True, alpha=alpha, diploid=diploid
    )
    trace_K2 = np.linalg.norm(U) ** 2 * (factor**2) / nVectors
    return factor, trace_K2


def chi2_cgf(t, lam, dof):
    result = 0
    for l, d in zip(lam, dof):
        result -= 0.5 * d * np.log(1 - 2 * l * t)
    return result


def chi2_cgf_prime(t, lam, dof):
    result = 0
    for l, d in zip(lam, dof):
        result += l * d / (1 - 2 * l * t)
    return result


def chi2_cgf_prime_prime(t, lam, dof):
    result = 0
    for l, d in zip(lam, dof):
        result += 2 * l**2 * d / (1 - 2 * l * t) ** 2
    return result


def chi2_pval(q, lam, dof):
    ld = 0
    for l, d in zip(lam, dof):
        ld += l * d
    t0 = scipy.optimize.bisect(
        lambda x: chi2_cgf_prime(x, lam, dof) - q,
        min(0, np.nextafter((q - ld) / (2 * q * min(lam)), -1)),
        np.nextafter(1 / 2 / max(lam), -1),
        maxiter=10000,
        xtol=1e-300,
    )
    w = np.sign(t0) * np.sqrt(2 * (t0 * q - chi2_cgf(t0, lam, dof)))
    v = t0 * np.sqrt(chi2_cgf_prime_prime(t0, lam, dof))
    return scipy.stats.norm.sf(w + np.log(v / w) / w)


def arg_rhe_with_error(arg, Traits, rng, alpha=-1, nVectors=200, diploid=True, debug=True):
    """
    Estimates for heritability and environment using RHE-reg as in Yue Wu and S. Sankararaman (2018),
    ie, for one genetic component.
    G should be standardised and Trait mean centered.
    """
    t0 = time.time()
    N, N_pheno = Traits.shape
    M = arg.num_mutations()

    if debug:
        t_trace = time.time()
        print("estimating trace of GRM...", end=" ", flush=True)
    factor, trace_K2 = cumulant_est(
        arg, rng, nVectors=nVectors, alpha=alpha, diploid=diploid, debug=debug
    )
    if debug:
        print(f"done [{time.time()-t_trace:.1f}s]", flush=True)

    GtY = arg_needle_lib.ARG_by_matrix_multiply_muts(
        arg, Traits.T, standardize=True, alpha=alpha, diploid=diploid
    )
    YGGtY = np.linalg.norm(GtY, axis=1) ** 2

    numerators = np.zeros(N_pheno)
    yTy = np.zeros(N_pheno)

    for i in range(N_pheno):
        y = Traits[:, i]
        yTy[i] = y.dot(y)
        numerators[i] = YGGtY[i] * factor - yTy[i]

    sigma_g_hat = numerators / (trace_K2 - N)
    sigma_e_hat = yTy / N - sigma_g_hat

    if debug:
        t_svd = time.time()
        print("estimating leading eigenvalues of GRM...", end=" ", flush=True)

    sigvals = rsvd(arg, 200, rng, n_subspace_iters=0, alpha=alpha)

    if debug:
        print(f"done [{time.time()-t_svd:.1f}s]", flush=True)

    sigvals = sigvals**2 * factor
    sigvals = sigvals[sigvals > 1e-5]

    l1 = N - sigvals.sum()
    l2 = trace_K2 - (sigvals**2).sum()
    if (l1 > 0) and (l2 > 0):
        s_dofs = [1] * sigvals.size + [l1**2 / l2]
        s_coefs = sigvals.tolist() + [l2 / l1]
    else:
        # for the unlikely case when estimated trace is less than sum of singular values
        s_dofs = [1] * sigvals.size
        s_coefs = sigvals.tolist()

    # if debug:
    #     print(f'{l1=}, {l2=}, {factor=}, {sigvals.sum()=}, {s_dofs=}, {s_coefs=}')

    pvals = np.zeros(N_pheno)
    for i in range(N_pheno):

        test_stat = YGGtY[i] * factor
        (first_estimate_pv, _, _, _) = liu_sf(
            test_stat, s_coefs, s_dofs, [0] * len(s_dofs)
        )
        if first_estimate_pv > 1e-2:
            pvals[i] = first_estimate_pv
        else:
            pvals[i] = chi2_pval(test_stat, s_coefs, s_dofs)

    if debug:
        print("ARG-h2 found by ARG-RHE:")
        print(" ".join(format(x, ".2e") for x in sigma_g_hat))
        print("Approx p-vals for sigma_g:")
        print(" ".join(format(x, ".2e") for x in pvals))
        print("Total duration = {0:.2f} secs".format(time.time() - t0))

    return sigma_g_hat, sigma_e_hat, pvals


@click.command()
@click.option("--arg", "arg_path", required=True, help="Input argn file")
@click.option("--pheno", "phe_path", required=True, help="Input phenotypes file")
@click.option("--out", "-o", required=True, help="Output file")
@click.option("--mu", default=1e-6, type=float, help="Mutation resampling rate")
@click.option("--alpha", default=-1.0, type=float, help="Alpha normalising exponent")
@click.option("--mac", default=1, type=float, help="Minimal MAC of resampled mutations to include")
@click.option("--seed", default=42, type=int, help="Random seed for reproducibility")
def arg_rhe(arg_path, phe_path, out, mu, alpha, mac, seed):
    start_time = time.time()
    logger.info(f"Starting arg_rhe with following parameters:")
    logger.info(f"  arg:        {arg_path}")
    logger.info(f"  phe:        {phe_path}")
    logger.info(f"  out:        {out}")
    logger.info(f"  mu:         {mu}")
    logger.info(f"  alpha:      {alpha}")
    logger.info(f"  mac:        {mac}")
    logger.info(f"  seed:       {seed}")

    arg = arg_needle_lib.deserialize_arg(arg_path)
    arg.populate_children_and_roots()
    arg_needle_lib.generate_mutations(arg, mu, seed)
    arg.populate_mutations_on_edges()
    arg_needle_lib.prepare_multiplication(arg)
    arg.keep_mutations_within_maf(
        mac / arg.num_samples(), 1.0 - mac / arg.num_samples()
    )
    arg.populate_mutations_on_edges()
    arg_needle_lib.prepare_multiplication(arg)

    phenotypes = np.loadtxt(phe_path)[:, 2:]
    phenotypes /= phenotypes.std(axis=0, keepdims=True)

    rng = np.random.default_rng(seed)

    sigma_g_hat, sigma_e_hat, pval = arg_rhe_with_error(
        arg, phenotypes, rng, alpha=alpha, diploid=True
    )

    df = pd.DataFrame(
        {
            "h2_g": sigma_g_hat,
            "h2_e": sigma_e_hat,
            "P": pval,
        }
    )
    df.to_csv(out, index=None)

    logger.info(f"Done, in {time.time() - start_time} seconds")
