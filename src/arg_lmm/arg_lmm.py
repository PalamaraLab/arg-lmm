#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program runs ARG-LOCO step1, assuming that the ARGs are already computed and stored on disk.
It estimates the total heritability using (genome-wide) ARG-RHE,
then computes the LOCO residuals using ARG-LOCO as well as the BLUP (if requested).
The results will be saved to disk in a format suitable for the next step (e.g. association testing).
It is assumed that the ARGs are diploid and that the phenotypes are in a tab-separated file with no header and no index column

Created on Fri Nov 25 16:45:18 2022
Updates:
     June 2023, add SE functionality for ARG-RHE-total-h2
     July 2025, clean code for release
     Sept 2025, refactor for ARG-LMM-fast package
@author: gkalantzis
"""
import argparse, time
import numpy as np
import pandas as pd
import arg_needle_lib
N_THREADS=1 #TODO: needs to be checked

def main():
    parser = argparse.ArgumentParser(description="ARG-LMM: estimate total-h2, compute LOCO residuals and BLUP.")
    parser.add_argument('--arg', nargs='+', required=True, help='List of ARG files, one per chromosome (e.g. chr1.arg chr2.arg ...)')
    parser.add_argument('--pheno', required=True, help='Phenotype file (tab-separated, FID as index)')
    parser.add_argument('-o', '--out', required=True, help='Output prefix')
    parser.add_argument('--ncalib', type=int, default=1, help='Number of calibration variants per chromosome')
    parser.add_argument('--alpha', type=float, default=-1.0, help='Alpha parameter for scaling')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--nThreads', type=int, default=1, help='Number of threads')
    parser.add_argument('--blup', action='store_true', help='Also compute genome-wide BLUPs')
    args = parser.parse_args()

    print("Loading ARGs...")
    arg_all = []
    for fname in args.arg:
        arg = arg_needle_lib.deserialize_arg(fname)
        arg.populate_children_and_roots()
        arg_all.append(arg)
    Nchr = len(arg_all)
    M_all = [x.num_mutations() for x in arg_all]
    print(f"Loaded {Nchr} ARGs with {sum(M_all)} mutations.")

    print("Loading phenotypes...")
    pheno = pd.read_csv(args.pheno, sep='\t', index_col=0).to_numpy()
    pheno_names = pd.read_csv(args.pheno, sep='\t', index_col=0).columns.tolist()
    N, Npheno = pheno.shape

    print(arg_all[0].num_samples())
    assert pheno.shape[0] == arg_all[0].num_samples()//2, "Number of samples in phenotype file does not match those in the ARG."

    print("\nEstimating genome-wide h2 with ARG-RHE...")
    tic = time.perf_counter()
    sigma_g_hat, sigma_e_hat = arg_rhe_total_no_se(
        arg_all, pheno, alpha=args.alpha, nVectors=30,
        diploid=True, seed=args.seed, nThreads=args.nThreads, debug=False
    )
    VC = np.array([sigma_g_hat, sigma_e_hat])
    print(f"Done. Mean h2 = {np.mean(VC[0]):.4f} ({np.std(VC[0])/np.sqrt(Npheno):.4f}). Duration = {time.perf_counter()-tic:.2f}s.")

    cg_params = {
        "N_pheno": Npheno,
        "N_calibr": args.ncalib,
        "max_iters": int(np.log10(N))**2 + N//2000,
        "alpha": args.alpha,
        "diploid": True
    }

    print("\nComputing LOCO residuals and calibration factors...")
    residuals, calibs, BLUP = arg_loco(
        arg_all, pheno, VC, cg_params, blup=args.blup
    )

    print(f"Saving results to {args.out}...")
    if args.blup:
        pd.DataFrame(BLUP).to_csv(f"{args.out}.BLUP.gz", index=None, sep='\t', compression='gzip')
    pd.DataFrame.from_dict(calibs, orient='index').to_csv(
        f"{args.out}.calibrators", header=None, sep='\t', float_format='%.6f'
    )
    for C in range(Nchr):
        df = pd.DataFrame({'FID': range(1, N+1), 'IID': range(1, N+1)})
        for T in range(Npheno):
            df[pheno_names[T]] = residuals[C][:, T]
        df.to_csv(f"{args.out}.chr{C+1}.residuals.gz", index=None, sep='\t', compression='gzip')
    print("All done.")


def arg_rhe_total_no_se( arg_all, Traits, alpha=-1, nVectors=30, diploid=False, nThreads=N_THREADS, seed=1, debug=True ):
    """
    Estimates the total heritability using RHE-reg as in Yue Wu and S. Sankararaman (2018), but using the ARG.
    This implementation supports multiple phenotypes and one genetic component.

    Input
    ------
    arg_all: a list of ARGs, one for each chromosome
    Traits: a matrix of shape (N, N_pheno) with the phenotypes
    alpha: the value of the alpha parameter for the standardization of the matrices
    nVectors: the number of random vectors to use for the Hutchinson's trace estimator
    diploid: whether the arg is diploid or not
    N_THREADS: the number of threads to use for the matrix multiplication
    debug: whether to print debug information or not

    Output
    _______
    sigma_g_hat: a vector of shape (N_pheno,) with the estimated genetic variances
    sigma_e_hat: a vector of shape (N_pheno,) with the estimated environmental variances

    """
    t0 = time.perf_counter()
    N, N_pheno = Traits.shape
    # M = np.sum([arg.num_mutations() for arg in arg_all])
    np.random.seed(seed)  # Set the seed for reproducibility
    Z = np.random.normal(0, 1, size=(N,nVectors)) # initilise random vectors for the trace estimation
    numerators = np.zeros(N_pheno)
    trace_KTK = 0

    # working with chromosome-specific GRMs
    for arg in arg_all:

        arg_needle_lib.prepare_fast_multiplication(arg)
        GtU = arg_needle_lib.ARG_by_matrix_multiply_muts_fast_mt(arg, np.concatenate([Traits,Z], axis=1).T, standardize=True, alpha=alpha, diploid=diploid, n_threads=nThreads)
        GGtU = arg_needle_lib.ARG_by_matrix_multiply_samples_faster_mt(arg, GtU.T, standardize=True, alpha=alpha, diploid=diploid, n_threads=nThreads)
        GGtU = np.array(GGtU) # this now contains both GGtY and GGtZ

        for i in range(N_pheno):
            numerators[i] += np.dot(Traits[:,i], GGtU[:,i]) / arg.num_mutations()

        # Hutchinson's trace estimator using vectors from the multivariate standard normal distribution.
        traces = []
        for b in range(nVectors):
            traces.append( np.dot(GGtU[:,N_pheno+b],GGtU[:,N_pheno+b]) )

        trace_KTK += np.mean(traces) / arg.num_mutations()**2

    yTy_all = np.array([np.dot(x,x) for x in Traits.T])
    numerators -= yTy_all
    sigma_g_hat = numerators / (trace_KTK - N)
    sigma_e_hat = yTy_all/N - sigma_g_hat

    if debug:
        print(f"Done with RHE-reg, duration: { time.perf_counter() - t0:.2f}")
        print("Numerators:", numerators)
        print("Trace:", trace_KTK)
        print("SNP-heritabilities found by RHE-reg:")
        print(" ".join( format(x,".4f") for x in sigma_g_hat ) )

    return sigma_g_hat, sigma_e_hat


def arg_rhe_total_with_se( arg, gen_length, Traits, alpha=-1, jackknife_blocks=20, nVectors=30, diploid=False, rand_seed=123, debug=True ):
    """
    Estimates the total heritability using RHE-reg with standard errors using jackknife.
    See `arg_rhe_total_no_se` for more details.
    """

    print('='*26,"ARG-RHE with SEs (and K=1)",'='*26,sep='\n')
    t0 = time.time()
    N, nPheno = Traits.shape
    # M = arg.num_mutations()
    R = np.random.normal(0, 1, size=(N,nVectors)) # initialise random vectors for the trace estimator
    if debug: print(" ".join( format(np.mean(R[:,p]),".2f") for p in range(nVectors) ) )

    Z_all = np.zeros((jackknife_blocks, N, nVectors)) # will contain all the XXtR matrices of shape [N*B]
    H_all = np.zeros((jackknife_blocks, nPheno)) # will contain all ytXXty scalars
    Z_sum = np.zeros((N, nVectors))
    H_sum = np.zeros(nPheno)
    # sigma_e_jack = np.zeros( (jackknife_blocks, nPheno) ) # not sure if needed
    BS = gen_length // jackknife_blocks
    chunk_sizes = []
    print("Muts in each arg chunk:")
    for j in range(jackknife_blocks):
        M_start = j * BS
        M_stop = min(M_start + BS, gen_length)
        # prepare this jackknife block
        arg_chunk = arg_needle_lib.trim_arg( arg, M_start, M_stop )
        arg_chunk.populate_children_and_roots()
        arg_needle_lib.generate_mutations( arg_chunk, mu=2*1e-8, random_seed=rand_seed ) #, sorted=True )
        # arg_chunk.populate_mutations_on_edges()
        chunk_sizes.append( arg_chunk.num_mutations() )
        print(arg_chunk.num_mutations(), end=', ')
        # go through this arg and calculate the Z and H matrices
        Z_all[j,:,:], H_all[j,:] = arg_rhe_chunk_calculations( arg_chunk, Traits, R, alpha=alpha, diploid=diploid )
        # get the total sums
        Z_sum += Z_all[j,:,:] # one vector per phenotype
        H_sum += H_all[j,:] # one scalar per phenotype

    print("total_new vs old = {0:.4f}".format( np.sum(chunk_sizes)/arg.num_mutations() ) )
    print("Done with processing the ARG. Duration = {0:.2f} secs".format(time.time() - t0))

    M = np.sum(chunk_sizes)
    yTy = np.zeros(nPheno)
    for p in range(nPheno):
        yTy[p] = np.dot( Traits[:,p], Traits[:,p] )

    if debug: print(" ".join( format(yTy[p],".2f") for p in range(nPheno) ) )

    sigma_g_hat_total, sigma_e_hat_total, traces = rhe_solve_normal_equations( Z_sum, H_sum, yTy, M )

    if debug: print(" ".join( format(traces[p],".2f") for p in range(jackknife_blocks) ) )

    if debug:
        # print(" ".join( format(x,".3f") for x in [np.mean(numerators), np.mean(yTy), np.mean(traces)]) )
        print("Total/intermediate SNP-h2 estimates:")
        print(" ".join( format(x,".4f") for x in sigma_g_hat_total ) )

    # repeat for each jackknife; LOBO: leave one block out
    # get the trace, the GGtY product, and solve the normal equations
    sigma_g_jack = np.zeros( (jackknife_blocks, nPheno) )
    mean_jack = np.zeros(nPheno)
    # print("Jack-knife trace estimates:")
    for j in range(jackknife_blocks):
        Z_LOBO = Z_sum - Z_all[j]
        H_LOBO = H_sum - H_all[j]

        sigma_g_jack[j,:], _, traces = rhe_solve_normal_equations( Z_LOBO, H_LOBO, yTy, M - chunk_sizes[j] )
        mean_jack += sigma_g_jack[j,:] / jackknife_blocks
        # mean_jack += ( 1 - chunk_sizes[j] / M ) * sigma_g_jack[j,:]

    sigma_g_hat = jackknife_blocks * sigma_g_hat_total - mean_jack * (jackknife_blocks - 1)
    # sigma_g_hat = jackknife_blocks * sigma_g_hat_total - mean_jack
    sigma_e_hat = yTy/N - sigma_g_hat

    if debug:
        for p in range(5):
            print("\nΕstimates for pheno -",p)
            print("{0:.3f} | {1:.3f} ({2:.3f}) | {3:.3f}".format( sigma_g_hat[p], np.mean(sigma_g_jack[:,p]), np.std(sigma_g_jack[:,p]), sigma_g_hat_total[p] ))

    SE_all = np.zeros(nPheno)
    print("\nFinal estimates (and SE)")
    for p in range(nPheno):
        SE_all[p] = np.std(sigma_g_jack[:,p]) * np.sqrt(jackknife_blocks - 1)
        print( "{0:.4f} ({1:.4f})".format(sigma_g_hat[p], SE_all[p]), end=', ')

    print( "\nTotal duration = {0:.2f} secs".format(time.time() - t0) )
    print("="*26)
    return sigma_g_hat_total, sigma_e_hat_total, SE_all

def arg_rhe_chunk_calculations( arg, Traits, R, alpha=-1, diploid=True ):
    """
    This function calculates Z = XXtR and H = VtV where v = XtY, for a given (chunk) arg.
    We follow Pazokitoroudi et al. (2020), but work for one component (K=1) and multiple phenotypes (P).
    Thus X, R, Z, and H have dimensions NxM, NxB, NxB, and P respectively
    """

    N, P = Traits.shape
    _, B = R.shape
    M = arg.num_mutations()

    # we'll get XtR and XtY with one pass over the arg
    arg_needle_lib.prepare_fast_multiplication(arg)
    # shape for input matrix : [num_vectors x samples]
    GtU = arg_needle_lib.ARG_by_matrix_multiply_muts_fast_mt(arg, np.concatenate([R,Traits], axis=1).T, standardize=True, alpha=alpha, diploid=diploid, n_threads=N_THREADS)
    # now calculate XXtR
    # shape for input matrix : [num_vectors x mutations]
    GGtU = arg_needle_lib.ARG_by_matrix_multiply_samples_faster_mt(arg, GtU.T, standardize=True, alpha=alpha, diploid=diploid, n_threads=N_THREADS)
    # calculate vTv
    H = np.zeros(P)
    for i in range(P):
        H[i] = np.dot( Traits[:,i], GGtU[:,B+i] )

    return GGtU[:,:B], H


def rhe_solve_normal_equations( U, H, yTy, nMuts ):
    nSamples, nVectors = U.shape
    nPheno = len(yTy)
    assert len(H) == len(yTy), "Wrong input for phenotypes!"
    traces=[]
    for b in range(nVectors):
        traces.append( np.dot(U[:,b],U[:,b]) / nMuts**2 )

    numerators = np.zeros(nPheno)
    for p in range(nPheno):
        numerators[p] = H[p]/nMuts - yTy[p]

    # here we assume that tr(K) = nSamples, which is approximately correct
    sigma_g_hat = numerators / ( np.mean(traces) - nSamples )
    # adjust for haploid/diploid samples when calculating sigma_e
    return sigma_g_hat, yTy/nSamples/2 - sigma_g_hat, traces

#%% LMM-LOCO stuff

def arg_loco( arg_all, Traits, VC, cg_params, blup=False ):
    """
    The main function for ARG-based LMM-LOCO step1.

    Input
    ------
    arg_all : list
        A list of ARGs, one for each chromosome.
    Traits : np.ndarray
        A matrix of shape (N, N_phen) with the phenotypes.
    VC : np.ndarray
        A matrix of shape (K, N_phen) with the variance components.
    cg_params : dict
        A dictionary with parameters used in the conjugate gradient iteration.

    Output
    -------
    residuals_all : dict
        A dictionary with the residuals for each chromosome.
    calibrators : dict
        A dictionary with the calibration factors for each phenotype.

    """
    ############## initialise stuff for the conjugate gradients ###############
    N_calibr = cg_params["N_calibr"]
    N_chr = len(arg_all)
    N, N_phen = Traits.shape
    assert N_phen == VC.shape[1], "Mismatch in the number of traits and VC estimates!"
    # select all the non_causal variants that will be used for calibration
    RHS = np.zeros((N, N_phen*(N_chr + N_calibr*N_chr))) # for all the systems we will solve Nt*Nc*(1+Ncalibr)
    # example for N_calibr=2:
    # [ [y1, x1;1, x1;2], [y1, x2;1, x2;2], ..., [y1, x22;1, x22;2] ] and repeat for y2, y3, ... yT
    print("Getting non-causal variants for estimating the calibration factors...")
    col=0
    for T in range(N_phen):
        for C in range(N_chr): # NOT equivalent to `for C in chr_map.keys():` as we follow index {0..21}
            RHS[:, col] = Traits[:,T];
            col += 1
            if N_calibr > 0:
                temp, _ = find_noncausal( arg_all[C], Traits[:,T], np.arange(N), N_calibr )
                for i in range(N_calibr):
                    RHS[:, col] = temp[:,i]
                    col += 1

    if blup:
        # add the genome-wide systems (one per phenotype)
        RHS = np.concatenate([RHS, Traits], axis=1)
        print("Will also calculate genome-wide BLUPs.")

    nrhs  = RHS.shape[1]
    print("{0} RHS in total are set for the linear systems.".format(nrhs))

    threshold = np.linalg.norm(RHS, ord=2, axis=0)*1e-5 # criteria for stopping the CG iteration; was x1e-5 originally
    start = time.time()
    CG_pack, norms, q = solve_conj_grad( arg_all, RHS, VC, threshold, cg_params, blup ) # run the main part
    print("CG is done; Total time for {1} systems and {2} iterations: {0:.2f} mins.\n".format( ( time.time() - start)/60, CG_pack.shape[1], q+1) )

    N=Traits.shape[0]
    calibrators = {}
    col = 0
    for T in range(N_phen):
        print("\nResults for pheno-{0} with total h2={1:.3f}".format( T, VC[0,T] ))
        y = RHS[:,col]
        Vinv_y = CG_pack[:,col];
        col += 1
        prospectiveStat = []; uncalibratedRetrospectiveStat = [] ## as in bolt
        gamma = []
        # unpack the results from CG; these are ordered as [chr1-res, chr1-x1, ..., chr2-res, chr2-x1,..., chr22-x3]
        for C in range(N_chr):
            for i in range(N_calibr):
                Xm = RHS[:, col] # get the original SNP
                temp = (Xm.T.dot(Vinv_y))**2
                # prospective = N * (x^T V^-1 y)^2 / (x^T V^-1 x * y^T V^-1 y)
                prospectiveStat.append( N*temp/ Xm.T.dot(CG_pack[:,col]) /np.dot(y, Vinv_y) )
                uncalibratedRetrospectiveStat.append( N*temp/Xm.T.dot(Xm)/Vinv_y.dot(Vinv_y) ) # norm(Xm)^2 * norm(VinvY)^2
                gamma.append( Xm.T.dot(CG_pack[:,col])/Xm.dot(Xm) )
                col += 1
        print("Grammar-gamma approximation = {0:.4f} ({1:.4f})".format( np.mean(gamma), np.std(gamma)) )

        calibration = (sum(uncalibratedRetrospectiveStat) / sum(prospectiveStat)).item()
        temp = np.array(uncalibratedRetrospectiveStat)/np.array(prospectiveStat)
        print("AvgPro: {0:.3f} AvgRetro: {1:.3f} Calibration: {2:.3f} ({3:.3f}) ({4} SNPs).".format(np.mean(prospectiveStat), np.mean(uncalibratedRetrospectiveStat), calibration, np.std(temp), len(prospectiveStat)))
        print("Ratio of medians: {0:.4f} | Median of ratios: {1:.4f}".format(  np.median(uncalibratedRetrospectiveStat)/np.median(prospectiveStat), np.median(temp))   )
        print("Mean, median and std of ratios = {0:.6f}, {1:.6f}, {2:.6f}".format(np.mean(temp), np.median(temp), np.std(temp) ))

        calibrators[ str(T+1) ] = calibration

    residuals_all = {}
    for C in range(N_chr):
        residuals_all[C] = np.zeros( Traits.shape )
        for T in range(N_phen):
            col = T * N_chr * (1+N_calibr) + C * (1+N_calibr)
            residuals_all[C][:,T] = CG_pack[:,col]

    if blup:
        print("\nFinally, calculating the BLUP for each phenotype...")
        print("This is obtained as K*Vinv*Y, where K is the GRM*sigma2/M.")
        blup_all = calculate_arg_blup( arg_all, CG_pack[:, nrhs - N_phen:], VC[0], alpha=-0.5, diploid=True, nThreads=N_THREADS )   # the last N_phen columns are the BLUPs
        return residuals_all, calibrators, blup_all
    else:
        return residuals_all, calibrators

def find_noncausal( arg, trait, samples, N_calibr=2, t_min=1, t_max=5 ):
    # TODO: needs to get the variants from the arg, not the geno array!
    #  samples should be in the {0,1,...,N-1} index
    # geno = arg_needle_lib.get_geno_chunk( arg, m_start=0, m_stop=1000, diploid=True )
    geno = arg_needle_lib.get_mutations_matrix( arg, from_pos=0, to_pos=1000 )
    geno = geno[:, np.arange(0,geno.shape[1],2)] + geno[:, np.arange(1,geno.shape[1],2)]
    varY = trait.dot(trait)
    X = np.zeros( (len(samples),N_calibr) )
    indx = np.random.permutation(geno.shape[0])
    scores = np.zeros(N_calibr)
    s = 0
    for v in indx:
        g = geno[ v, samples ]
        g = g - np.mean(g)
        varG = g.dot(g.T)
        if varG>0: # for the case we've sub-sampled and there's no variation
            test = (len(samples)-2)/( varY*varG/(g.T.dot(trait))**2 - 1 )
            if t_min < test <= t_max:
                X[:,s] = g
                scores[s] = test
                s += 1
        if s>=N_calibr:
            break

    return X, scores


def solve_conj_grad( arg, Z, VC, threshold, cg_params, blup=False ):
    """
    Solves systems of the form V*X = RHS using the conjugate gradients method, for V formed according to VC.
    Bottleneck: the number of iters is proportional to N, otherwise this is optimal for our purposes.
    """
    print("\nBegin CG iteration to calculate the residuals.")

    # Initialization
    solutions = np.zeros(Z.shape, dtype=np.float64) # this will converge to the solution
    residuals = Z.copy() # initialization, R(0) = B-A*X
    # Z = RHS.copy() # the "search directions", this is what we'll multiply the covar matrix with
    nrhs  = Z.shape[1]
    r2old = np.zeros((nrhs,1), Z.dtype)
    alpha = np.zeros((1,nrhs), Z.dtype) # for elementwise
    beta  = np.zeros((1,nrhs), Z.dtype) # for elementwise

    for q in range(cg_params["max_iters"]): # or until convergence
        tic = time.perf_counter()

        # form G*Gt*V (core calculations)
        VZ = covar_matmat_many( arg, Z, VC, cg_params, blup=blup )

        # check the residuals and update w.r.t CG;
        # note that the "residuals" here are the objects "r=b-Ax_k", not the Vinv_y things
        for j in range(nrhs):
            r2old[j] = residuals[:,j].dot( residuals[:,j] )
            alpha[0,j] = r2old[j] / Z[:,j].dot( VZ[:,j] )
        solutions += alpha * Z # update the solutions; elementwise
        residuals -= alpha * VZ
        norms = np.linalg.norm(residuals, ord=2, axis=0)
        if (norms <= threshold).all(): # TODO: avoid the extra calculations in some cases (more than necessary)
            break # is this the proper condition? we end up with more iterations otherwise
        for j in range(nrhs):
            beta[0,j] = residuals[:,j].dot(residuals[:,j].T) / r2old[j]
        Z = residuals + beta * Z # update Z; elementwise
        print("CG iteration {0}: time={1:.2f}, mean norm: {2:.6f}.".format(q+1, time.perf_counter()-tic, np.nanmean(norms) ), flush=True)

    return solutions, norms, q

def covar_matmat_many( arg_all, X, VC, cg_params, blup=False, N_THREADS=N_THREADS ):
    """
    Calculates the V*X, where V is formed according to VC, inside the ARG.
    Now supports genome-wide systems appended at the end of X.
    Here we work with a list of ARGs (e.g. different chunks or chroms).
    """
    N_calibr = cg_params["N_calibr"]
    N_phen = cg_params["N_pheno"]
    N_chr = len(arg_all)
    M_all = sum( [arg.num_mutations() for arg in arg_all ])
    nrhs = X.shape[1]
    nrhs_chrom_pheno = (N_chr-1)*(1+N_calibr) # the Ncolumns to update for one trait+one chrom

    VX = np.zeros(X.shape, dtype=np.float64)

    # Initialise with the environmental component, shared across all cases
    for T in range(N_phen):
        upd_indx_tmp = np.arange( T*N_chr*(N_calibr+1), (T+1)*N_chr*(N_calibr+1) )
        VX[:,upd_indx_tmp] = VC[-1,T]*X[:,upd_indx_tmp]
        if blup:
            # for genome-wide systems (last N_phen columns)
            VX[:, nrhs - N_phen + T] = VC[-1,T] * X[:, nrhs - N_phen + T]

    # create the weights for each GRM; this could go outside as it's constant wrt to the CG
    weights = np.zeros(nrhs)
    # the next could go outside as they are constant wrt to the CG iteration
    if N_chr > 1:
        i=0
        for T in range(N_phen):
            for C in range(N_chr):
                weights[i] = VC[0,T] / ( M_all - arg_all[C].num_mutations() )
                i+=1
                for S in range(N_calibr):
                    weights[i] = VC[0,T] / ( M_all - arg_all[C].num_mutations() )
                    i+=1
    else:
        weights = VC[0,:] / M_all

    # Genome-wide weights (used for BLUP)
    weights_gw = VC[0, :] / M_all

    for C, arg in enumerate(arg_all):
        # first get which cols correspond to this chromosome
        indx_chrom = np.arange( N_phen * N_chr * (1+N_calibr) )
        for T in range(N_phen):
            to_skip = np.arange( T*N_chr*(1+N_calibr) + C*(N_calibr+1), T*N_chr*(1+N_calibr) + (C+1)*(N_calibr+1))
            indx_chrom = np.setxor1d( indx_chrom, to_skip )

        # if we have genome-wide systems, then add the last N_phen columns
        if blup:
            indx_chrom = np.concatenate([indx_chrom, np.arange( N_phen * nrhs_chrom_pheno, N_phen * nrhs_chrom_pheno + N_phen )])

        # then get the chrom-based GRM
        arg_needle_lib.prepare_fast_multiplication(arg)
        GtX = arg_needle_lib.ARG_by_matrix_multiply_muts_fast_mt( arg, X[:,indx_chrom].T, standardize = True,
                                                             alpha=cg_params["alpha"], diploid=cg_params["diploid"], n_threads=N_THREADS )
        GGtX = arg_needle_lib.ARG_by_matrix_multiply_samples_faster_mt( arg, GtX.T, standardize = True,
                                                             alpha=cg_params["alpha"], diploid=cg_params["diploid"], n_threads=N_THREADS )

        # finally, update the corresponding cols of VX, for each trait
        for T in range(N_phen):
            indx_trait = np.arange( T*N_chr*(1+N_calibr), (T+1)*N_chr*(1+N_calibr))
            indx_chrom = np.arange( T*N_chr*(1+N_calibr) + C*(N_calibr+1), T*N_chr*(1+N_calibr) + (C+1)*(N_calibr+1))
            upd_indx_1 = np.setxor1d(indx_trait, indx_chrom)
            upd_indx_2 = np.arange( T*nrhs_chrom_pheno, (T+1)*nrhs_chrom_pheno )
            VX[:,upd_indx_1] += GGtX[:,upd_indx_2] * weights[upd_indx_2]
            if blup:
                # for genome-wide systems (last N_phen columns)
                VX[:, nrhs - N_phen + T] += GGtX[:, nrhs_chrom_pheno * N_phen + T] * weights_gw[T]

    return VX

def calculate_arg_blup( arg_all, Z, sigma_g, alpha=-0.5, diploid=True, nThreads=1 ):
    """
    Calculates K*R, where K is the GRM*sigma2/M
    sigma_g should have be of size R.shape[1]
    """
    assert len(sigma_g) == Z.shape[1], "Mismatch between the number of sigmas and columns!"
    n_muts = 0
    for arg in arg_all:
        n_muts += arg.num_mutations()
        arg_needle_lib.prepare_fast_multiplication(arg)
        U = arg_needle_lib.ARG_by_matrix_multiply_muts_fast_mt(arg, Z.T, standardize=True, alpha=alpha, diploid=diploid, n_threads=nThreads)
        U = arg_needle_lib.ARG_by_matrix_multiply_samples_faster_mt(arg, U.T, standardize=True, alpha=alpha, diploid=diploid, n_threads=nThreads)
        U = np.array(U)

    return U * sigma_g / n_muts


if __name__ == "__main__":
    main()
# end-of-script