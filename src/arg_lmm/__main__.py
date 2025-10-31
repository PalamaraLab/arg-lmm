import click
import logging
import time

from arg_lmm.arg_lmm import arg_lmm_main
from arg_lmm.arg_rhe import arg_rhe_main

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@click.command()
@click.argument("arg_paths", nargs=-1)
@click.option("--pheno", "pheno_path", required=True, help="Input phenotypes file")
@click.option("--out", "-o", required=True, help="Output filename or prefix")
@click.option("--mu", default=1e-6, type=float, help="Mutation resampling rate (RHE only)")
@click.option("--alpha", default=-1.0, type=float, help="Alpha normalising exponent")
@click.option("--mac", default=1, type=float, help="Minimal MAC of resampled mutations to include (RHE only)")
@click.option("--seed", default=42, type=int, help="Random seed for reproducibility")
@click.option("--rhe", is_flag=True, help="Enable Haseman-Elston estimation")
@click.option("--ncalib", default=1, type=int, help="Number of calibration variants per chromosome (non RHE)")
@click.option("--nthreads", default=1, type=int, help="Number of threads (non RHE)")
@click.option("--blup", is_flag=True, help="Also compute genome-wide BLUPs (non RHE)")
def arg_lmm(arg_paths, pheno_path, out, mu, alpha, mac, seed, rhe, ncalib, nthreads, blup):
    start_time = time.time()
    if rhe:
        if len(arg_paths) != 1:
            logger.error("arg-lmm --rhe expects only one ARG path")
            exit()

        if nthreads != 1:
            logger.warning("--nthreads argument not applicable to RHE")

        if ncalib != 1:
            logger.warning("--ncalib argument not applicable to RHE")

        if blup:
            logger.warning("--blup flag not applicable to RHE")

        logger.info(f"Starting arg-lmm RHE with following parameters:")
        logger.info(f"  arg:        {arg_paths[0]}")
        logger.info(f"  pheno:      {pheno_path}")
        logger.info(f"  out:        {out}")
        logger.info(f"  mu:         {mu}")
        logger.info(f"  alpha:      {alpha}")
        logger.info(f"  mac:        {mac}")
        logger.info(f"  seed:       {seed}")

        arg_rhe_main(arg_paths[0], pheno_path, out, mu, alpha, mac, seed)
    else:
        if len(arg_paths) < 2:
            logger.error("arg-lmm expects at least two ARG paths")
            exit()

        if mu != 1e-6:
            logger.warning("--mu argument only applicable to RHE")

        if mac != 1:
            logger.warning("--mac argument only applicable to RHE")

        logger.info(f"Starting arg-lmm with following parameters:")
        logger.info(f"  arg paths:  {arg_paths}")
        logger.info(f"  pheno:      {pheno_path}")
        logger.info(f"  out prefix: {out}")
        logger.info(f"  ncalib:     {ncalib}")
        logger.info(f"  alpha:      {alpha}")
        logger.info(f"  seed:       {seed}")
        logger.info(f"  nthreads:   {nthreads}")
        logger.info(f"  blup:       {blup}")

        arg_lmm_main(arg_paths, pheno_path, out, ncalib, alpha, seed, nthreads, blup)

    logger.info(f"Done, in {time.time() - start_time} seconds")
