import numpy as np
import treecorr
from astropy.io import fits
import fitsio
import numpy as np
import numpy as np
from astropy.table import Table
from multiprocessing import Pool
import pyarrow.parquet as pq
from multiprocessing import Pool, cpu_count
import pyarrow as pa
import matplotlib.pyplot as plt
import scienceplots
import argparse
from pathlib import Path
import multiprocessing
import sys
import healpy as hp
import os
from concurrent.futures import ThreadPoolExecutor, as_completed



def compute_w_theta(data, randoms, min_sep = 0.01,max_sep = 15, err_bars = True, n_bins = 50, n_threads = 100):
    if err_bars:
        cat_rand = treecorr.Catalog(ra=randoms['RA'], dec=randoms['DEC'], ra_units='degrees', dec_units='deg',npatch =30)
        cat_data = treecorr.Catalog(ra=data['RA'], dec=data['DEC'], ra_units='degrees', dec_units='deg',patch_centers=cat_rand.patch_centers)
        rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=n_bins, sep_units='deg', bin_type='Log', var_method='jackknife', cross_patch_weight='match')
        dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=n_bins, sep_units='deg', bin_type='Log', var_method='jackknife', cross_patch_weight='match')
        dr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=n_bins, sep_units='deg', bin_type='Log', var_method='jackknife', cross_patch_weight='match')
        rr.process(cat_rand, num_threads=n_threads)
        dd.process(cat_data, num_threads=n_threads)
        dr.process(cat_data, cat_rand, num_threads=n_threads)
        xi,varxi = dd.calculateXi(rr=rr, dr=dr)
        cov = dd.cov 
        sig = np.sqrt(varxi)
        return dd.meanr,xi, sig, cov
    else:
        cat_rand = treecorr.Catalog(ra=randoms['RA'], dec=randoms['DEC'], ra_units='degrees', dec_units='deg')
        cat_data = treecorr.Catalog(ra=data['RA'], dec=data['DEC'], ra_units='degrees', dec_units='deg')
        rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=n_bins, sep_units='deg', bin_type='Log')
        dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=n_bins, sep_units='deg', bin_type='Log')
        dr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=n_bins, sep_units='deg', bin_type='Log')
        rr.process(cat_rand, num_threads=n_threads)
        dd.process(cat_data, num_threads=n_threads)
        dr.process(cat_data, cat_rand, num_threads=n_threads)
        xi = dd.calculateXi(rr=rr, dr=dr)
        return dd.meanr,xi


