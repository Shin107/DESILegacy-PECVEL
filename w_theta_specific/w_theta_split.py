## 01/01/2025 A Sah:  Added weight option for treecorr calculation (only for full sample currently)
## [UPDATE]         Added --split_by support: compute w_theta in quantile or equal-width bins
##                  of any catalog column (MAG_R, MAG_G, Z, etc.)

import numpy as np
import treecorr
from astropy.io import fits
import fitsio
from astropy.table import Table
import pyarrow.parquet as pq
import pyarrow as pa
import matplotlib.pyplot as plt
import argparse
import healpy as hp
import os
import time
import concurrent.futures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Select survey part (north or south).')
parser.add_argument('-p', '--part', choices=['north', 'south', 'full'], default='full',
                    help='Which part to process (default: full)')
parser.add_argument('-m', '--method', choices=['treecorr', 'corrfunc'], default='treecorr',
                    help='Which method to use (default: treecorr)')
parser.add_argument('-e', '--error', choices=['jackknife', 'bootstrap', 'none'], default='none',
                    help='Which error estimation method to use (default: none)')
parser.add_argument('--add_weights', action='store_true', default=False,
                    help='Add systematic weights to the correlation function (treecorr only)')
parser.add_argument('--nside_weight', type=int, default=64,
                    help='HEALPix nside used for weights')
parser.add_argument('--multiple_patch', action='store_true', default=False,
                    help='Run for multiple patch numbers (treecorr + jackknife/bootstrap only)')
parser.add_argument('--kind', type=str,
                    choices=['equal_10', 'equal_5', 'partial_percentile', 'full_sample'],
                    default='partial_percentile')
parser.add_argument('--suffix', type=str, default='',
                    help='Suffix for output file names')
parser.add_argument('--covariance', action='store_true', default=False,
                    help='Save covariance matrix alongside w_theta values')

# --- NEW: split-by options ---------------------------------------------------
parser.add_argument('--split_by', type=str, default=None,
                    help='Catalog column to split on, e.g. MAG_R, MAG_G, Z_PHOT. '
                         'If not set, runs on the full (un-split) sample.')
parser.add_argument('--n_bins', type=int, default=5,
                    help='Number of bins to split into (default: 5)')
parser.add_argument('--bin_type', choices=['quantile', 'equal'], default='quantile',
                    help='"quantile" = equal-count bins (default); '
                         '"equal" = equal-width bins')
parser.add_argument('--bin_edges', type=float, nargs='+', default=None,
                    help='Optional: explicit bin edges, e.g. --bin_edges 16 17 18 19. '
                         'Overrides --n_bins and --bin_type.')
# -----------------------------------------------------------------------------

args = parser.parse_args()

part         = args.part
method       = args.method
error        = args.error
WGT          = args.add_weights
nside_weight = args.nside_weight
covariance   = args.covariance

if args.kind is not None and not WGT:
    import sys
    sys.exit('Error: --kind option can only be used if --add_weights is set to True')

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
if part == 'full':
    table_selection = Table(fitsio.FITS('table_match_final.fits')[1].read())
else:
    table_selection = Table(fitsio.FITS(f'../../FP_CUTS/table_{part}_unique.fits')[1].read())

if part == 'north':
    table_selection = table_selection[table_selection['DEC'] >= 32.375]
elif part == 'south':
    table_selection = table_selection[table_selection['DEC'] < 32.375]

# Load randoms
randoms = Table(fitsio.FITS('/user/animesh.sah/FP_CUTS/randoms_5M.fits')[1].read())
if part == 'north':
    rand_table = randoms[randoms['PHOTSYS'] == 'N']
elif part == 'south':
    rand_table = randoms[randoms['PHOTSYS'] == 'S']
else:
    rand_table = randoms
del randoms

rand_table = rand_table[rand_table['DEC'] > -30]

# ---------------------------------------------------------------------------
# Correlation parameters
# ---------------------------------------------------------------------------
nbins    = 50
nthreads = 192
min_sep  = 0.01   # degrees
max_sep  = 15     # degrees

print(f'Theta: {min_sep} to {max_sep} deg in {nbins} log-spaced bins')

# ---------------------------------------------------------------------------
# Helper: HEALPix pixel lookup
# ---------------------------------------------------------------------------
def coords_to_pix(ra_deg, dec_deg, nside, **kwargs):
    theta = np.radians(90.0 - dec_deg)
    phi   = np.radians(ra_deg)
    return hp.ang2pix(nside, theta, phi, **kwargs)


# ---------------------------------------------------------------------------
# Helper: build systematic weight arrays for a given data/random sub-sample
# ---------------------------------------------------------------------------
def get_weights(RA_d, DEC_d, RA_r, DEC_r):
    """Return (data_weights, rand_weights) arrays. None if WGT=False."""
    if not WGT:
        return None, None

    print(f'  Adding weights (nside={nside_weight})')
    if part == 'full':
        wn = hp.read_map('/user/animesh.sah/DESI_PECVEL/systematic_weights/'
                         'north_multilinear_systematic_weights_nside64_partial_percentile.fits')
        ws = hp.read_map('/user/animesh.sah/DESI_PECVEL/systematic_weights/'
                         'south_multilinear_systematic_weights_nside64_partial_percentile.fits')

        def _apply(ra, dec, wgt_n, wgt_s):
            wgt = np.empty(len(ra), dtype=np.float64)
            mn  = dec >= 32.375
            ms  = dec <  32.375
            wgt[mn] = wgt_n[coords_to_pix(ra[mn], dec[mn], nside_weight)]
            wgt[ms] = wgt_s[coords_to_pix(ra[ms], dec[ms], nside_weight)]
            return wgt

        dw = _apply(RA_d, DEC_d, wn, ws)
        rw = _apply(RA_r, DEC_r, wn, ws)
    else:
        if args.kind is not None:
            fname = (f'/user/animesh.sah/DESI_PECVEL/systematic_weights/'
                     f'{part}_multilinear_systematic_weights_nside64_{args.kind}.fits')
        elif part == 'south':
            fname = ('/user/animesh.sah/DESI_PECVEL/systematic_weights/'
                     'South_multilinear_systematic_weights_nside64.fits')
        else:
            fname = ('/user/animesh.sah/DESI_PECVEL/systematic_weights/'
                     'North_multilinear_systematic_weights_nside64.fits')

        wf = hp.read_map(fname)
        dw = wf[coords_to_pix(RA_d, DEC_d, nside_weight)]
        rw = wf[coords_to_pix(RA_r, DEC_r, nside_weight)]

    print(f'  Mean data weight: {np.mean(dw):.4f}')
    return dw, rw


# ---------------------------------------------------------------------------
# Core w_theta runner  (operates on a sub-table + matching random sub-sample)
# ---------------------------------------------------------------------------
def run_wtheta(RA_data, DEC_data, RA_random, DEC_random,
               data_weights, rand_weights,
               label='full'):
    """
    Compute w(theta) via treecorr for the supplied data/random arrays.
    label   : string used in printed output and saved filenames
    """
    N      = len(RA_data)
    rand_N = len(RA_random)
    print(f'\n--- Running w_theta  |  label={label}  |  N={N}  rand_N={rand_N}  ratio={rand_N/N:.1f} ---')

    # --- build treecorr catalogs ---
    cat_kw_d = dict(ra=RA_data,   dec=DEC_data,   ra_units='deg', dec_units='deg')
    cat_kw_r = dict(ra=RA_random, dec=DEC_random, ra_units='deg', dec_units='deg')
    if data_weights is not None:
        cat_kw_d['w'] = data_weights
    if rand_weights is not None:
        cat_kw_r['w'] = rand_weights

    corr_kw = dict(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                   sep_units='deg', bin_type='Log')

    out_dir = '/user/animesh.sah/w_theta_results'
    suf     = f'_{args.suffix}' if args.suffix else ''

    # -----------------------------------------------------------------------
    if method == 'treecorr' and error == 'none':
        cat_data = treecorr.Catalog(**cat_kw_d)
        cat_rand = treecorr.Catalog(**cat_kw_r)

        dd = treecorr.NNCorrelation(**corr_kw)
        dr = treecorr.NNCorrelation(**corr_kw)
        rr = treecorr.NNCorrelation(**corr_kw)

        dd.process(cat_data, num_threads=nthreads)
        dr.process(cat_data, cat_rand, num_threads=nthreads)
        rr.process(cat_rand, num_threads=nthreads)

        xi, varxi = dd.calculateXi(rr=rr, dr=dr)
        sig = np.sqrt(varxi)
        print('  sigma:', sig)

        fname = (f'{out_dir}/treecorr_{part}_{label}_{min_sep}_to_{max_sep}'
                 f'_w{WGT}{suf}.npy')
        np.save(fname, np.vstack([dd.meanr, xi, sig]))
        print(f'  Saved → {fname}')

    # -----------------------------------------------------------------------
    elif method == 'treecorr' and error == 'jackknife':
        npatch = 30

        cat_rand = treecorr.Catalog(**cat_kw_r, npatch=npatch)
        cat_data = treecorr.Catalog(**cat_kw_d, patch_centers=cat_rand.patch_centers)

        jk_kw = dict(**corr_kw, var_method='jackknife', cross_patch_weight='match')
        dd = treecorr.NNCorrelation(**jk_kw)
        dr = treecorr.NNCorrelation(**jk_kw)
        rr = treecorr.NNCorrelation(**jk_kw)

        dd.process(cat_data, num_threads=nthreads)
        dr.process(cat_data, cat_rand, num_threads=nthreads)
        rr.process(cat_rand, num_threads=nthreads)

        xi, varxi = dd.calculateXi(rr=rr, dr=dr)
        sig = np.sqrt(varxi)
        print('  sigma:', sig)

        base = (f'{out_dir}/treecorr_{part}_{label}_{error}_patches{npatch}'
                f'_{min_sep}_to_{max_sep}_w{WGT}{suf}')
        np.save(f'{base}.npy', np.vstack([dd.meanr, xi, sig]))
        print(f'  Saved → {base}.npy')
        if covariance:
            np.save(f'{base}_covariance.npy', dd.cov)
            print(f'  Saved → {base}_covariance.npy')

    # -----------------------------------------------------------------------
    elif method == 'treecorr' and error == 'bootstrap':
        N_bootstrap = 1000

        def _run_bootstrap(n_patch):
            cr = treecorr.Catalog(**cat_kw_r, npatch=n_patch)
            cd = treecorr.Catalog(**cat_kw_d, patch_centers=cr.patch_centers)
            bs_kw = dict(**corr_kw, var_method='bootstrap',
                         num_bootstrap=N_bootstrap, cross_patch_weight='geom')
            dd = treecorr.NNCorrelation(**bs_kw)
            dr = treecorr.NNCorrelation(**bs_kw)
            rr = treecorr.NNCorrelation(**bs_kw)
            dd.process(cd, num_threads=nthreads)
            dr.process(cd, cr, num_threads=nthreads)
            rr.process(cr, num_threads=nthreads)
            xi, varxi = dd.calculateXi(rr=rr, dr=dr)
            return n_patch, dd.meanr, xi, np.sqrt(varxi), dd.cov

        if args.multiple_patch:
            n_patch_list = [100]
            with concurrent.futures.ProcessPoolExecutor() as exe:
                futures = {exe.submit(_run_bootstrap, np): np for np in n_patch_list}
                for fut in concurrent.futures.as_completed(futures):
                    np_, meanr, xi, sig, cov = fut.result()
                    fname = (f'{out_dir}/treecorr_{part}_{label}_{error}'
                             f'_patches{np_}_{N_bootstrap}'
                             f'_{min_sep}_to_{max_sep}_w{WGT}{suf}.npy')
                    np.save(fname, np.vstack([meanr, xi, sig]))
                    print(f'  Saved → {fname}')
        else:
            npatch = 50
            _, meanr, xi, sig, cov = _run_bootstrap(npatch)
            base = (f'{out_dir}/treecorr_{part}_{label}_{error}_patches{npatch}'
                    f'_{N_bootstrap}_{min_sep}_to_{max_sep}_w{WGT}{suf}')
            np.save(f'{base}.npy', np.vstack([meanr, xi, sig]))
            print(f'  Saved → {base}.npy')
            if covariance:
                np.save(f'{base}_covariance.npy', cov)
                print(f'  Saved → {base}_covariance.npy')


# ---------------------------------------------------------------------------
# Build list of (sub_table, label) pairs to iterate over
# ---------------------------------------------------------------------------
def make_bins(table, col):
    """
    Return a list of (mask, label) tuples for the requested binning strategy.
    """
    values = np.array(table[col])

    # --- determine bin edges ---
    if args.bin_edges is not None:
        edges = np.array(args.bin_edges)
        print(f'Using manual bin edges for {col}: {edges}')
    elif args.bin_type == 'quantile':
        percentiles = np.linspace(0, 100, args.n_bins + 1)
        edges = np.nanpercentile(values, percentiles)
        print(f'Quantile edges for {col}: {edges}')
    else:  # equal-width
        edges = np.linspace(np.nanmin(values), np.nanmax(values), args.n_bins + 1)
        print(f'Equal-width edges for {col}: {edges}')

    bins = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        # include upper edge only in the last bin
        if i < len(edges) - 2:
            mask = (values >= lo) & (values < hi)
        else:
            mask = (values >= lo) & (values <= hi)
        label = f'{col}_{lo:.4g}_{hi:.4g}'
        count = np.sum(mask)
        print(f'  Bin {i+1}: [{lo:.4g}, {hi:.4g})  N={count}')
        bins.append((mask, label))

    return bins


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
t0 = time.time()

if args.split_by is None:
    # ---- no splitting: run on full sample ----
    sample_size = 5 * len(table_selection)
    idx         = np.random.choice(len(rand_table), sample_size, replace=False)
    rand_sub    = rand_table[idx]

    RA_d  = np.ascontiguousarray(np.array(table_selection['RA'],  dtype=np.float64))
    DEC_d = np.ascontiguousarray(np.array(table_selection['DEC'], dtype=np.float64))
    RA_r  = np.ascontiguousarray(np.array(rand_sub['RA'],         dtype=np.float64))
    DEC_r = np.ascontiguousarray(np.array(rand_sub['DEC'],        dtype=np.float64))

    dw, rw = get_weights(RA_d, DEC_d, RA_r, DEC_r)
    run_wtheta(RA_d, DEC_d, RA_r, DEC_r, dw, rw, label='full')

else:
    # ---- split by the requested column ----
    col = args.split_by
    if col not in table_selection.colnames:
        raise ValueError(f'Column "{col}" not found in table. '
                         f'Available columns: {table_selection.colnames}')

    print(f'\nSplitting by column: {col}  |  n_bins={args.n_bins}  '
          f'bin_type={args.bin_type}')
    bins = make_bins(table_selection, col)

    for mask, label in bins:
        sub_table = table_selection[mask]

        # draw a fresh random sub-sample sized to 5× this bin
        sample_size = 5 * len(sub_table)
        idx         = np.random.choice(len(rand_table), sample_size, replace=False)
        rand_sub    = rand_table[idx]

        RA_d  = np.ascontiguousarray(np.array(sub_table['RA'],  dtype=np.float64))
        DEC_d = np.ascontiguousarray(np.array(sub_table['DEC'], dtype=np.float64))
        RA_r  = np.ascontiguousarray(np.array(rand_sub['RA'],   dtype=np.float64))
        DEC_r = np.ascontiguousarray(np.array(rand_sub['DEC'],  dtype=np.float64))

        dw, rw = get_weights(RA_d, DEC_d, RA_r, DEC_r)
        run_wtheta(RA_d, DEC_d, RA_r, DEC_r, dw, rw, label=label)

# ---------------------------------------------------------------------------
elapsed = time.time() - t0
print(f'\nTotal time: {int(elapsed // 60)} min {int(elapsed % 60)} sec')