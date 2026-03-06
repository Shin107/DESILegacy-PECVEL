import glob
import fitsio
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import scienceplots
plt.style.use(['science', 'notebook', 'grid']) 
from scipy.optimize import curve_fit 


def _systematic_relation(ngal, nran, features, bin_edges,norm_factor):
    rands, _ = np.histogram(features, bins=bin_edges, weights=nran)

    data,  _ = np.histogram(features, bins=bin_edges, weights=ngal)
    vals = np.full(len(bin_edges) - 1, np.nan)
    good = (rands > 0) & (data > 0)
    vals[good] = (data[good] / rands[good]) * norm_factor
    return vals

def error_jackknife_patch(ngal, nran, features, num_bins=50, nside=6, jk_labels=None, ax=None,binning="linear", **kwargs):

    if binning == "percentile":
        bin_edges = np.nanpercentile(features, np.linspace(0, 100, num_bins+1))
    else:
        bin_edges = np.linspace(np.nanmin(features),np.nanmax(features),num_bins + 1)
    #bin_edges = np.nanpercentile(features, np.linspace(0, 100, num_bins+1))
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    if jk_labels is None:
        theta,phi  = hp.pix2ang(64, pix64, nest=False)
        jk_pix = hp.ang2pix(nside, theta, phi, nest=False)
        _,jk_labels = np.unique(jk_pix, return_inverse=True)
    n_jack = jk_labels.max() + 1
    #print(f'Number of jackknife patches: {n_jack}')
    # Create jackknife labels for the full dataset


    ###full_sample
    rands_full,_ = np.histogram(features, bins=bin_edges, weights=nran)
    data_full, _ = np.histogram(features, bins=bin_edges, weights=ngal)
    min_gals  = 50
    min_rands = 100
    valid_bins = (rands_full > min_rands) & (data_full > min_gals)
    norm_factor = rands_full[valid_bins].sum() / data_full[valid_bins].sum()
    full_vals = _systematic_relation(ngal, nran, features, bin_edges, norm_factor)
    full_vals[~valid_bins] = np.nan
    jk_vals = np.zeros((n_jack, len(full_vals)))  # shape if number of jackknife samples x number of bins
    # Perform jackknife resampling
    print(f'lentgh of ngal: {len(ngal)}, nran: {len(nran)}, features: {len(features)}, n_jack: {n_jack}, full_vals length: {len(full_vals)}, length of mask: {len(jk_labels)}')
    for k in range(n_jack):
        mask_jk = jk_labels != k
        jk_vals[k] = _systematic_relation(
            ngal[mask_jk],
            nran[mask_jk],
            features[mask_jk],
            bin_edges,
            norm_factor
        )




        
    jk_mean = np.nanmean(jk_vals, axis=0)
    jk_var  = (n_jack - 1) / n_jack * np.nansum(
        (jk_vals - jk_mean)**2, axis=0
    )
    jk_err = np.sqrt(jk_var)
    try:
        nanmask = np.isfinite(jk_mean) & np.isfinite(full_vals)
        
        assert np.allclose(jk_mean[nanmask], full_vals[nanmask], rtol=1e-2)
    except:
        print('Jackknife mean and full values do not match closely.\n Jackknife mean does not reproduce full estimate')
        print('Full vals:', full_vals[nanmask])
        print('Jackknife mean:', jk_mean[nanmask])

    if np.any(np.isnan(jk_err)):
        print('Warning: NaN values found in jackknife errors. Check data and binning.')
    if ax is None:
        return bin_centers,full_vals, jk_err, data_full, rands_full,n_jack,bin_edges
    # print('Jackknife errors:')
    # print(jk_err)
 
    ax.errorbar(
        bin_centers,
        full_vals,
        yerr=jk_err,
        fmt='.',
        
        alpha=0.5,capsize = 2,
        **kwargs
    )
    ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.5)

    return bin_centers,full_vals, jk_err , data_full, rands_full,n_jack,bin_edges



def finalize_data(full_vals,jk_err,data_full,rands_full,n_jack,max_frac_err=0.5,min_rand_factor=10):


    bad = np.zeros_like(full_vals, dtype=bool)
    bad_reasons = {}
    # ---NaNs--- 
    bad_nan = ~np.isfinite(full_vals) | ~np.isfinite(jk_err)
    bad |= bad_nan
    bad_reasons["nan"] = bad_nan

    # --- minimum galaxy counts ---
    bad_low_gal = data_full < 5 * n_jack
    bad |= bad_low_gal
    bad_reasons["low_gal"] = bad_low_gal

    # --- minimum random counts ---
    bad_low_rand = rands_full < min_rand_factor * n_jack
    bad |= bad_low_rand
    bad_reasons["low_rand"] = bad_low_rand

    # --- large fractional error ---
    frac_err = np.full_like(jk_err, np.nan)
    good = np.isfinite(full_vals) & (full_vals != 0)
    frac_err[good] = jk_err[good] / np.abs(full_vals[good])
    #print(data_full)
    #print(np.count_nonzero(bad_nan), np.count_nonzero(bad_low_gal), np.count_nonzero(bad_low_rand))
    
    # bad_frac_err = frac_err > max_frac_err
    # bad |= bad_frac_err
    # bad_reasons["large_frac_err"] = bad_frac_err

    usable = ~bad


    return usable, bad_reasons
