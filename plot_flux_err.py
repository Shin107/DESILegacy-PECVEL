import numpy as np
import os
from multiprocessing import Pool, cpu_count
import os
import resource
import time
from astropy.table import Table, vstack
import fitsio
import argparse
import matplotlib.pyplot as plt
from scipy.stats import norm
import scienceplots
plt.style.use(['science', 'notebook', 'grid'])
from functools import partial
start_time = time.time()

parser = argparse.ArgumentParser(description='Select survey part (north or south).')
parser.add_argument('-d', '--directory', choices=['north', 'south'], default='north',
                    help='Which part to process (default: full)')
parser.add_argument('-z', '--zeropoint_shift', type=float, default=0.0,
                    help='Zeropoint shift to apply to r band magnitudes (default: 0.0)')
parser.add_argument('-t','--test', action='store_true',
                    help='Run in test mode with limited files (default: False)')
parser.add_argument('-b','--bin_fluxes', action='store_true',
                    help='Bin fluxes after processing (default: False)')
parser.add_argument('-p','--process_files', action='store_true',
                    help='Process files to compute flux errors (default: False)')
parser.add_argument('--bin_type', choices=['quantile', 'log', 'linear'], default='quantile',
                    help='Method for binning fluxes (default: quantile)')
parser.add_argument('-m','--magnitude_space',action = 'store_true')
args = parser.parse_args()
directory= args.directory
zpt = args.zeropoint_shift
test_mode = args.test
bin_fluxes = args.bin_fluxes
process  = args.process_files
bin_type = args.bin_type
mag_space = args.magnitude_space


def get_representative_sample(file_list, flux_col='flux', target_size=10_000_000, 
                               max_files=None, random_sample=True):    
    flux_samples = []
    total_sampled = 0
    if max_files is None:
        max_files = len(file_list)
    
    if random_sample and len(file_list) > max_files:
        file_indices = np.random.choice(len(file_list), max_files, replace=False)
        files_to_sample = [file_list[i] for i in sorted(file_indices)]
    else:
        step = max(1, len(file_list) // max_files)
        files_to_sample = file_list[::step][:max_files]
    
    print(f"Sampling from {len(files_to_sample)} files to get {target_size:,} flux values...")
    
    per_file = target_size // len(files_to_sample)
    
    for i, file in enumerate(files_to_sample):
        try:
            t = Table.read(file)
            
            # Get flux column
            flux = np.array(t[flux_col])
            
            # Remove invalid values
            flux = flux[np.isfinite(flux) & (flux > 0)]
            
            if len(flux) == 0:
                continue
            
            # Random subsample from this file
            if len(flux) > per_file:
                idx = np.random.choice(len(flux), per_file, replace=False)
                flux = flux[idx]
            
            flux_samples.append(flux)
            total_sampled += len(flux)
            
            if (i + 1) % 10 == 0:
                print(f"  Sampled {i+1}/{len(files_to_sample)} files, {total_sampled:,} objects")
            
            # Stop if we have enough
            if total_sampled >= target_size:
                break
                
        except Exception as e:
            print(f"  Warning: Could not read {file}: {e}")
            continue

    if len(flux_samples) == 0:
        raise ValueError("No valid flux data found in sampled files!")
    
    flux_sample = np.concatenate(flux_samples)
    
    # Shuffle to remove any ordering bias
    np.random.shuffle(flux_sample)
    
    # Trim to target size
    if len(flux_sample) > target_size:
        flux_sample = flux_sample[:target_size]
    
    print(f"✓ Collected {len(flux_sample):,} flux values")
    print(f"  Range: {flux_sample.min():.3e} to {flux_sample.max():.3e}")
    print(f"  Median: {np.median(flux_sample):.3e}")
    
    return flux_sample

def create_robust_bins(flux_sample, n_bins=75, method='quantile', 
                       extend_range=True, min_flux=None, max_flux=None):
    # Remove invalid values
    flux_clean = flux_sample[np.isfinite(flux_sample) & (flux_sample > 0)]
    if len(flux_clean) == 0:
        raise ValueError("No valid flux values in sample!")
    
    # Determine range
    if min_flux is None:
        min_flux = flux_clean.min()
    if max_flux is None:
        max_flux = flux_clean.max()
    
    # Extend range to catch outliers in full catalog
    if extend_range:
        flux_range = max_flux - min_flux
        min_flux = max(1e-6, min_flux - 0.1 * flux_range)
        max_flux = max_flux + 0.1 * flux_range
    
    # Create bins
    if method == 'quantile':
        # Quantile-based (equal count)
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(flux_clean, percentiles)
        
        # Extend edges
        if extend_range:
            bin_edges[0] = min_flux
            bin_edges[-1] = max_flux
            
    elif method == 'log':
        # Log-spaced
        bin_edges = np.logspace(np.log10(min_flux), np.log10(max_flux), n_bins + 1)
        
    elif method == 'linear':
        # Linear
        bin_edges = np.linspace(min_flux, max_flux, n_bins + 1)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    bin_edges = np.unique(bin_edges)
    
    print(f"\nCreated {len(bin_edges)-1} bins ({method} method):")
    print(f"  Range: [{bin_edges[0]:.3e}, {bin_edges[-1]:.3e}]")
    print(f"  First few edges: {bin_edges[:5]}")
    print(f"  Last few edges: {bin_edges[-5:]}")
    
    return bin_edges


def bin_in_flux(paths,bin_dict):
        fits_path = paths
        cols1 = ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1','FLUX_W2','FLUX_IVAR_G','FLUX_IVAR_R','FLUX_IVAR_Z','FLUX_IVAR_W1','FLUX_IVAR_W2','MW_TRANSMISSION_G','MW_TRANSMISSION_R','MW_TRANSMISSION_Z','MW_TRANSMISSION_W1','MW_TRANSMISSION_W2']
        final_table = Table(fitsio.FITS(fits_path)[1].read(columns=cols1))
        n_bins = bin_dict['FLUX_G'].shape[0] - 1
        bin_medians = {}
        bin_sigma_dict = {}
        bin_sigma_p16_dict = {}
        bin_sigma_p84_dict = {}
        for flux_col, ivar_col in [('FLUX_G', 'FLUX_IVAR_G'),
                                   ('FLUX_R', 'FLUX_IVAR_R'),
                                   ('FLUX_Z', 'FLUX_IVAR_Z'),
                                   ('FLUX_W1', 'FLUX_IVAR_W1'),
                                   ('FLUX_W2', 'FLUX_IVAR_W2')]:
            flux_raw = np.array(final_table[flux_col])
            ivar_raw = np.array(final_table[ivar_col])

            flux, sigma = return_flux(flux_raw, ivar_raw)
            bin_edges = bin_dict[flux_col]
            bin_indices = np.digitize(flux, bin_edges) - 1  # -1 to convert to 0-based index
            
            mask = (bin_indices >= 0) & (bin_indices < len(bin_edges) - 1)
            flux = flux[mask]
            sigma = sigma[mask]
            bin_indices = bin_indices[mask]
            nbins = len(bin_edges) - 1
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            sigma_p_16 = np.array([np.percentile(sigma[bin_indices == i], 16) if np.sum(bin_indices == i) > 20 else np.nan for i in range(nbins)])
            sigma_p_84 = np.array([np.percentile(sigma[bin_indices == i], 84) if np.sum(bin_indices == i) > 20 else np.nan for i in range(nbins)])
            sigma_med = np.array([np.median(sigma[bin_indices == i]) if np.sum(bin_indices == i) > 20 else np.nan for i in range(nbins)])
            bin_medians[flux_col] = bin_centers
            bin_sigma_flux = np.vstack([sigma_med, sigma_p_16, sigma_p_84]).T
            bin_sigma_dict[flux_col] = sigma_med
            bin_sigma_p16_dict[flux_col] = sigma_p_16
            bin_sigma_p84_dict[flux_col] = sigma_p_84
        return bin_medians, bin_sigma_dict, bin_sigma_p16_dict, bin_sigma_p84_dict, len(final_table)

            
def bin_in_mag(paths):
    bin_edges = np.linspace(8,32,500)
    #print('Atleast this is loaded')
    cols = ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1','FLUX_W2','FLUX_IVAR_G','FLUX_IVAR_R','FLUX_IVAR_Z','FLUX_IVAR_W1','FLUX_IVAR_W2','MW_TRANSMISSION_G','MW_TRANSMISSION_R','MW_TRANSMISSION_Z','MW_TRANSMISSION_W1','MW_TRANSMISSION_W2']
    table = Table(fitsio.FITS(paths)[1].read(columns=cols))
    n_bins = len(bin_edges) - 1
    bands = ['G','R','Z','W1','W2']
    sigma_dict= {}
    for band in bands: 
        flux_raw = table[f'FLUX_{band}']
        ivar_raw = table[f'FLUX_IVAR_{band}']
        sigma_raw = 1 / np.sqrt(ivar_raw)
        mw_transmission_raw = table[f'MW_TRANSMISSION_{band}']
        mag,sigma_mag = flux_mag(flux_raw,  mw_transmission_raw,sigma_raw)
        sigma_mag= sigma_mag[np.isfinite(mag)]

        mag = mag[np.isfinite(mag)]
        bin_indices = np.digitize(mag,bin_edges) -1
        mask = (bin_indices >= 0) & (bin_indices < len(bin_edges) - 1)
        mag = mag[mask]
        sigma_mag = sigma_mag[mask]
        bin_indices = bin_indices[mask]
        nbins = len(bin_edges) - 1
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        sigma_med = np.array([np.median(sigma_mag[bin_indices == i]) if np.sum(bin_indices == i) > 20 else np.nan for i in range(nbins)])

        sigma_dict[band] = sigma_med

    return sigma_dict,bin_centers





            
def flux_mag(flux,mw_transmission,sigma=None):
    flux_corr = flux / mw_transmission
    mag = -2.5 * np.log10(flux_corr) + 22.5
    if sigma is not None:
        sigma = 2.5 / np.log(10) * (sigma / flux_corr)
    return mag, sigma




def return_flux(flux_raw,ivar_raw):
    flux  = flux_raw
    sigma = 1 / np.sqrt(ivar_raw)

    good = np.isfinite(flux) & np.isfinite(sigma) & (flux > 0) & (sigma > 0)
    flux, sigma = flux[good], sigma[good]
    return flux, sigma
def compute_flux_error_stats(flux, sigma):
    # Define logarithmic bins for flux
    # bins = np.logspace(
    #     np.log10(np.percentile(flux, 0.01)),
    #     np.log10(np.percentile(flux, 99.9)),
    #     1000
    # )
    n_bins = 100
    bins = np.percentile(flux, np.linspace(0, 100, n_bins + 1))
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Digitize flux values into bins
    bin_indices = np.digitize(flux, bins)

    # Vectorized computation of statistics per bin
    sigma_med = np.array([
        np.median(sigma[bin_indices == i]) if np.sum(bin_indices == i) > 20 else np.nan
        for i in range(1, len(bins))
    ])

    sigma_p_16 = np.array([
        np.percentile(sigma[bin_indices == i], 16) if np.sum(bin_indices == i) > 20 else np.nan
        for i in range(1, len(bins))
    ])

    sigma_p_84 = np.array([
        np.percentile(sigma[bin_indices == i], 84) if np.sum(bin_indices == i) > 20 else np.nan
        for i in range(1, len(bins))
    ])
    sigma_p_2l = np.array([
        np.percentile(sigma[bin_indices == i], 2.5) if np.sum(bin_indices == i) > 20 else np.nan
        for i in range(1, len(bins))
    ])

    sigma_p_2u = np.array([
        np.percentile(sigma[bin_indices == i], 97.5) if np.sum(bin_indices == i) > 20 else np.nan
        for i in range(1, len(bins))
    ])
    return bin_centers, sigma_med, sigma_p_16, sigma_p_84 , sigma_p_2l, sigma_p_2u

def process_flux_band(flux, file_pairs, directory, method='quantile'):
    """Process a single flux band"""
    flux_sample = get_representative_sample(
        file_list=file_pairs,
        flux_col=flux,
        target_size=20_000_000,   # 20M sample
        max_files=150,              # Sample from random files
        random_sample=True         # Randomly pick files
    )
    
    bin_edges = create_robust_bins(
        flux_sample,
        n_bins=500,
        method=method,
        extend_range=True      # Adds ±10% buffer for outliers
    )
    
    os.makedirs('flux_bins_DR9', exist_ok=True)
    np.save(f'flux_bins_DR9/{len(bin_edges)}_bins_{flux}_{directory}_{method}.npy', bin_edges)
    
    return bin_edges,flux
    
def main():
    input_dir1 = f"/storage/shadab/data/legacy_survey/dr9/{directory}/sweep/9.0/"
    file_pairs = []
    for fname in os.listdir(input_dir1):
        if fname.endswith('.fits'):
            file_pairs.append(os.path.join(input_dir1, fname))
    if test_mode:
        file_pairs = file_pairs[:20]  # Limit to first 20 files for testing
        print("Running in test mode with limited files.")
    print(f"Total matched files: {len(file_pairs)}")

    # Run in parallel



    # Store bins for later use
    fluxes = ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2']
    bands =     ['G','R','Z','W1','W2']

    flux_bins = {}

    # Process in parallel
    if not mag_space:
        if bin_fluxes:
            method = bin_type  # 'quantile', 'log', or 'linear'
            print('Processing flux bands in parallel...')

            # Create partial function with fixed arguments
            process_func = partial(process_flux_band, file_pairs=file_pairs, 
                                directory=directory, method=method)
            with Pool(min(cpu_count(), len(fluxes))) as pool:
                results = pool.map(process_func, fluxes)
            
            print("Completed processing:")
            for bin_edges, flux_name in results:
                print(f"  {flux_name}: {len(bin_edges)} bins")
                flux_bins[flux_name] = bin_edges
        else: 
            print('Loading precomputed flux bin edges...')
            for flux in fluxes:
                flux_bins[flux] = np.load(f'flux_bins_DR9/501_bins_{flux}_{directory}_{bin_type}.npy') 
            print("Flux bin edges loaded.")
    

    

    if process:
        if not mag_space:
            print('Processing files to compute flux error statistics in parallel...')
            with Pool(cpu_count()) as pool:
                process_func = partial(bin_in_flux, bin_dict=flux_bins)
                results = pool.map(process_func, file_pairs)
            print('Total objects processed:', sum(r[-1] for r in results if r is not None))
            bin_medians_list = [r[0] for r in results if r is not None]
            bin_sigma_dict_list = [r[1] for r in results if r is not None]
            bin_sigma_p16_dict_list = [r[2] for r in results if r is not None]
            bin_sigma_p84_dict_list = [r[3] for r in results if r is not None]
            # Combine results across all files
            combined_bin_medians = {}
            combined_bin_sigma = {}
            combined_bin_sigma_p16 = {}
            combined_bin_sigma_p84 = {}
            for flux in fluxes:
                combined_bin_medians[flux] = bin_medians_list[0][flux]
                combined_bin_sigma[flux] = np.nanmean(
                    np.array([d[flux] for d in bin_sigma_dict_list]), axis=0
                )
                combined_bin_sigma_p16[flux] = np.nanmean(
                    np.array([d[flux] for d in bin_sigma_p16_dict_list]), axis=0
                )
                combined_bin_sigma_p84[flux] = np.nanmean(
                    np.array([d[flux] for d in bin_sigma_p84_dict_list]), axis=0
                )
            # Save combined statistics
            os.makedirs('flux_error_stats_DR9', exist_ok=True)
            stats_table = Table()
            for flux in fluxes:
                stats_table[f'bin_center_{flux}'] = combined_bin_medians[flux]
                stats_table[f'sigma_median_{flux}'] = combined_bin_sigma[flux]
                stats_table[f'sigma_p16_{flux}'] = combined_bin_sigma_p16[flux]
                stats_table[f'sigma_p84_{flux}'] = combined_bin_sigma_p84[flux]
            print('\n\n\n',type(stats_table))
            print(stats_table)
            stats_table.write(f'flux_error_stats_DR9/flux_error_stats_{directory}_{bin_type}.fits', format='fits', overwrite=True)
            print(f"Saved flux error statistics to flux_error_stats_DR9/flux_error_stats_{directory}_{bin_type}.fits")
        else: 
            with Pool(cpu_count()) as pool:
                results = pool.map(bin_in_mag, file_pairs)
            sigma_dict = {}
            for band in bands:
                sigma_list = [r[0][band] for r in results if r[0] is not None]
                combined_sigma = np.nanmedian(np.array(sigma_list), axis=0)
                sigma_dict[band] = combined_sigma
            stats_table = Table()
            bin_centers = results[0][1]  # All have the same bin centers
            stats_table['flux_bin_center'] = bin_centers
            for band in bands:
                stats_table[f'sigma_median_{band}'] = sigma_dict[band]
            stats_table.write(f'flux_error_stats_DR9/mag_error_stats_{directory}_mag.fits', format='fits', overwrite=True)
            print(f"Saved flux error statistics to flux_error_stats_DR9/mag_error_stats_{directory}_mag.fits")
          
    # table = results
    


    # tables = []
    # merged = vstack(table)  # table can be a tuple
    # output_file = f"notebooks/{directory}_merged_test.fits"
    # FLUXES = ['G', 'R', 'Z', 'W1','W2']
    # for flux_name in FLUXES:
    #     flux_col = f'FLUX_{flux_name}'
    #     ivar_col = f'FLUX_IVAR_{flux_name}'
    #     mask = merged[ivar_col] > 0
    #     print(f'{np.count_nonzero(mask)} out of {len(merged)} have positive {ivar_col}')
    #     flux,sigma = return_flux(merged[flux_col], merged[ivar_col])
    #     bin_centers, sigma_med, sigma_p_lower, sigma_p_upper, sigma_p_2l, sigma_p_2u = compute_flux_error_stats(flux, sigma)
    #     t = Table()
    #     t['band'] = [flux_name] * len(bin_centers)
    #     t['flux_bin_center'] = bin_centers
    #     t['sigma_median']    = sigma_med
    #     t['sigma_p16']       = sigma_p_lower
    #     t['sigma_p84']       = sigma_p_upper
    #     t['sigma_p2.5']    = sigma_p_2l
    #     t['sigma_p97.5']   = sigma_p_2u
        
    #     tables.append(t)
    # stats_table = vstack(tables)
    # stats_table.write(f'flux_error_stats_{directory}.fits', format='fits', overwrite=True)
    # mask_r = merged['FLUX_IVAR_R'] > 0
    # mask_z = merged['FLUX_IVAR_Z'] > 0
    # mask_g = merged['FLUX_IVAR_G'] > 0
    # print(f'{np.count_nonzero(mask_r)} out of {len(merged)} have positive FLUX_IVAR_R')
    # print(f'{np.count_nonzero(mask_z)} out of {len(merged)} have positive FLUX_IVAR_Z')
    # print(f'{np.count_nonzero(mask_g)} out of {len(merged)} have positive FLUX_IVAR_G')
    # #plt.plot(merged['FLUX_R'][mask_r], 1/np.sqrt(merged['FLUX_IVAR_R'][mask_r]), '.', alpha=1, label='R band')
    # plt.plot(merged['FLUX_Z'][mask_z], 1/np.sqrt(merged['FLUX_IVAR_Z'][mask_z]), '.', alpha=0.1, label='Z band')
    # #plt.plot(merged['FLUX_G'][mask_g], 1/np.sqrt(merged['FLUX_IVAR_G'][mask_g]), '.', alpha=0.1, label='G band')
    # plt.xscale('log')
    # plt.ylim(-0.01,np.percentile(1/np.sqrt(merged['FLUX_IVAR_Z'][mask_z]),99.5)+0.1)
    # plt.xlabel('Flux (Nanomaggies)')
    # plt.ylabel('Flux Variance (Nanomaggies^2)')
    # plt.legend()
    # plt.title(f'Flux Error vs Magnitude ({directory.capitalize()})')
    # plt.grid()
    # plt.savefig(f'flux_error_vs_magnitude_{directory}.png', dpi=300)

    

if __name__ == "__main__":
    main()

usage = resource.getrusage(resource.RUSAGE_SELF)

end_time = time.time()

print(f"Wall time: {end_time - start_time:.2f} s")
