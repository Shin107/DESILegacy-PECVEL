import numpy as np
import os
from multiprocessing import Pool, cpu_count
import os
import psutil
import resource
import time
from astropy.table import Table, hstack
import fitsio
import argparse
import matplotlib.pyplot as plt
import scienceplots
from scipy.stats import norm
plt.style.use(['science', 'notebook', 'grid'])
from matplotlib.colors import LogNorm


parser = argparse.ArgumentParser(description='Select survey part (north or south).')
parser.add_argument('-d', '--directory', choices=['north', 'south'], default='north',
                    help='Which part to process (default: full)')
parser.add_argument('-z', '--zeropoint_shift', type=float, default=0.0,
                    help='Zeropoint shift to apply to r band magnitudes (default: 0.0)')
parser.add_argument('-t','--test', action='store_true',
                    help='Run in test mode with limited files (default: False)')
args = parser.parse_args()
directory= args.directory
zpt = args.zeropoint_shift
test_mode = args.test

def process_file_pair(paths):
        fits_path, pz_path = paths
    
    # try:
        cols1 = ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1','FLUX_W2','FLUX_IVAR_G','FLUX_IVAR_R','FLUX_IVAR_Z','FLUX_IVAR_W1','FLUX_IVAR_W2']
        
        # Load data
        dr9_chunk = Table(fitsio.FITS(fits_path)[1].read(columns=cols1))

        # Compute magnitude
        # MAG = {}
        # FIBERMAG = {}
        # MAG_NOEXT={}
        # for i in ['G','R','Z','W1']:
        #     flux = np.array(dr9_chunk[f'FLUX_{i}'])
        #     #fiberflux= np.array(dr9_chunk[f'FIBERFLUX_{i}'])
        #     trans = np.array(dr9_chunk[f'MW_TRANSMISSION_{i}'])
        #     frac = flux / trans
        #     #fracfiber= fiberflux /trans
        #     mag = np.empty_like(frac)
        #     #magfiber = np.empty_like(fracfiber)
        #     mag_noext= np.empty_like(frac)
        #     np.log10(flux, out=mag_noext, where=(frac > 0))
        #     np.log10(frac, out=mag, where=(frac > 0))
        #     #np.log10(fracfiber, out=magfiber, where=(fracfiber > 0))
        #     MAG[i] = 22.5 - 2.5 * mag
        #     #FIBERMAG[i] = 22.5 - 2.5 * magfiber
        #     MAG_NOEXT[i] = 22.5 - 2.5 * mag_noext
        # for i in ['G','R','Z']:
        #     fiberflux= np.array(dr9_chunk[f'FIBERFLUX_{i}'])
        #     trans = np.array(dr9_chunk[f'MW_TRANSMISSION_{i}'])
        #     fracfiber= fiberflux /trans
        #     magfiber = np.empty_like(fracfiber)
        #     np.log10(fracfiber, out=magfiber, where=(fracfiber > 0))
        #     FIBERMAG[i] = 22.5 - 2.5 * magfiber
        # if zpt != 0.0 and 'north' in fits_path:
        #     MAG['R'] -= zpt  # Apply zeropoint shift to R band in north only
        #     MAG['G']-= zpt
        #     print(f"Applied zeropoint shift of {zpt} to R band in {fits_path}")
        # # Compute r_circ
        # e1, e2 = dr9_chunk['SHAPE_E1'], dr9_chunk['SHAPE_E2']
        # epsilon = np.sqrt(e1**2 + e2**2)
        # bba = (1 - epsilon) / (1 + epsilon)
        # r_circ = np.sqrt(bba) * dr9_chunk['SHAPE_R']
      
        # dr9_chunk['FLUX_G'] = MAG['G']
        # dr9_chunk['FLUX_R'] = MAG['R']
        # dr9_chunk['FLUX_Z'] = MAG['Z']
        # dr9_chunk['FLUX_W1'] = MAG['W1']
        # dr9_chunk['FLUX_W2'] = MAG['W2']
        # dr9_chunk['FLUX_VAR_G'] = dr9_chunk['FLUX_IVAR_G']
        # dr9_chunk['FLUX_VAR_R'] = dr9_chunk['FLUX_IVAR_R']
        # dr9_chunk['FLUX_VAR_Z'] = dr9_chunk['FLUX_IVAR_Z']
        # dr9_chunk['FLUX_VAR_W1'] = dr9_chunk['FLUX_IVAR_W1']
        # dr9_chunk['FLUX_VAR_W2'] = dr9_chunk['FLUX_IVAR_W2']
  
        

        
        final_table = dr9_chunk
        name = os.path.basename(fits_path).replace('.fits', '')

            
        return final_table


def return_flux(flux_raw,ivar_raw):
    flux  = flux_raw
    sigma = 1 / np.sqrt(ivar_raw)

    good = np.isfinite(flux) & np.isfinite(sigma) & (flux > 0) & (sigma > 0)
    flux, sigma = flux[good], sigma[good]
    return flux, sigma
def compute_flux_error_stats(flux, sigma):
    # Define logarithmic bins for flux
    bins = np.logspace(
        np.log10(np.percentile(flux, 0.01)),
        np.log10(np.percentile(flux, 99.9)),
        100
    )
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Digitize flux values into bins
    bin_indices = np.digitize(flux, bins)

    # Vectorized computation of statistics per bin
    sigma_med = np.array([
        np.median(sigma[bin_indices == i]) if np.sum(bin_indices == i) > 20 else np.nan
        for i in range(1, len(bins))
    ])

    sigma_p_lower = np.array([
        np.percentile(sigma[bin_indices == i], 16) if np.sum(bin_indices == i) > 20 else np.nan
        for i in range(1, len(bins))
    ])

    sigma_p_upper = np.array([
        np.percentile(sigma[bin_indices == i], 84) if np.sum(bin_indices == i) > 20 else np.nan
        for i in range(1, len(bins))
    ])
    return bin_centers, sigma_med, sigma_p_lower, sigma_p_upper

def main():
    input_dir1 = f"/storage/shadab/data/legacy_survey/dr9/{directory}/sweep/9.0/"
    input_dir2 = f"/storage/shadab/data/legacy_survey/dr9/{directory}/sweep/9.0-photo-z/"
    # if zpt != 0.0:
    #     output_file = f"/user/animesh.sah/FP_CUTS/{directory}_cuts_v11.fits"
    # else:
    #     output_file = f"/user/animesh.sah/FP_CUTS/{directory}_cuts_v11.fits"
    #os.makedirs(output_dir, exist_ok=True)

    file_pairs = []
    for fname in os.listdir(input_dir1):
        if fname.endswith(".fits") and "-pz" not in fname:
            pz_name = fname.replace(".fits", "-pz.fits")
            if os.path.exists(os.path.join(input_dir2, pz_name)):
                #output_path = os.path.join(output_dir, fname.replace(".fits", "_selected.fits"))
                file_pairs.append((os.path.join(input_dir1, fname),
                                   os.path.join(input_dir2, pz_name)))

 

    if test_mode:
        file_pairs = file_pairs[:20]  # Limit to first 20 files for testing
        print("Running in test mode with limited files.")
    print(f"Total matched files: {len(file_pairs)}")

    # Run in parallel
    with Pool(cpu_count()) as pool:
        results= pool.map(process_file_pair, file_pairs)
    results = [r for r in results if r is not None]


    table = results
    

    from astropy.table import vstack

    tables = []
    merged = vstack(table)  # table can be a tuple
    output_file = f"notebooks/{directory}_merged_test.fits"
    FLUXES = ['G', 'R', 'Z', 'W1','W2']
    for flux_name in FLUXES:
        flux_col = f'FLUX_{flux_name}'
        ivar_col = f'FLUX_IVAR_{flux_name}'
        mask = merged[ivar_col] > 0
        print(f'{np.count_nonzero(mask)} out of {len(merged)} have positive {ivar_col}')
        flux,sigma = return_flux(merged[flux_col], merged[ivar_col])
        bin_centers, sigma_med, sigma_p_lower, sigma_p_upper = compute_flux_error_stats(flux, sigma)
        t = Table()
        t['band'] = [flux_name] * len(bin_centers)
        t['flux_bin_center'] = bin_centers
        t['sigma_median']    = sigma_med
        t['sigma_p16']       = sigma_p_lower
        t['sigma_p84']       = sigma_p_upper
        tables.append(t)
    stats_table = vstack(tables)
    stats_table.write(f'flux_error_stats_{directory}.fits', format='fits', overwrite=True)

    #flux,sigma = return_flux(merged['FLUX_Z'], merged['FLUX_IVAR_Z'])
    # merged.write(output_file, format='fits', overwrite=True)
    # print(f"Merged table saved to {output_file}")
    # Filter out zero or negative inverse variance to avoid divide by zero
    mask_r = merged['FLUX_IVAR_R'] > 0
    mask_z = merged['FLUX_IVAR_Z'] > 0
    mask_g = merged['FLUX_IVAR_G'] > 0
    print(f'{np.count_nonzero(mask_r)} out of {len(merged)} have positive FLUX_IVAR_R')
    print(f'{np.count_nonzero(mask_z)} out of {len(merged)} have positive FLUX_IVAR_Z')
    print(f'{np.count_nonzero(mask_g)} out of {len(merged)} have positive FLUX_IVAR_G')
    #plt.plot(merged['FLUX_R'][mask_r], 1/np.sqrt(merged['FLUX_IVAR_R'][mask_r]), '.', alpha=1, label='R band')
    plt.plot(merged['FLUX_Z'][mask_z], 1/np.sqrt(merged['FLUX_IVAR_Z'][mask_z]), '.', alpha=0.1, label='Z band')
    #plt.plot(merged['FLUX_G'][mask_g], 1/np.sqrt(merged['FLUX_IVAR_G'][mask_g]), '.', alpha=0.1, label='G band')
    plt.xscale('log')
    plt.ylim(-0.01,np.percentile(1/np.sqrt(merged['FLUX_IVAR_Z'][mask_z]),99.5)+0.1)
    plt.xlabel('Flux (Nanomaggies)')
    plt.ylabel('Flux Variance (Nanomaggies^2)')
    plt.legend()
    plt.title(f'Flux Error vs Magnitude ({directory.capitalize()})')
    plt.grid()
    plt.savefig(f'flux_error_vs_magnitude_{directory}.png', dpi=300)
    plt.show()
    plt.figure(figsize=(8,6))
    print(min(merged['FLUX_Z'][mask_z]), max(merged['FLUX_Z'][mask_z]))
    binx = np.logspace(min(merged['FLUX_Z'][mask_z]), max(merged['FLUX_Z'][mask_z]), 100)
    #binx = np.linspace(min(merged['FLUX_Z'][mask_z]), max(merged['FLUX_Z'][mask_z]), 100)
    biny = np.linspace(min(1/np.sqrt(merged['FLUX_IVAR_Z'][mask_z])), max(1/np.sqrt(merged['FLUX_IVAR_Z'][mask_z])), 100)
    hist,xedges,yedges=np.histogram2d(merged['FLUX_Z'][mask_z], 1/np.sqrt(merged['FLUX_IVAR_Z'][mask_z]), bins=(binx, biny))
    plt.imshow(hist.T, origin='lower',
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           aspect='auto',cmap='Blues',norm=LogNorm())
    plt.colorbar(label='Counts')
    plt.xscale('log')
    plt.ylim(-0.01,np.percentile(1/np.sqrt(merged['FLUX_IVAR_Z'][mask_z]),99.5)+0.1)
    plt.savefig(f'flux_error_vs_magnitude_density_{directory}.png', dpi=300)
    # Save the merged Astropy table to a FITS file
    

if __name__ == "__main__":
    main()

usage = resource.getrusage(resource.RUSAGE_SELF)

end_time = time.time()

print(f"Memory usage: {mem_info.rss / 1024**2:.2f} MB")
print(f"Wall time: {end_time - start_time:.2f} s")
print(f"CPU percent: {process.cpu_percent(interval=1.0)} %")
print(f"User CPU time: {usage.ru_utime:.2f} s")
print(f"System CPU time: {usage.ru_stime:.2f} s")
print(f"Max memory usage: {usage.ru_maxrss / 1024:.2f} MB")