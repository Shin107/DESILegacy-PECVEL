

## 01/01/2025 A Sah:  Added weight option for treecorr calculation (only for full sample currently)
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

plt.style.use(['science','notebook','grid'])

parser = argparse.ArgumentParser(description='Select survey part (north or south).')
parser.add_argument('-p', '--part', choices=['north', 'south','full'], default='full',
                    help='Which part to process (default: full)')
parser.add_argument('-m','--method', choices=['treecorr'], default='treecorr',
                    help='Which method to use (default: treecorr)')
parser.add_argument('-e','--error', choices=['jackknife','bootstrap','none'], default='none',
                    help='Which error estimation method to use (default: none)')



parser.add_argument('--add_weights', action='store_true',default=False,
                    help='If true, adds weights to the correlation function calculation (only for treecorr)')
parser.add_argument('--nside_weight', type=int, default=64,
                    help='nside used for  healpix weights')
parser.add_argument('--multiple_patch', action='store_true',default=False,
                    help='If true, runs for multiple patch numbers (only for treecorr with jackknife or bootstrap)')
parser.add_argument('--kind', type=str, choices=['equal_10','equal_5','partial_percentile','full_sample'], default=None,)
parser.add_argument('--suffix',type=str, default='',help='suffix for the output file name (only for treecorr with jackknife or bootstrap)')
parser.add_argument('--covariance', action='store_true', default=False, help='If true, saves the covariance matrix alongside the w_theta values (only for treecorr with jackknife or bootstrap)')

parser.add_argument('--file_path',action='store', default='/user/animesh.sah/FP_CUTS/data_cleaned_full_sample/cleaned_cut0.fits', help='Path to the input FITS file')
parser.add_argument('--base_path', help='Path to the input file or directory',action='store',default = None)
args = parser.parse_args()
part = args.part
method = args.method
error = args.error
WGT = args.add_weights
nside_weight = args.nside_weight
covariance = args.covariance
file_path = args.file_path
if args.kind != None and not WGT:
    import sys 
    sys.exit('Error: --kind option can only be used if --add_weights is set to True')

if args.base_path == None:
    base_path = Path(file_path).parent

else:
    base_path = Path(args.base_path)
    os.makedirs(base_path / 'w_theta_results', exist_ok=True)

table_selection = Table(fitsio.FITS(file_path)[1].read())


if part=='north':
    table_selection = table_selection[table_selection['DEC']>=32.375]
elif part=='south':
    table_selection = table_selection[table_selection['DEC']<32.375]


table_selection = table_selection[table_selection['DEC']>-30]



#table_selection = table_selection[table_selection['MAG_R']<17]
RA_data,DEC_data = table_selection['RA'], table_selection['DEC']
randoms = Table(fitsio.FITS('/user/animesh.sah/FP_CUTS/randoms_5M.fits')[1].read())
if part=='north':
    rand_table = randoms[randoms['PHOTSYS']=='N']
elif part=='south':
    rand_table = randoms[randoms['PHOTSYS']=='S']
else:
    rand_table = randoms
del(randoms)
rand_table = rand_table[rand_table['DEC'] > -30]

sample_size = 5*len(table_selection)

if len(rand_table) >= sample_size:
    idx = np.random.choice(len(rand_table), sample_size, replace=False)
    rand_table = rand_table[idx]
else:
    print('-'*20,'WARNING: len(randoms)<5*sample_size len randoms: '+str(len(rand_table)),'len(sample_size): '+str(sample_size),'-'*20)
RA_random,DEC_random = rand_table['RA'], rand_table['DEC'] 
RA_data   = np.ascontiguousarray(np.array(RA_data, dtype=np.float64))
DEC_data  = np.ascontiguousarray(np.array(DEC_data, dtype=np.float64))
RA_random = np.ascontiguousarray(np.array(RA_random, dtype=np.float64))
DEC_random= np.ascontiguousarray(np.array(DEC_random, dtype=np.float64))



print('-'*20,'Entered the main code','-'*20)
nbins = 50
nthreads=192
min_sep = 0.01  # degrees,
max_sep = 15  # degrees
N = len(RA_data)
rand_N = len(RA_random)
print('Theta value ranging from',min_sep,'to',max_sep,'in',nbins,'bins')
print('Length of data is',N)
print('Length of Random is',rand_N)
print("Ratio is:",rand_N/N)
import time 
a = time.time()

def coords_to_pix(ra_deg, dec_deg, nside,**kwargs):
    theta = np.radians(90.0 - dec_deg)  
    phi   = np.radians(ra_deg)
    ipix  = hp.ang2pix(nside, theta, phi,**kwargs)
    return ipix



if WGT:
    print('Adding weights to the calculation')
    print('Nside for weights is:',nside_weight)

    if part == 'full':
        print('Full sample: Applying north and south weights separately')
        weight_file_north = hp.read_map(f'/user/animesh.sah/DESI_PECVEL/systematic_weights/north_multilinear_systematic_weights_nside64_partial_percentile.fits')
        weight_file_south = hp.read_map(f'/user/animesh.sah/DESI_PECVEL/systematic_weights/south_multilinear_systematic_weights_nside64_partial_percentile.fits')
        north_mask_data = DEC_data >= 32.375
        south_mask_data = DEC_data < 32.375
        data_weights = np.empty(len(RA_data), dtype=np.float64)
        data_weights[north_mask_data] = weight_file_north[coords_to_pix(RA_data[north_mask_data], DEC_data[north_mask_data], nside_weight)]
        data_weights[south_mask_data] = weight_file_south[coords_to_pix(RA_data[south_mask_data], DEC_data[south_mask_data], nside_weight)]

        print('Mean of data weights:',np.mean(data_weights))
        data_weights = np.empty(len(RA_data), dtype=np.float64)
        data_weights[north_mask_data] = weight_file_north[coords_to_pix(RA_data[north_mask_data], DEC_data[north_mask_data], nside_weight)]
        data_weights[south_mask_data] = weight_file_south[coords_to_pix(RA_data[south_mask_data], DEC_data[south_mask_data], nside_weight)]
        print('Mean of data weights:',np.mean(data_weights))

        north_mask_rand = DEC_random >= 32.375
        south_mask_rand = DEC_random < 32.375
        rand_weights = np.empty(len(RA_random), dtype=np.float64)
        rand_weights[north_mask_rand] = weight_file_north[coords_to_pix(RA_random[north_mask_rand], DEC_random[north_mask_rand], nside_weight)]
        rand_weights[south_mask_rand] = weight_file_south[coords_to_pix(RA_random[south_mask_rand], DEC_random[south_mask_rand], nside_weight)]
    else:

        weight_file = hp.read_map(f'/user/animesh.sah/DESI_PECVEL/systematic_weights/{part}_multilinear_systematic_weights_nside64_full_sample.fits')

        if part=='south':
            # weight_file = hp.read_map(f'/user/animesh.sah/DESI_PECVEL/south_CSFD_systematic_weights_nside64.fits')
            weight_file = hp.read_map(f'/user/animesh.sah/DESI_PECVEL/systematic_weights/South_multilinear_systematic_weights_nside64.fits')
        elif part=='north':
            #weight_file = hp.read_map(f'/user/animesh.sah/DESI_PECVEL/north_CSFD_systematic_weights_nside64.fits')
            weight_file = hp.read_map(f'/user/animesh.sah/DESI_PECVEL/systematic_weights/North_multilinear_systematic_weights_nside64.fits')
        if args.kind is not None:
            weight_file = hp.read_map(f'/user/animesh.sah/DESI_PECVEL/systematic_weights/{part}_multilinear_systematic_weights_nside64_{args.kind}.fits')

        data_ipix = coords_to_pix(RA_data, DEC_data, nside_weight)
        data_weights = weight_file[data_ipix]
        rand_weights = weight_file[coords_to_pix(RA_random, DEC_random, nside_weight)]


if method=='treecorr' and error=='none':
    if WGT:
        cat_rand = treecorr.Catalog(ra=RA_random, dec=DEC_random, ra_units='deg', dec_units='deg')
        cat_data = treecorr.Catalog(ra=RA_data, dec=DEC_data, w=data_weights, ra_units='deg', dec_units='deg')
    else:
        cat_rand = treecorr.Catalog(ra=RA_random, dec=DEC_random, ra_units='deg', dec_units='deg')
        cat_data = treecorr.Catalog(ra=RA_data, dec=DEC_data, ra_units='deg', dec_units='deg')

    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins ,sep_units='deg'  ,  bin_type='Log'  )# bins are spaced evenly in log10(sep)
    dr =  treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins ,sep_units='deg' ,  bin_type='Log' )
    rr =  treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins ,sep_units='deg' ,  bin_type='Log'  )

    dd.process(cat_data,num_threads=nthreads)
    dr.process(cat_data, cat_rand,num_threads=nthreads)
    rr.process(cat_rand,num_threads=nthreads)



    xi, varxi = dd.calculateXi(rr=rr, dr=dr)
    sig = np.sqrt(varxi)
    print('sigma',sig)
    bins = np.logspace(np.log10(min_sep), np.log10(max_sep), nbins )

    #print(dd.meanr)
    #print(rr.meanr)
    #print(dd.meanr - dr.meanr)
    #print(dd.meanlogr)
    print(f'Saving the results to {base_path}/w_theta_results/theta_{min_sep}_to_{max_sep}_w_{WGT}_{args.suffix}.npy')
    np.save(f'{base_path}/w_theta_results/theta_{min_sep}_to_{max_sep}_w_{WGT}_{args.suffix}.npy', np.vstack([dd.meanr,xi,sig]))


elif method=='treecorr' and error!='none': 
    npatch=30

    if error=='jackknife':
        if WGT:
            cat_rand = treecorr.Catalog(ra=RA_random, dec=DEC_random, ra_units='deg', dec_units='deg',npatch=npatch)
            cat_data = treecorr.Catalog(ra=RA_data, dec=DEC_data, w=data_weights, ra_units='deg', dec_units='deg', patch_centers=cat_rand.patch_centers)
        else:
            cat_rand = treecorr.Catalog(ra=RA_random, dec=DEC_random, ra_units='deg', dec_units='deg',npatch=npatch)
            cat_data = treecorr.Catalog(ra=RA_data, dec=DEC_data, ra_units='deg', dec_units='deg',patch_centers=cat_rand.patch_centers)
        cat_rand.write_patch_centers('patch_centers.txt')
        print('Starting Jackknife error estimation')
        rr =  treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins ,sep_units='deg' ,  bin_type='Log'  , var_method='jackknife',cross_patch_weight='match')

        dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins ,sep_units='deg'  ,  bin_type='Log'  , var_method='jackknife',cross_patch_weight='match')# bins are spaced evenly in log10(sep)
        dr =  treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins ,sep_units='deg' ,  bin_type='Log'  , var_method='jackknife',cross_patch_weight='match')
        #print(rr)
        dd.process(cat_data,num_threads=nthreads)
        dr.process(cat_data, cat_rand,num_threads=nthreads)
        rr.process(cat_rand,num_threads=nthreads)
        xi, varxi = dd.calculateXi(rr=rr, dr=dr)
        cov = dd.cov
        sig = np.sqrt(varxi)
        print('sigma',sig)
        if args.covariance:
            np.save(f'{base_path}/w_theta_results/theta_{min_sep}_to_{max_sep}_jackknife_patches_{npatch}_w_{WGT}_{args.suffix}_covariance.npy', cov)
            #np.save(f'/user/animesh.sah/w_theta_results/treecorr_{part}_{error}_patches_{npatch}_{min_sep}_to_{max_sep}_w_{WGT}_{args.suffix}_covariance.npy', cov)   
        print(f'Saving the results to {base_path}/w_theta_results/theta_{min_sep}_to_{max_sep}_jackknife_patches_{npatch}_w_{WGT}_{args.suffix}.npy')
        #np.save(f'/user/animesh.sah/w_theta_results/treecorr_{part}_{error}_patches_{npatch}_{min_sep}_to_{max_sep}_w_{WGT}_{args.suffix}.npy', np.vstack([dd.meanr,xi,sig]))
        np.save(f'{base_path}/w_theta_results/theta_{min_sep}_to_{max_sep}_jackknife_patches_{npatch}_w_{WGT}_{args.suffix}.npy',  np.vstack([dd.meanr,xi,sig]))
        

    elif error =='bootstrap':
        
        N_bootstrap=1000
        import concurrent.futures
        def run_treecorr_bootstrap(n_patch):
            if WGT:
                cat_rand = treecorr.Catalog(ra=RA_random, dec=DEC_random, ra_units='deg', dec_units='deg', npatch=n_patch)
                cat_data = treecorr.Catalog(ra=RA_data, dec=DEC_data, w=data_weights, ra_units='deg', dec_units='deg', patch_centers=cat_rand.patch_centers)
            else:
                cat_rand = treecorr.Catalog(ra=RA_random, dec=DEC_random, ra_units='deg', dec_units='deg', npatch=n_patch)
                cat_data = treecorr.Catalog(ra=RA_data, dec=DEC_data, ra_units='deg', dec_units='deg', patch_centers=cat_rand.patch_centers)
            rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='deg', bin_type='Log', var_method='bootstrap', num_bootstrap=N_bootstrap, cross_patch_weight='geom')
            dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='deg', bin_type='Log', var_method='bootstrap', num_bootstrap=N_bootstrap, cross_patch_weight='geom')
            dr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='deg', bin_type='Log', var_method='bootstrap', num_bootstrap=N_bootstrap, cross_patch_weight='geom')
            dd.process(cat_data, num_threads=nthreads)
            dr.process(cat_data, cat_rand, num_threads=nthreads)
            rr.process(cat_rand, num_threads=nthreads)
            xi, varxi = dd.calculateXi(rr=rr, dr=dr)
            sig = np.sqrt(varxi)
            cov = dd.cov
            print(f'n_patch={n_patch}, sigma={sig}')
            return n_patch, dd.meanr, xi, sig, cov
        if args.multiple_patch:
            n_patch_list = [100]
            results = []
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(run_treecorr_bootstrap, n_patch) for n_patch in n_patch_list]
                for future in concurrent.futures.as_completed(futures):
                    n_patch, meanr, xi, sig, cov = future.result()
                    results.append((n_patch, meanr, xi, sig, cov))

            # Optionally, sort results by n_patch
            results.sort(key=lambda x: x[0])
            # Save each result separately or as a combined array as needed
            for n_patch, meanr, xi, sig, cov in results:
                #np.save(f'/user/animesh.sah/w_theta_results/treecorr_{part}_{error}_patches_{n_patch}_{N_bootstrap}_{min_sep}_to_{max_sep}_w_{WGT}_{args.suffix}.npy', np.vstack([meanr, xi, sig])) 
                print(f'Saving the results to {base_path}/w_theta_results/theta_{min_sep}_to_{max_sep}_bootstrap_patches_{n_patch}_w_{WGT}_{args.suffix}.npy')
                np.save(f'{base_path}/w_theta_results/theta_{min_sep}_to_{max_sep}_bootstrap_patches_{n_patch}_w_{WGT}_{args.suffix}.npy', np.vstack([meanr, xi, sig]))  
        else:
            n_patch  = 50 
            _,dd_meanr,xi,sig,cov = run_treecorr_bootstrap(n_patch)

                                     
            print(f'Saving the results to {base_path}/w_theta_results/theta_{min_sep}_to_{max_sep}_bootstrap_patches_{n_patch}_w_{WGT}_{args.suffix}.npy')  
            np.save(f'{base_path}/w_theta_results/theta_{min_sep}_to_{max_sep}_bootstrap_patches_{n_patch}_w_{WGT}_{args.suffix}.npy', np.vstack([dd_meanr,xi,sig]))
            if args.covariance:
                np.save(f'{base_path}/w_theta_results/theta_{min_sep}_to_{max_sep}_bootstrap_patches_{n_patch}_w_{WGT}_{args.suffix}_covariance.npy', cov)
        
b =time.time()
elapsed = b - a
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)
print(f"Total time taken: {minutes} min {seconds} sec")
