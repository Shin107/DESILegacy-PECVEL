import numpy as np
import treecorr
from astropy.io import fits
import fitsio
import numpy as np
import Corrfunc
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from Corrfunc.io import read_catalog
from Corrfunc.utils import convert_3d_counts_to_cf
from astropy.table import Table
from multiprocessing import Pool
import pyarrow.parquet as pq
from multiprocessing import Pool, cpu_count
import pyarrow as pa
import matplotlib.pyplot as plt
import scienceplots
import argparse
import multiprocessing
import sys
import healpy as hp
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

plt.style.use(['science','notebook','grid'])

parser = argparse.ArgumentParser(description='Select survey part (north or south).')
parser.add_argument('-p', '--part', choices=['north', 'south','full'], default='full',
                    help='Which part to process (default: full)')
parser.add_argument('-m','--method', choices=['treecorr','corrfunc'], default='treecorr',
                    help='Which method to use (default: treecorr)')
parser.add_argument('-e','--error', choices=['jackknife','bootstrap','none'], default='none',
                    help='Which error estimation method to use (default: none)')
args = parser.parse_args()
part = args.part
method = args.method
error = args.error

if part=='full':
    table_selection = Table(fitsio.FITS('table_match_final.fits')[1].read())
else:
    if part == 'north':
        table_selection = Table(fitsio.FITS('/user/animesh.sah/FP_CUTS/north_cuts_v9_zeropoint_shifted_0.0234.fits')[1].read())
    else:
        table_selection = Table(fitsio.FITS(f'../FP_CUTS/table_{part}_unique.fits')[1].read())

if part=='north':
    table_selection = table_selection[table_selection['DEC']>=32.375]
elif part=='south':
    table_selection = table_selection[table_selection['DEC']<32.375]
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

idx = np.random.choice(len(rand_table), sample_size, replace=False)

rand_table = rand_table[idx]
RA_random,DEC_random = rand_table['RA'], rand_table['DEC'] 
RA_data   = np.ascontiguousarray(np.array(RA_data, dtype=np.float64))
DEC_data  = np.ascontiguousarray(np.array(DEC_data, dtype=np.float64))
RA_random = np.ascontiguousarray(np.array(RA_random, dtype=np.float64))
DEC_random= np.ascontiguousarray(np.array(DEC_random, dtype=np.float64))

nbins = 200
nthreads=100
min_sep = 0.001  # degrees, example
max_sep = 10   # degrees
N = len(RA_data)
rand_N = len(RA_random)
print('Theta value ranging from',min_sep,'to',max_sep,'in',nbins,'bins')
print('Length of data is',N)
print('Length of Random is',rand_N)
print("Ratio is:",rand_N/N)


# def w_theta_corrfunc(ra_data,dec_data,ra_random,dec_random,min_sep=0.001,max_sep=100,nbins=0,nthreads=120):
#     bins = np.logspace(np.log10(min_sep), np.log10(max_sep), nbins+1 )
#     autocorr=1
#     N=len(ra)
#     N_rand = len(ra_random)
#     print(N_rand/N)
#     DD_counts = DDtheta_mocks(autocorr, nthreads, bins,ra_data, dec_data)
#     autocorr=0
#     DR_counts = DDtheta_mocks(autocorr, nthreads, bins,ra_data, dec_data,RA2=ra_random, DEC2=dec_random)
#     autocorr=1
#     RR_counts = DDtheta_mocks(autocorr, nthreads, bins,ra_random,dec_random)
#     wtheta = convert_3d_counts_to_cf(N, N, rand_N, rand_N, DD_counts, DR_counts,DR_counts, RR_counts)
#     return (bins[1:]+bins[:-1])/2,wtheta

def w_theta_corrfunc(ra_data,dec_data,ra_random,dec_random,min_sep=min_sep,max_sep=max_sep,nbins=nbins,nthreads=nthreads):
    #print(ra_data)
    #print("Shape of ra data",ra_data.shape)
    bins = np.logspace(np.log10(min_sep), np.log10(max_sep), nbins+1 )
    autocorr=1
    #N=len(ra)
    #N_rand = len(ra_random)
    #print(N_rand/N)

    DD_counts = DDtheta_mocks(autocorr, nthreads, bins,ra_data, dec_data)
    autocorr=0
    DR_counts = DDtheta_mocks(autocorr, nthreads, bins,ra_data, dec_data,RA2=ra_random, DEC2=dec_random)
    autocorr=1
    RR_counts = DDtheta_mocks(autocorr, nthreads, bins,ra_random,dec_random)
    wtheta = convert_3d_counts_to_cf(RA_data.size, RA_data.size, RA_random.size, RA_random.size, DD_counts, DR_counts,DR_counts, RR_counts)
    return (bins[1:]+bins[:-1])/2,wtheta
    




if method=='treecorr' and error=='none':
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
    np.save(f'/user/animesh.sah/w_theta_results/treecorr_{part}_{min_sep}_to_{max_sep}.npy', np.vstack([dd.meanr,xi,sig]))

elif method=='corrfunc' and error=='none':

    bins = np.logspace(np.log10(min_sep), np.log10(max_sep), nbins+1 )
    autocorr=1
    N=len(RA_data)
    print('Starting code')
    print('Length of data is',N)
    print('Length of Random is',rand_N)
    print("Ratio is:",rand_N/N)
    DD_counts = DDtheta_mocks(autocorr, nthreads, bins,RA_data, DEC_data)
    autocorr=0
    print('Done with DD counts')
    DR_counts = DDtheta_mocks(autocorr, nthreads, bins,RA_data, DEC_data,RA2=RA_random, DEC2=DEC_random)
    autocorr=1
    RR_counts = DDtheta_mocks(autocorr, nthreads, bins,RA_random,DEC_random)
    wtheta = convert_3d_counts_to_cf(RA_data.size, RA_data.size, RA_random.size, RA_random.size, DD_counts, DR_counts,DR_counts, RR_counts)
    print('W_theta',wtheta)
    np.save(f'/user/animesh.sah/w_theta_results/corrfunc_{part}_{min_sep}_to_{max_sep}_0.0234_v2.npy', np.vstack([(bins[1:]+bins[:-1])/2,wtheta]))

def hist_pix(ra,dec,nside=6):
    npix = hp.nside2npix(nside)
    pix = hp.ang2pix(nside, np.radians(90.-dec), np.radians(ra), lonlat=False)
    counts = np.bincount(pix, minlength=npix)
    return counts

def pixelize(ra,dec,nside=6):
    theta = np.ascontiguousarray((np.pi/2.0 - np.deg2rad(dec)).astype(np.float64))
    phi   = np.ascontiguousarray(np.deg2rad(ra).astype(np.float64))
    hpix = hp.ang2pix(nside, theta, phi,nest=False)
    return hpix
def prepare_jackknife(RA,DEC,nside=6):
    # takes in RA and DEC and returns the pixel indices and the usable indexes for jackknife
    counts=hist_pix(RA,DEC,nside=nside)
    counts_full = np.copy(counts)
    counts = counts[counts>0]
    mean = np.mean(counts)
    threshold = mean/2
    #counts_full = hist_pix(RA_data, DEC_data, nside=nside)
    mask = counts_full < threshold
    plot_map = mask.astype(float)   
    npix = pixelize(RA, DEC, nside=nside) ##pixels corresponding to RA AND DEC
    pixels = np.arange(0,12*nside**2)
    jk_indexes = pixels[~plot_map.astype(bool)]    ## pixels which can be used for jackknife
    indexes = np.where(np.isin(npix,jk_indexes))[0] ##indexes of the RA AND DEC which can be used for jackknife
    print('Total number of usable pixles are:',len(counts_full[~mask.astype(bool)]))
    return npix,jk_indexes, indexes

def leave_one_out_data(RA,DEC,index,nside=6):
    # takes in RA and DEC and the index of the pixel to be removed
    # returns the RA and DEC after removing the pixel
    npix,jk_indexes,indexes = prepare_jackknife(RA,DEC,nside=nside)
    mask = npix != index
    return RA[mask], DEC[mask]
def leave_one_out_random(RA,DEC,index,nside=6):
    # takes in RA and DEC and the index of the pixel to be removed
    # returns the RA and DEC after removing the pixel
    npix_random = pixelize(RA, DEC, nside=nside) ##pixels corresponding to RA AND DEC
    mask = npix_random != index
    return RA[mask], DEC[mask]


def w_bootstrapp(seed ):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(table_selection), int(len(table_selection)/2), replace=True)
    sampled = table_selection[idx]
    bins = np.logspace(np.log10(min_sep), np.log10(max_sep), nbins+1 )
    RA_data, DEC_data = sampled['RA'], sampled['DEC']
    RA_data = np.ascontiguousarray(np.array(RA_data, dtype=np.float64))
    DEC_data = np.ascontiguousarray(np.array(DEC_data, dtype=np.float64))
    print('done with sampling')
    print('max_sep',max_sep)
    print('min_sep',min_sep)
    print('nbins',nbins)
    print('length of RA data',len(RA_data)  )
    autocorr = 1
    DD_counts = DDtheta_mocks(autocorr, nthreads, bins, RA_data, DEC_data)
    print('Done with DD counts')
    autocorr = 0
    DR_counts = DDtheta_mocks(
        autocorr, nthreads, bins,
        RA_data, DEC_data,
        RA2=RA_random, DEC2=DEC_random
    )

    wtheta_boot = convert_3d_counts_to_cf(
        RA_data.size, RA_data.size, RA_random.size, RA_random.size,
        DD_counts, DR_counts,
        DR_counts, RR_counts
    )
    return wtheta_boot





def bootstrap_worker(seed):
    #print('began bootstrapping')
    # Ensure independent random sampling
    rng = np.random.default_rng(seed)
    #print(int(len(table_selection)))
    # print('Done with rng')
    # print(len(table_selection))
    # print('prblem:',int(len(table_selection)/10))
    idx = np.random.choice(len(table_selection), int(len(table_selection)), replace=True)
    sampled = table_selection[idx]
    # print(idx)
    # print(sampled)
    # print('length of sampled',len(sampled))
    bins = np.logspace(np.log10(min_sep), np.log10(max_sep), nbins+1 )
        
    RA_data,DEC_data = sampled['RA'], sampled['DEC']
    RA_data = np.ascontiguousarray(np.array(RA_data, dtype=np.float64))
    DEC_data = np.ascontiguousarray(np.array(DEC_data, dtype=np.float64))
    print('RA_data is', RA_data)
    np.save('RA_data.npy', RA_data)
    np.save('DEC_data.npy', DEC_data)
    print('DEC_data is', DEC_data)

    autocorr = 1
    DD_counts = DDtheta_mocks(autocorr, nthreads, bins, RA_data, DEC_data)
    
    print('DD counts',DD_counts)
    autocorr = 0
    DR_counts = DDtheta_mocks(autocorr, nthreads, bins,RA_data, DEC_data,RA2=RA_random, DEC2=DEC_random)
    print('DR counts',DR_counts)

    # Correlation function
    wtheta_boot = convert_3d_counts_to_cf(RA_data.size, RA_data.size, RA_random.size, RA_random.size,
                                          DD_counts, DR_counts,
                                          DR_counts, RR_counts)
    print('Done with wtheta')
    return wtheta_boot






if error!= 'none' and method == 'corrfunc':
    if error=='jackknife':
        nside=6
        _,jk_indexes,_= prepare_jackknife(RA_data,DEC_data,nside=nside)

        with multiprocessing.Pool(processes=50) as pool:
            data_jk = pool.starmap(leave_one_out_data, [(RA_data, DEC_data, index, nside) for index in jk_indexes])
            random_jk= pool.starmap(leave_one_out_random, [(RA_random, DEC_random, index, nside) for index in jk_indexes])
            w_jk=pool.starmap(w_theta_corrfunc, [(data_jk[i][0], data_jk[i][1], random_jk[i][0], random_jk[i][1]) for i in range(len(jk_indexes))])
        w_jk = np.array(w_jk)
        np.save(f'/user/animesh.sah/w_theta_results/corrfunc_{part}_{error}_patches_nside_{nside}_{len(jk_indexes)}_{min_sep}_to_{max_sep}_v2.npy', w_jk)
    elif error == 'bootstrap':
        num_bootstrap = 2000
        autocorr=1 
        bins = np.logspace(np.log10(min_sep), np.log10(max_sep), nbins+1 )

        RR_counts = DDtheta_mocks(autocorr, nthreads, bins,RA_random, DEC_random)
        print('Done with RR counts')
        seeds = np.random.SeedSequence().spawn(num_bootstrap)  # unique RNG seeds
        #print(seeds)
        #print([int(s.generate_state(1)[0]) for s in seeds])
        # with multiprocessing.Pool(processes=1) as pool:
        #     w_boot = pool.starmap(w_bootstrapp, [(s,) for s in seeds])
        total_cpus = os.cpu_count()  # e.g., 190
        with ThreadPoolExecutor(max_workers=total_cpus) as executor:
            futures = [executor.submit(bootstrap_worker, int(s.generate_state(1)[0])) for s in seeds]
            w_boot = [future.result() for future in as_completed(futures)]
        w_boot = np.array(w_boot)
        print('Done with bootstrapping')
        np.save(f'/user/animesh.sah/w_theta_results/corrfunc_{part}_{error}_num_bootstrap_{num_bootstrap}_{min_sep}_to_{max_sep}_test_by1.npy', w_boot)

        




        




elif method=='treecorr' and error!='none': 
    npatch=100

    if error=='jackknife':
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
        sig = np.sqrt(varxi)
        print('sigma',sig)

    elif error =='bootstrap':
        cat_rand = treecorr.Catalog(ra=RA_random, dec=DEC_random, ra_units='deg', dec_units='deg',npatch=100)
        cat_data = treecorr.Catalog(ra=RA_data, dec=DEC_data, ra_units='deg', dec_units='deg',patch_centers=cat_rand.patch_centers)
        print('Starting Bootstrap error estimation')
        rr =  treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins ,sep_units='deg' ,  bin_type='Log'  , var_method='bootstrap',num_bootstrap=5000,cross_patch_weight='geom')
        dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins ,sep_units='deg'  ,  bin_type='Log'  , var_method='bootstrap',num_bootstrap=5000,cross_patch_weight='geom')# bins are spaced evenly in log10(sep)
        dr =  treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins ,sep_units='deg' ,  bin_type='Log'  , var_method='bootstrap',num_bootstrap=5000,cross_patch_weight='geom')
        dd.process(cat_data,num_threads=nthreads)
        dr.process(cat_data, cat_rand,num_threads=nthreads)
        rr.process(cat_rand,num_threads=nthreads)
        xi, varxi = dd.calculateXi(rr=rr, dr=dr)
        sig = np.sqrt(varxi)
        print('sigma',sig)
    np.save(f'/user/animesh.sah/w_theta_results/treecorr_{part}_{error}_patches_{npatch}_{min_sep}_to_{max_sep}.npy', np.vstack([dd.meanr,xi,sig]))




        
