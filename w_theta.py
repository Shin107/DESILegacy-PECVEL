
import matplotlib.pyplot as plt
import numpy as np
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
import ray
import pyarrow as pa
import time
a= time.time()
random_compiled = fitsio.FITS('/user/animesh.sah/FP_CUTS/randoms_1M.fits')[1].read()
rand_table=Table(random_compiled)
sample_size =10*480896

rand_table = rand_table[rand_table['DEC'] > -30]
idx = np.random.choice(len(rand_table), sample_size, replace=False)

rand_table = rand_table[idx]

table_selection = Table(fitsio.FITS('table_match_final.fits')[1].read())
table_selection = table_selection
#rng = np.random.default_rng(34023)

#idx = rng.choice(len(table_selection), len(table_selection), replace=True)
RA_data,DEC_data = table_selection['RA'], table_selection['DEC']
RA_random,DEC_random = rand_table['RA'], rand_table['DEC']  
RA_data   = np.ascontiguousarray(np.array(RA_data, dtype=np.float64))
DEC_data  = np.ascontiguousarray(np.array(DEC_data, dtype=np.float64))
RA_random = np.ascontiguousarray(np.array(RA_random, dtype=np.float64))
DEC_random= np.ascontiguousarray(np.array(DEC_random, dtype=np.float64))
rand_N = len(RA_random)
nbins = 10
bins = np.logspace(-3, 1, nbins + 1) # note the +1 to nbins
nthreads = 10
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

# # Auto pairs counts in RR

# autocorr=1
# print('Done with DR counts')
autocorr=1
RR_counts = DDtheta_mocks(autocorr, nthreads, bins,RA_random,DEC_random)

# print('Done with RR counts')
wtheta = convert_3d_counts_to_cf(N, N, rand_N, rand_N, DD_counts, DR_counts,DR_counts, RR_counts)


# print('RR counts',RR_counts)
# print('DR counts',DR_counts)
# print('DD counts',DD_counts)
print('W_theta',wtheta)



import numpy as np
from multiprocessing import Pool

ray.init()

table_ref     = ray.put(table_selection)
RA_rand_ref   = ray.put(RA_random)
DEC_rand_ref  = ray.put(DEC_random)
RR_counts_ref = ray.put(RR_counts)
@ray.remote

def w_bootstrapp(seed ,table_ref, RA_rand_ref, DEC_rand_ref, RR_counts_ref):
    rng = np.random.default_rng(seed)

    # Pull large data back inside worker
    table_selection = table_ref
    RA_random = RA_rand_ref
    DEC_random = DEC_rand_ref
    RR_counts = RR_counts_ref

    # Bootstrap sample (with replacement)
    idx = rng.choice(len(table_selection), len(table_selection), replace=True)
    sampled = table_selection[idx]

    RA_data, DEC_data = sampled['RA'], sampled['DEC']
    RA_data = np.ascontiguousarray(np.array(RA_data, dtype=np.float64))
    DEC_data = np.ascontiguousarray(np.array(DEC_data, dtype=np.float64))

    # Pair counts
    autocorr = 1
    DD_counts = DDtheta_mocks(autocorr, nthreads, bins, RA_data, DEC_data)

    autocorr = 0
    DR_counts = DDtheta_mocks(
        autocorr, nthreads, bins,
        RA_data, DEC_data,
        RA2=RA_random, DEC2=DEC_random
    )

    # Correlation function
    wtheta_boot = convert_3d_counts_to_cf(
        N, N, rand_N, rand_N,
        DD_counts, DR_counts,
        DR_counts, RR_counts
    )
    return wtheta_boot





def bootstrap_worker(seed):
    print('began bootstrapping')
    # Ensure independent random sampling
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(table_selection), len(table_selection), replace=True)
    sampled = table_selection[idx]
    print(idx)
    print(sampled)
    print('length of sampled',len(sampled))
    RA_data,DEC_data = sampled['RA'], sampled['DEC']
    RA_data = np.ascontiguousarray(np.array(RA_data, dtype=np.float64))
    DEC_data = np.ascontiguousarray(np.array(DEC_data, dtype=np.float64))

    autocorr = 1
    DD_counts = DDtheta_mocks(autocorr, nthreads, bins, RA_data, DEC_data)
    
    print('DD counts',DD_counts)
    autocorr = 0
    DR_counts = DDtheta_mocks(autocorr, nthreads, bins,RA_data, DEC_data,RA2=RA_random, DEC2=DEC_random)
    print('DR counts',DR_counts)

    # Correlation function
    wtheta_boot = convert_3d_counts_to_cf(N, N, rand_N, rand_N,
                                          DD_counts, DR_counts,
                                          DR_counts, RR_counts)
    print('Done with wtheta')
    return wtheta_boot





    

n_bootstrap = 10000


seeds = np.random.SeedSequence().spawn(n_bootstrap)  # unique RNG seeds

print([int(s.generate_state(1)[0]) for s in seeds])
futures = [
    w_bootstrapp.remote(int(s.generate_state(1)[0]),table_ref, RA_rand_ref, DEC_rand_ref, RR_counts_ref)
    for s in seeds
]

dat = ray.get(futures)


table = pa.Table.from_arrays([pa.array(dat)], names=["bootstrap_test"])
pq.write_table(table, "bootstrap_results.parquet")
print("Saved bootstrap results to bootstrap_results.parquet")
b=time.time()
print('Time taken in seconds:', b-a)
print('Times taken in minutes:', (b-a)/60)
print('Times taken in hours:', (b-a)/3600)