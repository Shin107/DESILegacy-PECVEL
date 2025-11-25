import fitsio
import numpy as np
import astropy
from multiprocessing import Pool, cpu_count
from astropy.table import Table, vstack
import glob
#import matplotlib
import psutil

mem = psutil.virtual_memory()
print(f"Total: {mem.total / 1e9:.2f} GB")
print(f"Available: {mem.available / 1e9:.2f} GB")
print(f"Used: {mem.used / 1e9:.2f} GB")
import os


path ='/storage/shadab/data/legacy_survey/dr9/randoms/randoms-1-*.fits'
files = glob.glob(path)
files = [f for f in files if os.path.basename(f) != "randoms-1-7.fits"]
print("Detected CPUs:", os.cpu_count())
print(files)
#files=['/storage/shadab/data/legacy_survey/dr9/randoms/randoms-allsky-1-0.fits']
#
def extract_info(file):
    try:
        with fitsio.FITS(file) as f:  # ensures each process opens independently
            rows  = f[1].get_nrows()
            n = min(int(5e6), rows)  # number of random rows to sample
            idx = np.random.choice(rows, size=n, replace=False)

            data = Table(f[1].read(columns=['RA','DEC','PHOTSYS','NOBS_G','NOBS_R','NOBS_Z',
 'PSFDEPTH_G',
 'PSFDEPTH_R',
 'PSFDEPTH_Z',
 'GALDEPTH_G',
 'GALDEPTH_R',
 'GALDEPTH_Z',
 'PSFDEPTH_W1',
 'PSFDEPTH_W2',
 'PSFSIZE_G',
 'PSFSIZE_R',
 'PSFSIZE_Z',
 'MASKBITS',
 'EBV',
 'HPXPIXEL'],rows = idx.astype(int)))#,rows=range(10000000)))
        return data
    except Exception as e:
        print(f"Error reading {file}: {e}")
        return None


dat=[]
# for file in files:
#     dat.append(extract_info(file))

with Pool(processes=cpu_count()) as pool:
    dat = pool.map(extract_info, files)

final = vstack(dat)
#print(final)
#print(final.colnames)
#final.write('(/user/animesh.sah/FP_CUTS/randoms.fits', overwrite=True)
output_file='/user/animesh.sah/FP_CUTS/randoms_new.fits'
# fitsio.write(output_file,np.array(final),clobber=True)
final.write(output_file, overwrite=True)
# final_array = np.array(final.as_array())  # if final is an Astropy Table
# fitsio.write(output_file, final_array, clobber=True)
print('writing:',len(final),'lines to the table of which',len(final[final['DEC']>-30]),'are in the region of interest')
