
import fitsio
import numpy as np
from astropy.table import Table
import healpy as hp
import argparse
parser = argparse.ArgumentParser(description='Create mask with specified nside and part')
parser.add_argument('-n','--nside', type=int, default=1024, help='HEALPix nside parameter (default: 1024)')
parser.add_argument('-p','--part', type=str, choices=['north', 'south', 'full'], default='both', help='Part to process: north, south, or both (default: both)')
args = parser.parse_args()

nside = args.nside
part = args.part
randoms = Table(fitsio.FITS('/user/animesh.sah/FP_CUTS/randoms_5M.fits')[1].read())
if part=='north':
    rand_table = randoms[randoms['PHOTSYS']=='N']
elif part=='south':
    rand_table = randoms[randoms['PHOTSYS']=='S']
else:
    rand_table = randoms
del(randoms)
rand_table = rand_table[rand_table['DEC'] > -30]
RA_random,DEC_random = rand_table['RA'], rand_table['DEC'] 
def points_to_map(ra_deg, dec_deg, nside, weights=None):
    theta = np.radians(90.0 - dec_deg)  
    phi   = np.radians(ra_deg)
    ipix  = hp.ang2pix(nside, theta, phi, nest=False)
    npix  = hp.nside2npix(nside)
    if weights is None:
        weights = np.ones_like(ipix, dtype=np.float64)
    m = np.bincount(ipix, weights=weights, minlength=npix).astype(np.float64)
    return m
rand_map = points_to_map(RA_random,DEC_random,nside)
mask = (rand_map >0).astype(float)
np.save(f'mask_{part}',mask)
