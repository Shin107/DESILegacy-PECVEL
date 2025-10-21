import numpy as np
import os
from multiprocessing import Pool, cpu_count
import psutil
import resource
import time
from astropy.table import Table, hstack, vstack
import fitsio
import argparse
import pickle
from scipy.optimize import minimize
import logging 
import Corrfunc
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from Corrfunc.io import read_catalog
from Corrfunc.utils import convert_3d_counts_to_cf
import warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Minimizer:
    def __init__(self):
        self.north_path = "/storage/shadab/data/legacy_survey/dr9/north/sweep/9.0/"
        self.north_path2 = "/storage/shadab/data/legacy_survey/dr9/north/sweep/9.0-photo-z/"
        self.south_path = "/storage/shadab/data/legacy_survey/dr9/south/sweep/9.0/"
        self.south_path2 = "/storage/shadab/data/legacy_survey/dr9/south/sweep/9.0-photo-z/"
        self.nthreads= cpu_count() - 2
        self.w_theta = None
        self.theta_bins = None
        self.cols_dr9 =  [ 'BRICKID','OBJID','BRICKNAME','RA', 'DEC', 'FLUX_G','FLUX_R','FLUX_Z','MW_TRANSMISSION_G','MW_TRANSMISSION_R','MW_TRANSMISSION_Z','SHAPE_E1','SHAPE_E2','SHAPE_R','SERSIC','TYPE','NOBS_G','NOBS_R','NOBS_Z','MASKBITS','FRACMASKED_G','FRACMASKED_R','FRACMASKED_Z','FRACFLUX_G','FRACFLUX_R','FRACFLUX_Z','FRACIN_G', 'FRACIN_R',
       'FRACIN_Z','GAIA_PHOT_G_MEAN_MAG','NOBS_G','NOBS_R','NOBS_Z','FIBERFLUX_G', 'FIBERFLUX_R', 'FIBERFLUX_Z','FLUX_W1','MW_TRANSMISSION_W1']
        self.cols_pz = [
            'Z_PHOT_MEDIAN', 'Z_PHOT_L95', 'Z_PHOT_U95'
        ]
        
        


    def _load_and_preprocess(self, paths): 
        fits_path, pz_path = paths
        dr9_chunk = Table(fitsio.FITS(fits_path)[1].read(columns=self.cols_dr9))
        dr9_chunk_pz = Table(fitsio.FITS(pz_path)[1].read(columns=self.cols_pz))
        for band in ['G', 'R', 'Z', 'W1']:
            flux = np.array(dr9_chunk[f'FLUX_{band}'])
            trans = np.array(dr9_chunk[f'MW_TRANSMISSION_{band}'])
            frac = flux / trans
            
            mag = np.empty_like(frac)
            mag_noext = np.empty_like(flux)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                np.log10(frac, out=mag, where=(frac > 0))
                np.log10(flux, out=mag_noext, where=(flux > 0))
            mag_final = np.full_like(frac, np.nan)
            mag_noext_final = np.full_like(flux, np.nan)
            
            valid_frac = (frac > 0) & np.isfinite(frac)
            valid_flux = (flux > 0) & np.isfinite(flux)
            
            mag_final[valid_frac] = 22.5 - 2.5 * mag[valid_frac]
            mag_noext_final[valid_flux] = 22.5 - 2.5 * mag_noext[valid_flux]
            
            dr9_chunk[f'MAG_{band}'] = mag_final
            dr9_chunk[f'MAG_NOEXT_{band}'] = mag_noext_final
        for band in ['G', 'R', 'Z']:
            fiberflux = np.array(dr9_chunk[f'FIBERFLUX_{band}'])
            trans = np.array(dr9_chunk[f'MW_TRANSMISSION_{band}'])
            fracfiber = fiberflux / trans
            magfiber = np.empty_like(fracfiber)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                np.log10(fracfiber, out=magfiber, where=(fracfiber > 0))
            
            dr9_chunk[f'FIBERMAG_{band}'] = 22.5 - 2.5 * magfiber
        e1, e2 = dr9_chunk['SHAPE_E1'], dr9_chunk['SHAPE_E2']
        epsilon = np.sqrt(e1**2 + e2**2)
        bba = (1 - epsilon) / (1 + epsilon)
        r_circ = np.sqrt(np.clip(bba, 0, None)) * dr9_chunk['SHAPE_R']
        
        dr9_chunk['R_CIRC'] = r_circ
        dr9_chunk['BBA'] = bba
        dr9_chunk['EPSILON'] = epsilon
        
        # Merge with photo-z
        final_table = hstack([dr9_chunk, dr9_chunk_pz])
        self.table= final_table
        return final_table
    def load_data(self, n_jobs =10) -> None:
        print("Loading all data into RAM...")
        file_pairs = []
        for fname in os.listdir(self.north_path):
            if fname.endswith(".fits") and "-pz" not in fname:
                pz_name = fname.replace(".fits", "-pz.fits")
                if os.path.exists(os.path.join(self.north_path2, pz_name)):
                    #output_path = os.path.join(output_dir, fname.replace(".fits", "_selected.fits"))
                    file_pairs.append((os.path.join(self.north_path, fname),
                                    os.path.join(self.north_path2, pz_name)))
        logger.info(f"Total file pairs to process: {len(file_pairs)}")
        with Pool(n_jobs) as pool:
            results= pool.map(self._load_and_preprocess, file_pairs)
        results = [r for r in results if r is not None]
        logger.info("Combining all chunks into a single table...")
        logger.info("Loaded the data")
        self.table = vstack(results)
        #return results 
    def BGS_mask(self,table:Table) -> Table:
        cuts2 = (
        (
            ((table['FIBERMAG_R'] < (5.1 + table['MAG_R'])) & (table['MAG_R'] <= 17.8)) |
            ((table['FIBERMAG_R'] < 22.9) & (table['MAG_R'] > 17.8) & (table['MAG_R'] < 20))
        ) &
        ((-1 < (table['MAG_G'] - table['MAG_R'])) &
         ((table['MAG_G'] - table['MAG_R']) < 4) &
         (-1 < (table['MAG_R'] - table['MAG_Z'])) &
         ((table['MAG_R'] - table['MAG_Z']) < 4)) &

        (table['NOBS_G'] > 0) &
        (table['NOBS_R'] > 0) &
        (table['NOBS_Z'] > 0) &

        (table['FLUX_G'] > 0) &
        (table['FLUX_R'] > 0) &
        (table['FLUX_Z'] > 0) &

        (
            ((table['GAIA_PHOT_G_MEAN_MAG'] - table['MAG_NOEXT_R']) > 0.6) |
            (table['GAIA_PHOT_G_MEAN_MAG'] == 0)
        ) &

        (~np.isin(table['MASKBITS'], [1, 12, 13])) &

        (table['FRACMASKED_G'] < 0.4) &
        (table['FRACMASKED_R'] < 0.4) &
        (table['FRACMASKED_Z'] < 0.4) &

        (table['FRACFLUX_G'] < 5) &
        (table['FRACFLUX_R'] < 5) &
        (table['FRACFLUX_Z'] < 5) &

        (table['FRACIN_G'] > 0.2) &
        (table['FRACIN_R'] > 0.2) &
        (table['FRACIN_Z'] > 0.2)
    )
        self.table = table[cuts2]


    

    def apply_cuts(self,table:Table) -> Table:
        cuts = (
            (self.table['MAG_R'] < 18) &
            ((self.table['MAG_G'] - self.table['MAG_R']) > 0.68) &
            ((self.table['MAG_G'] - self.table['MAG_R']) > (1.3 * (self.table['MAG_R'] - self.table['MAG_Z']) - 0.05)) &
            ((self.table['MAG_G'] - self.table['MAG_R']) < (2.0 * (self.table['MAG_R'] - self.table['MAG_Z']) - 0.15)) &
            (self.table['R_CIRC'] > 0) &
            ((1 - bba) < 0.7) &
            (((self.table['TYPE'] == 'SER') & (self.table['SERSIC'] > 2.5)) | (self.table['TYPE'] == 'DEV')) &
            (self.table['Z_PHOT_MEDIAN'] < 0.15) &
            (self.table['Z_PHOT_L95'] < 0.1))
        
        self.table = table[cuts]
    def apply_zeropoint(self,zero_point_G,zero_point_R):
        table['MAG_G'] -= zero_point_G
        table['MAG_R'] -= zero_point_R
        return table
    def unique_table(self,table:Table) -> Table:
        _, unique_indices = np.unique(table[['BRICKID','BRICKNAME','OBJID']], return_index=True)
        self.table = table[unique_indices]
    
    def estimate_w_theta(self):
        randoms = Table(fitsio.FITS('/user/animesh.sah/FP_CUTS/randoms_5M.fits')[1].read())
        rand_table = randoms[randoms['PHOTSYS']=='N']
        rand_table = rand_table[rand_table['DEC'] > -30]
        sample_size = 5*len(self.table)
        idx = np.random.choice(len(rand_table), sample_size, replace=False)
        rand_table = rand_table[idx]
        RA_random,DEC_random = rand_table['RA'], rand_table['DEC'] 
        RA_data, DEC_data = self.table['RA'], self.table['DEC']
        RA_data   = np.ascontiguousarray(np.array(RA_data, dtype=np.float64))
        DEC_data  = np.ascontiguousarray(np.array(DEC_data, dtype=np.float64))
        RA_random = np.ascontiguousarray(np.array(RA_random, dtype=np.float64))
        DEC_random= np.ascontiguousarray(np.array(DEC_random, dtype=np.float64))
        nbins = 200
        min_sep = 0.001  # degrees, example
        max_sep = 10   # degrees
        N = len(RA_data)
        rand_N = len(RA_random)
        logger.info("Theta bins set up from %f to %f degrees with %d bins", min_sep, max_sep, nbins)
        logger.info("Length of data: %d, Length of randoms: %d", N, rand_N)
        logger.info("Ratio of randoms to data: %f", rand_N / N)
        logger.info("Calculating DD counts...")
        bins = np.logspace(np.log10(min_sep), np.log10(max_sep), nbins+1 )
        autocorr=1
        DD_counts = DDtheta_mocks(autocorr,self.nthreads, bins, RA_data, DEC_data)
        logger.info("Calculating RR counts...")
        autocorr=1
        RR_counts = DDtheta_mocks(autocorr,self.nthreads, bins, RA_random, DEC_random)
        logger.info("Calculating DR counts...")
        autocorr=0
        DR_counts = DDtheta_mocks(autocorr,self.nthreads, bins, RA_data, DEC_data, RA_random, DEC_random)
        logger.info("Converting counts to w(theta)...")
        wtheta = convert_3d_counts_to_cf(RA_data.size, RA_data.size, RA_random.size, RA_random.size, DD_counts, DR_counts,DR_counts, RR_counts)
        self.w_theta = wtheta
        self.theta_bins = (bins[1:]+bins[:-1])/2
        return self.theta_bins,self.w_theta
    def minimize_function(self,params):
        zero_point_G, zero_point_R = params
    







inst = Minimizer()
inst.load_data( n_jobs=50)
inst.apply_cuts(inst.table)








        # Placeholder for actual w(theta) estimation logic
        



        





        
        


