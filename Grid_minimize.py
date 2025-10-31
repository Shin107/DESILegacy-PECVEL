from turtle import pd
import numpy as np
import os
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import psutil
import resource
import time
from astropy.table import Table, hstack, vstack, unique
import fitsio
import argparse
import pickle
from scipy.optimize import minimize
import logging 
import gc 
from scipy.optimize import basinhopping
import Corrfunc
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from Corrfunc.io import read_catalog
from Corrfunc.utils import convert_3d_counts_to_cf
import warnings
import treecorr
from itertools import product
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import h5py
import time 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Grid minimize script")
parser.add_argument('--test', action='store_true', help='Run a test minimization')

args = parser.parse_args()
a = time.time()

class Dataprocessor:
    def __init__(self):
        self.north_path = "/storage/shadab/data/legacy_survey/dr9/north/sweep/9.0/"
        self.north_path2 = "/storage/shadab/data/legacy_survey/dr9/north/sweep/9.0-photo-z/"
        self.south_path = "/storage/shadab/data/legacy_survey/dr9/south/sweep/9.0/"
        self.south_path2 = "/storage/shadab/data/legacy_survey/dr9/south/sweep/9.0-photo-z/"
        self.nthreads= cpu_count() - 2
        self.w_theta = None
        
        self.table = None
        self.table_full = None
        self.theta_bins = None
        self.cols_dr9 =  [ 'BRICKID','OBJID','BRICKNAME','RA', 'DEC', 'FLUX_G','FLUX_R','FLUX_Z','MW_TRANSMISSION_G','MW_TRANSMISSION_R','MW_TRANSMISSION_Z','SHAPE_E1','SHAPE_E2','SHAPE_R','SERSIC','TYPE','NOBS_G','NOBS_R','NOBS_Z','MASKBITS','FRACMASKED_G','FRACMASKED_R','FRACMASKED_Z','FRACFLUX_G','FRACFLUX_R','FRACFLUX_Z','FRACIN_G', 'FRACIN_R',
       'FRACIN_Z','GAIA_PHOT_G_MEAN_MAG','NOBS_G','NOBS_R','NOBS_Z','FIBERFLUX_G', 'FIBERFLUX_R', 'FIBERFLUX_Z','FLUX_W1','MW_TRANSMISSION_W1']
        self.cols_pz = [
            'Z_PHOT_MEDIAN', 'Z_PHOT_L95', 'Z_PHOT_U95'
        ]
        
        
    def _compute_magnitude(self, flux, transmission,band):
        flux_corrected = flux / transmission
        mag = np.full_like(flux_corrected, np.nan,dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            valid = (flux_corrected > 0) & np.isfinite(flux_corrected)
            mag[valid] = 22.5 - 2.5 * np.log10(flux_corrected[valid])
        mag_noext = np.full_like(flux, np.nan,dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            valid_noext = (flux > 0) & np.isfinite(flux)
            mag_noext[valid_noext] = 22.5 - 2.5 * np.log10(flux[valid_noext])
        return mag, mag_noext
    def _compute_fibermagnitude(self, fiberflux, transmission,band):
        fiberflux_corrected = fiberflux / transmission
        magfiber = np.full_like(fiberflux_corrected, np.nan,dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            valid = (fiberflux_corrected > 0) & np.isfinite(fiberflux_corrected)
            magfiber[valid] = 22.5 - 2.5 * np.log10(fiberflux_corrected[valid])
        return magfiber



    def _load_and_preprocess(self, paths): 
        fits_path, pz_path = paths
        try:
            dr9_chunk = Table(fitsio.FITS(fits_path)[1].read(columns=self.cols_dr9))
            dr9_chunk_pz = Table(fitsio.FITS(pz_path)[1].read(columns=self.cols_pz))
        except Exception as e:
            logger.error(f"Error loading FITS files: {e}")
            return None

        for band in ['G', 'R', 'Z', 'W1']:
            flux = np.array(dr9_chunk[f'FLUX_{band}'],dtype=np.float32)
            trans = np.array(dr9_chunk[f'MW_TRANSMISSION_{band}'],dtype=np.float32)

            mag, mag_noext = self._compute_magnitude(flux, trans, band)
            dr9_chunk[f'MAG_{band}'] = mag
            dr9_chunk[f'MAG_NOEXT_{band}'] = mag_noext
            
    
   
 
        for band in ['G', 'R', 'Z']:
            fiberflux = np.array(dr9_chunk[f'FIBERFLUX_{band}'],dtype=np.float32)
            trans = np.array(dr9_chunk[f'MW_TRANSMISSION_{band}'],dtype=np.float32)
            magfiber = self._compute_fibermagnitude(fiberflux, trans, band)
            dr9_chunk[f'FIBERMAG_{band}'] = magfiber


        e1 = np.array(dr9_chunk['SHAPE_E1'],dtype=np.float32)
        e2 = np.array(dr9_chunk['SHAPE_E2'],dtype=np.float32)
        epsilon = np.sqrt(e1**2 + e2**2)
        bba = (1 - epsilon) / (1 + epsilon)
        r_circ = np.sqrt(np.clip(bba, 0, None)) * dr9_chunk['SHAPE_R']
        
        dr9_chunk['R_CIRC'] = r_circ
        dr9_chunk['BBA'] = bba
        dr9_chunk['EPSILON'] = epsilon
        
        # Merge with photo-z
        return hstack([dr9_chunk, dr9_chunk_pz])
    def load_data(self,region:str = 'north', n_jobs =10) -> Table:
        print("Loading all data into RAM...")
        file_pairs = []
        
        for fname in os.listdir(self.north_path):
            if fname.endswith(".fits") and "-pz" not in fname:
                pz_name = fname.replace(".fits", "-pz.fits")
                if os.path.exists(os.path.join(self.north_path2, pz_name)):
                    file_pairs.append((os.path.join(self.north_path, fname),
                                    os.path.join(self.north_path2, pz_name)))
        if args.test:
            file_pairs_ix = np.random.choice(len(file_pairs),20,replace=False)
            file_pairs= [file_pairs[i] for i in file_pairs_ix]
        logger.info(f"Total file pairs to process: {len(file_pairs)}")
        with Pool(n_jobs) as pool:
            results= pool.map(self._load_and_preprocess, file_pairs)
        results = [r for r in results if r is not None]
        logger.info("Combining all chunks into a single table...")
        logger.info("Loaded the data")

        self.table_full = vstack(results)
        del results
        gc.collect()

        #self.table_full = BGS_mask(self.table_full)
        #logger.info("After BGS mask, table length: %d", len(self.table_full))
        #return results 
    @staticmethod
    def get_BGS_mask(table: Table) -> np.ndarray:
        """Return boolean mask for BGS cuts"""
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
        mask1 = (table['FLUX_G'] != 0) & (table['FLUX_R'] != 0) & (table['FLUX_Z'] != 0)

        return cuts2 & mask1
    
    @staticmethod
    def precompute_selection_static_mask(table: Table):
        """Precompute static masks that do not depend on zero-point shifts."""
        cuts = (
            (table['R_CIRC'] > 0) &
            ((1 - table['BBA']) < 0.7) &
            (((table['TYPE'] == 'SER') & (table['SERSIC'] > 2.5)) | (table['TYPE'] == 'DEV')) &
            (table['Z_PHOT_MEDIAN'] < 0.15) &
            (table['Z_PHOT_L95'] < 0.1)
        )
        return cuts
    @staticmethod
    def selection_mag_mask(MAG_G,MAG_R,MAG_Z):
        """Precompute magnitude-related masks that depend on zero-point shifts."""
        cuts_mag = (
            (MAG_R < 18) &
            ((MAG_G - MAG_R) > 0.68) &
            ((MAG_G - MAG_R) > (1.3 * (MAG_R - MAG_Z) - 0.05)) &
            ((MAG_G - MAG_R) < (2.0 * (MAG_R - MAG_Z) - 0.15))
        )
        return cuts_mag
    @staticmethod
    def precompute_BGS_static_mask(table:Table):
        cuts2 = (
           
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
        mask1 = (table['FLUX_G'] != 0) & (table['FLUX_R'] != 0) & (table['FLUX_Z'] != 0)

        return cuts2 & mask1
    @staticmethod
    def BGS_mag_mask(MAG_R,FIBERMAG_R,MAG_G,MAG_Z):
        cuts2_mag = (
            (
                ((FIBERMAG_R < (5.1 + MAG_R)) & (MAG_R <= 17.8)) |
                ((FIBERMAG_R < 22.9) & (MAG_R > 17.8) & (MAG_R < 20))
            ) &
            ((-1 < (MAG_G - MAG_R)) &
             ((MAG_G - MAG_R) < 4) &
             (-1 < (MAG_R - MAG_Z)) &
             ((MAG_R - MAG_Z) < 4))
        )
        return cuts2_mag



        

    @staticmethod
    def get_selection_mask(table: Table) -> np.ndarray:
        """Return boolean mask for selection cuts"""
        cuts = (
            (table['MAG_R'] < 18) &
            ((table['MAG_G'] - table['MAG_R']) > 0.68) &
            ((table['MAG_G'] - table['MAG_R']) > (1.3 * (table['MAG_R'] - table['MAG_Z']) - 0.05)) &
            ((table['MAG_G'] - table['MAG_R']) < (2.0 * (table['MAG_R'] - table['MAG_Z']) - 0.15)) &
            (table['R_CIRC'] > 0) &
            ((1 - table['BBA']) < 0.7) &
            (((table['TYPE'] == 'SER') & (table['SERSIC'] > 2.5)) | (table['TYPE'] == 'DEV')) &
            (table['Z_PHOT_MEDIAN'] < 0.15) &
            (table['Z_PHOT_L95'] < 0.1)
        )
        
        return cuts
    @staticmethod
    def get_combined_mask(table: Table) -> np.ndarray:
        """Get combined selection + BGS mask"""
        return Dataprocessor.get_selection_mask(table) & Dataprocessor.get_BGS_mask(table)
    
    @staticmethod
    def remove_duplicates_mask(table: Table) -> np.ndarray:
        """Return mask for unique objects"""
        _, unique_indices = np.unique(table[['BRICKID', 'BRICKNAME', 'OBJID']],
                                     return_index=True)
        mask = np.zeros(len(table), dtype=bool)
        mask[unique_indices] = True
        return mask
    def BGS_mask(self,table:Table) -> Table:
        """DEPRECATED: Use get_BGS_mask instead. Modifies self.table in place."""
        logger.info("Applying BGS mask cuts...")
        logger.info("Initial table length: %d", len(table))
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
        logger.info("Table length after BGS mask cuts: %d", len(table))
        return table[cuts2]


    

    def apply_cuts(self,table:Table) -> Table:
        """DEPRECATED: Use get_selection_mask instead. Modifies self.table in place."""

        logger.info("Applying selection cuts...")
        logger.info("Initial table length: %d", len(table))
        cuts = (
            (table['MAG_R'] < 18) &
            ((table['MAG_G'] - table['MAG_R']) > 0.68) &
            ((table['MAG_G'] - table['MAG_R']) > (1.3 * (table['MAG_R'] - table['MAG_Z']) - 0.05)) &
            ((table['MAG_G'] - table['MAG_R']) < (2.0 * (table['MAG_R'] - table['MAG_Z']) - 0.15)) &
            (table['R_CIRC'] > 0) &
            ((1 - table['BBA']) < 0.7) &
            (((table['TYPE'] == 'SER') & (table['SERSIC'] > 2.5)) | (table['TYPE'] == 'DEV')) &
            (table['Z_PHOT_MEDIAN'] < 0.15) &
            (table['Z_PHOT_L95'] < 0.1))

        self.table = table[cuts]
        logger.info("Table length after selection cuts: %d", len(self.table))

        self.table = self._unique_table()
        #logger.info("Final table length after removing duplicates: %d", len(self.table))
        return self.table

    def apply_zeropoint(self,table:Table    ,zero_point_G,zero_point_R): 
        """DEPRECATED: This modifies table_full in place - dangerous for minimization!"""
        logger.info("Applying zero-point corrections...")   
        self.table_full['MAG_G'] -= zero_point_G
        self.table_full['MAG_R'] -= zero_point_R
    def _unique_table(self):
        """DEPRECATED"""
        logger.info("Removing duplicate entries...")
        logger.info("Table length before removing duplicates: %d", len(self.table))
        #_, unique_indices = np.unique(self.table[['BRICKID','BRICKNAME','OBJID']], return_index=True)
        self.table = unique(self.table, keys=['BRICKID','BRICKNAME','OBJID'])
        logger.info("Table length after removing duplicates: %d", len(self.table))
        return self.table
        logger.info("Table length after removing duplicates: %d", len(self.table))

    
    



class WthetaEstimator:
    def __init__(self,nthreads:int=100,w_theta_estimator:str='treecorr'):
        self.theta_bins = None
        self.nthreads = nthreads
        self.w_theta = None
        self._randoms_cache = None
        self._RR_counts_cache = None
        self._bins_cache = None
        self.w_theta_estimator = w_theta_estimator  #'treecorr' or 'corrfunc'
    def _initialize_randoms_and_bins(self, data_size_estimate=150000):
        """Initialize randoms and bins once """
        if self._randoms_cache is not None:
            logger.info("Using cached randoms and RR counts")
            return
            
        logger.info("Loading randoms for the first time...")
        randoms = Table(fitsio.FITS('/user/animesh.sah/FP_CUTS/randoms_5M.fits')[1].read())
        rand_table = randoms[randoms['PHOTSYS'] == 'N']
        rand_table = rand_table[rand_table['DEC'] > -30]
        
        # Use a fixed large sample 
        np.random.seed(42)
        sample_size = 5 * data_size_estimate
        idx = np.random.choice(len(rand_table), sample_size, replace=False)
        rand_table = rand_table[idx]
        
        self._randoms_cache = {
            'RA': np.ascontiguousarray(np.array(rand_table['RA'], dtype=np.float64)),
            'DEC': np.ascontiguousarray(np.array(rand_table['DEC'], dtype=np.float64))
        }
        
        # Set up bins once
        self.nbins = 200
        self.min_sep = 0.01
        self.max_sep = 10
        self._bins_cache = np.logspace(np.log10(self.min_sep), np.log10(self.max_sep), self.nbins + 1)

        # Compute RR counts once
        logger.info("Computing RR counts (one-time calculation)...")
        if  self.w_theta_estimator == 'corrfunc':
            self._RR_counts_cache = DDtheta_mocks(
                1, self.nthreads, self._bins_cache,
            
                self._randoms_cache['RA'], self._randoms_cache['DEC']
        )
        else:
            n_cpus = os.cpu_count()
            logger.info(f"Using {n_cpus} CPUs for RR counts computation")
            cat_rand = treecorr.Catalog(ra=self._randoms_cache['RA'], dec=self._randoms_cache['DEC'], ra_units='deg', dec_units='deg')
            self._RR_counts_cache = treecorr.NNCorrelation(min_sep=self.min_sep, max_sep=self.max_sep, nbins=self.nbins, sep_units='deg', bin_type='Log')
            self._RR_counts_cache.process(cat_rand,num_threads=n_cpus)

        self.theta_bins = (self._bins_cache[1:] + self._bins_cache[:-1]) / 2
        logger.info("Randoms and RR counts cached successfully")
    
    def estimate_w_theta(self,table:Table):
        if self._randoms_cache is None:
            self._initialize_randoms_and_bins(data_size_estimate=len(table))

        RA_data, DEC_data = table['RA'], table['DEC']
        if not RA_data.flags['C_CONTIGUOUS']:
            RA_data = np.ascontiguousarray(RA_data.astype(np.float64))
        else:
            RA_data = np.array(RA_data, dtype=np.float64)
            
        if not DEC_data.flags['C_CONTIGUOUS']:
            DEC_data = np.ascontiguousarray(DEC_data.astype(np.float64))
        else:
            DEC_data = np.array(DEC_data, dtype=np.float64)
        
        
        N = len(RA_data)

        rand_N = len(self._randoms_cache['RA'])
        logger.info("Data points: %d, Random points: %d (ratio: %.1f)", 
                   N, rand_N, rand_N / N)       
        
        logger.info("Calculating DD counts...")

        if self.w_theta_estimator == 'corrfunc':
            DD_counts = DDtheta_mocks(1, self.nthreads, self._bins_cache, RA_data, DEC_data)

            logger.info("Calculating DR counts...")
            DR_counts = DDtheta_mocks(0,self.nthreads, self._bins_cache, RA_data, DEC_data, RA2=self._randoms_cache['RA'], DEC2=self._randoms_cache['DEC'])
            logger.info("Using cached RR counts...")
            logger.info("Converting counts to w(theta)...")
            
            wtheta = convert_3d_counts_to_cf(RA_data.size, RA_data.size, self._randoms_cache['RA'].size, self._randoms_cache['RA'].size, DD_counts, DR_counts, DR_counts, self._RR_counts_cache)
            self.w_theta = wtheta
            #np.save('w_theta_minimized_v7',np.array([self.theta_bins,self.w_theta]))
        else: 
            cat_data = treecorr.Catalog(ra=RA_data, dec=DEC_data, ra_units='deg', dec_units='deg')
            dd = treecorr.NNCorrelation(min_sep=self.min_sep, max_sep=self.max_sep, nbins=self.nbins, sep_units='deg', bin_type='Log')
            dd.process(cat_data,num_threads=self.nthreads)
            logger.info("Calculating DR counts...")
            cat_rand = treecorr.Catalog(ra=self._randoms_cache['RA'], dec=self._randoms_cache['DEC'], ra_units='deg', dec_units='deg')
            dr = treecorr.NNCorrelation(min_sep=self.min_sep, max_sep=self.max_sep, nbins=self.nbins, sep_units='deg', bin_type='Log')
            dr.process(cat_data,cat_rand,num_threads=self.nthreads)
            logger.info("Using cached RR counts...")
            rr = self._RR_counts_cache
            logger.info("Converting counts to w(theta)...")
            wtheta  ,varxi = dd.calculateXi(rr= rr, dr=dr)
            #wtheta = (dd.npairs - 2 * dr.npairs * (N / rand_N) + rr.npairs * (N / rand_N)**2) / (rr.npairs * (N / rand_N)**2)
            self.w_theta = wtheta

            #np.save('w_theta_minimized_treecorr_v7',np.array([self.theta_bins,self.w_theta]))
        return self.theta_bins,self.w_theta
    
    def save_w_theta(self, filename: str):
        if self.w_theta is None or self.theta_bins is None:
            raise ValueError("w_theta has not been computed yet.")
        np.save(filename, np.array([self.theta_bins, self.w_theta]))
    #def minimize(self, pars):




class ZeroPointOptimizer:
    def __init__(self, data_processor: Dataprocessor, reference_w_theta: np.ndarray, nthreads:int=100):
        self.data_processor = data_processor
        self.reference_w_theta = reference_w_theta[1]
        self.reference_theta_bins = reference_w_theta[0]
        self.w_theta_estimator = WthetaEstimator(nthreads=nthreads)
        logger.info("Pre-initializing randoms and RR counts...")
        self.w_theta_estimator._initialize_randoms_and_bins()
        
        self.iteration = 0
        self.results_history = []  # Track all iterations
    def compute_w_theta_for_zeropoints(self, zero_points):
        """
        Compute w_theta for given zero points without calculating chi-squared.
        Returns: (zp_G, zp_R, theta_bins, w_theta_computed, n_objects)
        """
        
        self.iteration += 1
        zp_G, zp_R = zero_points
        logger.info(f"Iteration {self.iteration}: Computing w_theta for G={zp_G:.4f}, R={zp_R:.4f}")
        
        # Create copy and apply shifts
        table = self.data_processor.table
        print('Table length before applying zero-point shifts:', len(table))
        MAG_G = table['MAG_G'] - zp_G
        MAG_R = table['MAG_R'] - zp_R
        MAG_Z = table['MAG_Z']  # No shift in Z
        FIBERMAG_R = table['FIBERMAG_R'] - zp_R
        selection_color_mask = self.data_processor.selection_mag_mask(MAG_G=MAG_G,MAG_R=MAG_R,MAG_Z=MAG_Z)
        bgs_color_mask = self.data_processor.BGS_mag_mask(MAG_G=MAG_G,MAG_R=MAG_R,MAG_Z=MAG_Z,FIBERMAG_R=FIBERMAG_R)
        combined_color_mask = selection_color_mask & bgs_color_mask

        selected_table = table[combined_color_mask]
        n_objects = len(selected_table)
        logger.info(f"Selected {n_objects} objects after applying zero-point shifts and color cuts")
        
        
        try:
            theta_bins, w_theta_computed = self.w_theta_estimator.estimate_w_theta(selected_table)

            return (zp_G, zp_R, n_objects, theta_bins, w_theta_computed)
        except Exception as e:
            logger.error(f"Error computing w_theta: {e}")
            return (zp_G, zp_R, n_objects,None, None)
        finally:
            del table,selected_table
            gc.collect()
    def objective_function(self, zero_points):
        """zero_points:[zero_point_G,zero_point_R] """
        self.iteration += 1
        zp_G, zp_R = zero_points
        table_copy = self.data_processor.table_full.copy()
        # logger.info("Iteration %d: Testing zero points G=%.4f, R=%.4f", self.iteration, zp_G, zp_R)
        # table_copy = Table()
        # for col in self.data_processor.table_full.colnames:
        #     table_copy[col] = self.data_processor.table_full[col].copy()
        
        # # DEBUGGING: Check magnitudes BEFORE shift
        # mag_g_before = np.array(table_copy['MAG_G'][:5].copy())
        # mag_r_before = np.array(table_copy['MAG_R'][:5].copy())
        
        table_copy['MAG_G'] -= zp_G
        table_copy['MAG_R'] -= zp_R
        table_copy['FIBERMAG_R'] -= zp_R  
        
        # DEBUGGING: Check magnitudes AFTER shift
        # mag_g_after = table_copy['MAG_G'][:5]
        # mag_r_after = table_copy['MAG_R'][:5]
        #logger.info(f"MAG_G before: {mag_g_before}")
        #logger.info(f"MAG_G after:  {mag_g_after}")
        #logger.info(f"MAG_G change: {mag_g_after - mag_g_before}")
        #logger.info(f"MAG_R before: {mag_r_before}")
        #logger.info(f"MAG_R after:  {mag_r_after}")
        #logger.info(f"MAG_R change: {mag_r_after - mag_r_before}")

        selection_mask = Dataprocessor.get_selection_mask(table_copy)
        bgs_mask = Dataprocessor.get_BGS_mask(table_copy)
        combined_mask = selection_mask & bgs_mask
        #DEBUG
        n_selection = np.sum(selection_mask)
        n_bgs = np.sum(bgs_mask)
        n_combined = np.sum(combined_mask)
        logger.info(f"Selection mask: {n_selection}/{len(table_copy)} objects")
        logger.info(f"BGS mask: {n_bgs}/{len(table_copy)} objects")
        logger.info(f"Combined: {n_combined}/{len(table_copy)} objects")

        selected_table = table_copy[combined_mask]
        unique_mask = Dataprocessor.remove_duplicates_mask(selected_table)
        selected_table = selected_table[unique_mask]
        logger.info(f"Selected {len(selected_table)} objects after all cuts")
        logger.info(f"RA range: [{selected_table['RA'].min():.2f}, {selected_table['RA'].max():.2f}]")
        logger.info(f"DEC range: [{selected_table['DEC'].min():.2f}, {selected_table['DEC'].max():.2f}]")

        try:
            theta_bins, w_theta_computed = self.w_theta_estimator.estimate_w_theta(selected_table)
        except Exception as e:
            logger.error(f"Error computing w_theta: {e}")
            return 1e10
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.loglog(self.reference_theta_bins_filtered, self.reference_w_theta_filtered, 
                 '*', label='south')
        ax.loglog(theta_bins_filtered, w_theta_computed_filtered, 
                 '.', label='north')
        ax.grid(True, linestyle=':')
        ax.set_xlabel(r'$\theta$ (degrees)', fontsize=14)
        ax.set_ylabel(r'$w(\theta)$', fontsize=14)
        
        # Compute chi-squared
        

 
        chi_squared = np.sum((w_theta_computed - self.reference_w_theta)**2)
        logger.info("Chi-squared: %.6f", chi_squared)
        self.results_history.append({
            'iteration': self.iteration,
            'zp_G': zp_G,
            'zp_R': zp_R,
            'chi_squared': chi_squared,
            'n_objects': len(selected_table)
        })
        plt.title(f'Iteration {self.iteration}: Chi-squared={chi_squared:.4f}\nZP_G={zp_G:.4f}, ZP_R={zp_R:.4f}')
        plt.legend()
        plt.savefig(f'plots/w_theta_minimize/w_theta_{self.iteration}_v2.png')

        return chi_squared
    def run_minimization(self, initial_guess=(0.02, 0.04),bounds = [(-0.5,0.5),(-0.5,0.5)],minimizer_kwargs=None):
        logger.info("Starting minimization with initial guess G=%.4f, R=%.4f", initial_guess[0], initial_guess[1])
        if minimizer_kwargs is None:
 
            result = minimize(self.objective_function, initial_guess, method='Nelder-Mead',options={'disp': True,'maxiter':60000},tol = 1e-10,bounds=bounds)
        else: 
            result = basinhopping(self.objective_function, initial_guess, minimizer_kwargs=minimizer_kwargs or {'method': 'Nelder-Mead'}, niter=500, disp=True,stepsize=0.02)
        logger.info(f"\n{'='*60}")
        logger.info("Optimization Complete!")
        logger.info(f"{'='*60}")
        logger.info(f"Optimal zero-points: G={result.x[0]:.4f}, R={result.x[1]:.4f}")
        logger.info(f"Final chi-squared: {result.fun:.6e}")
        logger.info(f"Number of iterations: {self.iteration}")
        logger.info(f"Success: {result.success}")
        logger.info(f"Message: {result.message}")
        return result
    def save_results(self, filename: str):
        """Save optimization history"""
        import pandas as pd
        df = pd.DataFrame(self.results_history)
        df.to_csv(filename, index=False)
        logger.info(f"Results saved to {filename}")


class WThetaGridStorage:
    """Efficient storage for w_theta grid search results using HDF5"""
    
    @staticmethod
    def save_grid_hdf5(results_dict, filename, compression='gzip', compression_level=6):
        """
        Save grid results to HDF5 file with compression.
        
        Parameters:
        -----------
        results_dict : dict with structure from grid search
        filename : str, output HDF5 filename
        compression : str, 'gzip' (good compression) or 'lzf' (faster)
        compression_level : int, 1-9 for gzip (higher = better compression, slower)
        """
        with h5py.File(filename, 'w') as f:
            # Save metadata
            meta = f.create_group('metadata')
            meta.create_dataset('grid_values', data=results_dict['grid_values'])
            meta.create_dataset('reference_theta_bins', 
                              data=results_dict['reference_theta_bins'])
            meta.create_dataset('reference_w_theta', 
                              data=results_dict['reference_w_theta'])
            meta.attrs['n_points'] = len(results_dict['grid_values'])
            meta.attrs['creation_date'] = str(np.datetime64('now'))
            
            # Create group for grid data
            grid_group = f.create_group('grid_data')
            
            # Determine maximum theta bins size
            max_nbins = max(
                len(d['theta_bins']) if d['theta_bins'] is not None else 0
                for d in results_dict['data'].values()
            )
            
            n_grid = len(results_dict['grid_values'])
            
            # Pre-allocate arrays (2D for 40x40 grid)
            theta_bins_array = np.full((n_grid, n_grid, max_nbins), np.nan, dtype=np.float32)
            w_theta_array = np.full((n_grid, n_grid, max_nbins), np.nan, dtype=np.float32)
            n_objects_array = np.zeros((n_grid, n_grid), dtype=np.int32)
            valid_mask = np.zeros((n_grid, n_grid), dtype=bool)
            
            # Fill arrays
            for i, zp_G in enumerate(results_dict['grid_values']):
                for j, zp_R in enumerate(results_dict['grid_values']):
                    key = (zp_G, zp_R)
                    if key not in results_dict['data']:
                        continue
                    
                    data = results_dict['data'][key]
                    n_objects_array[i, j] = data['n_objects']
                    
                    if data['theta_bins'] is not None and data['w_theta'] is not None:
                        n_bins = len(data['theta_bins'])
                        theta_bins_array[i, j, :n_bins] = data['theta_bins']
                        w_theta_array[i, j, :n_bins] = data['w_theta']
                        valid_mask[i, j] = True
            
            # Save with compression
            grid_group.create_dataset('theta_bins', data=theta_bins_array, 
                                    compression=compression, 
                                    compression_opts=compression_level)
            grid_group.create_dataset('w_theta', data=w_theta_array,
                                    compression=compression,
                                    compression_opts=compression_level)
            grid_group.create_dataset('n_objects', data=n_objects_array,
                                    compression=compression,
                                    compression_opts=compression_level)
            grid_group.create_dataset('valid_mask', data=valid_mask,
                                    compression=compression)
            
            # Store actual number of bins used per grid point
            n_bins_array = np.zeros((n_grid, n_grid), dtype=np.int16)
            for i, zp_G in enumerate(results_dict['grid_values']):
                for j, zp_R in enumerate(results_dict['grid_values']):
                    key = (zp_G, zp_R)
                    if key in results_dict['data'] and results_dict['data'][key]['theta_bins'] is not None:
                        n_bins_array[i, j] = len(results_dict['data'][key]['theta_bins'])
            
            grid_group.create_dataset('n_bins', data=n_bins_array)
            
        logger.info(f"Saved grid to {filename}")
        logger.info(f"File size: {h5py.File(filename, 'r').id.get_filesize() / 1e6:.2f} MB")
    
    @staticmethod
    def load_grid_hdf5(filename):
        """
        Load grid results from HDF5 file.
        
        Returns:
        --------
        results_dict : dict with same structure as original
        """
        results_dict = {'data': {}}
        
        with h5py.File(filename, 'r') as f:
            # Load metadata
            results_dict['grid_values'] = f['metadata/grid_values'][:]
            results_dict['reference_theta_bins'] = f['metadata/reference_theta_bins'][:]
            results_dict['reference_w_theta'] = f['metadata/reference_w_theta'][:]
            
            # Load grid data
            theta_bins_array = f['grid_data/theta_bins'][:]
            w_theta_array = f['grid_data/w_theta'][:]
            n_objects_array = f['grid_data/n_objects'][:]
            n_bins_array = f['grid_data/n_bins'][:]
            
            # Reconstruct dictionary
            for i, zp_G in enumerate(results_dict['grid_values']):
                for j, zp_R in enumerate(results_dict['grid_values']):
                    n_bins = n_bins_array[i, j]
                    n_objects = n_objects_array[i, j]
                    
                    if n_bins > 0:
                        theta_bins = theta_bins_array[i, j, :n_bins]
                        w_theta = w_theta_array[i, j, :n_bins]
                    else:
                        theta_bins = None
                        w_theta = None
                    
                    results_dict['data'][(zp_G, zp_R)] = {
                        'theta_bins': theta_bins,
                        'w_theta': w_theta,
                        'n_objects': n_objects
                    }
        
        return results_dict
    
    @staticmethod
    def load_single_point_hdf5(filename, i, j):
        """Load data for a single grid point without loading entire file"""
        with h5py.File(filename, 'r') as f:
            grid_values = f['metadata/grid_values'][:]
            n_bins = f['grid_data/n_bins'][i, j]
            n_objects = f['grid_data/n_objects'][i, j]
            
            if n_bins > 0:
                theta_bins = f['grid_data/theta_bins'][i, j, :n_bins]
                w_theta = f['grid_data/w_theta'][i, j, :n_bins]
            else:
                theta_bins = None
                w_theta = None
            
            return {
                'zp_G': grid_values[i],
                'zp_R': grid_values[j],
                'theta_bins': theta_bins,
                'w_theta': w_theta,
                'n_objects': n_objects
            }
def run_grid_w_theta(minimizer,values,output_file='w_theta_grid.h5'):

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    n_points = len(values)
    from itertools import product
    import pandas as pd
    points = list(product(values, repeat=2))
    n_cpus = min(cpu_count(), 10) 
    print(cpu_count(),n_cpus)
    results_dict = {
    'grid_values': values,
    'reference_theta_bins': minimizer.reference_theta_bins,
    'reference_w_theta': minimizer.reference_w_theta,
    'data': {}
}
    logger.info(f"Starting grid search: {len(points)} points, {n_cpus} CPUs")
    with ThreadPoolExecutor(max_workers=n_cpus) as executor:
        for i, (point, result) in enumerate(zip(points, 
                                                executor.map(minimizer.compute_w_theta_for_zeropoints, 
                                                        points))):
            zp_G, zp_R, n_objects, theta_bins, w_theta = result
            results_dict['data'][(zp_G, zp_R)] = {
                'theta_bins': theta_bins,
                'w_theta': w_theta,
                'n_objects': n_objects
            }
    WThetaGridStorage.save_grid_hdf5(results_dict, output_file)
    return results_dict
    WThetaGridStorage.save_grid_hdf5(results_dict, output_file)
    return results_dict


if __name__ == '__main__':
    inst = Dataprocessor()
    inst.load_data(n_jobs=150)
    BGS_static_mask = inst.precompute_BGS_static_mask(table = inst.table_full)
    Selection_static_mask = inst.precompute_selection_static_mask(table = inst.table_full)
    static_mask = BGS_static_mask & Selection_static_mask
    inst.table = inst.table_full[static_mask]
    print('Length of full table is:',len(inst.table_full))
    print('Length after static mask is:',len(inst.table))
    unique_mask = Dataprocessor.remove_duplicates_mask(inst.table)
    inst.table = inst.table[unique_mask]   
    print('Length after unique mask is:',len(inst.table))

    values = np.linspace(0.1, -0.1, 10)

    minimizer = ZeroPointOptimizer(inst, reference_w_theta=np.load('/user/animesh.sah/w_theta_results/corrfunc_south_0.01_to_10.npy',allow_pickle=True), nthreads=15)
    results = run_grid_w_theta(minimizer,values,output_file='w_theta_grid_40x40.h5')
    b = time.time()
    elapsed = b - a
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"Elapsed time: {minutes} minutes {seconds:.2f} seconds")





'''

if __name__ == '__main__':


    logger.info("STEP 1: Loading all data")
    inst = Dataprocessor()
    inst.load_data( n_jobs=150)
    print('length of full table is:',len(inst.table_full))






    # logger.info("STEP 2: Computing w(theta)")
    # table_ref = inst.table_full.copy()
    # logger.info("Applying zero-point shifts for reference w(theta)")
    # table_ref['MAG_G']-=0.04
    # table_ref['MAG_R']-=0.0


    # ref_mask = Dataprocessor.get_combined_mask(table_ref)
    
    # table_ref = table_ref[ref_mask]
    # unique_mask = Dataprocessor.remove_duplicates_mask(table_ref)
    # table_ref = table_ref[unique_mask]
    # print('Length after cuts is:',len(table_ref))

    #theta_bins, reference_w_theta = w_theta_estimator.estimate_w_theta(table_ref)
    #np.save('0.0_0.04_zpshift',[theta_bins,reference_w_theta])




    # logger.info("STEP 2: Loading reference w(theta)")
    minimizer = ZeroPointOptimizer(inst, reference_w_theta=np.load('/user/animesh.sah/w_theta_results/corrfunc_south_0.01_to_10.npy',allow_pickle=True), nthreads=100)



    #minimizer = ZeroPointOptimizer(inst, reference_w_theta=np.load('/user/animesh.sah/w_theta_results/corrfunc_south_0.4_to_15_0.0234_v2.npy',allow_pickle=True), nthreads=18)
    # minimizer.run_minimization(initial_guess=(-0.02, 0.04),bounds = [(-0.2,0.2),(-0.2,0.2)],minimizer_kwargs={'method':'Nelder-Mead','options':{'disp': True,'maxiter':500},'tol':1e-12})
    # minimizer.save_results('minimization_folder/Minmizer_NM_basinhopping_v2.csv')

    print('Testing objective function at zp_G=0.022414, zp_R=0.05')
    print(minimizer.objective_function([0.022414, 0.05]))
    # print('Testing objective function at zp_G=0.029603, zp_R=0.114615')
    # print(minimizer.objective_function([0.029603, 0.114615]))



#     from multiprocessing import Pool, cpu_count
#     from concurrent.futures import ThreadPoolExecutor
#     n_points = 35
#     values = np.linspace(0.1, -0.1, n_points)
#     X, Y = np.meshgrid(values, values, indexing='xy')
#     grid_points = np.column_stack((X.ravel(), Y.ravel()))




# ###################################################################################
#     chi_squared = []
#     points = list(product(values, repeat=2))
#     import pandas as pd

#     # df_grid_loaded = pd.read_csv('/user/animesh.sah/DESI_PECVEL/degeneracy_grid_v2.csv')
#     # x = np.array(df_grid_loaded['zp_G'])
#     # y  = np.array(df_grid_loaded['zp_R'])
#     #points = list(zip(x,y))
#     n_points = 35
#     n_cpus = min(cpu_count(), 10) 
#     print(cpu_count(),n_cpus)
#     with open('minimization_folder/chi_squared_grid_half_v3.txt', 'w') as f:
#         f.write("zp_G,zp_R,chi_squared\n")
    
#     with ThreadPoolExecutor(max_workers=n_cpus) as executor:
#         for point, chi2 in zip(points, executor.map(minimizer.objective_function, points)):
#             with open('minimization_folder/chi_squared_grid_half_v3.txt', 'a') as f:
#                 f.write(f"{point[0]:.6f},{point[1]:.6f},{chi2:.6e}\n")
#             chi_squared.append(chi2)
#     chi_squared = np.array(chi_squared).reshape((n_points, n_points))
#     np.save('minimization_folder/chi_squared_grid_half_v3.npy', chi_squared)




# inst = Dataprocessor()
# inst.load_data( n_jobs=100)
# inst.apply_zeropoint(inst.table_full,zero_point_G=0.04,zero_point_R=0.04)
# inst.apply_cuts(inst.table_full)
# inst.BGS_mask(inst.table)

# print('length of full table is:',len(inst.table_full))
# print('length after cuts is:',len(inst.table))

# w_theta = WthetaEstimator(inst.table,nthreads=100)
# theta_bins, w_theta_values = w_theta.estimate_w_theta()









        # Placeholder for actual w(theta) estimation logic
        



        '''





        
        



