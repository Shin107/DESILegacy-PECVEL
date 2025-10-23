import numpy as np
import os
from multiprocessing import Pool, cpu_count
import psutil
import resource
import time
from astropy.table import Table, hstack, vstack, unique
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

parser = argparse.ArgumentParser(description="Grid minimize script")
parser.add_argument('--test', action='store_true', help='Run a test minimization')

args = parser.parse_args()
class Dataprocessor:
    def __init__(self):
        self.north_path = "/storage/shadab/data/legacy_survey/dr9/north/sweep/9.0/"
        self.north_path2 = "/storage/shadab/data/legacy_survey/dr9/north/sweep/9.0-photo-z/"
        self.south_path = "/storage/shadab/data/legacy_survey/dr9/south/sweep/9.0/"
        self.south_path2 = "/storage/shadab/data/legacy_survey/dr9/south/sweep/9.0-photo-z/"
        self.nthreads= cpu_count() - 2
        self.w_theta = None
        
        self.table_full = None
        self.theta_bins = None
        self.cols_dr9 =  [ 'BRICKID','OBJID','BRICKNAME','RA', 'DEC', 'FLUX_G','FLUX_R','FLUX_Z','MW_TRANSMISSION_G','MW_TRANSMISSION_R','MW_TRANSMISSION_Z','SHAPE_E1','SHAPE_E2','SHAPE_R','SERSIC','TYPE','NOBS_G','NOBS_R','NOBS_Z','MASKBITS','FRACMASKED_G','FRACMASKED_R','FRACMASKED_Z','FRACFLUX_G','FRACFLUX_R','FRACFLUX_Z','FRACIN_G', 'FRACIN_R',
       'FRACIN_Z','GAIA_PHOT_G_MEAN_MAG','NOBS_G','NOBS_R','NOBS_Z','FIBERFLUX_G', 'FIBERFLUX_R', 'FIBERFLUX_Z','FLUX_W1','MW_TRANSMISSION_W1']
        self.cols_pz = [
            'Z_PHOT_MEDIAN', 'Z_PHOT_L95', 'Z_PHOT_U95'
        ]
        
        
    def _compute_magnitude(self, flux, transmission,band):
        flux_corrected = flux / transmission
        mag = np.full_like(flux_corrected, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            valid = (flux_corrected > 0) & np.isfinite(flux_corrected)
            mag[valid] = 22.5 - 2.5 * np.log10(flux_corrected[valid])
        mag_noext = np.full_like(flux, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            valid_noext = (flux > 0) & np.isfinite(flux)
            mag_noext[valid_noext] = 22.5 - 2.5 * np.log10(flux[valid_noext])
        return mag, mag_noext
    def _compute_fibermagnitude(self, fiberflux, transmission,band):
        fiberflux_corrected = fiberflux / transmission
        magfiber = np.full_like(fiberflux_corrected, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            valid = (fiberflux_corrected > 0) & np.isfinite(fiberflux_corrected)
            magfiber[valid] = 22.5 - 2.5 * np.log10(fiberflux_corrected[valid])
        return magfiber



    def _load_and_preprocess(self, paths): 
        fits_path, pz_path = paths
        dr9_chunk = Table(fitsio.FITS(fits_path)[1].read(columns=self.cols_dr9))
        dr9_chunk_pz = Table(fitsio.FITS(pz_path)[1].read(columns=self.cols_pz))
        for band in ['G', 'R', 'Z', 'W1']:
            flux = np.array(dr9_chunk[f'FLUX_{band}'])
            trans = np.array(dr9_chunk[f'MW_TRANSMISSION_{band}'])
                
            mag, mag_noext = self._compute_magnitude(flux, trans, band)
            dr9_chunk[f'MAG_{band}'] = mag
            dr9_chunk[f'MAG_NOEXT_{band}'] = mag_noext
            
    
   
 
        for band in ['G', 'R', 'Z']:
            fiberflux = np.array(dr9_chunk[f'FIBERFLUX_{band}'])
            trans = np.array(dr9_chunk[f'MW_TRANSMISSION_{band}'])
            magfiber = self._compute_fibermagnitude(fiberflux, trans, band)
            dr9_chunk[f'FIBERMAG_{band}'] = magfiber
            
    
        e1, e2 = dr9_chunk['SHAPE_E1'], dr9_chunk['SHAPE_E2']
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
                    #output_path = os.path.join(output_dir, fname.replace(".fits", "_selected.fits"))
                    file_pairs.append((os.path.join(self.north_path, fname),
                                    os.path.join(self.north_path2, pz_name)))
        if args.test:
            file_pairs_ix = np.random.choice(len(file_pairs),20)
            file_pairs= [file_pairs[i] for i in file_pairs_ix]
        logger.info(f"Total file pairs to process: {len(file_pairs)}")
        with Pool(n_jobs) as pool:
            results= pool.map(self._load_and_preprocess, file_pairs)
        results = [r for r in results if r is not None]
        logger.info("Combining all chunks into a single table...")
        logger.info("Loaded the data")
        self.table_full = vstack(results)
        #self.table_full = BGS_mask(self.table_full)
        #logger.info("After BGS mask, table length: %d", len(self.table_full))
        #return results 
    @staticmethod
    def get_BGS_mask(table: Table) -> np.ndarray:
        """Return boolean mask for BGS cuts - does NOT modify table"""
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
    def get_selection_mask(table: Table) -> np.ndarray:
        """Return boolean mask for selection cuts - does NOT modify table"""
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
    def __init__(self,nthreads:int=100):
        self.theta_bins = None
        self.nthreads = nthreads
        self.w_theta = None
        self._randoms_cache = None
        self._RR_counts_cache = None
        self._bins_cache = None
    def _initialize_randoms_and_bins(self, data_size_estimate=150000):
        """Initialize randoms and bins once - call this before minimization"""
        if self._randoms_cache is not None:
            logger.info("Using cached randoms and RR counts")
            return
            
        logger.info("Loading randoms for the first time...")
        randoms = Table(fitsio.FITS('/user/animesh.sah/FP_CUTS/randoms_5M.fits')[1].read())
        rand_table = randoms[randoms['PHOTSYS'] == 'N']
        rand_table = rand_table[rand_table['DEC'] > -30]
        
        # Use a fixed large sample (5x your max expected data size)
        sample_size = 5 * data_size_estimate
        idx = np.random.choice(len(rand_table), sample_size, replace=False)
        rand_table = rand_table[idx]
        
        self._randoms_cache = {
            'RA': np.ascontiguousarray(np.array(rand_table['RA'], dtype=np.float64)),
            'DEC': np.ascontiguousarray(np.array(rand_table['DEC'], dtype=np.float64))
        }
        
        # Set up bins once
        nbins = 200
        min_sep = 0.01
        max_sep = 10
        self._bins_cache = np.logspace(np.log10(min_sep), np.log10(max_sep), nbins + 1)
        
        # Compute RR counts once
        logger.info("Computing RR counts (one-time calculation)...")
        autocorr = 1
        self._RR_counts_cache = DDtheta_mocks(
            autocorr, self.nthreads, self._bins_cache,
            self._randoms_cache['RA'], self._randoms_cache['DEC']
        )
        
        self.theta_bins = (self._bins_cache[1:] + self._bins_cache[:-1]) / 2
        logger.info("Randoms and RR counts cached successfully")
    
    def estimate_w_theta(self,table:Table):
        if self._randoms_cache is None:
            self._initialize_randoms_and_bins(data_size_estimate=len(table))

        RA_data, DEC_data = table['RA'], table['DEC']
        RA_data   = np.ascontiguousarray(np.array(RA_data, dtype=np.float64))
        DEC_data  = np.ascontiguousarray(np.array(DEC_data, dtype=np.float64))
        
        
        N = len(RA_data)
        rand_N = len(self._randoms_cache['RA'])
        logger.info("Data points: %d, Random points: %d (ratio: %.1f)", 
                   N, rand_N, rand_N / N)       
        
        logger.info("Calculating DD counts...")
        DD_counts = DDtheta_mocks(1, self.nthreads, self._bins_cache, RA_data, DEC_data)

        logger.info("Calculating DR counts...")
        DR_counts = DDtheta_mocks(0,self.nthreads, self._bins_cache, RA_data, DEC_data, RA2=self._randoms_cache['RA'], DEC2=self._randoms_cache['DEC'])
        logger.info("Using cached RR counts...")
        logger.info("Converting counts to w(theta)...")
        
        wtheta = convert_3d_counts_to_cf(RA_data.size, RA_data.size, self._randoms_cache['RA'].size, self._randoms_cache['RA'].size, DD_counts, DR_counts, DR_counts, self._RR_counts_cache)
        self.w_theta = wtheta
        return self.theta_bins,self.w_theta
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
    def objective_function(self, zero_points):
        """zero_points:[zero_point_G,zero_point_R] """
        self.iteration += 1
        zp_G, zp_R = zero_points
        logger.info("Iteration %d: Testing zero points G=%.4f, R=%.4f", self.iteration, zp_G, zp_R)
        table_copy = Table()
        for col in self.data_processor.table_full.colnames:
            table_copy[col] = self.data_processor.table_full[col].copy()
        
        # DEBUGGING: Check magnitudes BEFORE shift
        mag_g_before = np.array(table_copy['MAG_G'][:5].copy())
        mag_r_before = np.array(table_copy['MAG_R'][:5].copy())
        
        table_copy['MAG_G'] -= zp_G
        table_copy['MAG_R'] -= zp_R
        table_copy['FIBERMAG_R'] -= zp_R  
        
        # DEBUGGING: Check magnitudes AFTER shift
        mag_g_after = table_copy['MAG_G'][:5]
        mag_r_after = table_copy['MAG_R'][:5]
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
        
        w_theta_computed = w_theta_computed[theta_bins>0.01]
        self.reference_w_theta = self.reference_w_theta[self.reference_theta_bins>0.01]

 
        chi_squared = np.sum((w_theta_computed - self.reference_w_theta)**2)
        logger.info("Chi-squared: %.6f", chi_squared)
        self.results_history.append({
            'iteration': self.iteration,
            'zp_G': zp_G,
            'zp_R': zp_R,
            'chi_squared': chi_squared,
            'n_objects': len(selected_table)
        })
        return chi_squared
    def run_minimization(self, initial_guess=(0.02, 0.04),bounds = [(-0.5,0.5),(-0.5,0.5)]):
        logger.info("Starting minimization with initial guess G=%.4f, R=%.4f", initial_guess[0], initial_guess[1])

        result = minimize(self.objective_function, initial_guess, method='Nelder-Mead',options={'disp': True,'maxiter':60000},tol = 1e-10,bounds=bounds)

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
    logger.info("STEP 2: Loading reference w(theta)")
    minimizer = ZeroPointOptimizer(inst, reference_w_theta=np.load('/user/animesh.sah/w_theta_results/corrfunc_south_0.01_to_10.npy',allow_pickle=True), nthreads=100)



    minimizer.run_minimization(initial_guess=(0.04, -0.06),bounds = [(-0.2,0.2),(-0.2,0.2)])
    minimizer.save_results('minimization_folder/Minmizer_NM_6.csv')




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
        



        





        
        



