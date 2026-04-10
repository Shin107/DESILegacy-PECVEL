import numpy as np
from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from astropy.table import Table, vstack , unique
import fitsio
import argparse
import time
import logging
from pathlib import Path
from Metrics import PhotoZMetrics, SigmaNMAD  # Your custom metrics class
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# Setup logging
logging.basicConfig(level=logging.INFO)
def add_file_logger(run_dir: Path):
    """Add a file handler to the root logger"""
    log_path = run_dir / 'logs' / 'pipeline.log'
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    logger.info(f"File logging started at {log_path}")
logger = logging.getLogger(__name__)

def setup_run_dir(run_dir: str) -> Path:
    """Create run directory and all subdirectories"""
    run_dir = Path(run_dir)
    subdirs = ['models', 'plots', 'predictions', 'catboost_info', 'logs']
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in subdirs:
        (run_dir / sub).mkdir(exist_ok=True)
    logger.info(f"Run directory set up at: {run_dir.resolve()}")
    return run_dir
    
    
import yaml
class PhotoZCatBoostPipeline:
    """
    A complete pipeline for photometric redshift prediction using CatBoost
    """
    
    def __init__(self, use_gpu=False, random_state=8513,region ='north',run_dir='run',feature_names=None):
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.model = None
        self.feature_names = feature_names
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.X = None
        self.y = None
        self.region_tag = region
        self.run_dir =Path(run_dir)
        
        logger.info(f"Initialized PhotoZ pipeline using {'GPU' if use_gpu else 'CPU'}")
        logger.info(f"Number of CPUs available: {multiprocessing.cpu_count()}")
    @staticmethod
    def load_config(config_path:str ) -> dict:
        with open(config_path,'r') as f: 
            cfg = yaml.safe_load(f)
        required = [('data', 'region'), ('model', 'iterations'), ('pipeline', 'mode')]
        for section, key in required:
            if key not in cfg.get(section, {}):
                raise KeyError(f"Missing required config key: [{section}][{key}]")
        

        logger.info(f'Config loaded from {config_path}')
        return cfg
    def load_astronomical_data(self,cfg):
        """Load and process astronomical survey data"""
        logger.info("Loading astronomical data...")
        start_time = time.time()
        region = cfg['data'].get('region', 'both')

        dr1_paths = cfg['data']['dr1'] 
        legacy_paths = cfg['data']['legacy']
        logger.info(f'Loading data for the region: {region}')
        start_time = time.time()
        if region == 'north':
            DR1 = Table(fitsio.FITS(dr1_paths['north'])[1].read())
            legacy = Table(fitsio.FITS(legacy_paths['north'])[1].read())
        elif region == 'south':
            DR1    = Table(fitsio.FITS(dr1_paths['south'])[1].read())
            legacy = Table(fitsio.FITS(legacy_paths['south'])[1].read())

        elif region == 'both':
            DR1    = vstack([Table(fitsio.FITS(dr1_paths['north'])[1].read()),
                            Table(fitsio.FITS(dr1_paths['south'])[1].read())])
            legacy = vstack([Table(fitsio.FITS(legacy_paths['north'])[1].read()),
                            Table(fitsio.FITS(legacy_paths['south'])[1].read())]) 
        else: 
            raise ValueError(f'Unknown region')
        
                


       
        
        load_time = time.time() - start_time
        logger.info(f'Data loaded in {load_time:.2f} seconds')
        
        return DR1, legacy
    
    # def magnitude_weights_exponential(mag, bright_mag=19, scale=2):

    #     weights = np.exp(-scale * (mag - bright_mag))
    #     weights = np.clip(weights, 0.1, 10)  # Prevent extreme weights
    #     return weights
    def z_weight_FP(self, redshift, spline):
        def n_z_pdf(redshift):
            normed = np.maximum(0, spline(redshift))/np.mean(np.maximum(0, spline(redshift)))
            return normed
        return n_z_pdf(redshift)
    def weighted_sigmoid(self, center=19,steepness =3.0):
        #print(self.X[:,1])
        #print('shape of X[:,1]:',self.X[:,1].shape)
        x = np.asarray(self.X_train[:, 1], dtype=float)

        z = steepness * (x - center)

        weights = 1 / (1 + np.exp(z))
        normed_weights = weights / np.mean(weights)  # Normalize to mean of 1
        return normed_weights
    

    def compute_magnitudes(self, legacy):
        """Compute magnitudes and color indices from flux measurements"""
        logger.info("Computing magnitudes and colors...")
        
        MAG = {}
        MAG_NOEXT = {}
        
        for band in ['G', 'R', 'Z', 'W1', 'W2', 'W3']:
            flux = np.array(legacy[f'FLUX_{band}'])
            trans = np.array(legacy[f'MW_TRANSMISSION_{band}'])
            frac = flux / trans
            valid = (frac > 0) & np.isfinite(frac)

            mag = np.full_like(frac, fill_value=np.nan)
            mag_noext = np.full_like(frac, fill_value=np.nan)
            
            # Handle zero/negative fluxes
            
            np.log10(flux, out=mag_noext, where=valid)
            np.log10(frac, out=mag, where=valid)

            MAG[band] = 22.5 - 2.5 * mag
            MAG_NOEXT[band] = 22.5 - 2.5 * mag_noext
            
            legacy[f'MAG_{band}'] = MAG[band]

        # Compute color indicesEditor: Font Size


        color_combinations = [
            ('G', 'R'), ('R', 'Z'), ('G', 'Z'),
            ('W1', 'W2'), ('W2', 'W3'), ('W1', 'W3'),
            ('R', 'W1'), ('R', 'W2'), ('R', 'W3'),
            ('G', 'W1'), ('G', 'W2'), ('G', 'W3'),
            ('Z', 'W1'), ('Z', 'W2'), ('Z', 'W3')
        ]
        
        for band1, band2 in color_combinations:
            legacy[f'MAG_{band1}{band2}'] = legacy[f'MAG_{band1}'] - legacy[f'MAG_{band2}']
        legacy = self._add_features(legacy)
        legacy = self._add_features_shape_color(legacy)
        print('Available Columns:',legacy.colnames)
        return legacy
    def flux_ratios(self, legacy):
        """Compute flux ratios"""
        logger.info("Computing flux ratios...")
        
        ratio_combinations = [
            ('FLUX_G', 'FLUX_R'), ('FLUX_R', 'FLUX_Z'), ('FLUX_G', 'FLUX_Z'),
            ('FLUX_W1', 'FLUX_W2'), ('FLUX_W2', 'FLUX_W3'), ('FLUX_W1', 'FLUX_W3')
        ]
        
        FLUX_G = 10**(-(legacy['MAG_G'] - 22.5)/2.5)
        FLUX_R = 10**(-(legacy['MAG_R'] - 22.5)/2.5)
        flux_z = 10**(-(legacy['MAG_Z'] - 22.5)/2.5)
        flux_w1 = 10**(-(legacy['MAG_W1'] - 22.5)/2.5)
        flux_w2 = 10**(-(legacy['MAG_W2'] - 22.5)/2.5)
        flux_w3 = 10**(-(legacy['MAG_W3'] - 22.5)/2.5)


        for flux1, flux2 in ratio_combinations:
            ratio_name = f'RATIO_{flux1.split("_")[1]}{flux2.split("_")[1]}'
            legacy[ratio_name] = np.where(legacy[flux2] != 0, legacy[flux1] / legacy[flux2], np.nan)
        
        return legacy
    

    def _add_features(self, legacy):
        """Add additional features to the dataset"""
        e1,e2 = legacy['SHAPE_E1'], legacy['SHAPE_E2']
        epsilon = np.sqrt(e1**2 + e2**2)
        bba = (1 - epsilon) / (1 + epsilon)
        r_circ = np.sqrt(bba) * legacy['SHAPE_R']
        legacy['BBA'] = bba   # axial ratio 
        legacy['R_CIRC'] = r_circ  #Half light radius
        ## DCHISQ is shaped as PSF, REX, DEV, EXP, SER   (https://www.legacysurvey.org/dr10/files/)
        legacy['MODEL_WEIGHT'] = legacy['DCHISQ'][:,2]/(legacy['DCHISQ'][:,2]+legacy['DCHISQ'][:,3])    #(p = dchi2dev/(dchi2dev+dchi2exp)) 
        
        return legacy
    def _add_features_shape_color(self, legacy):
        """Add additional features to the dataset"""
        log_sersic = np.log1p(legacy['SERSIC'])
        legacy['LOG_SERSIC'] = log_sersic
        log_shape_r = np.log1p(legacy['SHAPE_R'])
        legacy['LOG_SHAPE_R'] = log_shape_r

        for col in legacy.colnames:
            if col.startswith('MAG_'):
                legacy[f'LOG_SERSIC_times_{col[4:]}'] = legacy['LOG_SERSIC'] * (legacy[col])
        legacy['LOG_SHAPE_R'] = np.log1p(legacy['SHAPE_R'])
        legacy['SNR_G'] = legacy['FLUX_G'] * np.sqrt(legacy['FLUX_IVAR_G'])
        legacy['SNR_R'] = legacy['FLUX_R'] * np.sqrt(legacy['FLUX_IVAR_R'])
        legacy['SNR_Z'] = legacy['FLUX_Z'] * np.sqrt(legacy['FLUX_IVAR_Z'])
        legacy['SNR_W1'] = legacy['FLUX_W1'] * np.sqrt(legacy['FLUX_IVAR_W1'])
        legacy['SNR_W2'] = legacy['FLUX_W2'] * np.sqrt(legacy['FLUX_IVAR_W2'])
        legacy['SNR_W3'] = legacy['FLUX_W3'] * np.sqrt(legacy['FLUX_IVAR_W3'])
        eps =1e-9 
        # legacy['fiber_frac_R'] = (legacy['FLUX_FIBER_R'] + eps) / (legacy['FLUX_R'] + eps)
        # legacy['fiber_frac_G'] = (legacy['FLUX_FIBER_G'] + eps) / (legacy['FLUX_G'] + eps)
        # legacy['fiber_color_gr'] = legacy['MAG_FIBER_G']  - legacy['MAG_FIBER_R']

        # legacy['color_gradinet_gr'] = (legacy['MAG_FIBER_G'] - legacy['MAG_G']) - (legacy['MAG_FIBER_R'] - legacy['MAG_R'])



        
        return legacy
    def filter_photometric_data(self, legacy,return_all_columns=False):
        """Apply photometric quality cuts to the data"""
        logger.info("Applying photometric quality cuts...")
        
        # Example cuts (these can be adjusted based on data characteristics)
        mask = (
            (legacy['FLUX_G'] > 0) &
            (legacy['FLUX_R'] > 0) &
            (legacy['FLUX_Z'] > 0) &
            (legacy['FLUX_W1'] > 0) &
            (legacy['FLUX_W2'] > 0) &
            (legacy['FLUX_W3'] > 0)
        )
        self.feature_names = feature_names

        filtered_legacy = legacy[mask]
        logger.info(f"Filtered from {len(legacy)} to {len(filtered_legacy)} objects based on photometric cuts")
        
        return filtered_legacy
    
    def filter_data(self, legacy, DR1,feature_names=None,return_all_columns=False,phot_only=False,diagnostics=False,keep_nan_vals = True):
        """Apply quality cuts and filtering to the data"""
        logger.info("Applying data quality cuts...")
        legacy_copy = legacy.copy()
        DR1_copy = DR1.copy()
        # Flux detection cuts
        if feature_names is None:
            #feature_names = ['MAG_GR', 'MAG_G', 'MAG_RZ', 'MAG_R', 'MAG_RW1', 'MAG_ZW1', 'MAG_W1W2']
            feature_names = ['MAG_G','MAG_R','MAG_Z','MAG_W1','MAG_W2','MAG_W3','MAG_GR','MAG_RZ','MAG_GZ','MAG_W1W2','MAG_W2W3','MAG_W1W3','MAG_RW1','MAG_RW2','MAG_RW3','MAG_GW1','MAG_GW2','MAG_GW3','MAG_ZW1','MAG_ZW2','MAG_ZW3','SERSIC','FLUX_IVAR_G','FLUX_IVAR_R','FLUX_IVAR_Z','FLUX_IVAR_W1','FLUX_IVAR_W2','FLUX_IVAR_W3', 'PSFSIZE_G',
        'PSFSIZE_R',
        'PSFSIZE_Z',
        'PSFDEPTH_G',
        'PSFDEPTH_R',
        'PSFDEPTH_Z', 'LOG_SERSIC', 'LOG_SHAPE_R', 'LOG_SERSIC_times_G', 'LOG_SERSIC_times_R', 'LOG_SERSIC_times_Z', 'LOG_SERSIC_times_W1', 'LOG_SERSIC_times_W2', 'LOG_SERSIC_times_W3', 'LOG_SERSIC_times_GR', 'LOG_SERSIC_times_RZ', 'LOG_SERSIC_times_GZ', 'LOG_SERSIC_times_W1W2', 'LOG_SERSIC_times_W2W3', 'LOG_SERSIC_times_W1W3', 'LOG_SERSIC_times_RW1', 'LOG_SERSIC_times_RW2', 'LOG_SERSIC_times_RW3', 'LOG_SERSIC_times_GW1', 'LOG_SERSIC_times_GW2', 'LOG_SERSIC_times_GW3', 'LOG_SERSIC_times_ZW1', 'LOG_SERSIC_times_ZW2', 'LOG_SERSIC_times_ZW3','TYPE']
        self.feature_names = feature_names
        logger.info(f"Using features: {feature_names}")

        missing_features = [f for f in feature_names if f not in legacy.colnames]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")

        mask1 = (legacy['FLUX_G'] != 0) & (legacy['FLUX_R'] != 0) & (legacy['FLUX_Z'] != 0)
        legacy = legacy[feature_names]

        # Finite value cuts
        numeric_cols = [c for c in legacy.colnames if np.issubdtype(legacy[c].dtype, np.number)]
        data = np.vstack([legacy[c] for c in numeric_cols]).T
        mask2 = np.isfinite(data).all(axis=1)
        # if diagnostics:
        #     logger.info(f"Objects with 0 flux G:{}, R:{}, Z:{}: {}".format(
        #         np.sum(legacy['FLUX_G'] == 0),
        #         np.sum(legacy['FLUX_R'] == 0),
        #         np.sum(legacy['FLUX_Z'] == 0),
        #         np.sum(mask1)
        #     ))
        #     logger.info(f"Objects with non-finite values in features: {np.sum(~mask2)}")
        if phot_only:
            combined_mask = mask1 & mask2
            logger.info(f"Filtered from {len(legacy)} to {np.sum(combined_mask)} objects based on photometric cuts only")
            if not return_all_columns:
                return legacy[combined_mask]
            else:
                return legacy_copy[combined_mask]
        # Valid redshift cuts - FIXED: added upper limit
        if diagnostics:
            logger.info(f"Objects with non-finite or non-positive redshifts: {np.sum(~(np.isfinite(DR1['Z']) & (DR1['Z'] > 0)))}")
        mask3 = np.isfinite(DR1['Z']) & (DR1['Z'] > 0)
        if keep_nan_vals:
            combined_mask = mask1  & mask3
        else: 
            combined_mask = mask1 & mask2 & mask3
        logger.info(f"Filtered from {len(legacy)} to {np.sum(combined_mask)} objects")
        
        DR1_final=DR1[combined_mask]
        _,indeces=np.unique(DR1_final['TARGETID'],return_index=True)
        DR1_final=unique(DR1_final,keys='TARGETID')

        legacy_final=legacy[combined_mask][indeces]
        logger.info(f"Filtered from {len(legacy)} to {len(legacy_final)} objects for unique TARGETID")
        if not return_all_columns:
            return legacy_final, DR1_final
        else:
            return legacy_copy[combined_mask], DR1_copy[combined_mask]


    def prepare_ml_data(self, legacy_filtered, DR1_filtered,return_phot_only = False):
        """Prepare machine learning dataset"""
        
        
        X = legacy_filtered.to_pandas().to_numpy()
        y = np.asarray(DR1_filtered['Z'],dtype = np.float32)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
        self.X, self.y = X, y
        # nan_feature = np.any(np.isnan(X),axis =0)
        # nan_counts = np.isnan(X).sum(axis=0)
        # mask = nan_counts > 0
        # if np.any(nan_feature):

        #     logger.info(f'Caution feature: {self.feature_names[nan_feature]} has nan values. Counts: {nan_counts[mask]}')
        return X, y

    def prepare_ml_data_phot_only(self, legacy_filtered, return_phot_only = False):
        """Prepare machine learning dataset"""
        
        
        X = legacy_filtered.to_pandas().to_numpy()
        
        
        # nan_feature = np.any(np.isnan(X),axis =0)
        # nan_counts = np.isnan(X).sum(axis=0)
        # mask = nan_counts > 0
        # if np.any(nan_feature):

        #     logger.info(f'Caution feature: {self.feature_names[nan_feature]} has nan values. Counts: {nan_counts[mask]}')
        return X

    def create_data_splits(self, X, y):
        """Create train/validation/test splits with consistent random states"""
        logger.info("Creating data splits...")
        
        # First split: 70% train, 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.20, random_state=self.random_state
        )
        
        # Second split: 15% validation, 15% test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_state
        )
        
        # Store splits in the class
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        
        logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_new_model(self, iterations=5000, depth=6, learning_rate=0.01,weight=None,train_dir=None):
        """Train a fresh model from scratch"""
        logger.info("Training new CatBoost model......")
        if weight == 'z_weight_FP':
            import pickle 
            with open('/Storage/animesh/PECVEL/Codes/n_z_spline.pkl','rb') as f:
                spline = pickle.load(f)
            weights = self.z_weight_FP(self.y_train, spline)
        if weight == 'weighted_sigmoid':
            weights = self.weighted_sigmoid(center=19,steepness=3.0)
        self.model = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function='RMSE',
            eval_metric=SigmaNMAD(),
            custom_metric=['MAE','R2'],
            task_type="GPU" if self.use_gpu else "CPU",
            random_seed=self.random_state,
            verbose=200, train_dir= train_dir
            #weights = weights if weight != None else None
        )
        
        # Create pools
        train_dir = train_dir or str(self.run_dir/'catboost_info')
        logger.info("Using weights for training" if weight != None else "No weights used for training")
        # self.X_train = self.X_train.astype(np.float32)
        # self.y_train = self.y_train.astype(np.float32)
        # self.X_val = self.X_val.astype(np.float32)
        # self.y_val = self.y_val.astype(np.float32)
        # self.X_test = self.X_test.astype(np.float32)
        # self.y_test = self.y_test.astype(np.float32)
        train_pool = Pool(self.X_train, self.y_train, weight=weights if weight != None else None, cat_features=[len(self.feature_names)-1] if 'TYPE' in self.feature_names else None)
        val_pool = Pool(self.X_val, self.y_val,cat_features=[len(self.feature_names)-1] if 'TYPE' in self.feature_names else None)

        # Train the model
        self.model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True
        )
        
        logger.info("Model training completed")
        return self.model

    def load_pretrained_model(self, model_path):
        """Load a pre-trained model"""
        logger.info(f"Loading model from {model_path}")
        
        self.model = CatBoostRegressor()
        self.model.load_model(model_path)
        
        logger.info(f"Loaded model with {self.model.get_param('iterations')} iterations")
        return self.model

    def continue_training(self, model_path, additional_iterations=1000):
        """Continue training from a saved model"""
        logger.info(f"Loading model from {model_path} for continued training")
        
        # Load existing model
        base_model = CatBoostRegressor()
        base_model.load_model(model_path)
        
        logger.info(f"Base model has {base_model.get_param('iterations')} iterations")
        
        prev_iters = base_model.get_param("iterations")
        logger.info(f"Base model had {prev_iters} iterations")

        # Create new model with more iterations

        params = base_model.get_params()
        params["iterations"] = prev_iters + additional_iterations  # override

        self.model = CatBoostRegressor(**params)
        # Create pools
        self.X_train = self.X_train.astype(np.float32)
        self.y_train = self.y_train.astype(np.float32)
        train_pool = Pool(self.X_train, self.y_train,cat_features=[len(self.feature_names)-1] if 'TYPE' in self.feature_names else None)
        val_pool = Pool(self.X_val, self.y_val,cat_features=[len(self.feature_names)-1] if 'TYPE' in self.feature_names else None)

        # Continue training
        self.model.fit(
            train_pool,
            eval_set=val_pool,
            init_model = base_model,
            use_best_model=True,
            verbose=200
        )
        
        logger.info(f"Continued training completed. Total iterations: {self.model.get_param('iterations')}")
        return self.model

    def cross_validate(self, fold_count=5, iterations=5000, depth=6, learning_rate=0.01):
        """Perform cross-validation"""
        logger.info(f"Running {fold_count}-fold cross-validation...")
        
        # Create a model for CV
        cv_model = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function='RMSE',
            eval_metric='RMSE',
            task_type="GPU" if self.use_gpu else "CPU",
            random_seed=self.random_state,
            verbose=100
        )
        cat_indices = [i for i, name in enumerate(self.feature_names) if name == 'TYPE'] if 'TYPE' in self.feature_names else None

        
        # Run cross-validation
        cv_results = cv(
            Pool(self.X, self.y, cat_features=cat_indices),
            params=cv_model.get_params(),
            fold_count=fold_count,
            early_stopping_rounds=200,
            verbose=100
        )
        
        logger.info("Cross-validation completed")
        return cv_results

    def evaluate_model(self, save_predictions=True):
        """Comprehensive model evaluation"""
        if self.model is None:
            raise ValueError("No model found. Train or load a model first.")
        
        logger.info("Evaluating model performance...")
        
        results = {}
        
        # Test set evaluation
        y_pred_test = self.model.predict(self.X_test)
        
        results['test'] = {
            'mae': mean_absolute_error(self.y_test, y_pred_test),
            'mse': mean_squared_error(self.y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
            'r2': r2_score(self.y_test, y_pred_test)
        }
        
        # Validation set evaluation
        y_pred_val = self.model.predict(self.X_val)
        
        results['validation'] = {
            'mae': mean_absolute_error(self.y_val, y_pred_val),
            'mse': mean_squared_error(self.y_val, y_pred_val),
            'rmse': np.sqrt(mean_squared_error(self.y_val, y_pred_val)),
            'r2': r2_score(self.y_val, y_pred_val)
        }
        
        # Custom PhotoZ metrics
        try:
            test_photoz_metrics = PhotoZMetrics(self.y_test, y_pred_test)  # FIXED: true values first
            val_photoz_metrics = PhotoZMetrics(self.y_val, y_pred_val)    # FIXED: true values first
            
            results['test']['photoz'] = test_photoz_metrics
            results['validation']['photoz'] = val_photoz_metrics
        except Exception as e:
            logger.warning(f"PhotoZ metrics calculation failed: {e}")
        
        # Print results
        self._print_evaluation_results(results)
        
        if save_predictions:
            # Save predictions for further analysis
            out_path = self.run_dir / 'predictions' / f'predictions_{self.region_tag}.npz'
            np.savez(out_path,
                    y_test=self.y_test, y_pred_test=y_pred_test,
                    y_val=self.y_val, y_pred_val=y_pred_val)
            logger.info(f"Predictions saved to {out_path}")
        
        return results

    def _print_evaluation_results(self, results):
        """Print formatted evaluation results"""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        for dataset in ['test', 'validation']:
            print(f"\n{dataset.upper()} SET METRICS:")
            print("-" * 30)
            metrics = results[dataset]
            print(f"MAE:  {metrics['mae']:.6f}")
            print(f"MSE:  {metrics['mse']:.6f}")
            print(f"RMSE: {metrics['rmse']:.6f}")
            print(f"R²:   {metrics['r2']:.6f}")
            
            if 'photoz' in metrics:
                print(f"\nCUSTOM PHOTOZ METRICS ({dataset.upper()}):")
                print(metrics['photoz'].summary())
        
        # Compare performance
        test_rmse = results['test']['rmse']
        val_rmse = results['validation']['rmse']
        rmse_diff = abs(test_rmse - val_rmse)
        
        print(f"\nPERFORMANCE COMPARISON:")
        print("-" * 30)
        print(f"Test RMSE:       {test_rmse:.6f}")
        print(f"Validation RMSE: {val_rmse:.6f}")
        print(f"Difference:      {rmse_diff:.6f}")
        
        if rmse_diff > 0.01:
            print("⚠️  Warning: Large difference between test and validation!")
        else:
            print("✅ Consistent performance across datasets")

    def save_model(self, filename):
        """Save the trained model"""
        path = self.run_dir/'models'/filename

        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Add timestamp if filename exists
        if path.exists():
            timestamp = int(time.time())
            path = self.run_dir / 'models' / f"{path.stem}_{timestamp}{path.suffix}"
        
        self.model.save_model(str(path))
        logger.info(f"Model saved to {path}")
        return str(path)

    def run_complete_pipeline(self, model_config=None):
        """Run the complete pipeline from data loading to evaluation"""
        if model_config is None:
            model_config = {
                'iterations': 5000,
                'depth': 6,
                'learning_rate': 0.01
            }
        
        logger.info("Starting complete PhotoZ pipeline...")
        
        
        legacy = self.compute_magnitudes(legacy)
        
        legacy_filtered, DR1_filtered = self.filter_data(legacy, DR1)
        
        X, y = self.prepare_ml_data(legacy_filtered, DR1_filtered)
        
        self.create_data_splits(X, y)
        
        #self.train_new_model(**model_config)

        results = self.evaluate_model()
        
        model_filename = self.save_model("catboost_photoz_model.cbm")
      
        logger.info("Complete pipeline finished successfully!")
        return results, model_filename
    def two_d_hist(self, x, y, bins=50):
        """Create 2D histogram"""
        binx = np.linspace(min(x), max(x), bins)
        biny = np.linspace(min(y), max(y), bins)
        hist, xedges, yedges = np.histogram2d(x, y, bins=[binx, biny])
        return hist, xedges, yedges

    def plot_2d_hist_validation(self, bins=50):
        """Plot 2D histogram"""
        x = self.y_val
        y = self.model.predict(self.X_val)
        save_path = self.run_dir / 'plots' / f'2d_hist_valid_{self.region_tag}.png'
        hist, xedges, yedges = self.two_d_hist(x, y, bins=bins)
        plt.figure(figsize=(8, 6))
        plt.imshow(hist.T, origin='lower', aspect='auto',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap='magma_r',norm = LogNorm())
        plt.plot(self.y_val, self.y_val, color='k', linestyle='--', label='y=x')
        plt.colorbar(label='Counts')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('2D Histogram')
        if save_path:
            plt.savefig(save_path)
            logger.info(f"2D histogram saved to {save_path}")
    def plot_2d_hist_test(self, bins=50):
        """Plot 2D histogram"""
        x = self.y_test
        y = self.model.predict(self.X_test)
        save_path = self.run_dir / 'plots' / f'2d_hist_test_{self.region_tag}.png'

        hist, xedges, yedges = self.two_d_hist(x, y, bins=bins)
        plt.figure(figsize=(8, 6))
        plt.imshow(hist.T, origin='lower', aspect='auto',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap='magma_r',norm=LogNorm())
        plt.plot(self.y_test,self.y_test,'k--')
        plt.colorbar(label='Counts')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('2D Histogram')
        if save_path:
            plt.savefig(save_path)
            logger.info(f"2D histogram saved to {save_path}")
    def plot_2d_hist_complete(self, bins=50):
        """Plot 2D histogram"""
        x = self.y
        y = self.model.predict(self.X)
        save_path = self.run_dir / 'plots' / f'2d_hist_complete_{self.region_tag}.png'


        hist, xedges, yedges = self.two_d_hist(x, y, bins=bins)
        plt.figure(figsize=(8, 6))
        plt.imshow(hist.T, origin='lower', aspect='auto',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap='magma_r', norm=LogNorm())
        plt.plot(self.y,self.y,'k--')
        plt.colorbar(label='Counts')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('2D Histogram')
        if save_path:
            plt.savefig(save_path)
            logger.info(f"2D histogram saved to {save_path}")
    
    
    
    def plot_feature_importance(self, model_path=None):
        if model_path is None:
            raise ValueError("No model found. Train or load a model first.")

        model = CatBoostRegressor()
        model.load_model(model_path)
        save_path = self.run_dir / 'plots' / f'feature_importance_{self.region_tag}.png'

        cat_idx = [self.feature_names.index('TYPE')] if 'TYPE' in self.feature_names else None

        # Sample to avoid memory explosion
        idx = np.random.choice(len(self.X_test), size=min(60000, len(self.X_test)), replace=False)
        sample_pool = Pool(self.X_test[idx], self.y_test[idx], cat_features=cat_idx)

        # SHAP values
        shap_values = model.get_feature_importance(type='ShapValues', data=sample_pool) #SHapley Additive exPlanations 
        shap_values = shap_values[:, :-1]
        shap_importance = np.mean(np.abs(shap_values), axis=0)

        # Loss importance
        loss_importance = model.get_feature_importance(type='LossFunctionChange', data=sample_pool) #change in the loss function if a particular feature is removed

        feature_names = self.feature_names

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.barh(feature_names, shap_importance)
        plt.title('SHAP Importance')

        plt.subplot(1, 2, 2)
        plt.barh(feature_names, loss_importance)
        plt.title('Loss Function Change')

        plt.tight_layout()
        plt.savefig(save_path)    

        

            

    


def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description="CatBoost PhotoZ Pipeline")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--run_dir",    type=str, required=True,   # <-- new
                        help="Directory to save all outputs")
    parser.add_argument("--region",   type=str,  default=None)
    parser.add_argument("--mode",     type=str,  default=None)
    parser.add_argument("--gpu",      action="store_true")
    parser.add_argument("--weight",   type=str,  default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--plot",     type=str,  default=None)
    args = parser.parse_args()

    cfg =     PhotoZCatBoostPipeline.load_config(args.config)

    if args.region:     cfg['data']['region']          = args.region
    if args.mode:       cfg['pipeline']['mode']        = args.mode
    if args.gpu:        cfg['model']['use_gpu']        = True
    if args.weight:     cfg['model']['weight']         = args.weight
    if args.model_path: cfg['pipeline']['model_path']  = args.model_path
    if args.plot:       cfg['pipeline']['plot']        = args.plot
    # Initialize pipeline
    run_dir = setup_run_dir(args.run_dir)
    add_file_logger(run_dir)
    import shutil
    shutil.copy(args.config, run_dir / 'logs' / 'config_used.yaml')
    region =  cfg['data']['region']
    pipeline = PhotoZCatBoostPipeline(
        use_gpu=cfg['model']['use_gpu'],
        random_state=cfg['model']['random_state'],region = cfg['data']['region'],run_dir=run_dir,feature_names=cfg['features']
    )    


    # Load and prepare data (always needed)
    DR1, legacy = pipeline.load_astronomical_data(cfg)
    legacy = pipeline.compute_magnitudes(legacy)
    legacy_filtered, DR1_filtered = pipeline.filter_data(legacy, DR1,feature_names = cfg.get('features'))
    X,y = pipeline.prepare_ml_data(legacy_filtered, DR1_filtered)
    pipeline.create_data_splits(X, y)
    
    mode       = cfg['pipeline']['mode']
    model_path = cfg['pipeline'].get('model_path')
    mcfg       = cfg['model']
    # Execute based on mode


    
    if mode == 'train':
        pipeline.train_new_model(
            iterations=mcfg['iterations'],
            depth=mcfg['depth'],
            learning_rate=mcfg['learning_rate'],
            weight=mcfg.get('weight'),train_dir = mcfg.get('train_dir')
        )
        pipeline.evaluate_model(save_predictions=cfg['pipeline']['save_predictions'])
        out_name = model_path or f"catboost_{region}_{mcfg.get('weight') or 'noweight'}.cbm"
        pipeline.save_model(out_name)

        
    elif mode == 'continue':
        if not model_path:
            raise ValueError("model_path required for 'continue' mode")
        pipeline.continue_training(model_path, additional_iterations=1000)
        pipeline.evaluate_model()
        pipeline.save_model(f"catboost_{region}_continued.cbm")

    elif mode == 'evaluate':
        if not model_path:
            raise ValueError("model_path required for 'evaluate' mode")
        pipeline.load_pretrained_model(model_path)
        pipeline.evaluate_model(save_predictions=cfg['pipeline']['save_predictions'])
        plot = cfg['pipeline'].get('plot')
        if plot == '2d_hist_valid':    pipeline.plot_2d_hist_validation()
        elif plot == '2d_hist_test':   pipeline.plot_2d_hist_test()
        elif plot == '2d_hist_complete': pipeline.plot_2d_hist_complete()
    elif mode == 'plot':
        if not model_path:
            raise ValueError("model_path required for 'plot' mode")
        pipeline.load_pretrained_model(model_path)
        pipeline.plot_feature_importance(model_path)


if __name__ == "__main__":
    main()
