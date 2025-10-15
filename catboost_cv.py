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
from Metrics import PhotoZMetrics  # Your custom metrics class
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhotoZCatBoostPipeline:
    """
    A complete pipeline for photometric redshift prediction using CatBoost
    """
    
    def __init__(self, use_gpu=False, random_state=8513):
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.X = None
        self.y = None
        
        logger.info(f"Initialized PhotoZ pipeline using {'GPU' if use_gpu else 'CPU'}")
        logger.info(f"Number of CPUs available: {multiprocessing.cpu_count()}")
    
    def load_astronomical_data(self):
        """Load and process astronomical survey data"""
        logger.info("Loading astronomical data...")
        start_time = time.time()
        
        # Load DR1 data
        north_DR1 = Table(fitsio.FITS('../LEGACY+DR1/TABLE_DR1_north_v1_sep_1.fits')[1].read())
        south_DR1 = Table(fitsio.FITS('../LEGACY+DR1/TABLE_DR1_south_v1_sep_1.fits')[1].read())
        DR1 = vstack([north_DR1, south_DR1])

        # Load Legacy data
        north_legacy = Table(fitsio.FITS('../LEGACY+DR1/TABLE_legacy_north_v1_sep_1.fits')[1].read())
        south_legacy = Table(fitsio.FITS('../LEGACY+DR1/TABLE_legacy_south_v1_sep_1.fits')[1].read())
        legacy = vstack([north_legacy, south_legacy])
        
        load_time = time.time() - start_time
        logger.info(f'Data loaded in {load_time:.2f} seconds')
        
        return DR1, legacy
    
    def compute_magnitudes(self, legacy):
        """Compute magnitudes and color indices from flux measurements"""
        logger.info("Computing magnitudes and colors...")
        
        MAG = {}
        MAG_NOEXT = {}
        
        for band in ['G', 'R', 'Z', 'W1', 'W2', 'W3']:
            flux = np.array(legacy[f'FLUX_{band}'])
            trans = np.array(legacy[f'MW_TRANSMISSION_{band}'])
            frac = flux / trans
            
            mag = np.empty_like(frac)
            mag_noext = np.empty_like(frac)
            
            # Handle zero/negative fluxes
            positive_mask = (frac > 0)
            np.log10(flux, out=mag_noext, where=positive_mask)
            np.log10(frac, out=mag, where=positive_mask)
            
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
        
        return legacy
    
    def filter_data(self, legacy, DR1):
        """Apply quality cuts and filtering to the data"""
        logger.info("Applying data quality cuts...")
        
        # Flux detection cuts
        mask1 = (legacy['FLUX_G'] != 0) & (legacy['FLUX_R'] != 0) & (legacy['FLUX_Z'] != 0)
        
        # Finite value cuts
        numeric_cols = [c for c in legacy.colnames if np.issubdtype(legacy[c].dtype, np.number)]
        data = np.vstack([legacy[c] for c in numeric_cols]).T
        mask2 = np.isfinite(data).all(axis=1)
        
        # Valid redshift cuts - FIXED: added upper limit
        mask3 = np.isfinite(DR1['Z']) & (DR1['Z'] > 0)
        
        combined_mask = mask1 & mask2 & mask3
        logger.info(f"Filtered from {len(legacy)} to {np.sum(combined_mask)} objects")
        
        DR1_final=DR1[combined_mask]
        _,indeces=np.unique(DR1_final['TARGETID'],return_index=True)
        DR1_final=unique(DR1_final,keys='TARGETID')

        legacy_final=legacy[combined_mask][indeces]
        logger.info(f"Filtered from {len(legacy)} to {len(legacy_final)} objects for unique TARGETID")

        return legacy_final, DR1_final

    def prepare_ml_data(self, legacy_filtered, DR1_filtered, feature_names=None):
        """Prepare machine learning dataset"""
        if feature_names is None:
            feature_names = ['MAG_GR', 'MAG_G', 'MAG_RZ', 'MAG_R', 'MAG_RW1', 'MAG_ZW1', 'MAG_W1W2']
        
        self.feature_names = feature_names
        logger.info(f"Using features: {feature_names}")
        
        X = legacy_filtered[feature_names].to_pandas().to_numpy()
        y = np.array(DR1_filtered['Z'])
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
        self.X, self.y = X, y
        return X, y

    def create_data_splits(self, X, y):
        """Create train/validation/test splits with consistent random states"""
        logger.info("Creating data splits...")
        
        # First split: 70% train, 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state
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

    def train_new_model(self, iterations=5000, depth=6, learning_rate=0.01):
        """Train a fresh model from scratch"""
        logger.info("Training new CatBoost model...")
        
        self.model = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function='RMSE',
            eval_metric='RMSE',
            task_type="GPU" if self.use_gpu else "CPU",
            random_seed=self.random_state,
            verbose=200
        )
        
        # Create pools
        train_pool = Pool(self.X_train, self.y_train)
        val_pool = Pool(self.X_val, self.y_val)
        
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
        train_pool = Pool(self.X_train, self.y_train)
        val_pool = Pool(self.X_val, self.y_val)
        
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
        
        # Run cross-validation
        cv_results = cv(
            Pool(self.X_train, self.y_train),
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
            np.savez('predictions.npz',
                    y_test=self.y_test, y_pred_test=y_pred_test,
                    y_val=self.y_val, y_pred_val=y_pred_val)
            logger.info("Predictions saved to predictions.npz")
        
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
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Add timestamp if filename exists
        path = Path(filename)
        if path.exists():
            timestamp = int(time.time())
            stem, suffix = path.stem, path.suffix
            filename = f"{stem}_{timestamp}{suffix}"
        
        self.model.save_model(filename)
        logger.info(f"Model saved to {filename}")
        return filename

    def run_complete_pipeline(self, model_config=None):
        """Run the complete pipeline from data loading to evaluation"""
        if model_config is None:
            model_config = {
                'iterations': 5000,
                'depth': 6,
                'learning_rate': 0.01
            }
        
        logger.info("Starting complete PhotoZ pipeline...")
        
        DR1, legacy = self.load_astronomical_data()
        
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

    def plot_2d_hist_validation(self, bins=50, save_path=f'catboost_2d_hist_valid.png'):
        """Plot 2D histogram"""
        x = self.y_val
        y = self.model.predict(self.X_val)

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
    def plot_2d_hist_test(self, bins=50, save_path='catboost_2d_hist_test.png'):
        """Plot 2D histogram"""
        x = self.y_test
        y = self.model.predict(self.X_test)

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
    def plot_2d_hist_complete(self, bins=50, save_path=f'catboost_2d_hist_complete.png'):
        """Plot 2D histogram"""
        x = self.y
        y = self.model.predict(self.X)



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
    
    
    



    
    
    


def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description="CatBoost PhotoZ Pipeline")
    parser.add_argument("--gpu", action="store_true", default=False, help="Use GPU if available")
    parser.add_argument("--mode", choices=['train', 'continue', 'evaluate'], default='train',
                       help="Pipeline mode")
    parser.add_argument("--model_path", type=str, help="Path to saved model")
    parser.add_argument("--iterations", type=int, default=5000, help="Number of iterations")
    parser.add_argument("--depth", type=int, default=7, help="Tree depth")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument('--plot',type=str,default=None,help='Plot type: 2d_hist_valid, 2d_hist_test, 2d_hist_complete')
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PhotoZCatBoostPipeline(use_gpu=args.gpu)
    
    # Load and prepare data (always needed)
    DR1, legacy = pipeline.load_astronomical_data()
    legacy = pipeline.compute_magnitudes(legacy)
    legacy_filtered, DR1_filtered = pipeline.filter_data(legacy, DR1)
    X, y = pipeline.prepare_ml_data(legacy_filtered, DR1_filtered, feature_names=['MAG_GR', 'MAG_G', 'MAG_RZ', 'MAG_R', 'MAG_RW1', 'MAG_ZW1', 'MAG_W1W2','TYPE','SERSIC'])
    pipeline.create_data_splits(X, y)
    
    # Execute based on mode
    if args.mode == 'train':
        # Train new model
        pipeline.train_new_model(
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.learning_rate
        )
        pipeline.evaluate_model()
        pipeline.save_model("catboost_model_new.cbm")
        
    elif args.mode == 'continue':
        if not args.model_path:
            raise ValueError("--model_path required for continue mode")
        
        # Continue training
        pipeline.continue_training(args.model_path, additional_iterations=1000)
        pipeline.evaluate_model()
        pipeline.save_model("catboost_model_continued.cbm")
        
    elif args.mode == 'evaluate':
        if not args.model_path:
            raise ValueError("--model_path required for evaluate mode")
        
        # Just evaluate existing model
        pipeline.load_pretrained_model(args.model_path)
        pipeline.evaluate_model()
        if args.plot == '2d_hist_valid':
            pipeline.plot_2d_hist_validation()
        elif args.plot == '2d_hist_test':
            pipeline.plot_2d_hist_test()
        elif args.plot == '2d_hist_complete':
            pipeline.plot_2d_hist_complete()


if __name__ == "__main__":
    main()
