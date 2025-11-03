import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from astropy.table import Table, vstack, unique
import fitsio
import argparse
import time
import logging
from pathlib import Path
from Metrics import PhotoZMetrics  # Your custom metrics class
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle
import os
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhotoZTabNetPipeline:
    """
    A complete pipeline for photometric redshift prediction using TabNet
    """

    def __init__(self, use_gpu=True, random_state=854, num_gpus=None):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.X = None
        self.y = None

        
        device = 'cuda' if self.use_gpu else 'cpu'
        logger.info(f"Initialized PhotoZ TabNet pipeline using {device.upper()}")
                # GPU setup
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            self.num_gpus = num_gpus if num_gpus else torch.cuda.device_count()
            self.device = 'cuda'
            logger.info(f"Using {self.num_gpus} GPUs")
            for i in range(self.num_gpus):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            self.num_gpus = 0
            self.device = 'cpu'
            logger.info("Using CPU")
    
    def load_astronomical_data(self):
        """Load and process astronomical survey data"""
        logger.info("Loading astronomical data...")
        start_time = time.time()
        
        # Load DR1 data
        # north_DR1 = Table(fitsio.FITS('../LEGACY+DR1/TABLE_DR1_north_v1_sep_1.fits')[1].read())
        # south_DR1 = Table(fitsio.FITS('../LEGACY+DR1/TABLE_DR1_south_v1_sep_1.fits')[1].read())
        north_DR1 = Table(fitsio.FITS('/Storage/animesh/PECVEL/LEGACY+DR1/TABLE_DR1_north_v1_sep_1.fits')[1].read())
        south_DR1 = Table(fitsio.FITS('/Storage/animesh/PECVEL/LEGACY+DR1/TABLE_DR1_south_v1_sep_1.fits')[1].read())
        DR1 = vstack([north_DR1, south_DR1])

        # Load Legacy data
        north_legacy = Table(fitsio.FITS('/Storage/animesh/PECVEL/LEGACY+DR1/TABLE_legacy_north_v1_sep_1.fits')[1].read())
        south_legacy = Table(fitsio.FITS('/Storage/animesh/PECVEL/LEGACY+DR1/TABLE_legacy_south_v1_sep_1.fits')[1].read())
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
            valid = (frac > 0) & np.isfinite(frac)

            mag = np.full_like(frac, fill_value=np.nan)
            mag_noext = np.full_like(frac, fill_value=np.nan)
            
            np.log10(flux, out=mag_noext, where=valid)
            np.log10(frac, out=mag, where=valid)

            MAG[band] = 22.5 - 2.5 * mag
            MAG_NOEXT[band] = 22.5 - 2.5 * mag_noext
            
            legacy[f'MAG_{band}'] = MAG[band]

        # Compute color indices
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
        
        return legacy
    
    def weighted_sigmoid(self, center=19,steepness =3.0):
        #print(self.X[:,1])
        #print('shape of X[:,1]:',self.X[:,1].shape)
        x = np.asarray(self.X_train[:, 1], dtype=float)

        z = steepness * (x - center)

        weights = 1 / (1 + np.exp(z))
        normed_weights = weights / np.mean(weights)  # Normalize to mean of 1
        return normed_weights

    def _add_features(self, legacy):
        """Add additional features to the dataset"""
        e1, e2 = legacy['SHAPE_E1'], legacy['SHAPE_E2']
        epsilon = np.sqrt(e1**2 + e2**2)
        bba = (1 - epsilon) / (1 + epsilon)
        r_circ = np.sqrt(bba) * legacy['SHAPE_R']
        legacy['BBA'] = bba
        legacy['R_CIRC'] = r_circ
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
        
        # Valid redshift cuts
        mask3 = np.isfinite(DR1['Z']) & (DR1['Z'] > 0)
        
        combined_mask = mask1 & mask2 & mask3
        logger.info(f"Filtered from {len(legacy)} to {np.sum(combined_mask)} objects")
        
        DR1_final = DR1[combined_mask]
        _, indices = np.unique(DR1_final['TARGETID'], return_index=True)
        DR1_final = unique(DR1_final, keys='TARGETID')

        legacy_final = legacy[combined_mask][indices]
        logger.info(f"Filtered from {len(legacy)} to {len(legacy_final)} objects for unique TARGETID")

        return legacy_final, DR1_final

    def prepare_ml_data(self, legacy_filtered, DR1_filtered, feature_names=None):
        """Prepare machine learning dataset"""
        if feature_names is None:
            feature_names = ['MAG_G', 'MAG_R', 'MAG_Z', 'MAG_W1', 'MAG_W2', 'MAG_W3',
                           'MAG_GR', 'MAG_RZ', 'MAG_GZ', 'MAG_W1W2', 'MAG_W2W3', 'MAG_W1W3',
                           'MAG_RW1', 'MAG_RW2', 'MAG_RW3', 'MAG_GW1', 'MAG_GW2', 'MAG_GW3',
                           'MAG_ZW1', 'MAG_ZW2', 'MAG_ZW3', 'SERSIC']
        
        self.feature_names = feature_names
        logger.info(f"Using features: {feature_names}")
        
        missing_features = [f for f in feature_names if f not in legacy_filtered.colnames]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")
        
        X = legacy_filtered[feature_names].to_pandas().to_numpy()
        y = np.array(DR1_filtered['Z'])
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
        
        self.X, self.y = X, y
        return X, y

    def create_data_splits(self, X, y, scale_features=True):
        """Create train/validation/test splits with optional feature scaling"""
        logger.info("Creating data splits...")
        
        # First split: 75% train, 25% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state
        )
        
        # Second split: 12.5% validation, 12.5% test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_state
        )
        
        # Scale features if requested
        if scale_features:
            logger.info("Scaling features...")
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)
            X = self.scaler.transform(X)
            self.X = X

        # Save scaler with same name as model (if model_path is provided)
        if hasattr(self, 'model_path') and self.model_path:
            scaler_filename = f"{Path(self.model_path).stem}_scaler.pkl"
            scaler_path = Path(self.model_path).parent / scaler_filename
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                logger.info(f"Scaler saved to {scaler_path}")
        
        # Store splits in the class
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        
        logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_new_model(self, n_d=32, n_a=32, n_steps=6, gamma=1.3, 
                       n_independent=2, n_shared=2, lambda_sparse=1e-4,
                       max_epochs=200, patience=20, batch_size=8192,virtual_batch_size=512,num_workers=8,weight=None):
        """Train a fresh TabNet model from scratch"""
        logger.info("Training new TabNet model...")
        effective_batch_size = batch_size * max(1, self.num_gpus // 2)
        logger.info(f"Using effective batch size: {effective_batch_size}")
        if weight=='sigmoid':
            weights = self.weighted_sigmoid(center = 19, steepness=3.0      )
            logger.info("Using weighted sigmoid for sample weights")
        self.model = TabNetRegressor(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            lambda_sparse=lambda_sparse,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=1e-3, weight_decay=1e-5),
            scheduler_params={"step_size": 50, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax',
            seed=self.random_state,
            verbose=10,
            device_name='cuda' if self.use_gpu else 'cpu'
        )
        
        # Train the model
        self.model.fit(
            X_train=self.X_train,
            y_train=self.y_train.reshape(-1, 1),
            eval_set=[(self.X_val, self.y_val.reshape(-1, 1))],
            eval_name=['validation'],
            eval_metric=['rmse', 'mae'],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True if self.use_gpu else False
        )
        
        logger.info("Model training completed")
        return self.model

    def load_pretrained_model(self, model_path):
        """Load a pre-trained model"""
        logger.info(f"Loading model from {model_path}")
        
        self.model = TabNetRegressor()
        self.model.load_model(model_path)
        
        # Load scaler if it exists
        scaler_path = Path(model_path).parent / 'scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("Scaler loaded successfully")
        
        logger.info("Model loaded successfully")
        return self.model

    def evaluate_model(self, save_predictions=True):
        """Comprehensive model evaluation"""
        if self.model is None:
            raise ValueError("No model found. Train or load a model first.")
        
        logger.info("Evaluating model performance...")
        
        results = {}
        
        # Test set evaluation
        y_pred_test = self.model.predict(self.X_test).flatten()
        
        results['test'] = {
            'mae': mean_absolute_error(self.y_test, y_pred_test),
            'mse': mean_squared_error(self.y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
            'r2': r2_score(self.y_test, y_pred_test)
        }
        
        # Validation set evaluation
        y_pred_val = self.model.predict(self.X_val).flatten()
        
        results['validation'] = {
            'mae': mean_absolute_error(self.y_val, y_pred_val),
            'mse': mean_squared_error(self.y_val, y_pred_val),
            'rmse': np.sqrt(mean_squared_error(self.y_val, y_pred_val)),
            'r2': r2_score(self.y_val, y_pred_val)
        }
        
        # Custom PhotoZ metrics
        try:
            test_photoz_metrics = PhotoZMetrics(self.y_test, y_pred_test)
            val_photoz_metrics = PhotoZMetrics(self.y_val, y_pred_val)
            
            results['test']['photoz'] = test_photoz_metrics
            results['validation']['photoz'] = val_photoz_metrics
        except Exception as e:
            logger.warning(f"PhotoZ metrics calculation failed: {e}")
        
        # Print results
        self._print_evaluation_results(results)
        
        if save_predictions:
            np.savez('predictions_tabnet.npz',
                    y_test=self.y_test, y_pred_test=y_pred_test,
                    y_val=self.y_val, y_pred_val=y_pred_val)
            logger.info("Predictions saved to predictions_tabnet.npz")
        
        return results

    def _print_evaluation_results(self, results):
        """Print formatted evaluation results"""
        print("\n" + "="*60)
        print("TABNET MODEL EVALUATION RESULTS")
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
        """Save the trained model and scaler"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Add timestamp if filename exists
        path = Path(filename)
        if path.exists():
            timestamp = int(time.time())
            stem, suffix = path.stem, path.suffix
            filename = f"{stem}_{timestamp}{suffix}"
        self.model_path = filename
        self.model.save_model(filename)
        
        # Save scaler
        scaler_path = path.parent / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"Model saved to {filename}")
        logger.info(f"Scaler saved to {scaler_path}")
        return filename

    def plot_feature_importance(self, save_path='tabnet_feature_importance.png'):
        """Plot feature importance from TabNet"""
        if self.model is None:
            raise ValueError("No model found. Train or load a model first.")
        
        logger.info("Computing feature importance...")
        
        # Get feature importances
        feature_importance = self.model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(feature_importance)[::-1]
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(indices)), feature_importance[indices])
        plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('TabNet Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        plt.close()

    def two_d_hist(self, x, y, bins=50):
        """Create 2D histogram"""
        binx = np.linspace(min(x), max(x), bins)
        biny = np.linspace(min(y), max(y), bins)
        hist, xedges, yedges = np.histogram2d(x, y, bins=[binx, biny])
        return hist, xedges, yedges

    def plot_2d_hist_validation(self, bins=50, save_path='tabnet_2d_hist_valid.png'):
        """Plot 2D histogram for validation set"""
        x = self.y_val
        y = self.model.predict(self.X_val).flatten()

        hist, xedges, yedges = self.two_d_hist(x, y, bins=bins)
        plt.figure(figsize=(8, 6))
        plt.imshow(hist.T, origin='lower', aspect='auto',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap='magma_r', norm=LogNorm())
        plt.plot(self.y_val, self.y_val, color='k', linestyle='--', label='y=x')
        plt.colorbar(label='Counts')
        plt.xlabel('True Redshift (z_spec)')
        plt.ylabel('Predicted Redshift (z_phot)')
        plt.title('TabNet: Validation Set Predictions')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"2D histogram saved to {save_path}")
        plt.close()

    def plot_2d_hist_test(self, bins=50, save_path='tabnet_2d_hist_test.png'):
        """Plot 2D histogram for test set"""
        x = self.y_test
        y = self.model.predict(self.X_test).flatten()

        hist, xedges, yedges = self.two_d_hist(x, y, bins=bins)
        plt.figure(figsize=(8, 6))
        plt.imshow(hist.T, origin='lower', aspect='auto',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap='magma_r', norm=LogNorm())
        plt.plot(self.y_test, self.y_test, 'k--', label='y=x')
        plt.colorbar(label='Counts')
        plt.xlabel('True Redshift (z_spec)')
        plt.ylabel('Predicted Redshift (z_phot)')
        plt.title('TabNet: Test Set Predictions')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"2D histogram saved to {save_path}")
        plt.close()

    def plot_2d_hist_complete(self, bins=50, save_path='tabnet_2d_hist_complete.png'):
        """Plot 2D histogram for complete dataset"""
        x = self.y
        y = self.model.predict(self.X).flatten()

        hist, xedges, yedges = self.two_d_hist(x, y, bins=bins)
        plt.figure(figsize=(8, 6))
        plt.imshow(hist.T, origin='lower', aspect='auto',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap='magma_r', norm=LogNorm())
        plt.plot(self.y, self.y, 'k--', label='y=x')
        plt.colorbar(label='Counts')
        plt.xlabel('True Redshift (z_spec)')
        plt.ylabel('Predicted Redshift (z_phot)')
        plt.title('TabNet: Complete Dataset Predictions')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"2D histogram saved to {save_path}")
        plt.close()


def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description="TabNet PhotoZ Pipeline")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU if available")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use (default: all available)")
    parser.add_argument("--mode", choices=['train', 'evaluate'], default='train',
                       help="Pipeline mode")
    parser.add_argument("--model_path", type=str, help="Path to saved model")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--virtual_batch_size", type=int, default=128, help="Virtual batch size")
    parser.add_argument("--n_d", type=int, default=32, help="Width of decision prediction layer")
    parser.add_argument("--n_a", type=int, default=32, help="Width of attention embedding")
    parser.add_argument("--n_steps", type=int, default=5, help="Number of steps in the architecture")
    parser.add_argument('--plot', type=str, default=None, 
                       help='Plot type: 2d_hist_valid, 2d_hist_test, 2d_hist_complete, feature_importance')
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PhotoZTabNetPipeline(use_gpu=args.gpu)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    
    # Load and prepare data (always needed)
    DR1, legacy = pipeline.load_astronomical_data()
    legacy = pipeline.compute_magnitudes(legacy)
    legacy_filtered, DR1_filtered = pipeline.filter_data(legacy, DR1)
    
    # Use all available features
    X, y = pipeline.prepare_ml_data(
        legacy_filtered, DR1_filtered,
        feature_names=['MAG_G', 'MAG_R', 'MAG_Z', 'MAG_W1', 'MAG_W2', 'MAG_W3',
                      'MAG_GR', 'MAG_RZ', 'MAG_GZ', 'MAG_W1W2', 'MAG_W2W3', 'MAG_W1W3',
                      'MAG_RW1', 'MAG_RW2', 'MAG_RW3', 'MAG_GW1', 'MAG_GW2', 'MAG_GW3',
                      'MAG_ZW1', 'MAG_ZW2', 'MAG_ZW3', 'R_CIRC', 'BBA', 
                      'SHAPE_E1', 'SHAPE_E2', 'SERSIC']
    )
    pipeline.create_data_splits(X, y, scale_features=True)
    
    # Execute based on mode
    if args.mode == 'train':
        # Train new model
        pipeline.train_new_model(
            n_d=args.n_d,
            n_a=args.n_a,
            n_steps=args.n_steps,
            max_epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )
        pipeline.evaluate_model()
        pipeline.save_model(args.model_path if args.model_path else "tabnet_photoz_model_sigmoid.zip")
        pipeline.plot_feature_importance()
        
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
        elif args.plot == 'feature_importance':
            pipeline.plot_feature_importance()


if __name__ == "__main__":
    main()