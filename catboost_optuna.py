import sys, site

# Manually add venv site-packages
#site.addsitedir('/Storage/animesh/env/lib/python3.10/site-packages')

#print("Python prefix:", sys.prefix)
#print("sys.path:", sys.path) 
import sys, site
import catboost 
import numpy as np
from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from astropy.table import Table, vstack
import fitsio
import matplotlib.pyplot as plt
from optuna.integration import CatBoostPruningCallback
import shap
import optuna
import argparse
import time
import multiprocessing
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def setup_args():
    """Setup command line arguments"""
    parser = argparse.ArgumentParser(description="CatBoost + Optuna tuner for photometric redshifts")
    parser.add_argument("--gpu", action="store_true", default=False, help="Use GPU if available")
    parser.add_argument("--n_trials", type=int, default=500, help="Number of Optuna trials")
    parser.add_argument("--db_name", type=str, default="optuna_catboost_photoz.db", help="Database name")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    return parser.parse_args()

def load_astronomical_data():
    """Load and process astronomical survey data"""
    logger.info("Loading astronomical data...")
    start_time = time.time()
    
    # Load DR1 data
    north_DR1 = Table(fitsio.FITS('/Storage/animesh/PECVEL/LEGACY+DR1/TABLE_DR1_north_v1_sep_1.fits')[1].read())
    south_DR1 = Table(fitsio.FITS('/Storage/animesh/PECVEL/LEGACY+DR1/TABLE_DR1_south_v1_sep_1.fits')[1].read())
    DR1 = vstack([north_DR1, south_DR1])

    # Load Legacy data
    north_legacy = Table(fitsio.FITS('/Storage/animesh/PECVEL/LEGACY+DR1/TABLE_legacy_north_v1_sep_1.fits')[1].read())
    south_legacy = Table(fitsio.FITS('/Storage/animesh/PECVEL/LEGACY+DR1/TABLE_legacy_south_v1_sep_1.fits')[1].read())
    legacy = vstack([north_legacy, south_legacy])
    
    load_time = time.time() - start_time
    logger.info(f'Data loaded in {load_time:.2f} seconds')
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
    color_combinations = [
        ('G', 'R'), ('R', 'Z'), ('G', 'Z'),
        ('W1', 'W2'), ('W2', 'W3'), ('W1', 'W3'),
        ('R', 'W1'), ('R', 'W2'), ('R', 'W3'),
        ('G', 'W1'), ('G', 'W2'), ('G', 'W3'),
        ('Z', 'W1'), ('Z', 'W2'), ('Z', 'W3')
    ]
    
    for band1, band2 in color_combinations:
        legacy[f'MAG_{band1}{band2}'] = legacy[f'MAG_{band1}'] - legacy[f'MAG_{band2}']
        
    """Apply quality cuts and filtering to the data"""
    logger.info("Applying data quality cuts...")
    
    # Flux detection cuts
    mask1 = (legacy['FLUX_G'] != 0) & (legacy['FLUX_R'] != 0) & (legacy['FLUX_Z'] != 0)
    
    # Finite value cuts
    numeric_cols = [c for c in legacy.colnames if np.issubdtype(legacy[c].dtype, np.number)]
    data = np.vstack([legacy[c] for c in numeric_cols]).T
    mask2 = np.isfinite(data).all(axis=1)
    
    # Valid redshift cuts
    mask3 = np.isfinite(DR1['Z']) & (DR1['Z'] > 0)   # Reasonable redshift range
    
    combined_mask = mask1 & mask2 & mask3
    logger.info(f"Filtered from {len(legacy)} to {np.sum(combined_mask)} objects")
    
    return legacy[combined_mask], DR1[combined_mask]
def prepare_ml_data(legacy_filtered, DR1_filtered, feature_names=None):

    if feature_names is None:
        feature_names = ['MAG_GR', 'MAG_G', 'MAG_RZ', 'MAG_R', 'MAG_RW1', 'MAG_ZW1', 'MAG_W1W2']
    
    logger.info(f"Using features: {feature_names}")
    
    X = legacy_filtered[feature_names].to_pandas().to_numpy()
    y = np.array(DR1_filtered['Z'])
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
    
    return X, y, feature_names
    
def create_data_splits(X, y, random_state=42):

    logger.info("Creating data splits...")
    
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )
    
    # Second split: 15% validation, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )
    
    logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def print_trial_ranges(study):
    """Print the range of hyperparameters tried in the study"""
    if len(study.trials) == 0:
        logger.info("No trials completed yet")
        return
    
    # Collect all parameter values
    param_ranges = {}
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            for param_name, param_value in trial.params.items():
                if param_name not in param_ranges:
                    param_ranges[param_name] = []
                param_ranges[param_name].append(param_value)
    
    # Print ranges
    logger.info("\n" + "="*50)
    logger.info("HYPERPARAMETER RANGES EXPLORED:")
    logger.info("="*50)
    
    for param_name, values in param_ranges.items():
        if isinstance(values[0], (int, float)):
            min_val = min(values)
            max_val = max(values)
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"{param_name:15} | Range: [{min_val:8.4f}, {max_val:8.4f}] | Mean: {mean_val:8.4f} ± {std_val:6.4f}")
        else:
            unique_vals = list(set(values))
            logger.info(f"{param_name:15} | Values: {unique_vals}")
    
    logger.info(f"\nTotal completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    logger.info("="*50 + "\n")

def objective(trial, train_pool, val_pool, use_gpu=False):
    """Optuna objective function"""
    # Suggest hyperparameters with wider, more reasonable ranges
    depth = trial.suggest_int("depth", 4, 12)  # Added missing range
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-1, log=True)
    iterations = trial.suggest_int("iterations", 1000, 15000)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-3, 50, log=True)
    border_count = trial.suggest_int("border_count", 32, 255)
    random_strength = trial.suggest_float("random_strength", 1e-3, 10, log=True)
    bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 10.0)
    
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    
    model = CatBoostRegressor(
        depth=depth,
        learning_rate=learning_rate,
        iterations=iterations,
        l2_leaf_reg=l2_leaf_reg,
        border_count=border_count,
        random_strength=random_strength,
        bagging_temperature=bagging_temperature,
        loss_function="RMSE",
        eval_metric="RMSE",
        task_type="GPU" if use_gpu else "CPU",
        devices=gpu_id if use_gpu else None,
        od_type="Iter",
        od_wait=600,
        verbose=False,
        random_seed=42
    )
    
    pruning_cb = CatBoostPruningCallback(trial, "RMSE")
    
    model.fit(
        train_pool, 
        eval_set=val_pool, 
        use_best_model=True, 
        callbacks=[pruning_cb]
    )
    
    rmse = model.get_best_score()["validation"]["RMSE"]
    return rmse    
   
    

def save_with_increment(filename='/Storage/animesh/PECVEL/Codes/optuna_catboost'):

    path = Path(filename)
    stem, suffix = path.stem, path.suffix

    counter = 0
    new_path = path
    while new_path.exists():
        counter += 1
        new_path = path.with_name(f"{stem}_{counter}{suffix}")

    return new_path
    
def main():
    print("Number of CPUs available:", multiprocessing.cpu_count())
    
    args = setup_args()
    logger.info(f'Using {"GPU" if args.gpu else "CPU"}')
    
    # Load and process data
    legacy,DR1 = load_astronomical_data()

    
    # Prepare ML data
    X, y, feature_names = prepare_ml_data(legacy, DR1)
    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(X, y, args.random_state)
    
    # Create CatBoost pools
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)
    
    
    # Setup Optuna study
    db_path = f"sqlite:///{args.db_name}"
    study = optuna.create_study(
        direction="minimize",
        study_name="catboost_photoz_tuning_v2",
        storage=db_path,
        load_if_exists=True
    )
    
    # Print existing trial ranges if study has previous trials
    if len(study.trials) > 0:
        print_trial_ranges(study)
    
    # Run optimization
    logger.info(f"Starting optimization with {args.n_trials} trials...")
    study.optimize(
        lambda trial: objective(trial, train_pool, val_pool, args.gpu), 
        n_trials=args.n_trials
    )
    
    # Print final results and ranges
    print_trial_ranges(study)
    
    if len(study.trials) > 0 and study.best_trial.state == optuna.trial.TrialState.COMPLETE:
        logger.info("="*50)
        logger.info("OPTIMIZATION RESULTS:")
        logger.info("="*50)
        logger.info("Best parameters:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"Best RMSE: {study.best_value:.6f}")
        
        # Test the best model
        best_model = CatBoostRegressor(**study.best_params)
        best_model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=False)
        
        test_pool = Pool(X_test, y_test)
        test_predictions = best_model.predict(test_pool)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        test_r2 = r2_score(y_test, test_predictions)
        
        logger.info(f"Test RMSE: {test_rmse:.6f}")
        logger.info(f"Test R²: {test_r2:.4f}")
        logger.info("="*50)
    else:
        logger.warning("No successful trial found.")
    
    logger.info(f'Results saved to {args.db_name}')

if __name__ == "__main__":
    main()

