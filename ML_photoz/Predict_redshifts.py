import sys

import numpy as np
import fitsio
import yaml
import argparse
import logging
import time
from astropy.table import Table, vstack
from catboost import CatBoostRegressor
from pathlib import Path
from joblib import Parallel, delayed
from catboost_cv import  PhotoZCatBoostPipeline
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from astropy.coordinates import SkyCoord
import astropy.units as u

def compute_magnitudes(legacy):
    """Same as pipeline — recompute mags from (perturbed) fluxes"""
    for band in ['G', 'R', 'Z', 'W1', 'W2', 'W3']:
        flux  = np.array(legacy[f'FLUX_{band}'])
        trans = np.array(legacy[f'MW_TRANSMISSION_{band}'])
        frac  = flux / trans
        valid = (frac > 0) & np.isfinite(frac)

        mag = np.full_like(frac, fill_value=np.nan)
        np.log10(frac, out=mag, where=valid)
        legacy[f'MAG_{band}'] = 22.5 - 2.5 * mag

    color_combinations = [
        ('G','R'),('R','Z'),('G','Z'),
        ('W1','W2'),('W2','W3'),('W1','W3'),
        ('R','W1'),('R','W2'),('R','W3'),
        ('G','W1'),('G','W2'),('G','W3'),
        ('Z','W1'),('Z','W2'),('Z','W3'),
    ]
    for b1, b2 in color_combinations:
        legacy[f'MAG_{b1}{b2}'] = legacy[f'MAG_{b1}'] - legacy[f'MAG_{b2}']

    return legacy

def add_features(legacy):
    e1, e2 = legacy['SHAPE_E1'], legacy['SHAPE_E2']
    epsilon = np.sqrt(e1**2 + e2**2)
    bba = (1 - epsilon) / (1 + epsilon)
    legacy['BBA']    = bba
    legacy['R_CIRC'] = np.sqrt(bba) * legacy['SHAPE_R']
    legacy['MODEL_WEIGHT'] = legacy['DCHISQ'][:,2]/(legacy['DCHISQ'][:,2]+legacy['DCHISQ'][:,3])    #(p = dchi2dev/(dchi2dev+dchi2exp)) 
    
    return legacy

def add_features_shape_color(legacy):
    legacy['LOG_SERSIC']  = np.log1p(legacy['SERSIC'])
    legacy['LOG_SHAPE_R'] = np.log1p(legacy['SHAPE_R'])

    for col in legacy.colnames:
        if col.startswith('MAG_'):
            legacy[f'LOG_SERSIC_times_{col[4:]}'] = legacy['LOG_SERSIC'] * legacy[col]

    for band in ['G', 'R', 'Z', 'W1', 'W2', 'W3']:
        legacy[f'SNR_{band}'] = legacy[f'FLUX_{band}'] * np.sqrt(legacy[f'FLUX_IVAR_{band}'])

    return legacy

def build_feature_matrix(legacy, feature_names):
    return legacy[feature_names].to_pandas().to_numpy()


# Core perturbation + prediction


FLUX_BANDS = ['G', 'R', 'Z', 'W1', 'W2', 'W3']



# def perturb_fluxes(legacy, rng):
#     """
#     Add Gaussian noise to each flux band.
#     sigma = 1 / sqrt(FLUX_IVAR_X),  noise ~ N(0, sigma)
#     Returns a *copy* of the table with perturbed fluxes.
#     """
#     perturbed = legacy.copy()
#     for band in FLUX_BANDS:
#         ivar  = np.array(legacy[f'FLUX_IVAR_{band}'])
#         # Guard against zero/negative ivar
#         sigma = np.where(ivar > 0, 1.0 / np.sqrt(np.maximum(ivar, 1e-30)), 0.0)
#         noise = rng.normal(0.0, sigma)
#         perturbed[f'FLUX_{band}'] = np.array(legacy[f'FLUX_{band}']) + noise
#     return perturbed


# def predict_with_uncertainty(model, legacy_clean, feature_names,
#                               n_perturbations=1000, seed=42):
#     """
#     For each object, run `n_perturbations` forward passes with
#     flux-perturbed copies, collect predictions, compute statistics.

#     Returns dict of arrays, each of shape (n_objects,):
#         mean, median, std, l68, u68, l95, u95
#     """
#     rng = np.random.default_rng(seed)
#     n_objects = len(legacy_clean)

#     # Store all predictions: shape (n_perturbations, n_objects)
#     all_preds = np.empty((n_perturbations, n_objects), dtype=np.float32)

#     logger.info(f"Running {n_perturbations} perturbations on {n_objects} objects...")
#     t0 = time.time()

#     for i in range(n_perturbations):
#         # 1. Perturb raw fluxes
#         legacy_pert = perturb_fluxes(legacy_clean, rng)

#         # 2. Recompute ALL derived features from perturbed fluxes
#         legacy_pert = compute_magnitudes(legacy_pert)
#         legacy_pert = add_features(legacy_pert)
#         legacy_pert = add_features_shape_color(legacy_pert)

#         # 3. Build feature matrix and predict
#         X_pert = build_feature_matrix(legacy_pert, feature_names)
#         all_preds[i] = model.predict(X_pert).astype(np.float32)

#         if (i + 1) % 100 == 0:
#             elapsed = time.time() - t0
#             logger.info(f"  {i+1}/{n_perturbations} done  ({elapsed:.1f}s elapsed)")

#     logger.info(f"Perturbation loop finished in {time.time()-t0:.1f}s")

#     # ── Compute statistics across perturbations (axis=0) ──
#     results = {
#         'mean'  : np.mean  (all_preds, axis=0),
#         'median': np.median(all_preds, axis=0),
#         'std'   : np.std   (all_preds, axis=0),
#         'l68'   : np.percentile(all_preds, 16, axis=0),
#         'u68'   : np.percentile(all_preds, 84, axis=0),
#         'l95'   : np.percentile(all_preds,  2.5, axis=0),
#         'u95'   : np.percentile(all_preds, 97.5, axis=0),
#     }
#     # Convenience: asymmetric error bars
#     results['err_lo68'] = results['median'] - results['l68']
#     results['err_hi68'] = results['u68']    - results['median']
#     results['err_lo95'] = results['median'] - results['l95']
#     results['err_hi95'] = results['u95']    - results['median']

#     return results, all_preds


# ─────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────

def _single_perturbation(legacy_arrays, ivar_arrays, non_flux_cols,
                          feature_names, model_path, seed):
    """
    Single worker function — loads model locally to avoid pickling issues,
    runs one perturbation and returns predictions.
    
    Accepts raw numpy arrays instead of astropy Table (faster to pickle/pass).
    """
    rng = np.random.default_rng(seed)

    # Rebuild a minimal Table from arrays
    legacy_pert = Table()
    for col, arr in non_flux_cols.items():
        legacy_pert[col] = arr.copy()

    # Perturb fluxes
    for band in FLUX_BANDS:
        flux  = legacy_arrays[band].copy()
        ivar  = ivar_arrays[band]
        sigma = np.where(ivar > 0, 1.0 / np.sqrt(np.maximum(ivar, 1e-30)), 0.0)
        legacy_pert[f'FLUX_{band}']      = flux + rng.normal(0.0, sigma)
        legacy_pert[f'FLUX_IVAR_{band}'] = ivar  # unchanged

    # Recompute derived features
    legacy_pert = compute_magnitudes(legacy_pert)
    legacy_pert = add_features(legacy_pert)
    legacy_pert = add_features_shape_color(legacy_pert)

    # Load model inside worker (avoids CatBoost pickle issues across processes)
    model = CatBoostRegressor()
    model.load_model(model_path)

    X = build_feature_matrix(legacy_pert, feature_names)
    return model.predict(X).astype(np.float32)

def predict_with_uncertainty_parallel(model_path, legacy_clean,feature_names,
                                       n_perturbations=1000, seed=42, n_jobs=-1):
    """
    Parallelized version using joblib.
    n_jobs=-1 uses all available CPUs.
    """
    rng = np.random.default_rng(seed)
    # Generate all seeds upfront — one per perturbation
    seeds = rng.integers(0, 2**31, size=n_perturbations)

    # Pre-extract arrays once — cheaper to pass to workers than full Table
    legacy_arrays = {band: np.array(legacy_clean[f'FLUX_{band}'])
                     for band in FLUX_BANDS}
    ivar_arrays   = {band: np.array(legacy_clean[f'FLUX_IVAR_{band}'])
                     for band in FLUX_BANDS}

    # All non-flux columns needed for feature engineering
    needed_cols = [name for name in legacy_clean.colnames if 'FLUX_' not in name]
    non_flux_cols = {col: np.array(legacy_clean[col])
                     for col in needed_cols if col in legacy_clean.colnames}

    logger.info(f"Running {n_perturbations} perturbations "
                f"on {len(legacy_clean)} objects using {n_jobs} jobs...")
    t0 = time.time()

    results_list = Parallel(n_jobs=n_jobs, verbose=10, backend='loky')(
        delayed(_single_perturbation)(
            legacy_arrays, ivar_arrays, non_flux_cols,
            feature_names, model_path, int(seeds[i])
        )
        for i in range(n_perturbations)
    )

    logger.info(f"Parallel perturbation finished in {time.time()-t0:.1f}s")

    # Stack into (n_perturbations, n_objects)
    all_preds = np.vstack(results_list)

    results = {
        'mean'  : np.mean(all_preds, axis=0),
        'median': np.median(all_preds, axis=0),
        'std'   : np.std(all_preds, axis=0),
        'l68'   : np.percentile(all_preds, 16,  axis=0),
        'u68'   : np.percentile(all_preds, 84,  axis=0),
        'l95'   : np.percentile(all_preds,  2.5, axis=0),
        'u95'   : np.percentile(all_preds, 97.5, axis=0),
    }
    results['err_lo68'] = results['median'] - results['l68']
    results['err_hi68'] = results['u68']    - results['median']
    results['err_lo95'] = results['median'] - results['l95']
    results['err_hi95'] = results['u95']    - results['median']

    return results, all_preds


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def load_data(cfg):
    region = cfg['data'].get('region', 'both')
    dp, lp = cfg['data']['dr1'], cfg['data']['legacy']

    if region == 'north':
        legacy = Table(fitsio.FITS(lp['north'])[1].read())
    elif region == 'south':
        legacy = Table(fitsio.FITS(lp['south'])[1].read())
    else:
        legacy = vstack([Table(fitsio.FITS(lp['north'])[1].read()),
                         Table(fitsio.FITS(lp['south'])[1].read())])
    logger.info(f"Loaded {len(legacy)} objects for region='{region}'")
    return legacy

def save_results(results, all_preds, legacy, cfg, out_path):
    """Save statistics + full posterior samples to .npz"""
    save_dict = {k: v for k, v in results.items()}
    save_dict['all_preds'] = all_preds          # shape (n_pert, n_obj) — full posterior

    # Optionally attach object IDs if available
    if 'TARGETID' in legacy.colnames:
        save_dict['TARGETID'] = np.array(legacy['TARGETID'])

    np.savez_compressed(out_path, **save_dict)
    logger.info(f"Results saved to {out_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PhotoZ uncertainty estimation via flux perturbation")
    parser.add_argument("--config",        type=str, required=True)
    parser.add_argument("--model_path",    type=str, required=True)
    parser.add_argument("--n_perturbations", type=int, default=1000)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--output",        type=str, default=None)
    parser.add_argument("--region",        type=str, default=None)
    parser.add_argument("--n_jobs", type=int, default=-1,
                    help="Number of parallel jobs (-1 = all CPUs)")
    args = parser.parse_args()  
    out_path = args.output
    if out_path is None:
        sys.exit("Error: --output path is required to save results.")
    cfg = load_config(args.config)
    if args.region:
        cfg['data']['region'] = args.region

    region = cfg['data']['region']

    # ── Load model ──

    logger.info(f"Loading model from {args.model_path}")
    model = CatBoostRegressor()
    model.load_model(args.model_path)

    # ── Load & prepare clean data (no perturbation yet) ──
    pipeline = PhotoZCatBoostPipeline(
        use_gpu=cfg['model']['use_gpu'],
        random_state=cfg['model']['random_state'],region = cfg['data']['region']
    )    


    north = Table(fitsio.FITS('/user/animesh.sah/FP_CUTS/table_north_unique.fits')[1].read())
    south = Table(fitsio.FITS('/user/animesh.sah/FP_CUTS/table_south_unique.fits')[1].read())
    DR1_SGC=Table(fitsio.FITS('../desi_dr1_SGC.fits')[1].read())
    DR1_NGC=Table(fitsio.FITS('../desi_dr1_NGC.fits')[1].read())
    DR1=vstack([DR1_NGC,DR1_SGC])
    coords_dr = SkyCoord(ra=DR1['RA']*u.deg, dec=DR1['DEC']*u.deg)
    coords_south = SkyCoord(ra=south['RA']*u.deg, dec=south['DEC']*u.deg)
    idx, d2d, _ = coords_south.match_to_catalog_sky(coords_dr)
    matched_south = south[d2d < 1*u.arcsec]
    matched_DR_south = DR1[idx[d2d < 1*u.arcsec]]
    coords_north = SkyCoord(ra=north['RA']*u.deg, dec=north['DEC']*u.deg)
    idx, d2d, _ = coords_north.match_to_catalog_sky(coords_dr)
    matched_north = north[d2d < 1*u.arcsec]
    matched_DR_north = DR1[idx[d2d < 1*u.arcsec]]

    data_instance = PhotoZCatBoostPipeline(feature_names=cfg['features'])

    matched_north = data_instance.compute_magnitudes(matched_north)
    
    matched_north = data_instance._add_features(matched_north)
    matched_north_filtered, DR1_north_filtered = data_instance.filter_data(matched_north, matched_DR_north, feature_names = cfg['features'],return_all_columns=True)

    matched_south = data_instance.compute_magnitudes(matched_south)
    
    matched_south = data_instance._add_features(matched_south)
    matched_south_filtered, DR1_south_filtered = data_instance.filter_data(matched_south, matched_DR_south, feature_names = cfg['features'],return_all_columns=True)
    print(matched_south_filtered.colnames)
    # DR1, legacy = pipeline.load_astronomical_data(cfg)
    # legacy = pipeline.compute_magnitudes(legacy)
    # legacy = pipeline._add_features(legacy)

    # ── Run perturbation loop ──
    results, all_preds = predict_with_uncertainty_parallel(
    model_path=args.model_path,
    legacy_clean=matched_south_filtered,
    feature_names=cfg['features'],
    n_perturbations=args.n_perturbations,
    seed=args.seed,
    n_jobs=args.n_jobs        # <-- add this CLI arg
)

    # ── Print quick summary ──
    print("\n── Redshift estimate summary ──")
    print(f"  Mean  photo-z : {results['mean'].mean():.4f}  ± {results['std'].mean():.4f}")
    print(f"  Median photo-z: {results['median'].mean():.4f}")
    print(f"  Avg 68% CI width: {(results['u68']-results['l68']).mean():.4f}")
    print(f"  Avg 95% CI width: {(results['u95']-results['l95']).mean():.4f}")

    # ── Save ──
    save_results(results, all_preds, matched_south_filtered , cfg, out_path)


if __name__ == "__main__":
    main()