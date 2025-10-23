import numpy as np
import fitsio
from astropy.table import Table,vstack,join,unique
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
import scienceplots
plt.style.use(['science','grid','notebook'])
from matplotlib.colors import LogNorm

from matplotlib.colors import LogNorm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from astropy.io import fits
import sys
from Metrics import PhotoZMetrics
import argparse

from catboost import CatBoostRegressor  
model = CatBoostRegressor() 

parser = argparse.ArgumentParser(description='Plot results for CatBoost model')
parser.add_argument('--model', type=str, required=True, 
                    help='Path to the CatBoost model file')
parser.add_argument('--output', type=str, default='plots',
                    help='Output directory for plots (default: plots)')
args = parser.parse_args()

DR1_SGC=Table(fitsio.FITS('desi_dr1_SGC.fits')[1].read())
DR1_NGC=Table(fitsio.FITS('desi_dr1_NGC.fits')[1].read())
DR1=vstack([DR1_NGC,DR1_SGC])

DR1_unique = unique(DR1,['TARGETID']) 
_, unique_indices = np.unique(DR1['TARGETID'], return_index=True)
print(f"Original DR1 length: {len(DR1)}")
print(f"DR1_unique length: {len(DR1_unique)}")
print(f"Unique indices shape: {unique_indices.shape}")
north = Table(fitsio.FITS('/user/animesh.sah/FP_CUTS/north_cuts_v10_flux_w1.fits')[1].read())
south = Table(fitsio.FITS('/user/animesh.sah/FP_CUTS/south_cuts_v10_flux_w1.fits')[1].read())
combined = vstack([north,south])
table_selection = Table(fitsio.FITS('table_match_final.fits')[1].read())
common = join(
    table_selection,
    combined,
    keys=['BRICKID','OBJID','BRICKNAME'],
    join_type='inner'
)
coords_dr = SkyCoord(ra=DR1['RA']*u.deg, dec=DR1['DEC']*u.deg)
coords_legacy = SkyCoord(ra=common['RA_1']*u.deg, dec=common['DEC_1']*u.deg)
idx_legacy, idx_dr, sep2d, _ = coords_dr.search_around_sky(coords_legacy, 1.5*u.arcsec)
matched_dr = DR1[idx_dr]
matched_FP = common[idx_legacy]
cols_to_remove = [col for col in matched_FP.colnames if col.endswith('_2')]
matched_FP.remove_columns(cols_to_remove)

cols_to_rename = {col: col[:-2] for col in matched_FP.colnames if col.endswith('_1')}
matched_FP.rename_columns(tuple(cols_to_rename.keys()), tuple(cols_to_rename.values()))
def two_d_hist(x,y):
    binx = np.linspace((min(x)), max(x), 50)
    biny = np.linspace((min(y)), max(y), 50)
    hist, xedges, yedges = np.histogram2d(x, y, bins=[binx, biny])
    return hist, xedges, yedges

def add_colors(table):
    MAG = {}
    FIBERMAG = {}
    MAG_NOEXT={}
    for i in ['G','R','Z','W1','W2','W3']:
        flux = np.array(table[f'FLUX_{i}'])
        trans = np.array(table[f'MW_TRANSMISSION_{i}'])
        frac = flux / trans
        mag = np.empty_like(frac)
        mag_noext = np.empty_like(frac)
        np.log10(flux, out=mag_noext, where=(frac > 0))
        np.log10(frac, out=mag, where=(frac > 0))
        MAG[i] = 22.5 - 2.5 * mag
        MAG_NOEXT[i] = 22.5 - 2.5 * mag_noext

    table['MAG_G'] = MAG['G']
    table['MAG_R'] = MAG['R']
    table['MAG_Z'] = MAG['Z']
    table['MAG_W1'] = MAG['W1']
    table['MAG_W2'] = MAG['W2']
    table['MAG_W3'] = MAG['W3']

    table['MAG_GR'] = table['MAG_G'] - table['MAG_R']
    table['MAG_RZ'] = table['MAG_R'] - table['MAG_Z']
    table['MAG_GZ'] = table['MAG_G'] - table['MAG_Z']
    table['MAG_W1W2'] = table['MAG_W1'] - table['MAG_W2']
    table['MAG_W2W3'] = table['MAG_W2'] - table['MAG_W3']
    table['MAG_W1W3'] = table['MAG_W1'] - table['MAG_W3']
    table['MAG_RW1'] = table['MAG_R'] - table['MAG_W1']
    table['MAG_RW2'] = table['MAG_R'] - table['MAG_W2']
    table['MAG_RW3'] = table['MAG_R'] - table['MAG_W3']
    table['MAG_GW1'] = table['MAG_G'] - table['MAG_W1']
    table['MAG_GW3'] = table['MAG_G'] - table['MAG_W3']
    table['MAG_GW2'] = table['MAG_G'] - table['MAG_W2']
    table['MAG_ZW1'] = table['MAG_Z'] - table['MAG_W1']
    table['MAG_ZW2'] = table['MAG_Z'] - table['MAG_W2']
    table['MAG_ZW3'] = table['MAG_Z'] - table['MAG_W3']
    return table

def compute_magnitudes( legacy):
        """Compute magnitudes and color indices from flux measurements"""
        print("Computing magnitudes and colors...")
        
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
        #legacy = self._add_features(legacy)
        
        return legacy

#matched_FP = add_colors(matched_FP)

matched_FP = compute_magnitudes(matched_FP)

def _add_features( legacy):
    """Add additional features to the dataset"""
    e1,e2 = legacy['SHAPE_E1'], legacy['SHAPE_E2']
    epsilon = np.sqrt(e1**2 + e2**2)
    bba = (1 - epsilon) / (1 + epsilon)
    r_circ = np.sqrt(bba) * legacy['SHAPE_R']
    legacy['BBA'] = bba
    legacy['R_CIRC'] = r_circ
    return legacy

matched_FP = _add_features( matched_FP)


model.load_model(args.model)


y =matched_dr['Z']
feature_names =['MAG_G','MAG_R','MAG_Z','MAG_W1','MAG_W2','MAG_W3','MAG_GR','MAG_RZ','MAG_GZ','MAG_W1W2','MAG_W2W3','MAG_W1W3','MAG_RW1','MAG_RW2','MAG_RW3','MAG_GW1','MAG_GW2','MAG_GW3','MAG_ZW1','MAG_ZW2','MAG_ZW3','R_CIRC','BBA','SHAPE_E1','SHAPE_E2','SERSIC']
X = matched_FP[feature_names].to_pandas().to_numpy()
z_pred=model.predict(X)



binx = np.linspace(min(matched_dr['Z']), 
                    max(matched_dr['Z']), 50)
biny = np.linspace(min(z_pred), 
                    max(z_pred), 50)
hist, xedges, yedges = np.histogram2d(matched_dr['Z'], 
                                        z_pred, 
                                        bins=(binx, biny))

# Log scale plot
plt.figure(figsize=(10, 8))
plt.imshow(hist.T, origin='lower',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect='auto', cmap='Blues', norm=LogNorm())
plt.plot(z_pred, z_pred, 'w--', lw=0.7)
plt.colorbar(label = 'CatBoost predicted')
plt.title('FP sample')
plt.xlabel('Z (DR1)')


biny = np.linspace(min(matched_FP['Z_PHOT_MEDIAN']), max(matched_FP['Z_PHOT_MEDIAN']), 50)
binx = np.linspace(min(matched_dr['Z']), max(matched_dr['Z']), 50)
hist,xedges,yedges=np.histogram2d(matched_dr['Z'], matched_FP['Z_PHOT_MEDIAN'], bins=(binx, biny))
plt.imshow(hist.T, origin='lower',
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           aspect='auto',cmap='magma_r',norm=LogNorm(),alpha=0.5)

plt.plot(matched_FP['Z_PHOT_MEDIAN'],matched_FP['Z_PHOT_MEDIAN'],'k--',lw=0.7)
plt.colorbar(label = 'Photo Z legacy')
plt.xlabel('Spec Z (DR1)')
plt.savefig(f"{args.output}/2dhist_zvsphotz_FP_vs_CatBoost_{model}.png", dpi=300)

metric = PhotoZMetrics(matched_dr['Z'], z_pred)


print('MSE for Catboost :',mean_squared_error(matched_dr['Z'], z_pred))
print('MAE for Catboost :',mean_absolute_error(matched_dr['Z'], z_pred))
print('R2 for Catboost :',r2_score(matched_dr['Z'], z_pred))
print('Photo Z metrics')
for key, value in metric.summary().items():
     print(f"{key} for Catboost: {value}")

print('------------------------------------------------------------')
print('MSE for Legacy :',mean_squared_error(matched_dr['Z'], matched_FP['Z_PHOT_MEDIAN']))
print('MAE for Legacy :',mean_absolute_error(matched_dr['Z'], matched_FP['Z_PHOT_MEDIAN']))
print('R2 for Legacy :',r2_score(matched_dr['Z'], matched_FP['Z_PHOT_MEDIAN']))
print('Photo Z metrics')
metric_legacy = PhotoZMetrics(matched_dr['Z'], matched_FP['Z_PHOT_MEDIAN'])
for key, value in metric_legacy.summary().items():
     print(f"{key} for Legacy: {value}")


nbins = 5 
zbins = np.linspace(np.min(matched_dr['Z']), np.max(matched_dr['Z']), nbins+1)
bin_indices = np.digitize(matched_dr['Z'], zbins)
fig,axes = plt.subplots(1,5,figsize = (25,5))
axes = axes.flatten()
for i in range(1, nbins+1):
    mask = bin_indices == i
    phot_z_bin  = z_pred[mask]
    spec_z_bin = matched_dr['Z'][mask]
    bin_hist = np.linspace(zbins[0],zbins[-1],60)
    axes[i-1].hist(spec_z_bin,bins =bin_hist ,alpha =0.7, label='SpecZ',edgecolor='blue')
    axes[i-1].hist(phot_z_bin,bins =bin_hist ,alpha =0.55, label='CatBoost',edgecolor='green')
    axes[i-1].axvline(np.median(spec_z_bin), color='k', linestyle='--', label='SpecZ Median')
    axes[i-1].set_title(f'SpecZ bin: {zbins[i-1]:.3f} - {zbins[i]:.3f}')
    axes[i-1].legend()
    axes[i-1].set_xlabel('Predicted PhotoZ')
    axes[i-1].set_ylabel('Count')   
plt.tight_layout()
plt.savefig(f"{args.output}/zbin_histograms_FP_CatBoost_{model}.png", dpi=300)



DR1_north= Table(fitsio.FITS('/user/animesh.sah/DESI_PECVEL/TABLE_DR1_north_v1_sep_1.fits')[1].read())
DR1_south= Table(fitsio.FITS('/user/animesh.sah/DESI_PECVEL/TABLE_DR1_south_v1_sep_1.fits')[1].read())
DR1_full = vstack([DR1_north,DR1_south])
legacy_north = Table(fitsio.FITS('/user/animesh.sah/DESI_PECVEL/TABLE_legacy_north_v1_sep_1.fits')[1].read())
legacy_south = Table(fitsio.FITS('/user/animesh.sah/DESI_PECVEL/TABLE_legacy_south_v1_sep_1.fits')[1].read())
legacy_full = vstack([legacy_north,legacy_south])
mask1 = np.isfinite(DR1_full['Z'])
mask2  = DR1_full['Z']>0
mask3 = np.isfinite(legacy_full['Z_PHOT_MEDIAN'])
mask4 = legacy_full['Z_PHOT_MEDIAN']>0
DR1_full = DR1_full[mask1 & mask2 & mask4 &mask3]
legacy_full = legacy_full[mask1 & mask2 & mask4 &mask3]

legacy_full = add_colors(legacy_full)
legacy_full = _add_features( legacy_full)

legacy_full=add_colors(legacy_full)
#feature_names = ['MAG_GR', 'MAG_G', 'MAG_RZ', 'MAG_R', 'MAG_RW1', 'MAG_ZW1', 'MAG_W1W2']
X_full = legacy_full[feature_names].to_pandas().to_numpy()
z_pred_full=model.predict(X_full)

metric = PhotoZMetrics(DR1_full['Z'], z_pred_full)
print('MSE for Catboost :',mean_squared_error(DR1_full['Z'], z_pred_full))
print('MAE for Catboost :',mean_absolute_error(DR1_full['Z'], z_pred_full))
print('R2 for Catboost :',r2_score(DR1_full['Z'], z_pred_full))
print('Photo Z metrics')
for key, value in metric.summary().items():
     print(f"{key} for Catboost: {value}")

print('------------------------------------------------------------')
print('MSE for Legacy :',mean_squared_error(DR1_full['Z'], legacy_full['Z_PHOT_MEDIAN']))
print('MAE for Legacy :',mean_absolute_error(DR1_full['Z'], legacy_full['Z_PHOT_MEDIAN']))
print('R2 for Legacy :',r2_score(DR1_full['Z'], legacy_full['Z_PHOT_MEDIAN']))
print('Photo Z metrics')
metric_legacy = PhotoZMetrics(DR1_full['Z'], legacy_full['Z_PHOT_MEDIAN'])
for key, value in metric_legacy.summary().items():
     print(f"{key} for Legacy: {value}")

nbins = 5 
zbins = np.linspace(np.min(DR1_full['Z']), np.max(DR1_full['Z']), nbins+1)
bin_indices = np.digitize(DR1_full['Z'], zbins)
fig,axes = plt.subplots(1,5,figsize = (25,5))
axes = axes.flatten()
for i in range(1, nbins+1):
    mask = bin_indices == i
    phot_z_bin  = z_pred_full[mask]
    spec_z_bin = DR1_full['Z'][mask]
    bin_hist = np.linspace(zbins[0],zbins[-1],60)
    axes[i-1].hist(spec_z_bin,bins =bin_hist ,alpha =0.7, label='SpecZ',edgecolor='blue')
    axes[i-1].hist(phot_z_bin,bins =bin_hist ,alpha =0.55, label='CatBoost',edgecolor='green')
    axes[i-1].axvline(np.median(spec_z_bin), color='k', linestyle='--', label='SpecZ Median')
    axes[i-1].set_title(f'SpecZ bin: {zbins[i-1]:.3f} - {zbins[i]:.3f}')
    #axes[i-1].legend()
    axes[i-1].set_xlabel('Predicted PhotoZ')
    axes[i-1].set_ylabel('Count')   
plt.legend()
plt.tight_layout()
plt.savefig(f"{args.output}/zbin_histograms_full_CatBoost_{model}.png", dpi=300)

