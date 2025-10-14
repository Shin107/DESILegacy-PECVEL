import fitsio 
from astropy.table import Table,vstack,hstack
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
from multiprocessing import Pool, cpu_count
import fitsio
import argparse
from functools import partial
DR1_SGC=Table(fitsio.FITS('/user/animesh.sah/DESI_PECVEL/desi_dr1_SGC.fits')[1].read())
#DR1_v12=Table(fitsio.FITS('/user/animesh.sah/DESI_PECVEL/desi_dr1_v1.2.fits')[1].read())
DR1_NGC=Table(fitsio.FITS('/user/animesh.sah/DESI_PECVEL/desi_dr1_NGC.fits')[1].read())
DR1=vstack([DR1_NGC,DR1_SGC])



def cross_match(paths,angular_separation):
    fits_path, pz_path = paths
    name = os.path.basename(fits_path).replace('.fits', '')
    cols1 = [ 'BRICKID','OBJID','BRICKNAME','RA', 'DEC','SHAPE_E1','SHAPE_E2','SHAPE_R' 'FLUX_G','FLUX_R','FLUX_Z','MW_TRANSMISSION_G','MW_TRANSMISSION_R','MW_TRANSMISSION_Z','SHAPE_E1','SHAPE_E2','SHAPE_R','SERSIC','TYPE','NOBS_G','NOBS_R','NOBS_Z','MASKBITS','FRACMASKED_G','FRACMASKED_R','FRACMASKED_Z','FRACFLUX_G','FRACFLUX_R','FRACFLUX_Z','FRACIN_G', 'FRACIN_R',
       'FRACIN_Z','GAIA_PHOT_G_MEAN_MAG','NOBS_G','NOBS_R','NOBS_Z','FIBERFLUX_G', 'FIBERFLUX_R', 'FIBERFLUX_Z','FLUX_W1','FLUX_W2','FLUX_W3','FLUX_W4','MW_TRANSMISSION_W1','MW_TRANSMISSION_W2','MW_TRANSMISSION_W3','MW_TRANSMISSION_W4','PSFDEPTH_G','PSFDEPTH_R','PSFDEPTH_Z']
    cols2 = ['Z_PHOT_MEDIAN','Z_PHOT_L95','Z_PHOT_MEAN','Z_SPEC','SURVEY']
    dr9_chunk = Table(fitsio.FITS(fits_path)[1].read(columns=cols1))
    dr9_chunk_pz = Table(fitsio.FITS(pz_path)[1].read(columns=cols2))
    final_table = hstack([dr9_chunk, dr9_chunk_pz])
    coords_dr = SkyCoord(ra=DR1['RA']*u.deg, dec=DR1['DEC']*u.deg)
    coords_legacy = SkyCoord(ra=final_table['RA']*u.deg, dec=final_table['DEC']*u.deg)
    idx_legacy, idx_dr, sep2d, _ = coords_dr.search_around_sky(coords_legacy, angular_separation*u.arcsec)
    matched_dr1 = DR1[idx_dr]
    matched_legacy = final_table[idx_legacy]
    return matched_dr1, matched_legacy










def main(directory,runtime=1,angular_separation=1):
    input_dir1 = f"/storage/shadab/data/legacy_survey/dr9/{directory}/sweep/9.0/"
    input_dir2 = f"/storage/shadab/data/legacy_survey/dr9/{directory}/sweep/9.0-photo-z/"
    output_file = f"/user/animesh.sah/FP_CUTS/{directory}_cuts_SHAPE_R.fits"
    #os.makedirs(output_dir, exist_ok=True)
    file_pairs = []
    for fname in os.listdir(input_dir1):
        if fname.endswith(".fits") and "-pz" not in fname:
            pz_name = fname.replace(".fits", "-pz.fits")
            if os.path.exists(os.path.join(input_dir2, pz_name)):
                #output_path = os.path.join(output_dir, fname.replace(".fits", "_selected.fits"))
                file_pairs.append((os.path.join(input_dir1, fname),
                                   os.path.join(input_dir2, pz_name)))
    if runtime == 0:
        import random
        file_pairs=[random.choice(file_pairs),random.choice(file_pairs),random.choice(file_pairs)]
 

 
    with Pool(cpu_count()) as pool:
        results= pool.map(partial(cross_match, angular_separation=angular_separation), file_pairs)
    results = [r for r in results if r is not None]

    table_DR1, table_legacy = zip(*results)
    filename='test' if runtime==0 else next(f'v{i}' for i in range(1,100) if not os.path.exists(f'TABLE_DR1_{directory}_v{i}'))
    table_DR1_final = vstack(table_DR1)
    table_legacy_final = vstack(table_legacy)
    fitsio.write(f'TABLE_DR1_{directory}_{filename}_sep_{angular_separation}.fits',table_DR1_final.as_array(),clobber=True)
    fitsio.write(f'TABLE_legacy_{directory}_{filename}_sep_{angular_separation}.fits',table_legacy_final.as_array(),clobber=True)



    #     results= pool.map(process_file_pair, file_pairs)
    # import pandas as pd
    # df = pd.DataFrame(results)
    
    # df.to_parquet(f"summary_cuts_{directory}_SHAPE_R.parquet")
    

    


		




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory name for the DR9 data south or north", type=str, default="north")
    parser.add_argument("runtime", help="Debug mode, if set to 0, will only run one file", type=int, default=1)
    parser.add_argument("angular_separation", help="Angular separation for cross-matching", type=int, default=1)

    args = parser.parse_args()

    main(args.directory, args.runtime, args.angular_separation)
