import fitsio
import numpy as np
import os
from multiprocessing import Pool, cpu_count
import os
import psutil
import resource
import time
from astropy.table import Table, hstack
import argparse


process = psutil.Process(os.getpid())
start_time = time.time()
print("Detected CPUs:", os.cpu_count())


def process_file_pair(paths):
    fits_path, pz_path = paths
    name = os.path.basename(fits_path).replace('.fits', '')
    cols1 = [ 'BRICKID','OBJID','BRICKNAME','RA', 'DEC', 'FLUX_G','FLUX_R','FLUX_Z','MW_TRANSMISSION_G','MW_TRANSMISSION_R','MW_TRANSMISSION_Z','SHAPE_E1','SHAPE_E2','SHAPE_R','SERSIC','TYPE','NOBS_G','NOBS_R','NOBS_Z','MASKBITS','FRACMASKED_G','FRACMASKED_R','FRACMASKED_Z','FRACFLUX_G','FRACFLUX_R','FRACFLUX_Z','FRACIN_G', 'FRACIN_R',
       'FRACIN_Z','GAIA_PHOT_G_MEAN_MAG','NOBS_G','NOBS_R','NOBS_Z','FIBERFLUX_G', 'FIBERFLUX_R', 'FIBERFLUX_Z']
    cols2 = ['Z_PHOT_MEDIAN','Z_PHOT_L95']
    dr9_chunk = Table(fitsio.FITS(fits_path)[1].read(columns=cols1))
    dr9_chunk_pz = Table(fitsio.FITS(pz_path)[1].read(columns=cols2))
    mask = (
    (dr9_chunk_pz['Z_PHOT_MEDIAN'] == -99) |
    (dr9_chunk_pz['Z_PHOT_L95'] == -99) |
    (dr9_chunk['FLUX_G'] == 0) |
    (dr9_chunk['FLUX_R'] == 0) |
    (dr9_chunk['FLUX_Z'] == 0))
	
    total_num=len(dr9_chunk)
    num_masked = np.sum(mask)
    #dr9_chunk= dr9_chunk[~mask]
    #dr9_chunk_pz = dr9_chunk_pz[~mask]
    
    MAG = {}
    FIBERMAG = {}
    MAG_NOEXT={}
    for i in ['G','R','Z']:
        flux = np.array(dr9_chunk[f'FLUX_{i}'])
        fiberflux= np.array(dr9_chunk[f'FIBERFLUX_{i}'])
        trans = np.array(dr9_chunk[f'MW_TRANSMISSION_{i}'])
        frac = flux / trans
        fracfiber= fiberflux /trans
        mag = np.empty_like(frac)
        magfiber = np.empty_like(fracfiber)
        mag_noext= np.empty_like(frac)
        np.log10(flux, out=mag_noext, where=(frac > 0))
        np.log10(frac, out=mag, where=(frac > 0))
        np.log10(fracfiber, out=magfiber, where=(fracfiber > 0))
        MAG[i] = 22.5 - 2.5 * mag
        FIBERMAG[i] = 22.5 - 2.5 * magfiber
        MAG_NOEXT[i] = 22.5 - 2.5 * mag_noext
    e1, e2 = dr9_chunk['SHAPE_E1'], dr9_chunk['SHAPE_E2']
    epsilon = np.sqrt(e1**2 + e2**2)
    bba = (1 - epsilon) / (1 + epsilon)
    r_circ = np.sqrt(bba) * dr9_chunk['SHAPE_R']
    dr9_chunk['R_CIRC'] = r_circ
    dr9_chunk['MAG_G'] = MAG['G']
    dr9_chunk['MAG_R'] = MAG['R']
    dr9_chunk['MAG_Z'] = MAG['Z']
    dr9_chunk['FIBERMAG_G'] = FIBERMAG['G']
    dr9_chunk['FIBERMAG_R'] = FIBERMAG['R']
    dr9_chunk['FIBERMAG_Z'] = FIBERMAG['Z']
    dr9_chunk['MAG_NOEXT_G'] = MAG_NOEXT['G']
    dr9_chunk['MAG_NOEXT_R'] = MAG_NOEXT['R']
    dr9_chunk['MAG_NOEXT_Z'] = MAG_NOEXT['Z']


    cut1 = (MAG['R'] < 18)
    cut2 =  ((MAG['G'] - MAG['R']) > 0.68)
    cut3 = ((MAG['G'] - MAG['R']) > (1.3 * (MAG['R'] - MAG['Z']) - 0.05))
    cut4 =  ((MAG['G'] - MAG['R']) < (2.0 * (MAG['R'] - MAG['Z']) - 0.15))
    cut5 =  (r_circ > 0)
    cut6 = ((1 - bba) < 0.7)
    cut7 =  (((dr9_chunk['TYPE'] == 'SER') & (dr9_chunk['SERSIC'] > 2.5)) | (dr9_chunk['TYPE'] == 'DEV'))
    cut8 =   (dr9_chunk_pz['Z_PHOT_MEDIAN'] < 0.15)
    cut9  = (dr9_chunk_pz['Z_PHOT_L95'] < 0.1)
    num_cut1= np.count_nonzero(cut1)
    num_cut2= np.count_nonzero(cut2)
    num_cut3= np.count_nonzero(cut3)
    num_cut4= np.count_nonzero(cut4)
    num_cut5= np.count_nonzero(cut5)
    num_cut6= np.count_nonzero(cut6)
    num_cut7= np.count_nonzero(cut7)
    num_cut8= np.count_nonzero(cut8)
    num_cut9= np.count_nonzero(cut9)
    k = np.array([num for num in [num_cut1, num_cut2, num_cut3, num_cut4, num_cut5, num_cut6, num_cut7, num_cut8, num_cut9]])


    num_cum_cut1 = np.sum(cut1&cut2);num_cum_cut2 = np.sum(cut1&cut2&cut3);num_cum_cut3 = np.sum(cut1&cut2&cut3&cut4);num_cum_cut4 = np.sum(cut1&cut2&cut3&cut4&cut5);num_cum_cut5 = np.sum(cut1&cut2&cut3&cut4&cut5&cut6)
    num_cum_cut6 = np.sum(cut1&cut2&cut3&cut4&cut5&cut6&cut7)
    num_cum_cut7 = np.sum(cut1&cut2&cut3&cut4&cut5&cut6&cut7&cut8)
    num_cum_cut8 = np.sum(cut1&cut2&cut3&cut4&cut5&cut6&cut7&cut8&cut9)
    num_cum = np.array([num for num in [num_cut1,num_cum_cut1, num_cum_cut2, num_cum_cut3, num_cum_cut4, num_cum_cut5, num_cum_cut6, num_cum_cut7, num_cum_cut8]])
    final=dr9_chunk[cut1&cut2&cut3&cut4&cut5&cut6&cut7&cut8&cut9]
    final_mask=cut1&cut2&cut3&cut4&cut5&cut6&cut7&cut8&cut9

    l=np.count_nonzero(dr9_chunk['SHAPE_R']==0)
    dct={'name':name,'total': total_num,'Zero_r':l, 
	  'masked': num_masked,
		  'cuts': k,
		  'cumulative': num_cum,'final:': len(final),'final_mask':final_mask}
    return dct



def main(directory,runtime=1):
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
        file_pairs=[random.choice(file_pairs),random.choice(file_pairs)]

 
    with Pool(cpu_count()) as pool:
        results= pool.map(process_file_pair, file_pairs)
    import pandas as pd
    df = pd.DataFrame(results)
    #print(df['Negative_r'],df['total'])
    
    df.to_parquet(f"summary_cuts_{directory}_SHAPE_R.parquet")
    

    


		




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory name for the DR9 data south or north", type=str, default="north")
    parser.add_argument("runtime", help="Debug mode, if set to 0, will only run one file", type=int, default=1)
    args = parser.parse_args()

    main(args.directory, args.runtime)

	
	







    


    
    
    

        