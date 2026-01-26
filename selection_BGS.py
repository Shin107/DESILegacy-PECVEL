import numpy as np
import os
from multiprocessing import Pool, cpu_count
import os
import psutil
import resource
import time
from astropy.table import Table, hstack, vstack
import fitsio
import argparse




parser = argparse.ArgumentParser(description='Select survey part (north or south).')
parser.add_argument('-d', '--directory', choices=['north', 'south'], default='north',
                    help='Which part to process (default: full)')
parser.add_argument('-t','--test', action='store_true', help='Run on a test subset of the data')
args = parser.parse_args()
directory= args.directory
test = args.test

def BGS_mask(dr9_chunk):
    cuts2 = (
        (
            ((dr9_chunk['FIBERMAG_R'] < (5.1 + dr9_chunk['MAG_R'])) & (dr9_chunk['MAG_R'] <= 17.8)) |
            ((dr9_chunk['FIBERMAG_R'] < 22.9) & (dr9_chunk['MAG_R'] > 17.8) & (dr9_chunk['MAG_R'] < 20))
        ) &
        ((-1 < (dr9_chunk['MAG_G'] - dr9_chunk['MAG_R'])) &
         ((dr9_chunk['MAG_G'] - dr9_chunk['MAG_R']) < 4) &
         (-1 < (dr9_chunk['MAG_R'] - dr9_chunk['MAG_Z'])) &
         ((dr9_chunk['MAG_R'] - dr9_chunk['MAG_Z']) < 4)) &
        
        (dr9_chunk['NOBS_G'] > 0) &
        (dr9_chunk['NOBS_R'] > 0) &
        (dr9_chunk['NOBS_Z'] > 0) &
        
        (dr9_chunk['FLUX_G'] > 0) &
        (dr9_chunk['FLUX_R'] > 0) &
        (dr9_chunk['FLUX_Z'] > 0) &
        
        (
            ((dr9_chunk['GAIA_PHOT_G_MEAN_MAG'] - dr9_chunk['MAG_NOEXT_R']) > 0.6) |
            (dr9_chunk['GAIA_PHOT_G_MEAN_MAG'] == 0)
        ) &
        
        (~np.isin(dr9_chunk['MASKBITS'], [1, 12, 13])) &
        
        (dr9_chunk['FRACMASKED_G'] < 0.4) &
        (dr9_chunk['FRACMASKED_R'] < 0.4) &
        (dr9_chunk['FRACMASKED_Z'] < 0.4) &
        
        (dr9_chunk['FRACFLUX_G'] < 5) &
        (dr9_chunk['FRACFLUX_R'] < 5) &
        (dr9_chunk['FRACFLUX_Z'] < 5) &
        
        (dr9_chunk['FRACIN_G'] > 0.2) &
        (dr9_chunk['FRACIN_R'] > 0.2) &
        (dr9_chunk['FRACIN_Z'] > 0.2)
    )
    return cuts2


def process_file_pair(paths):
        fits_path= paths
    
    # try:
        cols1 = [ 'BRICKID','OBJID','BRICKNAME','RA', 'DEC','SHAPE_E1','SHAPE_E2','SHAPE_R' ,'FLUX_G','FLUX_R','FLUX_Z','MW_TRANSMISSION_G','MW_TRANSMISSION_R','MW_TRANSMISSION_Z','SHAPE_E1','SHAPE_E2','SERSIC','TYPE','NOBS_G','NOBS_R','NOBS_Z','MASKBITS','FRACMASKED_G','FRACMASKED_R','FRACMASKED_Z','FRACFLUX_G','FRACFLUX_R','FRACFLUX_Z','FRACIN_G', 'FRACIN_R',
       'FRACIN_Z','GAIA_PHOT_G_MEAN_MAG','NOBS_G','NOBS_R','NOBS_Z','FIBERFLUX_G', 'FIBERFLUX_R', 'FIBERFLUX_Z','FLUX_W1','FLUX_W2','FLUX_W3','FLUX_W4','MW_TRANSMISSION_W1','MW_TRANSMISSION_W2','MW_TRANSMISSION_W3','MW_TRANSMISSION_W4','PSFDEPTH_G','PSFDEPTH_R','PSFDEPTH_Z','FLUX_IVAR_G','FLUX_IVAR_R','FLUX_IVAR_Z','FLUX_IVAR_W1','FLUX_IVAR_W2','FLUX_IVAR_W3','PSFSIZE_G','PSFSIZE_R','PSFSIZE_Z','EBV',]
        cols2 = ['Z_PHOT_MEDIAN','Z_PHOT_L95']
        
        # Load data
        dr9_chunk = Table(fitsio.FITS(fits_path)[1].read(columns=cols1))

        # Compute magnitude
        MAG = {}
        FIBERMAG = {}
        MAG_NOEXT={}
        for i in ['G','R','Z','W1']:
            flux = np.array(dr9_chunk[f'FLUX_{i}'])
            #fiberflux= np.array(dr9_chunk[f'FIBERFLUX_{i}'])
            trans = np.array(dr9_chunk[f'MW_TRANSMISSION_{i}'])
            frac = flux / trans
            #fracfiber= fiberflux /trans
            mag = np.empty_like(frac)
            #magfiber = np.empty_like(fracfiber)
            mag_noext= np.empty_like(frac)
            np.log10(flux, out=mag_noext, where=(frac > 0))
            np.log10(frac, out=mag, where=(frac > 0))
            #np.log10(fracfiber, out=magfiber, where=(fracfiber > 0))
            MAG[i] = 22.5 - 2.5 * mag
            #FIBERMAG[i] = 22.5 - 2.5 * magfiber
            MAG_NOEXT[i] = 22.5 - 2.5 * mag_noext
        for i in ['G','R','Z']:
            fiberflux= np.array(dr9_chunk[f'FIBERFLUX_{i}'])
            trans = np.array(dr9_chunk[f'MW_TRANSMISSION_{i}'])
            fracfiber= fiberflux /trans
            magfiber = np.empty_like(fracfiber)
            np.log10(fracfiber, out=magfiber, where=(fracfiber > 0))
            FIBERMAG[i] = 22.5 - 2.5 * magfiber
        # Compute r_circ
        e1, e2 = dr9_chunk['SHAPE_E1'], dr9_chunk['SHAPE_E2']
        epsilon = np.sqrt(e1**2 + e2**2)
        bba = (1 - epsilon) / (1 + epsilon)
        r_circ = np.sqrt(bba) * dr9_chunk['SHAPE_R']
        dr9_chunk['R_CIRC'] = r_circ
        dr9_chunk['MAG_G'] = MAG['G']
        dr9_chunk['MAG_R'] = MAG['R']
        dr9_chunk['MAG_Z'] = MAG['Z']
        dr9_chunk['MAG_W1'] = MAG['W1']
        dr9_chunk['FIBERMAG_G'] = FIBERMAG['G']
        dr9_chunk['FIBERMAG_R'] = FIBERMAG['R']
        dr9_chunk['FIBERMAG_Z'] = FIBERMAG['Z']
        dr9_chunk['MAG_NOEXT_G'] = MAG_NOEXT['G']
        dr9_chunk['MAG_NOEXT_R'] = MAG_NOEXT['R']
        dr9_chunk['MAG_NOEXT_Z'] = MAG_NOEXT['Z']
        


        
        result={}
        



        cuts = BGS_mask(dr9_chunk)
        final_table = dr9_chunk[cuts]   
        
            
        

        return final_table

def main():
    input_dir1 = f"/storage/shadab/data/legacy_survey/dr9/{directory}/sweep/9.0/"
    
    output_file = f"/user/animesh.sah/FP_CUTS/{directory}_BGS_cuts_DR8_v0.fits"
    #os.makedirs(output_dir, exist_ok=True)

    file_pairs = []
    for fname in os.listdir(input_dir1):
        if fname.endswith('.fits'):
            file_pairs.append(os.path.join(input_dir1, fname))

    print(f"Total matched files: {len(file_pairs)}")
    if test:
        file_pairs = file_pairs[:10]
        print("Running in test mode. Processing only 10 files.")
    with Pool(cpu_count()) as pool:
        results= pool.map(process_file_pair, file_pairs)
    results = [r for r in results if r is not None]


    table = results
    merged = {}
    
    l=np.array([len(r) for r in table])
    print(l)
    print(sum(l))


    merged = vstack(table)  
    print(merged)

    print('length of the final table:',len(merged['RA']))

    fitsio.write(output_file,merged.as_array(),clobber=True)
if __name__ == "__main__":
    main()

usage = resource.getrusage(resource.RUSAGE_SELF)

end_time = time.time()

