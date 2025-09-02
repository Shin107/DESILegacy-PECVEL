import numpy as np
import os
from multiprocessing import Pool, cpu_count
import os
import psutil
import resource
import time
from astropy.table import Table, hstack
import fitsio


process = psutil.Process(os.getpid())
start_time = time.time()
print("Detected CPUs:", os.cpu_count())

def process_file_pair(paths):
        fits_path, pz_path = paths
    
    # try:
        cols1 = [ 'BRICKID','OBJID','BRICKNAME','RA', 'DEC', 'FLUX_G','FLUX_R','FLUX_Z','MW_TRANSMISSION_G','MW_TRANSMISSION_R','MW_TRANSMISSION_Z','SHAPE_E1','SHAPE_E2','SHAPE_R','SERSIC','TYPE','NOBS_G','NOBS_R','NOBS_Z','MASKBITS','FRACMASKED_G','FRACMASKED_R','FRACMASKED_Z','FRACFLUX_G','FRACFLUX_R','FRACFLUX_Z','FRACIN_G', 'FRACIN_R',
       'FRACIN_Z','GAIA_PHOT_G_MEAN_MAG','NOBS_G','NOBS_R','NOBS_Z','FIBERFLUX_G', 'FIBERFLUX_R', 'FIBERFLUX_Z','FLUX_W1','MW_TRANSMISSION_W1']
        cols2 = ['Z_PHOT_MEDIAN','Z_PHOT_L95']
        
        # Load data
        dr9_chunk = Table(fitsio.FITS(fits_path)[1].read(columns=cols1))
        dr9_chunk_pz = Table(fitsio.FITS(pz_path)[1].read(columns=cols2))

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
        #dr9_chunk['FIBERMAG_W1'] = FIBERMAG['W1']
        dr9_chunk['MAG_NOEXT_G'] = MAG_NOEXT['G']
        dr9_chunk['MAG_NOEXT_R'] = MAG_NOEXT['R']
        dr9_chunk['MAG_NOEXT_Z'] = MAG_NOEXT['Z']
        

        
        # Apply cuts
        initial_count = len(dr9_chunk)
        cut_counts = {}

        # Cut 1: Basic MAG_R cut
        cut1 = (MAG['R'] < 18)
        cut_counts['R_mag'] = np.count_nonzero(cut1)
      
        # Cut 2: MAG_G - MAG_R color cut
        cut2 = cut1 & ((MAG['G'] - MAG['R']) > 0.68)
     
        cut_counts['G-R > 0.68'] = np.count_nonzero(cut2)

        # Cut 3: Sloped color cut 1
        cut3 = cut2 & ((MAG['G'] - MAG['R']) > (1.3 * (MAG['R'] - MAG['Z']) - 0.05))
        cut_counts['Slope1'] = np.count_nonzero(cut3)

        # Cut 4: Sloped color cut 2
        cut4 = cut3 & ((MAG['G'] - MAG['R']) < (2.0 * (MAG['R'] - MAG['Z']) - 0.15))
        cut_counts['Slope2'] = np.count_nonzero(cut4)

        # Cut 5: R_CIRC > 0
        cut5 = cut4 & (r_circ > 0)
        cut_counts['R_CIRC > 0'] = np.count_nonzero(cut5)

        # Continue similarly...
        # Example: combine all logical conditions as you already do
        cut6 = cut5 & ((1 - bba) < 0.7)
        cut_counts['BBA < 0.7'] = np.count_nonzero(cut6)
        # Example: Sersic type and Sersic index cut
        cut7 = cut6 & (((dr9_chunk['TYPE'] == 'SER') & (dr9_chunk['SERSIC'] > 2.5)) | (dr9_chunk['TYPE'] == 'DEV'))
        cut_counts['Sersic type and index'] = np.count_nonzero(cut7)
        # Example: Photo-z median and lower limit cuts
        #dr9_chunk_pz = dr9_chunk_pz[cut7]  # Apply previous cuts to photo-z data

        cut8 = cut7 & (dr9_chunk_pz['Z_PHOT_MEDIAN'] < 0.15)
        cut_counts['Z_PHOT_MEDIAN < 0.15'] = np.count_nonzero(cut8)
        cut9 = cut8&(dr9_chunk_pz['Z_PHOT_L95'] < 0.1)
        
        cut_counts['Z_PHOT_L95 < 0.1'] = np.count_nonzero(cut9)

        

        cuts = (
            (MAG['R'] < 18) &
            ((MAG['G'] - MAG['R']) > 0.68) &
            ((MAG['G'] - MAG['R']) > (1.3 * (MAG['R'] - MAG['Z']) - 0.05)) &
            ((MAG['G'] - MAG['R']) < (2.0 * (MAG['R'] - MAG['Z']) - 0.15)) &
            (r_circ > 0) &
            ((1 - bba) < 0.7) &
            (((dr9_chunk['TYPE'] == 'SER') & (dr9_chunk['SERSIC'] > 2.5)) | (dr9_chunk['TYPE'] == 'DEV')) &
            (dr9_chunk_pz['Z_PHOT_MEDIAN'] < 0.15) &
            (dr9_chunk_pz['Z_PHOT_L95'] < 0.1)
        )

#         cuts2= ( ((dr9_chunk['FIBERMAG_G']< (5.1+dr9_chunk['MAG_R']))& (dr9_chunk['MAG_R']<=17.8)) | 
                
             
#                 (dr9_chunk['NOBS_G']>0) & (dr9_chunk['NOBS_R']>0) & (dr9_chunk['NOBS_Z']>0) &
#         (dr9_chunk['FLUX_G']>0) & (dr9_chunk['FLUX_R']>0) & (dr9_chunk['FLUX_Z']>0) &
#         ((dr9_chunk['GAIA_PHOT_G_MEAN_MAG'] -dr9_chunk['MAG_NOEXT_R']>0.6 ) | (dr9_chunk['GAIA_PHOT_G_MEAN_MAG']==0))  &
#         ((dr9_chunk['MASKBITS']!=1)& (dr9_chunk['MASKBITS']!=12) & (dr9_chunk['MASKBITS']!=13)  )
#    &
#             (dr9_chunk['FRACMASKED_G']<0.4) & (dr9_chunk['FRACMASKED_R']<0.4) & (dr9_chunk['FRACMASKED_Z']<0.4) & (dr9_chunk['FRACFLUX_G']<5) & (dr9_chunk['FRACFLUX_R']<5) & (dr9_chunk['FRACFLUX_Z']<5)
#                 & (dr9_chunk['FRACIN_G']>0.2)& (dr9_chunk['FRACIN_R']>0.2) & (dr9_chunk['FRACIN_Z']>0.2) 
#                 )

        # Filter data

        result={}
        

        cut_counts['Final all cuts'] = np.count_nonzero(cuts)

        # Now print individual and cumulative rejections
        print(f"→ Initial: {initial_count}")
        prev = initial_count
        for label, count in cut_counts.items():
            rejected = prev - count
            print(f"{label:20s}: kept = {count}, rejected = {rejected}, cumulative loss = {initial_count - count}")
            prev = count
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
        e1, e2 = dr9_chunk['SHAPE_E1'], dr9_chunk['SHAPE_E2']
        epsilon = np.sqrt(e1**2 + e2**2)
        bba = (1 - epsilon) / (1 + epsilon)
        r_circ = np.sqrt(bba) * dr9_chunk['SHAPE_R']
        dr9_chunk['R_CIRC'] = r_circ
        dr9_chunk['MAG_G'] = MAG['G']
        dr9_chunk['MAG_R'] = MAG['R']
        dr9_chunk['MAG_Z'] = MAG['Z']
        dr9_chunk['MAG_W1'] = MAG['W1']
        # dr9_chunk['FIBERMAG_G'] = FIBERMAG['G']
        # dr9_chunk['FIBERMAG_R'] = FIBERMAG['R']
        # dr9_chunk['FIBERMAG_Z'] = FIBERMAG['Z']
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
        final_dr=dr9_chunk[cut1&cut2&cut3&cut4&cut5&cut6&cut7&cut8&cut9]

        dr9_chunk_sel = dr9_chunk[cut1&cut2&cut3&cut4&cut5&cut6&cut7&cut8&cut9]
        dr9_chunk_pz_sel = dr9_chunk_pz[cut1&cut2&cut3&cut4&cut5&cut6&cut7&cut8&cut9]

        # Merge and save
        # final = {}
        # for col in dr9_chunk_sel.dtype.names:
        #     final[col] = dr9_chunk_sel[col]
        # for col in dr9_chunk_pz_sel.dtype.names:
        #     final[col] = dr9_chunk_pz_sel[col]
        final_table = hstack([dr9_chunk_sel, dr9_chunk_pz_sel])
        name = os.path.basename(fits_path).replace('.fits', '')
        dct={'name':name,'total': total_num, 
        'masked': num_masked,
            'cuts': k,
            'cumulative': num_cum,'final_table_length': len(final_table)}
            
        

    # except Exception as e:
    #     print(f"✗ Failed: {os.path.basename(fits_path)} → {e}")
    #print('\n\n ',len(final_table),'\n\n')
        return final_table,  dct

def main():
    input_dir1 = "/storage/shadab/data/legacy_survey/dr9/south/sweep/9.0/"
    input_dir2 = "/storage/shadab/data/legacy_survey/dr9/south/sweep/9.0-photo-z/"
    output_file = "/user/animesh.sah/FP_CUTS/south_cuts_v9_w1.fits"
    #os.makedirs(output_dir, exist_ok=True)

    file_pairs = []
    for fname in os.listdir(input_dir1):
        if fname.endswith(".fits") and "-pz" not in fname:
            pz_name = fname.replace(".fits", "-pz.fits")
            if os.path.exists(os.path.join(input_dir2, pz_name)):
                #output_path = os.path.join(output_dir, fname.replace(".fits", "_selected.fits"))
                file_pairs.append((os.path.join(input_dir1, fname),
                                   os.path.join(input_dir2, pz_name)))
    #print(file_pairs[:20])
    #file_pairs=file_pairs[:]  
    #import random
    #file_pairs=[random.choice(file_pairs),random.choice(file_pairs)]

 
    print(f"Total matched files: {len(file_pairs)}")
    #print(file_pairs[:20])
    #print(hi)


    # Run in parallel
    with Pool(cpu_count()) as pool:
        results= pool.map(process_file_pair, file_pairs)
    results = [r for r in results if r is not None]


    table, cut_counts = zip(*results)
    merged = {}
    
    l=np.array([len(r) for r in table])
    print(l)
    print(sum(l))
    #np.save('lenghts_mask.npy',l)
    # for key in table[0].keys():
    #     merged[key] = np.concatenate([r[key] for r in table])

    from astropy.table import vstack


    merged = vstack(table)  # table can be a tuple

    print(len(merged['RA']))
    print(len(merged.as_array()))
    import pickle
    with open("south_dict_v1.pkl", "wb") as f:
        pickle.dump(merged, f)
    fitsio.write(output_file,merged.as_array(),clobber=True)
    #summary = {}
    #import pandas as pd
    #df = pd.DataFrame(cut_counts)
    
    #df.to_parquet(f"summary_cuts_south_v5_selection_code.parquet")
    
    # for d in cut_counts:
    #     for k, v in d.items():
    #         summary[k] = summary.get(k, 0) + v

    # # Print summary
    # print("Summary of selection cuts:")
    # for key, value in summary.items():
    #     print(f"  {key}: {value}")

if __name__ == "__main__":
    main()

usage = resource.getrusage(resource.RUSAGE_SELF)

end_time = time.time()
mem_info = process.memory_info()

print(f"Memory usage: {mem_info.rss / 1024**2:.2f} MB")
print(f"Wall time: {end_time - start_time:.2f} s")
print(f"CPU percent: {process.cpu_percent(interval=1.0)} %")
print(f"User CPU time: {usage.ru_utime:.2f} s")
print(f"System CPU time: {usage.ru_stime:.2f} s")
print(f"Max memory usage: {usage.ru_maxrss / 1024:.2f} MB")