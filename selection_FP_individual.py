import numpy as np
import os
from multiprocessing import Pool, cpu_count, process
import os
import psutil
import resource
import time
from astropy.table import Table, hstack, vstack
import fitsio
import argparse
process = psutil.Process(os.getpid())
start_time = time.time()



parser = argparse.ArgumentParser(description='Select survey part (north or south).')
parser.add_argument('-d', '--directory', choices=['north', 'south'], default='north',
                    help='Which part to process (default: full)')
parser.add_argument('-z', '--zeropoint_shift', type=float, default=0.0,
                    help='Zeropoint shift to apply to r band magnitudes (default: 0.0)')
parser.add_argument('-c','--cuts',type = int, default=0,help='Number of cuts to apply ( default =0 , i.e all cuts, 1 means apply only R band cut, 2 means R and G-R cut and so on till 4 , 10=z cuts )' )

parser.add_argument('-m', '--mode', choices=['default', 'cumulative', 'individual'], default='default',
                    help=(
                        'Cut mode:\n'
                        '  default     : existing behaviour, one output table\n'
                        '  cumulative  : 9 tables; table_i has cuts 1..i applied in sequence\n'
                        '  individual  : 9 tables; table_i has only cut i applied alone'
                    ))
args = parser.parse_args()
directory= args.directory
zpt = args.zeropoint_shift
cut_args = args.cuts
mode = args.mode


CUT_LABELS = [
    'cut1_R_lt_18',
    'cut2_GR_gt_068',
    'cut3_slope1',
    'cut4_slope2',
    'cut5_Rcirc_gt_0',
    'cut6_BBA_lt_07',
    'cut7_sersic',
    'cut8_Zmed_lt_015',
    'cut9_ZL95_lt_01',
]

def _compute_mags_and_shapes(dr9_chunk, fits_path):
    """Reusable helper: compute MAG, FIBERMAG, MAG_NOEXT, r_circ, bba."""
    MAG, FIBERMAG, MAG_NOEXT = {}, {}, {}
    for i in ['G', 'R', 'Z', 'W1']:
        flux  = np.array(dr9_chunk[f'FLUX_{i}'])
        trans = np.array(dr9_chunk[f'MW_TRANSMISSION_{i}'])
        frac  = flux / trans
        mag       = np.full_like(frac, np.nan)
        mag_noext = np.full_like(frac, np.nan)
        np.log10(flux, out=mag_noext, where=(frac > 0))
        np.log10(frac, out=mag,       where=(frac > 0))
        MAG[i]       = 22.5 - 2.5 * mag
        MAG_NOEXT[i] = 22.5 - 2.5 * mag_noext
    for i in ['G', 'R', 'Z']:
        fiberflux  = np.array(dr9_chunk[f'FIBERFLUX_{i}'])
        trans      = np.array(dr9_chunk[f'MW_TRANSMISSION_{i}'])
        fracfiber  = fiberflux / trans
        magfiber   = np.full_like(fracfiber, np.nan)
        np.log10(fracfiber, out=magfiber, where=(fracfiber > 0))
        FIBERMAG[i] = 22.5 - 2.5 * magfiber
 
    if zpt != 0.0 and 'north' in fits_path:
        MAG['R'] -= zpt
        MAG['G'] -= zpt
 
    e1, e2   = dr9_chunk['SHAPE_E1'], dr9_chunk['SHAPE_E2']
    epsilon  = np.sqrt(e1**2 + e2**2)
    bba      = (1 - epsilon) / (1 + epsilon)
    r_circ   = np.sqrt(bba) * dr9_chunk['SHAPE_R']
    return MAG, FIBERMAG, MAG_NOEXT, r_circ, bba
 



def selector(MAG,r_circ,bba,dr9_chunk,dr9_chunk_pz):
    return [MAG['R']<18,
            (MAG['G'] - MAG['R']) > 0.68,
            (MAG['G'] - MAG['R']) > (1.3 * (MAG['R'] - MAG['Z']) - 0.05),
            (MAG['G'] - MAG['R']) < (2.0 * (MAG['R'] - MAG['Z']) - 0.15),
            (r_circ > 0),
            ((1 - bba) < 0.7),
            (((dr9_chunk['TYPE'] == 'SER') & (dr9_chunk['SERSIC'] > 2.5)) | (dr9_chunk['TYPE'] == 'DEV')),
            (dr9_chunk_pz['Z_PHOT_MEDIAN'] < 0.15),
            (dr9_chunk_pz['Z_PHOT_L95'] < 0.1)]
def process_file_pair(paths):
        fits_path, pz_path = paths
    
    # try:
        cols1 = [ 'BRICKID','OBJID','BRICKNAME','RA', 'DEC','SHAPE_E1','SHAPE_E2','SHAPE_R' ,'FLUX_G','FLUX_R','FLUX_Z','MW_TRANSMISSION_G','MW_TRANSMISSION_R','MW_TRANSMISSION_Z','SHAPE_E1','SHAPE_E2','SERSIC','TYPE','NOBS_G','NOBS_R','NOBS_Z','MASKBITS','FRACMASKED_G','FRACMASKED_R','FRACMASKED_Z','FRACFLUX_G','FRACFLUX_R','FRACFLUX_Z','FRACIN_G', 'FRACIN_R',
       'FRACIN_Z','GAIA_PHOT_G_MEAN_MAG','NOBS_G','NOBS_R','NOBS_Z','FIBERFLUX_G', 'FIBERFLUX_R', 'FIBERFLUX_Z','FLUX_W1','FLUX_W2','FLUX_W3','FLUX_W4','MW_TRANSMISSION_W1','MW_TRANSMISSION_W2','MW_TRANSMISSION_W3','MW_TRANSMISSION_W4','PSFDEPTH_G','PSFDEPTH_R','PSFDEPTH_Z','FLUX_IVAR_G','FLUX_IVAR_R','FLUX_IVAR_Z','FLUX_IVAR_W1','FLUX_IVAR_W2','FLUX_IVAR_W3','PSFSIZE_G','PSFSIZE_R','PSFSIZE_Z','EBV',]
        cols2 = ['Z_PHOT_MEDIAN','Z_PHOT_L95']
        
        # Load data
        dr9_chunk = Table(fitsio.FITS(fits_path)[1].read(columns=cols1))
        dr9_chunk_pz = Table(fitsio.FITS(pz_path)[1].read(columns=cols2))

        # Compute magnitude
        MAG, FIBERMAG, MAG_NOEXT, r_circ, bba = _compute_mags_and_shapes(dr9_chunk, fits_path)
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
        

        
        # Apply cuts
        initial_count = len(dr9_chunk)
        cut_counts = {}

        cuts_diag = selector(MAG,r_circ,bba,dr9_chunk,dr9_chunk_pz)
        combined = np.ones(len(dr9_chunk), dtype=bool)
        for label, cut in zip(CUT_LABELS, cuts_diag):
            combined &= cut
            cut_counts[label] = np.sum(combined)
        print(f'Initial : {initial_count}' )
        prev = initial_count
        for label, count in cut_counts.items():
            print(f"  {label:30s}: kept={count:7d}  rejected={prev-count:7d}  cumulative_loss={initial_count-count}")
            prev = count
        total_num = len(dr9_chunk)
        cuts_all = selector(MAG,r_circ,bba,dr9_chunk,dr9_chunk_pz)
        
        k       = np.array([np.count_nonzero(c) for c in cuts_all])
        num_cum = np.array([np.count_nonzero(np.logical_and.reduce(cuts_all[:i+1]))
                            for i in range(len(cuts_all))])
    
        name = os.path.basename(fits_path).replace('.fits', '')
        dct  = {'name': name, 'total': total_num,
                'cuts': k, 'cumulative': num_cum}
        
        if mode == 'cumulative':
            tables = []
            combined = np.ones(len(dr9_chunk), dtype=bool)
            for c in cuts_all:
                combined &= c
                t = hstack([dr9_chunk[combined], dr9_chunk_pz[combined]])
                tables.append(t)
            return tables, dct
        elif mode == 'individual':
            tables = []
            for c in cuts_all:
                t = hstack([dr9_chunk[c], dr9_chunk_pz[c]])
                tables.append(t)
            return tables, dct
        else:
            if cut_args > 0 and cut_args <= 4:
                combined_cut = np.logical_and.reduce(cuts_all[:cut_args])
            elif cut_args == 10:
                combined_cut = cuts_all[0] & cuts_all[7] & cuts_all[8]
            elif cut_args == 11:
                combined_cut = cuts_all[0] & cuts_all[8]
            else:
                combined_cut = np.logical_and.reduce(cuts_all)
    
            final_table = hstack([dr9_chunk[combined_cut], dr9_chunk_pz[combined_cut]])
            dct['final_table_length'] = len(final_table)
            return final_table, dct




def main():
    input_dir1 = f"/storage/shadab/data/legacy_survey/dr9/{directory}/sweep/9.0/"
    input_dir2 = f"/storage/shadab/data/legacy_survey/dr9/{directory}/sweep/9.0-photo-z/"
    base_out = "/user/animesh.sah/FP_CUTS"
    if mode == 'cumulative':
        out_dir = os.path.join(base_out, f"cumulative_{directory}_test")
        os.makedirs(out_dir, exist_ok=True)
    elif mode == 'individual':
        out_dir = os.path.join(base_out, f"individual_{directory}_test")
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = base_out
        os.makedirs(out_dir, exist_ok=True)
        if cut_args == 0:
            output_file = os.path.join(out_dir, f"{directory}_cuts_v12.fits")
        else:
            output_file = os.path.join(out_dir, f"{directory}_cuts_only_{cut_args}.fits")
    file_pairs = []


    for fname in os.listdir(input_dir1):
        if fname.endswith(".fits") and "-pz" not in fname:
            pz_name = fname.replace(".fits", "-pz.fits")
            if os.path.exists(os.path.join(input_dir2, pz_name)):
                file_pairs.append((os.path.join(input_dir1, fname),
                                   os.path.join(input_dir2, pz_name)))
    print(f"Total matched files: {len(file_pairs)}")
    with Pool(cpu_count()) as pool:
        results= pool.map(process_file_pair, file_pairs)
    results = [r for r in results if r is not None]

    if mode in ('cumulative', 'individual'):
        # results is a list of  (list_of_9_tables, dct)
        all_tables, _ = zip(*results)          # all_tables: N_files × 9
        n_cuts = len(CUT_LABELS)
 
        for i in range(n_cuts):
            per_cut_tables = [all_tables[f][i] for f in range(len(all_tables))]
            merged = vstack(per_cut_tables)
            label  = CUT_LABELS[i]
            out_path = os.path.join(out_dir, f"{directory}_{mode}_{label}.fits")
            fitsio.write(out_path, merged.as_array(), clobber=True)
            print(f"[{mode}] Saved cut {i+1}/9 → {out_path}  (N={len(merged)})")
 
    else:
        tables, _ = zip(*results)
        merged = vstack(list(tables))
        print(f"Total objects with {cut_args} cuts in {directory}: {len(merged)}")
        fitsio.write(output_file, merged.as_array(), clobber=True)
        print(f"Saved → {output_file}")

if __name__ == "__main__":
    main()


    

usage    = resource.getrusage(resource.RUSAGE_SELF)
end_time = time.time()
mem_info = process.memory_info()
print(f"Memory usage:    {mem_info.rss / 1024**2:.2f} MB")
print(f"Wall time:       {end_time - start_time:.2f} s")
print(f"CPU percent:     {process.cpu_percent(interval=1.0)} %")
print(f"User CPU time:   {usage.ru_utime:.2f} s")
print(f"System CPU time: {usage.ru_stime:.2f} s")
print(f"Max memory:      {usage.ru_maxrss / 1024:.2f} MB")
 