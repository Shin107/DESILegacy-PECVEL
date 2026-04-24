import subprocess
import argparse
import re
import os 
import glob
parser = argparse.ArgumentParser(description='Run w_theta_specific with different input paths')
parser.add_argument('--input_path', help='Path to the input file',default='/user/animesh.sah/FP_CUTS/data_cleaned_south_cumulative')
parser.add_argument('-e','--error', choices=['jackknife','bootstrap','none'], default='none',
                    help='Which error estimation method to use (default: none)')
parser.add_argument('-i','--individual_files', action='store_true', help='Individual separate files')
parser.add_argument('--name_external', type=str, default='', help='Additional name to add to the suffix')
parser.add_argument('--dir', type=str, default='w_theta_specific_results', help='Directory to save the results')
parser.add_argument('-p','--part', choices=['north', 'south'], default=None, help='Part of the sky (north or south). If not provided, it will be inferred from the input path.')
args = parser.parse_args()
path = args.input_path
dir_name = args.dir
if dir_name is not  None:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        os.makedirs(os.path.join(dir_name, 'w_theta_results'), exist_ok=True)
    #os.chdir(dir_name)
if not args.individual_files:
    part = path.split('/')[-1]

    if 'south' in part:
        part = 'south'
    elif 'north' in part:
        part = 'north'
    else:
        raise ValueError("Path must contain either 'south' or 'north' to determine the part of the sky.")
    print(f'Running w_theta_specific for {part} part of the sky.')
if not args.individual_files:
    for file in glob.glob(path+'/*.fits'): 
        print('='*10,f'Processing file: {file}','='*10)
        match = re.search(r'cut(.*)\.fits',file)
        if match: 
            name = match.group(1)
            #subprocess.run(['python3', 'w_theta_specific.py', '--file_path', file, '--suffix', name,'-p',part,'-e',args.error,'--add_weights','--kind', 'partial_percentile'])
            subprocess.run(['python3', 'w_theta_specific.py', '--file_path', file, '--suffix', name,'-p',part,'-e',args.error])

            
else: 
    subprocess.run(['python3', 'w_theta_specific.py', '--file_path', args.input_path, '--suffix', args.name_external,'-p',args.part,'-e',args.error,'--base_path',dir_name])



