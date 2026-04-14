import subprocess
import argparse
import re
import os 
import glob
parser = argparse.ArgumentParser(description='Run w_theta_specific with different input paths')
parser.add_argument('--input_path', help='Path to the input file',default='/user/animesh.sah/FP_CUTS/data_cleaned_south_cumulative')
parser.add_argument('-e','--error', choices=['jackknife','bootstrap','none'], default='none',
                    help='Which error estimation method to use (default: none)')
args = parser.parse_args()
path = args.input_path

part = path.split('/')[-1]
if 'south' in part:
    part = 'south'
elif 'north' in part:
    part = 'north'
else:
    raise ValueError("Path must contain either 'south' or 'north' to determine the part of the sky.")
print(f'Running w_theta_specific for {part} part of the sky.')
for file in glob.glob(path+'/*.fits'): 
    print('='*10,f'Processing file: {file}','='*10)
    match = re.search(r'cut(.*)\.fits',file)
    if match: 
        name = match.group(1)
        subprocess.run(['python3', 'w_theta_specific.py', '--file_path', file, '--suffix', name,'-p',part,'-e',args.error])

        





