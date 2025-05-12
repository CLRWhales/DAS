#this script reads in csv files, breaks down the timestamps and recreates them for copying into the drive
#%%

import numpy as np
import pandas as pd
import shutil
import glob
import os

csv_directory = 'C:\\Users\\Calder\\Outputs\\E7_cleaning\\NORSAR01v2_cleaning_outer\\ID_flags'
source = '\\some other directory'
destination = '\\b;ah'


csvs = glob.glob(os.path.join(csv_directory, '*.csv'))

for f in csvs:
    tmp = pd.read_csv(f)
    minutes = tmp['file_name'][tmp['whale_flag'] == 'W'].tolist()
    date = minutes[0].split('_')[1].split('T')[0]
    times = [min.split('_')[1].split('T')[1].split('Z')[0] for min in minutes]
    times= times[0:3]
    fnames = []
    vals = ['0', '1', '2', '3', '4', '5']
    for t in times:
        print(t)
        print(t[0:4])
        for v in vals:
            filename = t[0:4] + v + t[5]
            fnames.append(filename)

    for k in fnames:
        sourcepath = os.path.join(source,date,'dphi',k + '.hdf5')
        destpath = os.path.join(destination,date,'dphi',k + '.hdf5')
        shutil.copyfile(sourcepath,destpath)




# %%
