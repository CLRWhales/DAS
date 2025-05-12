#this script simulates a hyperbola
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, resample, butter, sosfiltfilt
from simpleDASreader4 import load_DAS_file
from Calder_utils import faststack
import glob, os

directory = 'E:\\NORSAR01v2\\20220821\\dphi'
files=  glob.glob(os.path.join(directory, '*.hdf5'))
start = '182007'
stop = '190007'

low = 25, 
high = 99

start_idx = [i for i, s in enumerate(files) if start in s][0]
stop_idx = [i for i, s in enumerate(files) if stop in s][0]

files = files[int(start_idx):int(stop_idx)]



chans = range(15000)
i = 0
for f in files:
    signal, meta = load_DAS_file(f, chIndex=chans)
    name = os.path.basename(f).split('.')[0]
    #signal= signal[:,0:15000]

    signal = faststack(signal,5)
    dt = meta['header']['dt']

    sost = butter(N = 6,
            Wn = (5,30),
            btype = 'bandpass',
            fs = 1/dt,
            output = 'sos')
    
    signal = sosfiltfilt(sos = sost,
                        x = signal,
                        axis = 0)

    

    vis = 10*np.log10(abs(signal))
    min = np.percentile(vis,low)
    max = np.percentile(vis,high)
    plt.imshow(vis, cmap = 'turbo', aspect = 'auto')
    plt.colorbar()
    plt.clim(min,max)
    path = 'C:\\Users\\Calder\\Outputs\\waveformtx'
    fname = os.path.join(path,name + '.png')
    plt.savefig(fname)
    plt.close()
    i = i+1

# %%
