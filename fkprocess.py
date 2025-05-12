#this file looks to begin processing into the fk domain using 2d fft
#%%
import numpy as np
from simpleDASreader4 import load_DAS_file
import glob
import os
import matplotlib.pyplot as plt
from scipy.signal import detrend, resample, butter, sosfiltfilt


directory = 'D:\\DAS\\ship'

files = glob.glob(os.path.join(directory, '*.hdf5'))
files = ["E:\\NORSAR01v2\\20220821\\dphi\\182607.hdf5"]
i = 0
for f in files:
    signal, meta = load_DAS_file(f)
    signal = signal[:,2000:6000]
    dt = meta['header']['dt']
    dx = 4
    sost = butter(N = 6,
                Wn = (5,30),
                btype = 'bandpass',
                fs = 1/dt,
                output = 'sos')


    signal = sosfiltfilt(sos = sost,
                        x = signal,
                        axis = 0)
    

    tmp= np.array_split(signal, 5, axis = 0)
    for t in tmp:
        i +=1
        freqs = np.fft.rfftfreq(n=t.shape[0],d=dt)
        wavenumber= np.fft.rfftfreq(n=t.shape[1],d=dx)
        fk = np.abs(np.fft.rfft2(t))
        fk /=np.min(fk)

        plt.figure(figsize = (5,5))
        plt.imshow(10*np.log10(fk), cmap = 'turbo', origin = 'lower', aspect = 'auto', extent = (np.min(wavenumber),np.max(wavenumber),np.min(freqs),np.max(freqs)))
        plt.colorbar(label = 'strain (dB)')
        plt.clim(40,60)
        plt.ylim(0,50)
        plt.xlabel('Wavenumber (1/m)')
        plt.ylabel('Frequency (Hz)')
        fname = os.path.join('C:\\Users\\Calder\\Outputs\\CornellPosterplots\\ship',str(i) + '.png')
        # plt.savefig(fname)
        # plt.close()

    del signal
    del meta

# %%
