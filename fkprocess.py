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
import numpy as np
from simpleDASreader4 import load_DAS_file
import glob
import os
import matplotlib.pyplot as plt
from scipy.signal import detrend, resample, butter, sosfiltfilt
from numpy.lib.stride_tricks import as_strided

def sliding_window_fft2d(arr, window_shape):
    step_y, step_x = window_shape[0] // 2, window_shape[1] // 2
    shape = (
        (arr.shape[0] - window_shape[0]) // step_y + 1,
        (arr.shape[1] - window_shape[1]) // step_x + 1,
        window_shape[0],
        window_shape[1]
    )
    strides = (
        arr.strides[0] * step_y,
        arr.strides[1] * step_x,
        arr.strides[0],
        arr.strides[1]
    )
    
    windows = as_strided(arr, shape=shape, strides=strides)
    
    results = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            win = windows[i, j]
            fft_result = np.fft.fftshift(np.abs((np.fft.rfft2(win,axes = (-1,-2)))),axes = 1)
            position = (i * step_y, j * step_x)
            results.append((position, fft_result))
    
    return results


files = ["C:\\Users\\Calder\\OneDrive - NTNU\\Desktop\\183137.hdf5"]
dst = "C:\\Users\\Calder\\Outputs\\FKtest"
signal,meta = load_DAS_file(files[0])

dt = meta['header']['dt']
dx = 4
windowshape = (1024,2048)
freqs = np.fft.rfftfreq(n=windowshape[0],d=dt)
wavenumber= np.fft.fftshift(np.fft.fftfreq(n=windowshape[0],d=dx))

fks = sliding_window_fft2d(signal,windowshape)
del signal

for i,f in enumerate(fks):
    relT = f[0][0] *dt
    cidx = f[0][1]
    channel = meta['header']['channels'][cidx]
    plt.figure() 
    plt.imshow(10*np.log10(f[1]), origin = 'lower',aspect = 'auto',extent = (np.min(wavenumber),np.max(wavenumber),np.min(freqs),np.max(freqs)))
    fname = os.path.join(dst,'FK_X' + str(channel) + '_T' + str(relT)+'.png')
    plt.ylim(0,50)
    plt.savefig(fname)
    plt.close()

# %%
#trying to vis some of the fks 

import numpy as np
import matplotlib.pyplot as plt

tmp = np.load("C:\\Users\\Calder\\Outputs\\FK_test20250528T163924\\FK\\FK512_T0_X3072_20220821T180017Z.npy")
plt.figure()
plt.imshow(tmp, aspect = 'auto', origin = 'lower')

# %%
