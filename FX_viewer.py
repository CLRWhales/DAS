#%%
import glob, os
import numpy as np
import matplotlib.pyplot as plt
directory = 'C:\\Users\\Calder\\Outputs\\BF_out_full_20250514T202112\\Complex'
freqs = np.loadtxt("C:\\Users\\Calder\\Outputs\\BF_out_full_20250514T202112\\Dim_Frequency.txt")
chans = np.loadtxt("C:\\Users\\Calder\\Outputs\\BF_out_full_20250514T202112\\Dim_Channel.txt")
times = np.loadtxt("C:\\Users\\Calder\\Outputs\\BF_out_full_20250514T202112\\Dim_Time.txt")
files = glob.glob(os.path.join(directory,'*.npy'))

start = '183537'
stop = '183547'
# start the routine
start_idx = [i for i, s in enumerate(files) if start in s][0]
stop_idx = [i for i, s in enumerate(files) if stop in s][0]
files = files[int(start_idx):int(stop_idx)]
FTX = np.load(files[0])
chans = np.arange(FTX.shape[2])*12/1000

for i in np.arange(FTX.shape[1]):
    slice = 10*np.log10(abs(FTX[:,i,:]))
    slice = 10*np.log10(np.mean(abs(FTX), axis = 1))
    
    plt.imshow(slice, aspect = 'auto', cmap = 'turbo', origin='lower', extent = (np.min(chans),np.max(chans),np.min(freqs),np.max(freqs)))
    plt.clim(low,high)
    plt.ylim(0,100)
    plt.savefig(os.path.join('C:\\Users\\Calder\\Outputs\\FX_call', str(i) + '.png'))
    plt.close
# %%
slice = 10*np.log10(np.mean(abs(FTX), axis = 1))
low = np.percentile(slice,25)
high = np.percentile(slice,99)
plt.imshow(slice, aspect = 'auto', cmap = 'turbo', origin='lower', extent = (np.min(chans),np.max(chans),np.min(freqs),np.max(freqs)))
plt.clim(low,high)
plt.ylim(0,100)

# %%
