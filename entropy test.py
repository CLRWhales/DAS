#%% this is an entropy detector based on Erbe et al

import numpy as np
import matplotlib.pyplot as plt
import glob, os

def compute_entropy(arr):
    '''
    computes the local timewise spectral entropy of FTX after prewhitening with the mean
    inputs
        arr: FTX np.array
    outputs:
        entropy: 2d np array of the spectral entropy through time, relative to the timewise means of the FTX block, same dimension of TX

    
    '''
    arr = abs(arr)
    arr /= np.mean(arr, axis = 1)[:,None,:] #prewhiten
    arr /=np.sum(arr,axis = 0) #normalize to psd
    denom = np.log2(arr.shape[0]-1)
    num = np.sum(arr * np.log2(arr), axis = 0)
    entropy = -num/denom
    return entropy

path = 'C:\\Users\\Calder\\Outputs\\BF_out_full_20250514T202112\\Complex'
dst = 'C:\\Users\\Calder\\Outputs\\entropyplot'
files = glob.glob(os.path.join(path,'*.npy'))
chans = np.loadtxt("C:\\Users\\Calder\\Outputs\\BF_out_full_20250514T202112\\Dim_Channel.txt")
times = np.loadtxt("C:\\Users\\Calder\\Outputs\\BF_out_full_20250514T202112\\Dim_Time.txt")

for i,f in enumerate(files):
    name = os.path.basename(f).split('.')[0].split('_')[1]
    arr = np.load(f)
    ent = compute_entropy(arr)
    plt.figure()
    plt.imshow(1-ent,aspect='auto', extent = (np.min(chans),np.max(chans),np.max(times),np.min(times)))
    plt.colorbar()
    plt.clim(0,0.1)
    fname = os.path.join(dst,name + '.png')
    plt.savefig(fname)
    plt.close()


# %%
