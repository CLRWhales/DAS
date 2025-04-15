#this script looks to try and visualize the tx data for cleaning 
#%%
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import glob
import os

def load_DASout(directory,dtype,nworkers = 1): 
    """
    This is a function that can be used to quickly load FTX files in a directory. if there are too many files/too large, can overload the ram. 

    inputs:
    X: directory of files, without trailing slash
    dtype: are you loading  FTX of TX ?
    nworkers: how many threads do you want running? default is 1

    returns:
    data: concatenated np array of data in the directory of dimension FTX
    freqs: np. array of freqs for the data
    times: np array of relative time since the start of the array, ignores gaps
    channels: np array of channel distances of the array

    TODO:
    build in gap handelling in the time index by parsing the file names?
    """
    tmpdir = directory
    for i in range(4):
        clist = glob.glob(tmpdir + '/Dim_Channel.txt')
        if clist:
            break
        else:
            tmpdir = os.path.split(tmpdir)[0]
    
    if i == 4:
        raise ValueError("could not find channel, frequency, or time, files within 4 levels, check directory")

    clist = tmpdir + '/Dim_Channel.txt'
    flist = tmpdir + '/Dim_Frequency.txt'
    tlist = tmpdir + '/Dim_Time.txt'

    print(clist)
    print(flist)
    print(tlist)

    filepaths = sorted( glob.glob(directory + '/*.npy') )#[0:10]
    
    if len(filepaths) == 0:
        raise ValueError("no files found at directory.")
    
    channels = np.loadtxt(clist)
    freqs = np.loadtxt(flist)
    times = np.loadtxt(tlist)
    lt = len(times)
    bt = np.arange(len(times),dtype= 'int')
    match dtype:
        case 'FTX':
            data = np.empty((len(freqs),len(times)*len(filepaths)+1,len(channels)))
            with ThreadPoolExecutor(max_workers=nworkers) as exe:
                futures = exe.map(np.load,filepaths)
                for i, ff in enumerate(futures):
                    tidx = i*lt + bt
                    data[:,tidx,:] = ff

            times = np.arange(start = 0, stop = data.shape[1])*0.25
            return data, freqs,times,channels

        case 'TX':
            data = np.empty((len(times)*len(filepaths)+1,len(channels)))
            with ThreadPoolExecutor(max_workers=nworkers) as exe:
                futures = exe.map(np.load,filepaths)
                for i, ff in enumerate(futures):
                    tidx = i*lt + bt
                    data[tidx,:] = ff

            times = np.arange(start = 0, stop = data.shape[0])*0.25
            return data,times,channels
        case _:
            raise TypeError('dtype must be either "FTX", "TX"')


# path = "C:/Users/Calder/Outputs/ShipWhaleClean20250204T181740/5Hz_30Hz"
# tmp, times, channels= load_DASout(path,'TX',2)
# times = times/60
# #%%
# mtmp = np.mean(tmp,axis = 0)
# mt2 = tmp-mtmp[None,:]
# mt2std = np.std(mt2,axis = 0)
# mt3 = mt2/mt2std[None,:]
# plt.figure(figsize= (10,15))
# plt.imshow(np.fliplr(mt3), aspect= 'auto',cmap = 'seismic',extent = (np.max(channels),np.min(channels),np.max(times),np.min(times)))
# plt.colorbar()
# plt.clim(-3,3)


# %%
