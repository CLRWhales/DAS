#this script contains the helper functions developed by calder robinson
#%%
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import glob
import h5pydict

def faststack(X,n, wind = 1):
     """ fast adjacent channel stack through time
     Inputs:
     Parameters
    ----------
    X : 2d np.array 
        2d data array with dimension (T,X).
    n : int
        number of channels to stack together
    wind : 1d np.array
        optional cross channel windowshape, default is square

    Returns
    -------
    stacked : 2d np.array
        data array containing the mean of n adjacent channels 
    
    Note: when using a window other than 1, if n does not divide into the number of X, the final group  average will be returend without the window applied, 
    and it contains fewerthan requested channels.
     """
     rows,cols = X.shape
     trimval = np.mod(cols,n)

     if wind == 1:
        wind = np.ones(n)
     elif len(wind) != n:
         raise ValueError("window not same size as stack number.")
         

     if trimval!=0:
          trimmedmean = X[:,-trimval:].mean(axis = 1)
          X = X[:,:-trimval]
          stacked = X.reshape(-1,n)
          stacked = np.average(stacked, axis = 1, weights = wind).reshape(rows,-1)
          stacked = np.c_[stacked,trimmedmean]
     else:
          stacked = X.reshape(-1,n)
          stacked = np.average(stacked, axis = 1, weights=wind).reshape(rows,-1)

     return stacked

def loadFTX(directory,nworkers = 1): 
    """
    This is a function that can be used to quickly load FTX files in a directory. if there are too many files/too large, can overload the ram. 

    inputs:
    X: directory of files, without trailing slash

    returns:
    data: concatenated np array of data in the directory of dimension FTX
    freqs: np. array of freqs for the data
    times: np array of relative time since the start of the array, ignores gaps
    channels: np array of channel distances of the array

    TODO:
    build in gap handelling in the time index by parsing the file names?

    """

    clist = directory + '/Dim_Channel.txt'
    flist = directory + '/Dim_Frequency.txt'
    tlist = directory + '/Dim_Time.txt'

    filepaths = sorted( glob.glob(directory + '*.npy') )#[0:10]
    if len(filepaths) == 0:
        raise ValueError("no files found at directory.")
    
    channels = np.loadtxt(clist)
    freqs = np.loadtxt(flist)
    times = np.loadtxt(tlist)
    lt = len(times)
    bt = np.arange(len(times),dtype= 'int')

    data = np.empty((len(freqs),len(times)*len(filepaths)+1,len(channels)))

    with ThreadPoolExecutor(max_workers=nworkers) as exe:
        futures = exe.map(np.load,filepaths)
        for i, ff in enumerate(futures):
            tidx = i*lt + bt
            data[:,tidx,:] = ff

    times = np.arange(start = 0, stop = data.shape[1])*0.25


    return data, freqs,times,channels

def load_meta(filename,metaDetail = 1):
    """
    Extracted from ASN load das file, see for details
    """
    with h5pydict.DictFile(filename,'r') as f:
        # Load metedata (all file contents except data field)
        m = f.load_dict(skipFields=['data']) 
        if metaDetail==1:
            ds=m['demodSpec']
            mon=m['monitoring']
            meta=dict(fileVersion = m['fileVersion'],
                      header     = m['header'],
                      timing     = m['timing'],
                      cableSpec  = m['cableSpec'],
                      monitoring = dict(Gps = mon['Gps'],
                                        Laser=dict(itu=mon['Laser']['itu'])),
                      demodSpec  = dict(roiStart = ds['roiStart'],
                                        roiEnd   = ds['roiEnd'],
                                        roiDec   = ds['roiDec'],
                                        nDiffTau = ds['nDiffTau'],
                                        nAvgTau  = ds['nAvgTau'],
                                        dTau     = ds['dTau']))
        else:
            meta = m
    return meta
