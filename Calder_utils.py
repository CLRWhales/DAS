#this script contains the helper functions developed by calder robinson
#%%
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import glob
import h5pydict
import os
from numpy.lib.stride_tricks import as_strided

def faststack(X,n, wind = 1):
     """ fast adjacent channel stack through time
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

def loadFTX(directory,nworkers = 1,start_f=None, stop_f=None, start_cidx=None, stop_cidx=None): 
    """
    This is a function that can be used to quickly load FTX files in a directory. if there are too many files/too large, can overload the ram. 

    inputs:
    X: directory of files, without trailing slash
    nworkers: how many workers do you want to deploy?
    startf: index of frequency start for slicing
    stopf: index of frequency stop for slicing
    start_cidx: index of channel start for slicing
    stop_cidx: index of channel stop for slicing


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
        clist = glob.glob(os.path.join(tmpdir, 'Dim_Channel.txt'))
        if clist:
            break
        else:
            tmpdir = os.path.split(tmpdir)[0]

    if i == 4:
        raise ValueError("could not find channel, frequency, or time files within 4 levels, check directory")

    channels = np.loadtxt(os.path.join(tmpdir, 'Dim_Channel.txt'))[start_cidx:stop_cidx]
    times =  np.loadtxt(os.path.join(tmpdir, 'Dim_Time.txt'))
    freqs = np.loadtxt(os.path.join(tmpdir, 'Dim_Frequency.txt'))[start_f:stop_f]

    filepaths = sorted( glob.glob(os.path.join(directory, '*.npy')))#[0:10]
    if len(filepaths) == 0:
        raise ValueError("no files found at directory.")
    
    lt = len(times)
    bt = np.arange(len(times),dtype= 'int')

    data = np.empty((len(freqs),len(times)*len(filepaths)+1,len(channels)))

    with ThreadPoolExecutor(max_workers=nworkers) as exe:
        futures = exe.map(np.load,filepaths)
        for i, ff in enumerate(futures):
            tidx = i*lt + bt
            data[:,tidx,:] = ff[start_f:stop_f,:,start_cidx:stop_cidx]

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

def window_rms(a, windowsize):
    a2 = np.square(a)
    window = np.ones(windowsize)/float(windowsize)
    return np.sqrt(np.convolve(a2, window, 'valid'))

def compute_entropy(arr):
    '''
    computes the local timewise spectral entropy of FTX after prewhitening with the mean
    inputs
        arr: FTX np.array
    outputs:
        entropy: 2d np array
        spectral entropy through time, relative to the timewise means of the FTX block, same dimension of TX
    '''
    arr = abs(arr)
    arr /= np.mean(arr, axis = 1)[:,None,:] #prewhiten
    arr /=np.sum(arr,axis = 0) #normalize to psd
    denom = np.log2(arr.shape[0]-1)
    num = np.sum(arr * np.log2(arr), axis = 0)
    entropy = -num/denom
    return entropy

def sliding_window_FK(arr, window_shape,overlap = 2, rescale = False):
    step_y, step_x = window_shape[0] // overlap, window_shape[1] // overlap
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
            if rescale:
                np.log10(fft_result,out = fft_result)
                fft_result *=10
                fft_result -=np.min(fft_result)
                fft_result /=np.max(fft_result)
                array_2d_uint8 = (255 * fft_result).clip(0, 255).astype(np.uint8)
                position = (i * step_y, j * step_x)
                results.append((position, array_2d_uint8))
            else:
                position = (i * step_y, j * step_x)
                results.append((position, fft_result))

    return results