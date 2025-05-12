#this is to explore potential for lossy data compression
#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
from Calder_utils import load_meta
import cProfile

def fast_load(filename):
    with h5py.File(filename, "r") as f:
        signal = f['data'][:,:]
        # ds = f['demodSpec']
        # mon = f['monitoring']
        # meta=dict(fileVersion = f['fileVersion'][()],
        #                 header     = f['header'],
        #                 timing     = f['timing'],
        #                 cableSpec  = f['cableSpec'],
        #                 monitoring = dict(Gps = mon['Gps'],
        #                                     Laser=dict(itu=mon['Laser']['itu'])),
        #                 demodSpec  = dict(roiStart = ds['roiStart'],
        #                                     roiEnd   = ds['roiEnd'],
        #                                     roiDec   = ds['roiDec'],
        #                                     nDiffTau = ds['nDiffTau'],
        #                                     nAvgTau  = ds['nAvgTau'],
        #                                     dTau     = ds['dTau']))

        return signal#, meta

# %%
import cupy as cp
filename = "D:\\DAS\\shipwhale2\\163007.hdf5"
signal = fast_load(filename)

def gpu_cumsum_cpu_io(np_array):
    # Step 1: Transfer NumPy array to GPU (CuPy)
    cp_array = cp.asarray(np_array)
    
    # Step 2: Perform cumsum on the GPU
    cp_result = cp.cumsum(cp_array,axis=0)
    
    # Step 3: Transfer result back to CPU (NumPy)
    np_result = cp.asnumpy(cp_result)
    
    return np_result


cumsum_result = gpu_cumsum_cpu_io(signal)
normal = np.cumsum(signal, axis = 0)
 
# %%
