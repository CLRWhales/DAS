#this is to explore potential for lossy data compression
#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
from Calder_utils import load_meta
import cProfile
from pstats import Stats
from simpleDASreader4 import load_DAS_file, unwrap
import matplotlib.pyplot as plt
def fast_load(filename):
    meta = load_meta(filename)
    with h5py.File(filename, "r") as f:
        signal = f['data'][:,:] * np.float32(meta['header']['dataScale'])
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
        signal = unwrap(signal,meta['header']['spatialUnwrRange'],axis=1)
        signal = signal.T.cumsum(1) * meta['header']['dt']
        signal/=meta['header']['sensitivities']
        return signal, meta
#%%
filename = "D:\\DAS\\shipwhale2\\163007.hdf5"
with cProfile.Profile() as pr:
    signal1,meta1 = fast_load(filename)
    stats = Stats(pr)
    stats.sort_stats('cumtime').print_stats(10)

with cProfile.Profile() as pr:
    signal2,meta2 = load_DAS_file(filename)
    stats = Stats(pr)
    stats.sort_stats('cumtime').print_stats(10)


print(np.array_equal(signal1,signal2))
# %%
# import cupy as cp
# filename = "D:\\DAS\\shipwhale2\\163007.hdf5"
# signal = fast_load(filename)

# def gpu_cumsum_cpu_io(np_array):
#     # Step 1: Transfer NumPy array to GPU (CuPy)
#     cp_array = cp.asarray(np_array)
    
#     # Step 2: Perform cumsum on the GPU
#     cp_result = cp.cumsum(cp_array,axis=0)
    
#     # Step 3: Transfer result back to CPU (NumPy)
#     np_result = cp.asnumpy(cp_result)
    
#     return np_result


# cumsum_result = gpu_cumsum_cpu_io(signal)
# normal = np.cumsum(signal, axis = 0)
 
# %%
