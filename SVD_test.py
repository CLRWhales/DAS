#this script explores the idea of SVD as a denoising algorithm for DAS stuff like the talks at asa

#%%
import numpy as np
import matplotlib.pyplot as plt
from simpleDASreader4 import load_DAS_file
from Calder_utils import faststack
from scipy.signal import butter, sosfiltfilt
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm


file = "E:\\NORSAR01v2\\20220821\\dphi\\183137.hdf5"
#file = "E:\\NORSAR01v2\\20220821\\dphi\\182847.hdf5"
file = "C:\\Users\\Calder\\OneDrive - NTNU\\Desktop\\183137.hdf5"
nchans = 6000
nstack = 5

chans = range(nchans)
signal,meta = load_DAS_file(file, chIndex=chans)

signal = signal/1e-9 #moving to microstrain
#signal = faststack(signal,nstack)
dt = meta['header']['dt']
#signal = signal[:,2000:6000]
sost = butter(N = 6,
        Wn = (3),
        btype = 'highpass',
        fs = 1/dt,
        output = 'sos')

signal_filt = sosfiltfilt(sos = sost,
                    x = signal,
                    axis = 0)

norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

plt.figure(figsize=(12,12))
plt.imshow(signal_filt, aspect = 'auto', cmap = 'seismic', norm = norm)
plt.colorbar()
#plt.clim(-0.5,0.5)

# noise = np.random.randn(signal.shape[0],signal.shape[1])
# signal = signal + noise
# plt.figure(figsize=(12,12))
# plt.imshow(signal, aspect = 'auto', cmap = 'seismic')
# plt.colorbar()



# %%svd

U, S, VH = np.linalg.svd(signal_filt, full_matrices=False)

k = 100
plt.figure()
plt.plot(S)
plt.yscale('log')
plt.xscale('log')
plt.vlines([k],0,1000)


S_k = np.zeros_like(S)
S_k[:k] = S[:k]
denoised_image = (U * S_k) @ VH

plt.figure(figsize=(12,12))
plt.imshow(denoised_image, aspect = 'auto', cmap = 'seismic', norm = norm)
plt.colorbar()
# %%
