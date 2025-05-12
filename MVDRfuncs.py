#this is a group of functions that performs MVDR beamforming on data
#%%
import numpy as np
from numba import njit

def gen_steering_vector(freq, spacing, N, theta, Cw = 1470): #theta is relative to the perpendicular
    wavelength = Cw/freq
    d = spacing/wavelength
    out = np.exp(-2j * np.pi * d * np.arange(N) * np.sin(theta))
    out = out[:,None]
    return(out)


def w_mvdr(block,s, dload = 0):
   """
   compute mvdr weights based on a block of data where each row is a sensor, and each column is a time step.

   inputs:
   block, 2d numpy array where rows are different spatial sensors, and columns are time observations across sensors.
   s, 1d column wise np.array,
   dload: diagonal loading value to assist in inversion, default 0

   output:
   w, 1d column np array adaptive weights of beamformer within the block of data
   """
   diag = np.diag(np.ones(s.shape[0]*dload))
   R = np.cov(block) # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
   Rinv = np.linalg.pinv(R+diag) # 3x3. pseudo-inverse tends to work better/faster than a true inverse
   w = (Rinv @ s)/(s.conj().T @ Rinv @ s) # MVDR/Capon equation! numerator is 3x3 * 3x1, denominator is 1x3 * 3x3 * 3x1, resulting in a 3x1 weights vector
   return w


#%%TESTING
fid = 20
tmp = np.load("C:\\Users\\Calder\\Outputs\\DASdata2\\FTX512_20220721T010617Z.npy")
tmp = tmp[fid,:,10:20]
nchans = 5
train = 10
nfreqs = tmp.shape[0]
freqs = np.loadtxt("C:\\Users\\Calder\\Outputs\\DASdata2\\Dim_Frequency.txt")
t2 = gen_steering_vector(freqs[fid],4,10,0)


#transpose to match cov input
tmp = tmp.T[:,0:15]

weights = w_mvdr(tmp,t2,0.1)
print(weights)

# %%
