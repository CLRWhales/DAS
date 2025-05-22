#this is a group of functions that performs MVDR beamforming on data
#%%
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import glob, os
from scipy.ndimage import rotate

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
   diag = np.diag(np.ones(s.shape[1])*dload)
   R = np.cov(block) # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
   Rinv = np.linalg.pinv(R+diag) # 3x3. pseudo-inverse tends to work better/faster than a true inverse
   w = (Rinv @ s)/(s.conj().T @ Rinv @ s) # MVDR/Capon equation! numerator is nxn * nx1, denominator is 1xn * nxn * nx1, resulting in a nx1 weights vector
   return w

def idealspacing(freq,chanspace,Cw = 1470):
    lamda = Cw/freq
    ideal = lamda/2
    ideal = np.floor(ideal/chanspace)
    if ideal < 1:
        ideal = False
    return(ideal,lamda)

@njit
def FTX_beamformer(FTX,freqs,spacing,wf,wt,wx,theta,Cw = 1470,dload = 0):
    F, T, X = FTX.shape
    
    out_F = F - wf + 1
    out_T = T - wt + 1
    out_X = X - wx + 1
    output = np.empty((out_F, out_T, out_X), dtype = 'complex128')

    wavelength = Cw/freqs
    d = spacing/wavelength
    theta_rad = theta / 180 * 3.141592653589793
    #print(theta_rad)
    
    for f in range(F):
        svec = np.exp(-2j * np.pi * d[f] * np.arange(wx) * np.sin(theta_rad))[:,None]
        
        for t in range(T - wt + 1):
            for x in range(X - wx + 1):
                window = FTX[f, t:t+wt, x:x+wx].T#[:,None]
                # R = np.cov(window.T)
                # diag = np.diag(np.ones(R.shape[1])*dload)
                # Rinv = np.linalg.pinv(R+diag)
                # w = (Rinv @ svec)/(svec.conj().T @ Rinv @ svec)
                # print(window.shape)
                # print(svec.shape)
                tmp = svec.conj().T @ window
                # print(tmp.shape)
                output[f,t,x] = tmp[0,0]
    return output

# %%
# directory = 'C:\\Users\\Calder\\Outputs\\BF_out_full_20250514T202112\\Complex'
# freqs = np.loadtxt("C:\\Users\\Calder\\Outputs\\BF_out_full_20250514T202112\\Dim_Frequency.txt")
# chans = np.loadtxt("C:\\Users\\Calder\\Outputs\\BF_out_full_20250514T202112\\Dim_Channel.txt")
# times = np.loadtxt("C:\\Users\\Calder\\Outputs\\BF_out_full_20250514T202112\\Dim_Time.txt")
# files = glob.glob(os.path.join(directory,'*.npy'))


# start = '180007'
# stop = '185957'
# blocksizeT = 1
# blocksizeC = 30
# fidmin = 20
# fidmax = 31
# freqs = freqs[fidmin:fidmax]
# # start the routine
# start_idx = [i for i, s in enumerate(files) if start in s][0]
# stop_idx = [i for i, s in enumerate(files) if stop in s][0]

# files = files[int(start_idx):int(stop_idx)]

# chanidx = np.arange(len(chans))*2
# chanidx = np.int16(chanidx[chanidx<len(chans)])

# wf, wt, wx = 1, blocksizeT, blocksizeC  # Window sizes

# for i,file in enumerate(files):
#     name = os.path.basename(file).split('.')[0]
#     FTX = np.load(file)
#     FTX = FTX[fidmin:fidmax,:,chanidx]

#     output1 = FTX_beamformer(FTX,freqs,24,wf,wt,wx,45)
#     output2 = FTX_beamformer(FTX,freqs,24,wf,wt,wx,-45)
#     output3 = FTX_beamformer(FTX,freqs,24,wf,wt,wx,0)

#     omean1 = np.mean(abs(output1), axis = 1)
#     omean2 = np.mean(abs(output2), axis = 1)
#     omean3 = np.mean(abs(output3), axis = 1)
#     omean1 = omean1.mean(0)
#     omean2 = omean2.mean(0)
#     omean3 = omean3.mean(0)
#     wind = np.ones(30)
#     om1 = np.convolve(wind,omean1)
#     om2 = np.convolve(wind,omean2)
#     om3 = np.convolve(wind,omean3)

#     #tmp = omean1[5,:][:,None]*omean2[5,:][None,:]
#     tmp = om1[:,None]*om2[None,:]
#     #tmp = om1[:,None]*om2[None,:]
#     # chans = np.arange(tmp.shape[1])*12/1000
#     # low = np.percentile(10*np.log10(tmp),25)
#     # high = np.percentile(10*np.log10(tmp),99)
#     # plt.imshow(10*np.log10(abs(tmp)), origin = 'lower',extent=(np.min(chans),np.max(chans),np.min(chans),np.max(chans)))
#     # plt.plot([0,np.max(chans)], [0, np.max(chans)], color = 'white')
#     # plt.clim(low,high)
#     # plt.xlabel('Distance along fiber (km), bearing +45')
#     # plt.ylabel('Distance along fiber (km), bearing -45')
#     t4 = name.split('_')[1]
#     fname = os.path.join('C:\\Users\\Calder\\Outputs\\DFT ranging', t4)
#     fname2 = os.path.join('C:\\Users\\Calder\\Outputs\\DFT ranging', t4 + '_zero')
#     np.save(fname,tmp)
#     np.save(fname2,om3)
#     plt.title(t4)
    
#     plt.savefig(fname, dpi = 500)
#     plt.close()
# plt.figure()
# plt.plot(10*n, dpi = p.log10(omean1[5].T))

# test = omean1[1:10,:].mean(0)
# plt.figure()
# plt.plot(10*np.log10(test))

#%%

directory = 'C:\\Users\\Calder\\Outputs\\DFT ranging'
files = glob.glob(os.path.join(directory,'*.npy'))
zero = [f for f in files if 'zero' in f]
nonzero = [f for f in files if 'zero' not in f]


#%%
from scipy.ndimage import convolve1d
def cfar(array, guardsize,sample_size):
    kernalsize = 2*(sample_size + guardsize)+1
    kernal = np.zeros(kernalsize)
    kernal[:sample_size] = 1
    kernal[-sample_size:]=1
    kernal = kernal/np.sum(kernal)
    thresh = convolve1d(array,kernal,axis = 0,mode = 'reflect')
    return thresh


zstack = []
for z in zero:
    zstack.append(np.load(z))
tmp = np.vstack(zstack)
tmp = tmp - np.mean(tmp, axis = 0)
tmp = tmp/ np.std(tmp, axis = 0)
tmp = tmp - np.min(tmp, axis = 0)
tmp[:,0:225] = 0
kernal = np.ones(50)/50
tmp = convolve1d(tmp,kernal,axis=1, mode = 'reflect')
t3 = tmp.T
ns = 10
thresh = cfar(t3,30,ns)
thresh = thresh.T
lowthresh = ns*((40/100)**(-1/ns)-1)
mask = np.greater(tmp,thresh*lowthresh)


plt.imshow(10*np.log10(tmp), aspect = 'auto')
t2 = np.argmax(tmp,axis = 1)
t2[t2>600] = 0
tdiff = np.diff(t2)

m2 = np.max(tmp,axis = 1)
time = np.arange(tmp.shape[0])*10/60
plt.figure()
plt.scatter(time,t2,c = m2)
plt.hlines(y = [450,1200], xmin = 0, xmax = np.max(time))


mt = m2>9
tdt = abs(tdiff)<100
idx2 = np.where(tdt)
plt.figure()
plt.scatter(time,t2)
plt.scatter(time[1:][tdt],t2[1:][tdt])
plt.hlines(y = 600, xmin = 0,xmax = np.max(time))

#%%
def max_anti_diagonals_np(arr):
    if arr.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")

    rows, cols = arr.shape
    diagonalmaxidx = []
    diagonalmaxval = []
    flipped = np.fliplr(arr)
    for offset in range(-rows + 1, cols):
        anti_diag = flipped.diagonal(offset=offset)
        diagonalmaxidx.append(np.nanargmax(anti_diag))
        diagonalmaxval.append(np.nanmax(anti_diag))
    
    idx = np.vstack(diagonalmaxidx)
    val = np.vstack(diagonalmaxval)
    return idx,val

def symmetric_array(n):
    half = n // 2
    if n % 2 == 0:
        first_half = np.linspace(0, half, half, endpoint=False)
        second_half = np.linspace(half, 0, half)
    else:
        first_half = np.linspace(0, half, half + 1)
        second_half = np.linspace(half, 0, half + 1)[1:]
    return np.concatenate((first_half, second_half))[:n]

diagmaxidx = []
diagmaxval = []
for f in nonzero:
    data = np.load(f)
    #nan the 'upper'
    iu1 = np.triu_indices(data.shape[0],1)
    iu2 = np.tril_indices(data.shape[0],-500)
    data[iu1] = 0
    data[iu2] = 0
    idx,val = max_anti_diagonals_np(data)
    diagmaxidx.append(idx)
    diagmaxval.append(val)


    # chans = np.arange(data.shape[0])*12/1000
    # diag = rotate(data,45, cval=np.nan)
    # fiber = diag.shape()
    # offset= np.arange(diag.shape[0])*12
    # plt.figure()
    # plt.imshow(10*np.log10(data),aspect='auto')#, extent = (np.min(chans),np.max(chans),np.min(offset),np.max(offset)))
    # t4 = os.path.basename(f).split('.')[0]
    # fname = os.path.join('C:\\Users\\Calder\\Outputs\\DFT ranging_diag', t4 + '_diag.png')
    # plt.title(t4)
    # plt.xlabel('Fiber distance (km)')
    # plt.ylabel('Fiber offset (m)')
    # plt.savefig(fname, dpi = 500)
    # plt.close()



#%%
adj = np.bartlett(2499)*625
crossrange = []
plt.figure()
for i,k in enumerate(diagmaxidx):
    tmp = np.float64(k)
    
    tmp = np.squeeze(tmp)-adj
    plt.plot(tmp)
    cr = tmp[-t2[i]]
    crossrange.append(cr)

shiptrack = np.hstack(crossrange)*np.sin(np.pi/4)/np.sin(np.pi/2)*24
fiber= t2*24/1000
plt.figure()
plt.scatter(fiber,shiptrack, c= time)

np.save(file = 'D:\\DAS\\DASsourceLOC\\shipfiber',arr = fiber)
np.save(file = 'D:\\DAS\\DASsourceLOC\\shiptrack',arr = shiptrack)
# %%
