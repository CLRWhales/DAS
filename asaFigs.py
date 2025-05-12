#%% this is plots for the ASA presnetation
import numpy as np
import matplotlib.pyplot as plt
from simpleDASreader4 import load_DAS_file
from Calder_utils import faststack
from scipy.signal import detrend, resample, butter, sosfiltfilt
from scipy.ndimage import convolve1d, binary_closing, generate_binary_structure, binary_dilation
from matplotlib import cm

file = "E:\\NORSAR01v2\\20220821\\dphi\\183137.hdf5"
nchans = 15000
nstack = 5

chans = range(nchans)
signal,meta = load_DAS_file(file, chIndex=chans)
signal = signal/1e-9 #moving to microstrain
signal = faststack(signal,nstack)
dt = meta['header']['dt']
waveform = signal[:,760]
x = np.arange(signal.shape[0])*dt

plt.plot(x,waveform)
plt.ylabel('Nano strain (ε)')
plt.xlabel('Time (s)')


colors = cm.Blues(np.linspace(0.3, 1, signal.shape[1]))
# %%
sost = butter(N = 6,
        Wn = (5,30),
        btype = 'bandpass',
        fs = 1/dt,
        output = 'sos')

waveform_filt = sosfiltfilt(sos = sost,
                    x = signal,
                    axis = 0)

plt.plot(x,waveform_filt[:,760])
plt.xlabel('Time (s)')
plt.ylabel('Nano strain (ε)')
 

# %%
windowsize=125
kernal = np.ones(windowsize)/windowsize
sig2 = np.square(waveform_filt)
vis = np.sqrt(convolve1d(sig2,kernal,axis = 0,mode = 'reflect'))
plt.plot(x,vis[:,760])
plt.xlabel('Time (s)')
plt.ylabel('RMS Nano strain (ε)')
# %%
plt.plot(x,vis)
plt.ylim(0,1)
 # %%

def cfar(array, guardsize,sample_size):
    kernalsize = 2*(sample_size + guardsize)+1
    kernal = np.zeros(kernalsize)
    kernal[:sample_size] = 1
    kernal[-sample_size:]=1
    kernal = kernal/np.sum(kernal)
    thresh = convolve1d(array,kernal,axis = 0,mode = 'reflect')
    return thresh

nguard = 625
nsample = 300
maskthresh = 1.2
truethresh = 4

struc = generate_binary_structure(2,2)

tmp = cfar(vis, nguard,nsample)
mask = np.greater(vis,tmp*maskthresh)
#mask = binary_dilation(input = mask,structure=struc, iterations=1)
output = np.greater(vis,tmp*truethresh)



# %%
plt.plot(x,output*vis)
plt.ylim(0,2)
# %%

colors = cm.viridis(np.linspace(0, 1, signal.shape[1]))

fig, ax = plt.subplots(figsize=(10, 6))
lines = ax.plot(x,vis)

# Apply the colors to the lines
for line, color in zip(lines, colors):
    line.set_color(color)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Sensor Reading')
ax.set_title('Sensor Data from Multiple Positions')
plt.ylim(0,4)
plt.tight_layout()
plt.show()
 # %% 

ns = 2*nsample
mt = 30
tt=1
lowthresh = ns*((mt/100)**(-1/ns)-1)
highthresh = ns*((tt/100)**(-1/ns)-1)

plt.plot(x,vis[:,760])
plt.plot(x,tmp[:,760]*highthresh)
plt.plot(x,tmp[:,760])
plt.plot(x,tmp[:,760]*lowthresh)

# %%
def crossings_nonzero_all(data):
    pos = data > 0
    npos = ~pos
    return ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]

t2 = vis[:,760]-(tmp[:,760]*0.3)

zc = crossings_nonzero_all(t2)
t3 =np.greater(vis,tmp*4)
t4 = vis>4

# %%
