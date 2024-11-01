#this script looks to take coherent signals and highlight them while getting the "bearing"


#%%

import sys, glob, time #,os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import decimate,detrend, resample, ShortTimeFFT
from scipy.signal.windows import hamming
import utils_tmp, utils_time
from simpleDASreader4 import load_multiple_DAS_files
#from skimage import measure, filters,morphology
from scipy.ndimage import gaussian_filter

#data info:
path_data = 'C:/Users/calderr/Downloads/tmpData8m/'
nfiles = 5

#processing info:
n_synthetic = 130 #how many synthetic receivers to you want 
synthetic_spacing = 250 #channels between synthetic recievers (where channelspacing is half the guage length)
n_stack = 5 #number of channels to stack at each synthetic reciever
fstarget = 256 #resample to this fs
nfft = 512 #num sampls in fft
windowtime = 0.5 #seconds
bp_paras =[5, 127, 6, 0]             # bandpass parameters [lowcut,highcut,orders,zerophase] could make this a highpass
alpha_taper=0.1

#savelocation
plot_path = 'C:/Users/calderr/Downloads/plot_outputs/'


#find files
filepaths = sorted( glob.glob(path_data + '*.hdf5') )
files = [f.split('\\')[-1] for f in filepaths]; 
fileIDs = [int(item.split('.')[0]) for item in files]

if len(fileIDs)<nfiles:
    nfiles = len(fileIDs)

fileIDs = fileIDs[0:nfiles]

#find channels
channels = []
for i in range(n_synthetic):
    channels.extend([x+i*synthetic_spacing for x in range(0,n_stack)])

#load data + pre process
t_ex_start=time.perf_counter()  

data_raw, meta = load_multiple_DAS_files(path_data, fileIDs, chIndex=channels, roiIndex=None,
                            integrate=True, unwr=True, metaDetail=1, useSensitivity=True, spikeThr=None)

t_ex_end=time.perf_counter(); print(f'data loading time: {t_ex_end-t_ex_start}s'); 
t_ex_start=time.perf_counter()  
dt = meta['header']['dt']

#data = utils_tmp.stack_channels(data_raw, n_stack, axis_stack=1)
data = data_raw
num = data.shape[0]/(1/fstarget)*dt
num = num.astype(int)
data = resample(data,num ,axis = 0);
data=detrend(data, axis=0, type='linear');

dt = 1/fstarget
bp_lowcut, bp_highcut, bp_order, bp_zerophase = bp_paras
data = utils_tmp.bandpass_2darray(data, 1/dt, bp_lowcut, bp_highcut, axis_filt=0, order=bp_order, 
                                      zerophase=bp_zerophase, taper_data=False, alpha_tape=alpha_taper)

t_ex_end=time.perf_counter(); print(f'preprocess time: {t_ex_end-t_ex_start}s');
t_ex_start=time.perf_counter()  

windowsize = windowtime*fstarget
w = hamming(round(windowsize))
hop = round(windowsize/2)
SFT = ShortTimeFFT(w,hop = hop,fs = fstarget,mfft = nfft)

#define a target for loop output
sig = data[:,0]
Sx = SFT.stft(sig)
nf,nt = Sx.shape
spec = np.empty([nf,nt, data.shape[1]],dtype=np.complex64)

for col in range(data.shape[1]):
    sig = data[:,col]
    spec[:,:,col] = SFT.stft(sig)

t_ex_end=time.perf_counter(); print(f'fftprocess time: {t_ex_end-t_ex_start}s'); 

#this section does some funky phase difference summations?
t_ex_start=time.perf_counter()  

spec_mag = np.abs(spec)
nstacks = int(spec.shape[2]/n_stack)


data_stacked = np.empty((nf,nt,nstacks))

N = len(sig)
c = 1470
k = (2 * np.pi * SFT.f)/c * 4



# for s in range(nstacks): 
#         ch_start = s*n_stack
#         pdiff = np.diff(spec[:,:,ch_start:ch_start+n_stack], axis=2)
#         tmp = np.mean(pdiff, axis=2)
#         tmp = np.angle(tmp)

#         theta = np.arcsin(tmp / k[:,None])
#         deg = np.degrees(theta)

#         data_stacked[:,:,s] = deg

#         fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
#         t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
#         ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
#                rf"$\Delta t = {SFT.delta_t:g}\,$s)",
#             ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
#                rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
#             xlim=(t_lo, t_hi))

#         im1 = ax1.imshow(deg, origin='lower', aspect='auto',
#                  extent=SFT.extent(N), cmap='seismic')
#         fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")

#         ax1.legend()
#         fig1.tight_layout()
#         plt.savefig(plot_path + 'fig_'+str(s)+'.png')
#         plt.close()


t_ex_end=time.perf_counter(); print(f'phase time: {t_ex_end-t_ex_start}s'); 





# %% This portion loks to try and speed up the fft operations on das shape data

def break_and_stack_rows(arr, n):
    # Check if the number of rows is divisible by n
    test = arr.shape[0] % n

    if test != 0:
        #raise ValueError("Number of rows must be divisible by n.")
        arr = arr[:-test,:]
    
    # Calculate the number of groups
    num_groups = arr.shape[0] // n
    
    # Create a list to hold the new groups
    new_groups = []
    
    # Iterate through the number of groups and extract slices
    for i in range(num_groups):
        start_row = i * n
        end_row = start_row + n
        new_groups.append(arr[start_row:end_row])
    
    # Concatenate the new groups horizontally
    result = np.hstack(new_groups)

    return result

#break and stack to windowsize
t_ex_start=time.perf_counter()  
tmp2 = break_and_stack_rows(data,50)
#t_ex_end=time.perf_counter(); print(f're-section time: {t_ex_end-t_ex_start}s'); 

#multiply by window + with broadcasting?
wind = np.hamming(50)
tmp2 = tmp2*wind[:,None]

#perform fft
#t_ex_start=time.perf_counter()  
fftout = np.fft.rfft(tmp2,nfft,0)
#t_ex_end=time.perf_counter(); print(f'fft time: {t_ex_end-t_ex_start}s'); 

#unbreak into stacks of specrograms?
#t_ex_start=time.perf_counter() 
nt_slices = fftout.shape[1]//data.shape[1]
target = np.empty((fftout.shape[0],nt_slices,data.shape[1]),dtype=np.complex64)

for i in range(nt_slices):
     start = i*data.shape[1]
     end = start + data.shape[1]
     target[:,i,:] = fftout[:,start:end]

t_ex_end=time.perf_counter(); print(f'section back into fft time: {t_ex_end-t_ex_start}s'); 

#coherent sum back into the 130 channels?

# target2 = np.empty((257,153,130))

# nstacks = int(spec.shape[2]/n_stack)
# for s in range(nstacks): 
#         ch_start = s*n_stack
#         tmp = abs(np.sum(target[:,:,ch_start:ch_start+n_stack], axis=2))

        
#         target2[:,:,s] = tmp

#         tmp = tmp - np.mean(tmp)
#         tmp = tmp/np.std(tmp)
        
#         fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
#         t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
#         ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
#                rf"$\Delta t = {SFT.delta_t:g}\,$s)",
#             ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
#                rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
#             xlim=(t_lo, t_hi))

#         im1 = ax1.imshow(tmp, origin='lower', aspect='auto',
#                  extent=SFT.extent(N), cmap='turbo', vmin = 0,vmax = 15)
#         fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")

#         ax1.legend()
#         fig1.tight_layout()
#         plt.savefig(plot_path + 'fig_'+str(s)+'.png')
#         plt.close()

# %%
