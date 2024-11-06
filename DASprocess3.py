#this script looks to take coherent signals and highlight them while getting the "bearing"


#%%

import sys, glob, time #,os
import numpy as np # pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import decimate,detrend, resample, ShortTimeFFT
from scipy.signal.windows import hamming
import utils_tmp, utils_time
from simpleDASreader4 import load_multiple_DAS_files
#from skimage import measure, filters,morphology
#from scipy.ndimage import gaussian_filter
from DASFFT import sneakyfft


#data info:
path_data = 'D:/DAS/tmpData8m/'
nfiles = 5

#processing info:
n_synthetic = 130 #how many synthetic receivers to you want 
synthetic_spacing = 250 #channels between synthetic recievers (where channelspacing is half the guage length)
n_stack = 1 #number of channels to stack at each synthetic reciever
fstarget = 256 #resample to this fs
nfft = 512 #num sampls in fft
bp_paras =[5, 127, 6, 0]             # bandpass parameters [lowcut,highcut,orders,zerophase] could make this a highpass
alpha_taper=0.1


#fft info:
N_fft = 512
N_overlap = 64
N_samp = 128


#savelocation
plot_path = 'C:/Users/Calder/Ouputs/DASplots1'


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

#compute spectrogram
t_ex_start=time.perf_counter()  

window = np.hamming(N_samp)

t_ex_start=time.perf_counter()  
fftoutput = sneakyfft(data,N_samp,N_overlap,N_fft, window)
t_ex_end=time.perf_counter(); print(f'sneakyfft_time: {t_ex_end-t_ex_start}s'); 

freqs = fstarget/N_fft*np.arange(fstarget+1)
fftout = abs(fftoutput)

for f in range(fftout.shape[2]):
    plt.imsave(plot_path + '/fig_'+str(f)+'.png', fftout[:,:,f],cmap = 'turbo',origin = 'lower',)

#this section does some funky phase difference summations?
#t_ex_start=time.perf_counter()  

# spec_mag = np.abs(spec)
# nstacks = int(spec.shape[2]/n_stack)


# data_stacked = np.empty((nf,nt,nstacks))

# N = len(sig)
# c = 1470
# k = (2 * np.pi * SFT.f)/c * 4



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


#t_ex_end=time.perf_counter(); print(f'phase time: {t_ex_end-t_ex_start}s'); 


# %%
#trying out gpu methods
from DASFFT import reshape_array_with_overlap
import cupy as cp 
from cupyx.profiler import benchmark
reshaped = reshape_array_with_overlap(data,128,64)

reshaped = reshaped*window[:,None]

# t_ex_start=time.perf_counter()  
# cpu_fft = np.fft.rfft(reshaped,n = N_FFT, axis =0)
# t_ex_end=time.perf_counter(); print(f'cpu fft: {t_ex_end-t_ex_start}s'); 

reshaped_gpu = cp.asarray(reshaped)
gpu_fft = cp.fft.rfft(reshaped_gpu,N_fft,axis = 0)



# print(benchmark(test_func,(reshaped,),n_repeat = 20))

# %%
