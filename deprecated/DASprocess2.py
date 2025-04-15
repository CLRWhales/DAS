#this script tries to streamline the process of finding multiple calls in a das data set
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
from datetime import datetime

#data info:
path_data = 'D:/DAS/tmpData8m/'
nfiles = 16

#processing info:
n_synthetic = 200 #how many synthetic receivers to you want 
synthetic_spacing = 125 #channels between synthetic recievers (where channelspacing is half the guage length)
ch_space = 4 #m, spacing between adjacent "receivers" on the fiber
n_stack = 5 #number of channels to stack at each synthetic reciever
fstarget = 256 #resample to this fs
nfft = 512 #num sampls in fft
windowtime = 0.5 #seconds
bp_paras =[5, 127, 6, 0]             # bandpass parameters [lowcut,highcut,orders,zerophase] could make this a highpass
alpha_taper=0.1

#savelocation
plot_path = 'C:/Users/Calder/Ouputs/DASplots1/'


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
#t_ex_start=time.perf_counter()  
dt = meta['header']['dt']

data = utils_tmp.stack_channels(data_raw, n_stack, axis_stack=1)
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

#stft and post process
windowsize = windowtime*fstarget
w = hamming(round(windowsize))
hop = round(windowsize/2)
SFT = ShortTimeFFT(w,hop = hop,fs = fstarget,mfft = nfft)

timestamp = datetime.fromtimestamp(int(meta['header']['time']))

for col in range(data.shape[1]):
    sig = data[:,col]
    Sx = abs(SFT.stft(sig))

    dist = col*synthetic_spacing*ch_space#/1000 # turns the ch index into km for the plot title

    #normalize across time? good but to sqishy in dynamic range
    # Sx = Sx-np.mean(Sx,axis = 1)[:,None] 
    # Sx = Sx/np.std(Sx,axis = 1)[:,None]

    #this is full norm?
    Sx = Sx - np.mean(Sx)
    Sx = Sx/np.std(Sx)

    N = len(sig)
    Sxb = gaussian_filter(Sx, sigma = 3)

    # thresh2 = filters.threshold_otsu(Sxb,nbins=256) # could do differnt thresholding 
    # spec_bin = (Sxb > thresh2).astype(int)
    # labels = measure.label(spec_bin)
    # regions = measure.regionprops(labels)

    c_pos = meta['header']['channels'][i]

    fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
    t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
    ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
               rf"$\Delta t = {SFT.delta_t:g}\,$s)",
        ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
               rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
        xlim=(t_lo, t_hi))

    im1 = ax1.imshow(abs(Sx), origin='lower', aspect='auto',
                 extent=SFT.extent(N), cmap='turbo', vmin = 0, vmax = 20)
    fig1.colorbar(im1, label="Magnitude")
    fig1.suptitle(str(dist) + ' m, ' + timestamp.strftime("%Y-%m-%d %H:%M:%S"))

    # # for props in regions:
    # #     minr,minc,maxr,maxc = props.bbox
    # #     bx = (minc*SFT.delta_t,maxc*SFT.delta_t,maxc*SFT.delta_t,minc*SFT.delta_t,minc*SFT.delta_t) 
    # #     by = (minr*SFT.delta_f,minr*SFT.delta_f,maxr*SFT.delta_f,maxr*SFT.delta_f,minr*SFT.delta_f)
    # #     ax1.plot(bx,by,'-w',linewidth = 0.5)
    

    ax1.legend()
    fig1.tight_layout()
    plt.savefig(plot_path + 'fig_'+str(dist)+'m.png')
    plt.close()

t_ex_end=time.perf_counter(); print(f'fftprocess time: {t_ex_end-t_ex_start}s'); 


 # %%
