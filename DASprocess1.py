#this script is an initial attempt to build up a fast das core alg
#Calder Robinson, 20240916

#%%

import sys, glob, time #,os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import decimate,detrend, resample, ShortTimeFFT
from scipy.signal.windows import hamming
import utils_tmp, utils_time
from simpleDASreader4 import load_multiple_DAS_files
from skimage import measure, filters,morphology
from scipy.ndimage import gaussian_filter


path_data = 'C:/Users/calderr/Downloads/tempData/'
filepaths = sorted( glob.glob(path_data + '*.hdf5') )
files = [f.split('\\')[-1] for f in filepaths]; 

channel_target_rel=11460             # target channel (rel) around which data should be extracted
nchs_before=500; nchs_after=500;    #  load n channels (relative) before and after the target channel
dec_chs_extr = 1
fstarget = 256
nfft = 512
fileIDs = [int(item.split('.')[0]) for item in files]
windowtime = 0.5 #seconds
bp_paras =[10, 100, 6, 0]             # bandpass parameters [lowcut,highcut,orders,zerophase]
alpha_taper=0.1

# low = 40
# high = 99
channels = np.arange(channel_target_rel-nchs_before,channel_target_rel+nchs_after)[::dec_chs_extr]


t_ex_start=time.perf_counter()  

data_raw, meta = load_multiple_DAS_files(path_data, fileIDs, chIndex=channels, roiIndex=None,
                            integrate=True, unwr=True, metaDetail=1, useSensitivity=True, spikeThr=None)

t_ex_end=time.perf_counter(); print(f'data loading time: {t_ex_end-t_ex_start}s'); 

dt = meta['header']['dt']
bp_lowcut, bp_highcut, bp_order, bp_zerophase = bp_paras
data_raw = utils_tmp.bandpass_2darray(data_raw, 1/dt, bp_lowcut, bp_highcut, axis_filt=0, order=bp_order, 
                                      zerophase=bp_zerophase, taper_data=False, alpha_tape=alpha_taper)


# %%
#this portion does some preliminary processing and then creates a spectrogram

num = data_raw.shape[0]/(1/fstarget)*dt
num = num.astype(int)
data = resample(data_raw,num ,axis = 0);
data=detrend(data, axis=0, type='linear');

windowsize = windowtime*fstarget
w = hamming(round(windowsize))
hop = round(windowsize/2)
SFT = ShortTimeFFT(w,hop = hop,fs = fstarget,mfft = nfft)

nchan = 10
chan = 500
sig = data[:,chan:chan+nchan]
sig = sig.sum(axis = 1)/(nchan+1)
plt.plot(sig)

Sx = SFT.stft(sig)
Sx = abs(Sx)

N = len(sig)
#plotting
fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
               rf"$\Delta t = {SFT.delta_t:g}\,$s)",
        ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
               rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
        xlim=(t_lo, t_hi))

im1 = ax1.imshow(abs(Sx), origin='lower', aspect='auto',
                 extent=SFT.extent(N), cmap='turbo')
fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")

ax1.legend()
fig1.tight_layout()
plt.show()

Sxb = gaussian_filter(Sx, sigma = 2 )

fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
               rf"$\Delta t = {SFT.delta_t:g}\,$s)",
        ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
               rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
        xlim=(t_lo, t_hi))

im1 = ax1.imshow(abs(Sxb), origin='lower', aspect='auto',
                 extent=SFT.extent(N), cmap='turbo')
fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")

ax1.legend()
fig1.tight_layout()
plt.show()

#%%
#generating a binary spectrogram then detection blobs
#remove small objects and holes
thresh2 = filters.threshold_otsu(Sxb,nbins=128) # could do differnt thresholding 
spec_bin = (Sxb > thresh2).astype(int)
#spec_bin = morphology.remove_small_objects(spec_bin,150)
#spec_bin = morphology.remove_small_holes(spec_bin,150)

labels = measure.label(spec_bin)
regions = measure.regionprops(labels)

fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
               rf"$\Delta t = {SFT.delta_t:g}\,$s)",
        ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
               rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
        xlim=(t_lo, t_hi))

im1 = ax1.imshow(abs(Sx), origin='lower', aspect='auto',
                 extent=SFT.extent(N), cmap='turbo')
fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")




for props in regions:
    minr,minc,maxr,maxc = props.bbox

    bx = (minc*SFT.delta_t,maxc*SFT.delta_t,maxc*SFT.delta_t,minc*SFT.delta_t,minc*SFT.delta_t) 
    by = (minr*SFT.delta_f,minr*SFT.delta_f,maxr*SFT.delta_f,maxr*SFT.delta_f,minr*SFT.delta_f)
    ax1.plot(bx,by,'-w',linewidth = 0.5)
    

ax1.legend()
fig1.tight_layout()
plt.show()

#this portion looks to detect blobs in the spec and label them and get idxs
# labels = measure.label(spec_bin)
# props = measure.regionprops(labels)
# blob_coords = [p.coords for p in props]

# blobsize = np.array([p.coords.size/2 for p in props]) #divide by 2 due to coordinate pairs to get number of pixels
# #plt.hist(blobsize, bins = 20)

# sizethresh = 150
# idx = np.nonzero(blobsize>150)
# blobsize2 = blobsize[idx] # get rid of all the small annoying ones
# short_props = props[idx]


#sort blobs by size?

# %%

