#this script is the first steps into a consistent processing pipeline for bulk data runs based on an ini file io

#%%
import sys, glob, time #,os
import numpy as np # pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend, resample, butter, sosfiltfilt
#import utils_tmp, utils_time
from simpleDASreader4 import load_multiple_DAS_files
#from skimage import measure, filters,morphology
#from scipy.ndimage import gaussian_filter
from DASFFT import sneakyfft
import configparser
import argparse
import Calder_utils
import math
import datetime
# prepping and reading the ini file
# parser = argparse.ArgumentParser()
# parser.add_argument("iniPath",help = 'path to to the ini file')
# args = parser.parse_args()
# config_name = args.iniPath

config_name = 'example.ini'
config = configparser.ConfigParser()
config.read(filenames=config_name)


#find files
filepaths = sorted( glob.glob(config['DataInfo']['Directory'] + '*.hdf5') )
files = [f.split('\\')[-1] for f in filepaths]; 
fileIDs = [int(item.split('.')[0]) for item in files]

#trim to time of interest
starttime = int(config['ProcessingInfo']['startime'])
stoptime = int(config['ProcessingInfo']['stoptime'])
fileIDs = fileIDs[fileIDs>=starttime and fileIDs <=stoptime]


if len(fileIDs)<int(config['DataInfo']['n_files']):
    nfiles = len(fileIDs)
else: 
    nfiles = int(config['DataInfo']['n_files'])



fileIDs = fileIDs[0:nfiles]

#find channels
channels = []
for i in range(int(config['ProcessingInfo']['n_synthetic'])):
    channels.extend([x+i*int(config['ProcessingInfo']['synthetic_spacing']) for x in range(0,int(config['ProcessingInfo']['n_stack']))])

#load data + pre process
t_ex_start=time.perf_counter()  

data_raw, meta = load_multiple_DAS_files(config['DataInfo']['directory'], fileIDs, chIndex=channels, roiIndex=None,
                            integrate=True, unwr=True, metaDetail=1, useSensitivity=True, spikeThr=None)

t_ex_end=time.perf_counter(); print(f'data loading time: {t_ex_end-t_ex_start}s'); 
t_ex_start=time.perf_counter()  
dt = meta['header']['dt']

n_stack = int(config['ProcessingInfo']['n_stack'])

if config['ProcessingInfo'].getboolean('stack'):
    data = Calder_utils.faststack(data_raw,n_stack)
    channels = np.array(channels)[np.arange(0,len(channels),n_stack)]
else:
    data = data_raw

if config['ProcessingInfo']['fs_target'] == 'auto':
    fs_target = 2**math.floor(math.log(1/dt,2))
else:
    fs_target = int(config['ProcessingInfo']['fs_target'])

data /= 10E-12 #scaling into units of strain is handled, this moves it to pico strain? 
num = data.shape[0]/(1/fs_target)*dt
num = num.astype(int)
data = resample(data,num ,axis = 0);
data=detrend(data, axis=0, type='linear');

dt_new = 1/fs_target

#filering
cuts = [int(config['FilterInfo']['lowcut']),int(config['FilterInfo']['highcut'])]

if(config['FilterInfo']['type'] == 'lowpass' or config['FilterInfo']['type'] == 'highpass'):
    cuts = cuts[0]

sos = butter(N = int(config['FilterInfo']['order']),
             Wn = cuts,
             btype = config['FilterInfo']['type'],
             fs = fs_target,
             output = 'sos')

data = sosfiltfilt(sos = sos,
                   x = data,
                   axis = 0)

# data = utils_tmp.bandpass_2darray(data, fs_target, bp_lowcut, bp_highcut, axis_filt=0, order=bp_order, 
#                                       zerophase=bp_zerophase, taper_data=True, alpha_tape=alpha_taper)

t_ex_end=time.perf_counter(); print(f'preprocess time: {t_ex_end-t_ex_start}s');

#compute spectrogram
t_ex_start=time.perf_counter()  

N_samp= int(config['FFTInfo']['n_samp'])
N_overlap = int(config['FFTInfo']['n_overlap'])
N_fft = int(config['FFTInfo']['n_fft'])
window = np.hamming(N_samp)
spec, freqs, times = sneakyfft(data,N_samp,N_overlap,N_fft, window,fs_target)

t_ex_end=time.perf_counter(); print(f'sneakyfft_time: {t_ex_end-t_ex_start}s'); 

# saving the output:
#need to save vector F
#need to save vector T
#need to save vector channels
#need to save UTC start times
#need to save channel spacing
#need to save complex spec?

savetype = config['SaveInfo']['data_type']
if savetype == 'fast':
    maxspec = np.max(abs(spec),axis = 1)
    meanspec = np.mean(abs(spec),axis = 1)
    sdspec = np.std(abs(spec),axis = 1)

    norm = (maxspec-meanspec)/sdspec

    plt.figure(figsize=(8, 8))
    plt.imshow(norm, origin='lower', extent=(min(channels),max(channels),min(freqs),max(freqs)), aspect = 'auto')
    plt.colorbar()
elif savetype == 'magnitude':
    #do something else
    magspec = 10*np.log10(abs(spec))
    date = meta['header']['time']
    fdate = datetime.datetime.fromtimestamp(int(date),tz = datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
    data_name = config['SaveInfo']['data_directory'] + '/' + fdate
    np.save(data_name,magspec)


    frequencies = freqs
    c_idx = channels
    channeldistance = meta['header']['channels'][c_idx]
    





# %% this is a simple delay and sum beamformer
c_spacing =  meta['header']['gaugeLength'] #meters
c = 1450 #m/s in the water
thetas = np.linspace(-1*np.pi/2, np.pi/2, 64) #angles to search over 
#precompute the farfeild steering Array? dim(F, Chan,theta)
s = np.empty(shape = (len(freqs),n_stack,len(thetas)),dtype = "complex64")

d = c_spacing*freqs/c
wind = np.hamming(n_stack)

#adding fiber directionality?
w2 = np.cos(thetas)**2
plt.plot(thetas,w2)

for f in range(len(d)):
    for thet in range(len(thetas)):
        s[f,:,thet] = np.exp(-2j * np.pi * d[f] * np.arange(n_stack) * np.sin(thetas[thet]))*wind

sub_chans = np.arange(n_stack)

bf_stacked = np.empty((spec.shape[0],spec.shape[1],int(config['ProcessingInfo']['n_synthetic']),2),dtype = "complex64")
BEARING = thetas*180/np.pi
for chan in range(0,spec.shape[2],n_stack):
    out_idx = int(chan/n_stack)
    for t in range(spec.shape[1]):
         tmp = spec[:,t,chan+sub_chans][:,:,None]*s #apply Delay and sum weights, 
         tmp = np.sum(tmp,axis = 1) #and then combine across subchannels
         Spectral_maxBEARING = BEARING[np.squeeze(np.argmax(tmp,axis =1))] #find the max energy bearing
         Spectral_maxVAL = np.squeeze(np.max(tmp,axis = 1))
         bf_stacked[:,t,out_idx,0] = Spectral_maxVAL
         bf_stacked[:,t,out_idx,1] = Spectral_maxBEARING


# %%

stacked = abs(bf_stacked)

for channel in range(stacked.shape[2]):
    #plt.imsave(config['SaveInfo']['plot_directory'] + 'fig_'+str(channel)+'.png', stacked[:,:,channel,0],cmap = 'turbo',origin = 'lower',)
    #plot_false_color_arrays(stacked[:,:,channel,1],stacked[:,:,channel,0],config['SaveInfo']['plot_directory'] + 'fig_'+str(channel)+'.png')
    hue = (stacked[:,:,channel,1])# - np.min(stacked[:,:,channel,1])) / (np.ptp(stacked[:,:,channel,1]))
    intensity = (stacked[:,:,channel,0] - np.min(stacked[:,:,channel,0])) / (np.ptp(stacked[:,:,channel,0]))

    fname = config['SaveInfo']['plot_directory'] + 'fig_'+str(channel)+'.png'
    plt.figure(figsize=(8, 8))
    plt.imshow(X = hue, alpha= intensity, cmap = 'turbo',origin = 'lower', extent=(min(times),max(times),min(freqs),max(freqs)))
    plt.colorbar()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
# her we save the output to some kind of constant HDF5 structure with fields: F,T,Chan,spacing, 

# %%this runs the 2d hysteretic cfar threshold on the generic spectrograms
from CFAR2D import cfar_2D
from skimage import filters
cfar_stack = abs(np.copy(spec))*1E9
b1 = 2
b2 = 4

nf_guard = 4
nf_ref = 4
nt_guard = 10
nt_ref = 10



for i in range(spec.shape[2]):
    thresh = cfar_2D(cfar_stack[:,:,i],nf_guard,nt_guard,nf_ref,nt_ref)
    map = np.zeros(shape = (thresh.shape))
    map[np.where(cfar_stack[:,:,i]>=thresh*b1)] = 1
    all_zeros = not np.any(map) 
    if(all_zeros):
        cfar_stack[:,:,i] = map
        continue 

    map[np.where(cfar_stack[:,:,i]>=thresh*b2)] = 3
    new = filters.apply_hysteresis_threshold(map, 0.5,2)
    cfar_stack[:,:,i] = cfar_stack[:,:,i]*new


spec2 = abs(spec)
for channel in range(cfar_stack.shape[2]):

    # #only plot the things with something
    # all_zeros = not np.any(cfar_stack[:,:,channel])   

    # if(all_zeros):
    #     continue 

    fname = config['SaveInfo']['plot_directory'] + 'fig_'+str(channel)+'.png'
    plt.figure(figsize=(8, 8))
    plt.imshow(X = cfar_stack[:,:,channel], cmap = 'turbo', origin = 'lower')
    plt.colorbar()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


# %% exponential moving average detone
import scipy.signal as sig


aspec = (abs(spec))
plt.imshow((aspec[:,:,228]+aspec[:,:,227])/2,origin='lower', cmap = 'magma')
plt.colorbar()

smoother = np.array(((1,2,1),(2,4,2),(1,2,1)))[:,:,None]
aspec = sig.convolve(aspec,smoother,mode = 'same')

n = spec.shape[1]
y = np.empty_like(aspec)
y[:,0,:] = aspec[:,0,:]
decay_const = 0.15
dt = 0.5
T = 5
epsilon = 1 - np.exp(np.log(decay_const)*(dt/T))

for t in range(1,n):
     y[:,t,:] = (1-epsilon)*y[:,t-1,:] + epsilon*aspec[:,t,:]

cleaned = aspec - y
plt.imshow(cleaned[:,:,228],origin='lower', cmap = 'magma')



 # %%
