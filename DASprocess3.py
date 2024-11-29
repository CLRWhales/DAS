#this script is the first steps into a consistent processing pipeline for bulk data runs based on an ini file io

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
import configparser
import argparse

# prepping and reading the ini file
# parser = argparse.ArgumentParser()
# parser.add_argument("iniPath",help = 'path to to the ini file')
# args = parser.parse_args()
# config_name = args.iniPath

config_name = 'example.ini'

config = configparser.ConfigParser()
config.read(filenames=config_name)

bp_paras =[5, 127, 6, 0]             # bandpass parameters [lowcut,highcut,orders,zerophase] could make this a highpass
alpha_taper=0.1



#find files
filepaths = sorted( glob.glob(config['DataInfo']['Directory'] + '*.hdf5') )
files = [f.split('\\')[-1] for f in filepaths]; 
fileIDs = [int(item.split('.')[0]) for item in files]

if len(fileIDs)<int(config['DataInfo']['n_files']):
    nfiles = len(fileIDs)

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

fs_target = int(config['ProcessingInfo']['fs_target'])
data = data_raw
num = data.shape[0]/(1/fs_target)*dt
num = num.astype(int)
data = resample(data,num ,axis = 0);
data=detrend(data, axis=0, type='linear');

dt_new = 1/fs_target
bp_lowcut, bp_highcut, bp_order, bp_zerophase = bp_paras
data = utils_tmp.bandpass_2darray(data, fs_target, bp_lowcut, bp_highcut, axis_filt=0, order=bp_order, 
                                      zerophase=bp_zerophase, taper_data=True, alpha_tape=alpha_taper)

t_ex_end=time.perf_counter(); print(f'preprocess time: {t_ex_end-t_ex_start}s');

#compute spectrogram
t_ex_start=time.perf_counter()  

N_samp= int(config['FFTInfo']['n_samp'])
N_overlap = int(config['FFTInfo']['n_overlap'])
N_fft = int(config['FFTInfo']['n_fft'])
window = np.hamming(N_samp)
spec, freqs, times = sneakyfft(data,N_samp,N_overlap,N_fft, window,fs_target)

t_ex_end=time.perf_counter(); print(f'sneakyfft_time: {t_ex_end-t_ex_start}s'); 


fftout = abs(spec)

for f in range(fftout.shape[0]):
    plt.imsave(config['SaveInfo']['plot_directory'] + 'fig_'+str(f)+'.png', fftout[f,:,:],cmap = 'turbo',origin = 'lower',)




# %%
