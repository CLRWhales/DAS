#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
"""
Created on Wed Jan 25 12:08:43 2023

@author: keving
"""


import sys, glob, time #,os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.signal import decimate,detrend #resample

import utils_tmp, utils_time
from simpleDASreader4 import load_multiple_DAS_files


# paths for input
path_info = '../../BaneNOR/Info/'
date='20210831'                     # date of data acquisition in format of YYYYMMDD
path_data = 'C:/Users/calderr/Downloads/tmpData8m/'
headerfile = None  # path_info + f'Headerinfo/header_info_{date}.csv'

#constants
# time delay between DAS System and UTC
TIME_OFFSET_DAS = 0                 # time difference between mearsuring device and actual UTC time
DX_ORIG = 1.02                      # physical spatial sampling of DAS equipment: stored in headers under dx
DEC_CHS=4;                          # interval of spatial data extraction during experiment, stored in headers und demodSpec/roiDec
LEN_REC=10                          # recording length for each file in seconds

# flags and settings
detrend_data=1                      # remove the (linear) trend from the data
dec_chs_extr=2;                     # decimation in space (e.g. if 2, every second channel is extracted)
dec_time = 1                        # decimation in time (e.g. if 2, every second sample is extracted)
ch_dec_stack=int(2);                     # apply averaging in space (e.g. if 2, every two channels are averaged)
apply_bandpass=1                    # apply bandpass frequency filter to the data
bp_paras =[30, 130, 6, 0]             # bandpass parameters [lowcut,highcut,orders,zerophase]
alpha_taper=0.1

# select data
time_selected =  None #'00:16:37' #None #'11:06:43' #'11:06:43' #'14:08:03'
seconds_off =  0                    # offset from time in seconds
file_idx=1                         # index of target file that is to be read / only used if time_selected is None
channel_target_rel=1             # target channel (rel) around which data should be extracted
nchs_before=0; nchs_after=16000;    #  load n channels (relative) before and after the target channel
nfiles_before=1; nfiles_after=2;    # load nfiles before and after the target file 

# plot settings 
plot_data_raw=0; plot_data_pro=1;

# create list of files
filepaths = sorted( glob.glob(path_data + '*.hdf5') )
files = [f.split('\\')[-1] for f in filepaths]; 


# obtain file_idx for selected time 
if time_selected is not None and headerfile is not None: 
    # read in header data
    df_head = pd.read_csv(headerfile, sep='\t', header=0, dtype={'file_id': object})
    year, month, day = utils_time.date2ints(date)
    hh, mm, ss = utils_time.time2ints(time_selected)
    dto = datetime(year=year, month=month, day=day, hour=hh, minute=mm, second=ss)
    dto_utc = utils_time.assign_utc(dto) + timedelta(seconds=seconds_off)
    timestamp = dto_utc.timestamp()
    file_idx, toff = utils_tmp.get_idx_closest_smaller_value(df_head["timestamp_rec"], timestamp)
    time_rel_event = toff + nfiles_before*LEN_REC

# create list of file idxes to load
file_idxes = np.arange(file_idx-nfiles_before,file_idx+nfiles_after,1)

# obtain fileIDs from selected files
fileIDs = [int(files[fidx].split('.')[0]) for fidx in file_idxes]; 
nfids=len(fileIDs)

# create array of channels to load
channels = np.arange(channel_target_rel-nchs_before,channel_target_rel+nchs_after)[::dec_chs_extr]

# load data
t_ex_start=time.perf_counter()  
data_raw, meta = load_multiple_DAS_files(path_data, fileIDs, chIndex=channels, roiIndex=None,
                            integrate=True, unwr=True, metaDetail=1, useSensitivity=True, spikeThr=None)

#data_raw = data_raw.astype('float32', order='F')
t_ex_end=time.perf_counter(); print(f'data loading time: {t_ex_end-t_ex_start}s'); 

# gather headerinfo
header = meta.get("header")
dt_orig = header["dt"].item(); # original sample spacing in time
dec_chs_sampled = meta["demodSpec"]["roiDec"].item();  # channel decimation while sampling (e.g. if 2, every second channel is stored)
dx_phy=header["dx"].item();  # original physical channels spacing
dx=dx_phy*dec_chs_sampled*dec_chs_extr # actual channel spacing resulting from physical spacing and the decimations
channels_abs = meta["appended"]["channels"] # actual (absolut) channels that are extracted
GL = meta['demodSpec']['nDiffTau'] * dx_phy # gauge length of acquisition

# extract and process time info
timestamp = header.get("time").item(); # timestamp of extracted recording
dto_rec = datetime.fromtimestamp(timestamp);
dto_rec = utils_time.dto_loc2utc(dto_rec, local=False); 
dto_utc = dto_rec + timedelta(seconds=-TIME_OFFSET_DAS)
trec, trec_utc  = [dto.isoformat().split("T")[1].split('+')[0].split('.')[0] for dto in [dto_rec, dto_utc]]; 

if plot_data_raw: 
    ns, nx = data_raw.shape
    times = np.arange(0,ns)*dt_orig
    fig = plt.figure()
    mesh = plt.pcolormesh(meta["appended"]["channels"], times, np.log10(np.abs(data_raw)), vmin=-10, vmax=None, cmap='viridis')
    ax=plt.gca(); ax.invert_yaxis()
    plt.title('raw DAS, {}, UTC: {}'.format(date,trec_utc))
    plt.xlabel('channels_abs'); plt.ylabel('Time [s]')
    cbar = plt.colorbar(mappable=mesh); cbar.set_label('log(|Strain|)')
    plt.show()
      
#%% process data
t_ex_start=time.perf_counter()  
if ch_dec_stack: # apply stacking (averaging) in space (among channels)
    data = utils_tmp.stack_channels(data_raw, ch_dec_stack, axis_stack=1); channels_abs=channels_abs[::ch_dec_stack]
    dx = dx * ch_dec_stack
else: 
    data=data_raw.copy()
if dec_time > 1:  # apply decimation in time 
    data = decimate(data, dec_time, axis=0); print("decimate data in time"); dt=dt_orig*dec_time
else: 
    dt=dt_orig; 
if detrend_data: 
    data=detrend(data, axis=0, type='linear'); print("detrend data")
if apply_bandpass:
    print('apply bandpass')
    if type(bp_paras) != list or (type(bp_paras) == list and len(bp_paras) != 4): 
        raise ValueError("! bandpass params need to be a list of 4 items: lowcut,highcut,order,zerophase")
    bp_lowcut, bp_highcut, bp_order, bp_zerophase = bp_paras
    data = utils_tmp.bandpass_2darray(data, 1/dt, bp_lowcut, bp_highcut, axis_filt=0, order=bp_order, 
                                      zerophase=bp_zerophase, taper_data=False, alpha_tape=alpha_taper)
t_ex_end=time.perf_counter(); print(f'data processing time: {t_ex_end-t_ex_start}s'); 

# store header info after processing
info = {"dt_orig":dt_orig,"dt":dt,"dx":dx, "dx_phy":dx_phy,"dec_chs_sampled":dec_chs_sampled, 
        "dec_chs_extr":dec_chs_extr, "ch_stack":ch_dec_stack, "trec":trec, "trec_utc":trec_utc,
        "channels_abs":channels_abs, "channels_rel_extr":channels,"gauge_length":GL,
        "timestamp_rec":timestamp, "timestamp_utc":dto_utc.timestamp()}

if plot_data_pro: 
    
    ns, nx = data.shape
    times = np.arange(0,ns)*dt_orig
    fig = plt.figure()
    mesh = plt.pcolormesh(channels_abs, times, np.log10(np.abs(data)), vmin=-10, vmax=None, cmap='turbo')
    ax=plt.gca(); ax.invert_yaxis()
    plt.title('DAS_pro, {}, UTC: {}'.format(date,trec_utc))
    plt.xlabel('channels_abs'); plt.ylabel('Time [s]')
    cbar = plt.colorbar(mappable=mesh); cbar.set_label('log(|Strain|)')
    plt.show()
    

print("done"); #sys.exit()


# %%
#this portion tries to apply a sliding RMS filter to it.