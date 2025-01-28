#this script is the first steps into a consistent processing pipeline for bulk data runs based on an ini file io
import sys, glob, time #,os
import numpy as np # pandas as pd
#import matplotlib.pyplot as plt
from scipy.signal import detrend, resample, butter, sosfiltfilt
#import utils_tmp, utils_time
from simpleDASreader4 import load_DAS_file, unwrap, combine_units #nned this if the other functions are uncommented
#from skimage import measure, filters,morphology
#from scipy.ndimage import gaussian_filter
from DASFFT import sneakyfft
import configparser
import argparse
import Calder_utils
import math
import datetime
#from load_DAS_conc import load_files, preprocess_DAS
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial

def load_file(channels, verbose, filepath):
    """
    load a single DAS file -> from kevinG
    """
    if verbose: 
        fid = filepath.split('/')[-1];  print(f"/nloading file {fid}")
    data, meta = load_DAS_file(filepath, chIndex=channels, roiIndex=None, samples=None,
                      integrate=False, unwr=False, metaDetail=1, useSensitivity=False,
                      spikeThr=None)
    return data, meta


def load_files(path_data, channels, verbose, fileIDs):
    """
    distribute multiple file loading over threads -> from kevinG
    """
    # create a thread pool
    with ThreadPoolExecutor(len(fileIDs)) as exe:
        
        file_paths = [path_data + str(fid) + '.hdf5' for fid in fileIDs]
        # load files
        results = exe.map(partial(load_file,channels, verbose), file_paths)
        # collect data
        list_data=[]; list_meta=[];
        for listx in results: 
            list_data.append(listx[0]); list_meta.append(listx[1])

        return (list_data,list_meta, fileIDs)



def preprocess_DAS(data, list_meta, unwr=True, integrate=True, useSensitivity=True, spikeThr=None): 
    """
    preprocess loaded DAS data, for details see SimpleDASreader4  -> from kevinG
    """
    meta = list_meta[0]
    # pre-process raw data
    if unwr or spikeThr or integrate:
        if meta['header']['dataType']<3 or meta['demodSpec']['nDiffTau']==0:
            raise ValueError('Options unwr, spikeThr or integrate can only be\
                             used with time differentiated phase data')
    if unwr and meta['header']['spatialUnwrRange']:
        data=unwrap(data,meta['header']['spatialUnwrRange'],axis=1)
    unit=meta['appended']['unit']
    #print(f"DEBUG: unit={unit}")
    if spikeThr is not None:
        data[np.abs(data)>spikeThr] = 0
    if useSensitivity:
        if 'sensitivity' in meta["header"].keys():
            data/=meta['header']['sensitivity']
            sensitivity_unit=meta['header']['sensitivityUnit']
        elif 'sensitivities' in meta["header"].keys(): 
            data/=meta['header']['sensitivities'].item()
            sensitivity_unit=meta['header']['sensitivityUnits'].item()
        else: 
            raise KeyError("!! no sensitivity keys in meta dict")
        unit=combine_units([unit, sensitivity_unit],'/')
        #print(f"DEBUG: sensitivity_unit={sensitivity_unit}, unit={unit}")
    if integrate:
        data=np.cumsum(data,axis=0)*meta['header']['dt']
        unit=combine_units([unit, unit.split('/')[-1]])
        #print(f"DEBUG: unit={unit}")
    meta['appended']['unit']=unit
   
    #update all metas
    for metax in list_meta: 
        metax['appended']['unit']=unit
        
    return (data, list_meta)


def LPS_block(path_data,channels,verbose,config, fileIDs):
    """
    Load, Process, and Save, a single block of das data files into FTX for further analysis and visualization 
    """
    #load into list
    list_data, list_meta, _ = load_files(path_data = path_data,
                                      channels = channels,
                                      verbose = verbose,
                                      fileIDs= fileIDs)
    data =  np.concatenate(list_data, axis=0)
    data, list_meta = preprocess_DAS(data, list_meta)
    data /= 10E-12 #scaling into units of strain is handled, this moves it to pico strain? 

    #do stacking
    n_stack = int(config['ProcessingInfo']['n_stack'])
    chans = list_meta[0]['appended']['channels']

    if config['ProcessingInfo'].getboolean('stack'):
        data = Calder_utils.faststack(data,n_stack)
        chIDX = np.arange(start = 0,stop = len(chans), step = n_stack)
    else:
        chIDX = np.arange(0,len(chans))

    #do resampling
    dt = list_meta[0]['header']['dt']
    if config['ProcessingInfo']['fs_target'] == 'auto':
        fs_target = 2**math.floor(math.log(1/dt,2))
    else:
        fs_target = int(config['ProcessingInfo']['fs_target'])

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
    

    # STFT
    match config['FFTInfo']['input_type']:
        case 'point':
            N_samp= int(config['FFTInfo']['n_samp'])
            N_overlap = int(config['FFTInfo']['n_overlap'])
            N_fft = int(config['FFTInfo']['n_fft'])
            window = np.hamming(N_samp)
            spec, freqs, times = sneakyfft(data,N_samp,N_overlap,N_fft, window,fs_target)
        case 'time':
            N_samp= fs_target*int(config['FFTInfo']['n_samp'])
            N_overlap = N_samp*int(config['FFTInfo']['n_overlap'])
            N_fft = fs_target/int(config['FFTInfo']['n_fft'])
            window = np.hamming(N_samp)
            spec, freqs, times = sneakyfft(data,N_samp,N_overlap,N_fft, window,fs_target)
        case _:
            raise TypeError('input must be either "point" or "time"')

    

    #saving info
    date = list_meta[0]['header']['time']
    fdate = datetime.datetime.fromtimestamp(int(date),tz = datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
    data_name = config['SaveInfo']['data_directory'] + '/FTX' + str(fs_target) + '_' + fdate +'Z'

    if fileIDs[0] == config['Append']['first']:
        freqname = config['SaveInfo']['data_directory'] + '/Dim_Frequency.txt'
        np.savetxt(freqname,freqs)

        channeldistance = list_meta[0]['appended']['channels'][chIDX]
        channelname= config['SaveInfo']['data_directory'] + '/Dim_Channel.txt'
        np.savetxt(channelname,channeldistance)

        timename = config['SaveInfo']['data_directory'] + '/Dim_Time.txt'
        np.savetxt(timename,times)

        cfgname = config['SaveInfo']['data_directory'] + '/config.ini'
        with open(cfgname, 'w') as configfile:
            config.write(configfile)

    match config['SaveInfo']['data_type']:
        case 'fast':
            maxspec = np.max(abs(spec),axis = 1)
            meanspec = np.mean(abs(spec),axis = 1)
            sdspec = np.std(abs(spec),axis = 1)
            norm = (maxspec-meanspec)/sdspec

            # plt.figure(figsize=(8, 8))
            # plt.imshow(norm, origin='lower', extent=(min(c2),max(c2),min(freqs),max(freqs)), aspect = 'auto')
            # plt.colorbar()
        
        case 'magnitude':
            #print('mag')
            spec = 10*np.log10(abs(spec))
            np.save(data_name,spec)
        case 'complex':
            #print('complex')
            spec = 10*np.log10(spec)
            np.save(data_name,spec)

def DAS_processor(X):
    config = configparser.ConfigParser()
    config.read(filenames=X)

    #setup 
    filepaths = sorted( glob.glob(config['DataInfo']['Directory'] + '*.hdf5') )
    files = [f.split('\\')[-1] for f in filepaths] 
    fileIDs = [int(item.split('.')[0]) for item in files]

    if len(fileIDs)== 0:
        raise ValueError("Data files cannot be found, check path ends with slash")
    

    if len(config['ProcessingInfo']['starttime'])>0:
        starttime = int(config['ProcessingInfo']['starttime'])
        stoptime = int(config['ProcessingInfo']['stoptime'])
        fileIDs = [i for i in fileIDs if i >= starttime and i <= stoptime]

    if type(fileIDs[0]) == int: 
            fileIDs = ['{:06d}'.format(fid) for fid in fileIDs]  
    fileIDs_int = np.array(fileIDs, dtype=np.int32)
    assert fileIDs_int[0] == fileIDs_int.min() and fileIDs_int[-1] == fileIDs_int.max()

    config['Append'] = {'first':fileIDs[0]}

    n_files = int(config['DataInfo']['n_files'])
    list_fids = [fileIDs[x:x+n_files] for x in range(0, len(fileIDs), n_files)]

    #find channels
    channels = []
    if int(config['ProcessingInfo']['n_synthetic']) == -1:
        channels = None
    else:
        for i in range(int(config['ProcessingInfo']['n_synthetic'])):
            channels.extend([x+i*int(config['ProcessingInfo']['synthetic_spacing']) for x in range(0,int(config['ProcessingInfo']['n_stack']))])


    n_workers = int(config['DataInfo']['n_workers'])
    path_data = config['DataInfo']['directory']
    verbose = config['ProcessingInfo'].getboolean('verbose')

    if verbose:
        print(path_data)
        print(list_fids)
        print(channels) 

    with ProcessPoolExecutor(max_workers= n_workers) as executor:
        executor.map(partial(LPS_block, path_data,channels,verbose, config), list_fids)
       
            
        
if __name__ == '__main__':
    config_name = "C:/Users/Calder/Workspace/Python_Env/DAS/example.ini"
    t_ex_start=time.perf_counter()  
    DAS_processor(config_name)
    t_ex_end=time.perf_counter(); print(f'duration: {t_ex_end-t_ex_start}s'); 


