#this script is the first steps into a consistent processing pipeline for bulk data runs based on an ini file io
#%%
import os
import glob,time
import numpy as np 
from scipy.signal import detrend, resample, butter, sosfiltfilt
from simpleDASreader4 import load_DAS_file, unwrap, combine_units #nned this if the other functions are uncommented
from DASFFT import sneakyfft
import configparser
import argparse
import Calder_utils
import math
import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial


#handling lack of tk on some linux distross. shoddy, need to fix in the future
try:
    import tkinter as tk
except ImportError:
    available = False
else:
    available = True
    from tkinter import filedialog

def load_INI():
    """
    this function loads an ini file from the terminal or file browser
    TODO: 
        build in a default/imput checking portion so that we can make sure that it is all good to run. 
    """

    parser = argparse.ArgumentParser(description="Process a filename.")
    parser.add_argument("filename", nargs="?", type=str, help="The name of the file to process")
    
    args = parser.parse_args()
    
    if args.filename is None and available:
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        args.filename = filedialog.askopenfilename(title="Select a file")
        root.destroy()
    
    config = None
    if args.filename:
        config = configparser.ConfigParser()
        config.read(filenames=args.filename)
    
    return config



def load_file(channels, verbose, filepath):
    """
    load a single DAS file -> from kevinG
    """
    if verbose: 
        fid = os.path.basename(filepath);  print(f"/nloading file {fid}")
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
        
        file_paths = [os.path.join(path_data, str(fid) + '.hdf5') for fid in fileIDs]
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
    cuts = [float(config['FilterInfo']['lowcut']),float(config['FilterInfo']['highcut'])]
    dofilt = True
    

    match config['FilterInfo']['type']:
        case 'lowpass':
            cuts = cuts[0]
        case 'highpass':
            cuts = cuts[0]
        case 'bandpass':
            cuts = cuts
        case 'bandstop':
            cuts = cuts
        case 'none':
            dofilt = False
        case _:
            TypeError('input must be either "lowpass", "highpass","bandpass","bandstop", or "none')
    
    
    
    if dofilt:
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
            N_samp= int(fs_target*float(config['FFTInfo']['n_samp']))
            N_overlap = int(N_samp*float(config['FFTInfo']['n_overlap']))
            N_fft = int(fs_target/float(config['FFTInfo']['n_fft']))
            window = np.hamming(N_samp)
            spec, freqs, times = sneakyfft(data,N_samp,N_overlap,N_fft, window,fs_target)
            
        case _:
            raise TypeError('input must be either "point" or "time", if time, make sure divisions yield power of 2 for speed')

    

    #saving info
    date = list_meta[0]['header']['time']
    fdate = datetime.datetime.fromtimestamp(int(date),tz = datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')


    if fileIDs[0] == config['Append']['first']:

        freqname = os.path.join(config['Append']['outputdir'] , 'Dim_Frequency.txt')
        np.savetxt(freqname,freqs)

        channeldistance = list_meta[0]['appended']['channels'][chIDX]
        channelname= os.path.join(config['Append']['outputdir'] ,'Dim_Channel.txt')
        np.savetxt(channelname,channeldistance)

        timename = os.path.join(config['Append']['outputdir'] , 'Dim_Time.txt')
        np.savetxt(timename,times)

        cfgname = os.path.join(config['Append']['outputdir'] , 'config.ini')
        with open(cfgname, 'w') as configfile:
            config.write(configfile)

        
    match config['SaveInfo']['data_type']: 
        case 'magnitude':
            magdir = os.path.join(config['Append']['outputdir'] , 'Magnitude')
            #Path(magdir).mkdir()
            os.makedirs(magdir, exist_ok=True)
            spec = 10*np.log10(abs(spec))
            fname = 'FTX' + str(fs_target) + '_' + fdate +'Z'
            data_name = os.path.join(magdir,fname)
            np.save(data_name,spec)

            
        case 'complex':
            compdir = os.path.join(config['Append']['outputdir'] , 'Complex')
            #Path(compdir).mkdir(exist_ok=True)
            os.makedirs(compdir, exist_ok=True)
            #spec = 10*np.log10(spec)
            fname = 'FTX' + str(fs_target) + '_' + fdate +'Z'
            data_name = os.path.join(compdir,fname)
            np.save(data_name,spec)

        case 'cleaning':
            #these thresholds are based on Robins values.
            mfreq = np.max(freqs)
            cuts = [0.5,5,30,130,mfreq+1] 
            for (l,h) in zip(cuts[:-1],cuts[1:]):
                if l > mfreq:
                    break
                f_idx = np.where(np.logical_and(freqs >= l,freqs <= h))
                TX = 10*np.log10(np.mean(abs(spec[f_idx[0],:,:]),axis = 0))
                cleandir = os.path.join(config['Append']['outputdir'] , str(l) + 'Hz_' + str(h) + 'Hz')
                #Path(cleandir).mkdir(exist_ok=True)
                os.makedirs(cleandir, exist_ok= True)
                fname = 'TX' + str(fs_target) + '_' + fdate +'Z'
                fout = os.path.join(cleandir, fname)
                np.save(fout,TX)

        case 'LTSA':
            LTSAdir = os.path.join(config['Append']['outputdir'] , 'LTSA')
            os.makedirs(LTSAdir, exist_ok=True)
            LTSA = np.mean(abs(spec),axis = 1)
            fname = 'FX_LTSA' + str(fs_target) + '_' + fdate +'Z'
            data_name = os.path.join(LTSAdir,fname)
            np.save(data_name,LTSA)
            
        case _:
            raise TypeError('input must be either "magnitude", "complex","cleaning" ')

def DAS_processor():
    config = None
    config = load_INI()

    if not config:
        raise ValueError("could not find the config, check file path")

    #setup 
    filepaths = sorted( glob.glob(os.path.join(config['DataInfo']['Directory'], '*.hdf5')))
    files = [os.path.basename(f) for f in filepaths] 
    fileIDs = [int(item.split('.')[0]) for item in files]

    if len(fileIDs)== 0:
        print(os.path.join(config['DataInfo']['Directory'], '*.hdf5'))
        #raise ValueError("Data files cannot be found, check path ends with slash")
    
    

    if len(config['ProcessingInfo']['starttime'])>0:
        starttime = int(config['ProcessingInfo']['starttime'])
        stoptime = int(config['ProcessingInfo']['stoptime'])
        fileIDs = [i for i in fileIDs if i >= starttime and i <= stoptime]
        if len(fileIDs) == 0:
            raise ValueError("time snippet requested does not exist in File list, check if correct")

    if type(fileIDs[0]) == int: 
            fileIDs = ['{:06d}'.format(fid) for fid in fileIDs]  
    fileIDs_int = np.array(fileIDs, dtype=np.int32)
    assert fileIDs_int[0] == fileIDs_int.min() and fileIDs_int[-1] == fileIDs_int.max()

    n_files = int(config['DataInfo']['n_files'])
    list_fids = [fileIDs[x:x+n_files] for x in range(0, len(fileIDs), n_files)]

    #find channels
    firstfile = os.path.join(config['DataInfo']['directory'] , fileIDs[0] + '.hdf5')
    channels = []
    meta = Calder_utils.load_meta(firstfile)
    chans = meta['header']['channels']
    match config['ProcessingInfo']['n_synthetic']:
        case '-1':
            channels = None

        case 'auto':
            spacing = int(config['ProcessingInfo']['synthetic_spacing'])
            nstack = int(config['ProcessingInfo']['n_stack'])
            
            n_synthetic = np.floor(len(chans)/spacing)
            n_synthetic = int(n_synthetic)
            for i in range(n_synthetic):
                channels.extend([x+i*spacing for x in range(0,nstack)])
            channels[:] = [x for x in channels if x <= np.max(chans)]

        case 'meter':
            dx = int(chans[1]-chans[0])
            spacing = int(np.rint(int(config['ProcessingInfo']['synthetic_spacing'])/dx))
            nstack = int(config['ProcessingInfo']['n_stack'])
            n_synthetic = np.floor(len(chans)/spacing)
            n_synthetic = int(n_synthetic)

            if nstack > spacing:
                nstack = nstack - (nstack - spacing - 1) #makes the nstacks fit nicely within spacing
                print('reducing stack size to fit in spacing.')
            
            for i in range(n_synthetic):
                channels.extend([x+i*spacing for x in range(0,nstack)])
            channels[:] = [x for x in channels if x <= np.max(chans)]

        case _:
            for i in range(int(config['ProcessingInfo']['n_synthetic'])):
                channels.extend([x+i*int(config['ProcessingInfo']['synthetic_spacing']) for x in range(0,int(config['ProcessingInfo']['n_stack']))])
            channels[:] = [x for x in channels if x <= np.max(chans)]


    n_workers = int(config['DataInfo']['n_workers'])
    verbose = config['ProcessingInfo'].getboolean('verbose')
    path_data = config['DataInfo']['directory'] 

    #making the output directory
    tnow = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    outputdir = os.path.join(config['SaveInfo']['directory'],config['SaveInfo']['run_name']+tnow)
    os.makedirs(outputdir)
    
    config['Append'] = {'first':fileIDs[0],
                        'outputdir':outputdir}




    if verbose:
        print(path_data)
        print(outputdir)
        print(channels) 


    # with ProcessPoolExecutor(max_workers= n_workers) as executor:
    #     executor.map(partial(LPS_block, path_data,channels,verbose, config), list_fids)
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(partial(LPS_block, path_data,channels,verbose, config), lf)for lf in list_fids]
        for future in as_completed(futures):
            future.result()
        
if __name__ == '__main__':

    #config_name = "C:/Users/Calder/Workspace/Python_Env/DAS/example.ini"
    t_ex_start=time.perf_counter()  
    DAS_processor()
    t_ex_end=time.perf_counter(); print(f'duration: {t_ex_end-t_ex_start}s'); 



# %%
