#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:43:47 2022
@author: keving
load many files concurrently with processes and threads
yields a speedup of up to 40% (depends on machine)
2-4 cpus are usually enough as the bottleneck is the reading from disk
idea from superfastpython.com
"""
import time
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from simpleDASreader4 import load_multiple_DAS_files, load_DAS_file, unwrap, combine_units
 
LEN_REC = 10.0 # recording length of files

def distribute_items_workers(items, n_workers): 
    """
    helper function to distribute items evenly over workers

    Parameters
    ----------
    items : list
        list of items to distribute.
    n_workers : int
        number of workers.

    Returns
    -------
    list_items : list
        list with sublists of items for each worker.

    """
    nitems = len(items)
    nitems_per_worker = int(nitems/n_workers) 
    nitems_remainder = nitems % n_workers
    
    idx_min=0; list_items=[]; 
    for i in range(n_workers):
        # distribute div remainder over data chunks 
        if i < nitems_remainder:
            div_remainder=1
        else: 
            div_remainder=0
        if i ==0: 
            idx_min = idx_min + i*nitems_per_worker 
            idx_max = idx_min + nitems_per_worker + div_remainder
        else: 
            idx_min = idx_max
            idx_max = idx_min + nitems_per_worker + div_remainder
        list_items.append(items[idx_min:idx_max])
    
    return list_items


def load_file(channels, verbose, filepath):
    """
    load a single DAS file
    """
    if verbose: 
        fid = filepath.split('/')[-1];  print(f"\nloading file {fid}")
    data, meta = load_DAS_file(filepath, chIndex=channels, roiIndex=None, samples=None,
                      integrate=False, unwr=False, metaDetail=1, useSensitivity=False,
                      spikeThr=None)
    return data, meta

 
def load_files(path_data, channels, verbose, fileIDs):
    """
    distribute file loading over threads
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

        return (list_data, list_meta, fileIDs)
    

def preprocess_DAS(data, list_meta, unwr=True, integrate=True, useSensitivity=True, spikeThr=None): 
    """
    preprocess loaded DAS data, for details see SimpleDASreader4
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

 
def load_data_conc(path_data, fileIDs, channels=None, n_workers=4, unwr=True, integrate=True, df_head=None,
                   useSensitivity=True, spikeThr=None, verbose=False, full_output=False, fill_gaps=False,
                   preprocess_data=True, record_time=True, colnames_gap_rem=None, gap_fill_threshold=60,
                   mem_order="F"):
    """
    load files concurrently over multiple processors and threads and concatenate

    Parameters
    ----------
    fileIDs : list of strings
        list of fileIDs to load.
    path_data : str
        data path.
    channels : 1d np.array or list
        channels to load.
    n_workers : int, optional
        number of processors to use for loading. The default is 4.
    unwr : bool, optional
        unwrap DAS signal. The default is True.
    integrate : bool, optional
        integrate DAS signal. The default is True.
    useSensitivity : bool, optional
        use proper amplitude scaling. The default is True.
    spikeThr : float, optional
        substitute high amplitude values. The default is None.
    fill_gaps: bool, optional
        flag for filling potential gaps in between the loaded files with zeros. Requires df_head (see description). Default is False.
    df_head: pd.DataFrame
        pd.DataFrame containing columns header info for each file: fileIDs, flags for gaps and 
        time difference between subsequent files. The default is None.
    colnames_gap_rem: dict, optional
        dictionary containing the column names for df_head used in the gap_removal. 
        example: {"fids":YOUR_COLUMN_NAME,"flag_gap":YOUR_COLUMN_NAME,"tdiff_file":YOUR_COLUMN_NAME}.
        Default is None.
    gap_fill_threshold: float, optional
        max gapsize to fill with zeros, in seconds 
     preprocess_data: bool, optional
         apply DAS pre-processing to data if true. Default is True.
    full_output: bool, optional
        returns list of meta data for all loaded files and dict for gap removal if true. 
        Default is False.
    mem_order: str, optional
        memory order of the array. Default ist F (Fortran) and allows for faster columnwise processing
    Raises
    ------
    ValueError
        for too large gaps. The loading should be splitted to avoid the gap

    Returns
    -------
    2d np.array, list, list
        2d data array, list of meta data, list of fileIDs.

    """
    if record_time:
        tex_start=time.perf_counter(); 
        
    # prepare
    if type(fileIDs[0]) == int: 
            fileIDs = ['{:06d}'.format(fid) for fid in fileIDs]  
    fileIDs_int = np.array(fileIDs, dtype=np.int32)
    assert fileIDs_int[0] == fileIDs_int.min() and fileIDs_int[-1] == fileIDs_int.max()
    
    if n_workers>len(fileIDs):
        n_workers=len(fileIDs); print(f"WARNING: workers reduced to {n_workers}")
    # create the process pool
    with ProcessPoolExecutor(n_workers) as executor:
        
        list_fids = distribute_items_workers(fileIDs, n_workers)    
        results = executor.map( partial(load_files,path_data,channels,verbose), list_fids) 
        list_data=[]; list_meta=[]; list_ids=[]; 
        for listx in results:  # sort results into dedicated lists
            for ll in range(len(listx[0])):
               [ listx_results.append( listx[idx][ll] ) for idx, listx_results in enumerate((list_data,list_meta,list_ids)) ]
                #list_data.append(listx[0][ll]) ; #list_meta.append(listx[1][ll]); #list_ids.append(listx[2][ll])
    if fill_gaps: 
        assert isinstance(df_head, pd.DataFrame), \
            "dataframe with header info / gap info not provided. Abort."
        # get labels for header data
        if colnames_gap_rem is None: 
            colnames_gap_rem = {"fid":"fid_int","flag_gap":"flag_gap","tdiff_file":"tdiff_prev_file"}
        label_fids_int, label_flag_gaps, label_tdiff_file = [colnames_gap_rem[label] \
                                                             for label in ["fid", "flag_gap","tdiff_file"]]
            
        df_head_cut = df_head[(df_head[label_fids_int]>=fileIDs_int[0]) & \
                              (df_head[label_fids_int]<=fileIDs_int[-1])]
        assert df_head_cut.shape[0]==len(list_data), \
            "number of headerinfo deviates from number of data files. Abort."
        flags_gaps = df_head_cut[label_flag_gaps].values[1::] # remove 0-idx as no zeros should be appended before the first file
        #print(f"DEBUG: flags_gaps: {flags_gaps}")
        if np.any(flags_gaps.astype(bool)):
            print("gaps detected, proceeed to removal")
            dt = df_head_cut.iloc[0]["dt"]
            fidxes_gaps = np.where(flags_gaps==1)[0]+1
            # if fidxes_gaps[0]==0: # remove 0-idx as no zeros should be appended before the first file
            #     fidxes_gaps = np.delete(fidxes_gaps, 0)
            ns_gaps, tdiffs_gaps = [np.zeros(fidxes_gaps.shape[0]).astype(int) for k in range(2)]
            for i,idx in enumerate(fidxes_gaps):
                #if flags_gaps[idx-1]==1: # -1 because is first file is excluded 
                tdiffs_gaps[i] = df_head_cut.iloc[idx][label_tdiff_file] - LEN_REC
                if np.abs(tdiffs_gaps[i]) > gap_fill_threshold:
                    raise ValueError(f"! ABORT: Gap size exceeds threshold of {gap_fill_threshold}, \
                                     increase threshold or split the loading of these files")
                ns_gaps[i] = round(tdiffs_gaps[i]/dt)
                print(f"i={i}, tfdiff={tdiffs_gaps[i]}, ns_gap={ns_gaps[i]}")
                # concat zeros for gaps
                list_data[idx] = np.concatenate([np.zeros((ns_gaps[i],list_data[idx].shape[1])),\
                                                 list_data[idx]],axis=0
                                                )
            dict_gaps={"flag_gaps":1,"fids":df_head_cut.iloc[fidxes_gaps][label_fids_int].values, \
                       "tdiff":tdiffs_gaps, "ns_gaps":ns_gaps}
        else:
            dict_gaps={"flag_gaps":0}
    else:
        dict_gaps={"flag_gaps":0}
        
    data = np.array( np.concatenate(list_data, axis=0) ,  order=mem_order)
    
    # pre-process raw data
    if preprocess_data:
        data, list_meta = preprocess_DAS(data, list_meta, unwr=unwr, integrate=integrate, 
                                    useSensitivity=useSensitivity, spikeThr=spikeThr)
    if record_time:
         print('data loading time: {:.1f}s'.format(time.perf_counter()-tex_start) )

    if full_output:         
        return (data, list_meta, list_ids, dict_gaps)
    else: 
        return (data, list_meta[0] )
        
    

if __name__ == '__main__':
    
    import sys, glob #, time
    import matplotlib.pyplot as plt
    from scipy.signal import resample_poly, butter, filtfilt
    
    date=20220705
    path_data = '/media/keving/Ula/fsi/exps/CGF_Ula_Tampnet/{}/dphi/'.format(date)
    #path_data = '/media/keving/Green/Rissa-DAS/{}/dphi/'.format(date)
    #path_data = './Data/Rissa/'
    
    headerfile = None #path_info + f'Headerinfo/header_info_pro_{date}.csv'
    
    use_multipro=0;                     # set to zero to switch off multiprocessing/threading
    n_workers=4;                        # number of processors used for concurrent loading
    idx_file=100;                       # target file index to load
    nfiles_before, nfiles_after = 8, 8  # number of files to load before and after target file
    chmin, chmax = 10, 800;             # min and max channels to load
    gap_fill_threshold=60               # max gap length to fill with zeros, in seconds
    mem_order="F"                       # columnwise memory order (fortran style), allows faster columnwise processing

    # flags 
    unwr=1;             # unwrap DAS signal
    integrate=1;        # integrate from strain rate to strain
    useSensitivity=1;   # convert from differential phase to strain rate
    spikeThr=None;      # replace spikes
    fill_gaps=0;        # fill gaps between files with zeros
    plot_data=0;
    
    
    # list files in dir
    files = sorted([f.split('/')[-1] for f in glob.glob(path_data +'*.hdf5')]); 
    
    #print("done"); sys.exit()
    
    # select files from list
    idxes_files = np.arange(idx_file-nfiles_before,idx_file+nfiles_after,1)
    files_used = [files[fidx] for fidx in idxes_files]
    fileIDs = [f.split('.')[0] for f in files_used]
    fileIDs_int = [int(fid) for fid in fileIDs]
    
    if headerfile is not None:
        df_head = pd.read_csv(headerfile, header=0, sep='\t', dtype={'file_id': object})
        df_head["fid_int"]=df_head["file_id"].values.astype(int)
        df_head_cut = df_head[df_head["fid_int"].isin(fileIDs_int)]
    else:
        df_head_cut=None
        
    channel_idxes=np.arange(chmin,chmax,1) 

    #print("done"); sys.exit()
    
    tex_start=time.perf_counter()
    if use_multipro: 
        data, list_meta, fids_completed, dict_gaps = load_data_conc(path_data, fileIDs, channel_idxes, n_workers=n_workers, 
                unwr=unwr, integrate=integrate, useSensitivity=useSensitivity, spikeThr=spikeThr, 
                fill_gaps=fill_gaps, df_head=df_head_cut, full_output=True, record_time=True, colnames_gap_rem=None, 
                gap_fill_threshold=gap_fill_threshold, mem_order=mem_order
                ) #data, list_meta, fileIDs_completed
        
        meta=list_meta[0]
    else: 
        fileIDs_int = [ int(fid) for fid in fileIDs ]
        data, meta = load_multiple_DAS_files(path_data, fileIDs_int, chIndex=channel_idxes, roiIndex=None,
                                  integrate=integrate, unwr=unwr, metaDetail=1,
                                 useSensitivity=useSensitivity, spikeThr=spikeThr)
    print(f'mp={use_multipro} execution time: {time.perf_counter()-tex_start:.1f}s'); 
    
    channels_abs = meta["appended"]["channels"]
    dt_orig = meta["header"]["dt"] #.item()
    
    
    
    if plot_data:
        dec_time=4
        f_cutoff=5
        
        data_plot = resample_poly(data, 1, dec_time)
        dt = dt_orig*dec_time
        
        b, a = butter(4, f_cutoff, btype="high", fs=1/dt)
        data_plot = filtfilt(b, a, data_plot, axis=0)
        times=np.arange(0,data_plot.shape[0]*dt,dt)
        
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        mesh = ax.pcolormesh(channel_idxes, times, np.abs(data_plot), vmax=1.0e-8 )
        plt.colorbar(mesh, ax=ax, label="Strain")
        ax.set(xlabel="channel index", ylabel="Time [s]")
        ax.invert_yaxis()
        plt.show()
    
    
    print("done")
    