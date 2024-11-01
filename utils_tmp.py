#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:48:59 2023

@author: keving
"""
import numpy as np
from scipy.signal import  butter, lfilter #sosfilt, freqz
from scipy.signal.windows import tukey #sosfilt, freqz

def get_idx_closest_smaller_value(arr, val, full_output=False):
    """ get idx for closest value in list 
    """
    diffs = (arr - val)*(-1)
    val_min =min([i for i in diffs if i >= 0])  # find the smallest positive value
    idx =diffs.tolist().index(val_min)    # get index of determined value -> index of file for the shot
    
    if full_output:     
        return idx, val_min, arr[idx]
    else: 
        return idx, val_min
    

def stack_channels(data, stack_fac, axis_stack=1, stack_leftovers=True):
    """
    stack adjacent channels of a 2d np.array
    ! if number of channels does perfectly fit the stack window length channels at the end of the array will be discarded, may be improved later 

    Parameters
    ----------
    data : 2d np.array
        2d data array.
    stack_fac : int, optional
        number of traces to stack together
    axis_stack : 0 or 1, optional
        axis to stack along (will be reduced). The default is 1.
    stack_leftovers : bool, optional
        leftover channels from the division will be stacked together and not discarded. Default is True.

    Returns
    -------
    data_stacked : 2d np.array
        data array of stacked channels.

    """
    if stack_fac <= 1: 
        return data
    if axis_stack == 0: 
        data = data.T
        
    ns, nx = data.shape
    nstacks = int(nx/stack_fac)
    nleft = nx % stack_fac
    
    if nleft > 0 and stack_leftovers==True:
        nstacks+=1; 
        
    data_stacked = np.zeros((ns,nstacks))
    for s in range(nstacks): 
        ch_start = s*stack_fac
        #print("DEBUG: run {},start: {}, end: {}".format(s,ch_start,ch_start+stack_fac))
        data_stacked[:,s] = np.mean(data[:,ch_start:ch_start+stack_fac], axis=1)
        
    if axis_stack == 0: 
        data_stacked = data_stacked.T
    
    return data_stacked



def tukey_taper(ns, alpha, ns_shift=0, sym=True, onesided=False):
    """
    design tukey taper window

    Parameters
    ----------
    ns : int
        number of samples.
    alpha : scalar from 0-1
        tuning parameter for steepness of the taper.
    ns_shift : int, optional
        shift the taper onset to later samples, samples before the shift are zeroed. The default is 0.
    sym : bool, optional
        apply symmetric taper. The default is True.
    onesided : bool, optional
        apply taper only to the beginning of the trace. The default is False.

    Returns
    -------
    taper : 1d np.array
        taper function.

    """
    taper = np.zeros(ns)
    ns_taper = ns-ns_shift
    taper[ns_shift:ns] = tukey(ns_taper,alpha, sym=sym);
    if onesided==True: 
        taper[int(ns/2):ns]=1
    
    return taper


def taper_2darray(data, alpha, axis=0, ns_shift=0, sym=True, onesided=False, save_taper=False):
    """
    apply taper to 2d array 
    """
    if axis==1: 
        data = data.T
        
    ns, nx = data.shape
    taper = tukey_taper(ns, alpha, ns_shift=ns_shift, sym=sym, onesided=True)
    Taper = np.tile(taper,(nx,1)).T
    
    if onesided==False: 
        taper_flip = np.flip(Taper,axis=0)
        Taper = Taper * taper_flip
        
    data_tape = data * Taper; 
    if axis==1: 
        data_tape = data_tape.T; Taper=Taper.T
    
    if save_taper: 
        return data_tape, Taper
    else: 
        return data_tape
    
    
def butter_bandpass(lowcut, highcut, fs, order):
    """
    design butterworth bandpass filter

    Parameters
    ----------
    lowcut : float
        lowcut frequency.
    highcut : float
        highcut frequency.
    fs : float
        sampling frequency.
    order : int
        order of filter.

    Returns
    -------
    b : floats
        numerators polynomials of filter.
    a : floats
        denominators polynomials of filter.

    """
    f_nyq = fs/2
    f_low_norm = lowcut/ f_nyq
    f_high_norm = highcut/ f_nyq
    b, a = butter(order, [f_low_norm,f_high_norm], btype='band')
    return b, a 
    
    
    
def bandpass_2darray(data, fs, lowcut, highcut, axis_filt=0, order=4, zerophase=False, taper_data=True,
                     alpha_tape=0.1, ns_shift_tape=0, sym_tape=True, onesided_tape=False):
    """
    Parameters
    ----------
    data : 2d np.array
        2d dataset, time in vertical axis.
    fs : scalar
        sampling frequency.
    lowcut : scalar
        low corner frequency for bandpass.
    highcut : scalar
        high corner frequency for bandpass.
    order : int, optional
        number of poles for bandpass design. The default is 4.
    zerophase : bool, optional
        apply zerophase filter. The default is False.
    taper_data : bool, optional
        apply taper before and after filtering. The default is True.
    alpha_tape : scalar, optional
        tuning parameter for steepness of the taper. The default is 0.1.
    ns_shift_tape : int, optional
        shift the taper onset to later samples, samples before the shift are zeroed. The default is 0.
    sym_tape : bool, optional
        apply symmetric taper. The default is True.
    onesided_tape : bool, optional
        apply taper only to the beginning of the trace. The default is False.

    Returns
    -------
    data_filt : 2d np.array
        filtered data.

    """
    
    b, a =  butter_bandpass(lowcut,highcut,fs, order)
    ns, nx = data.shape
    
    if taper_data==True: 
        data, Taper = taper_2darray(data, alpha_tape, ns_shift=ns_shift_tape, sym=sym_tape, onesided=onesided_tape, save_taper=True)
        
    data_filt = lfilter(b, a, data, axis=axis_filt)  
    
    if taper_data==True:      
        data_filt = data_filt * Taper
         
    return data_filt