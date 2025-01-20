#this script contains the helper functions developed by calder robinson

import numpy as np
import configparser


def faststack(X,n, wind = 1):
     """ fast adjacent channel stack through time
     Inputs:
     Parameters
    ----------
    X : 2d np.array 
        2d data array with dimension (T,X).
    n : int
        number of channels to stack together
    wind : 1d np.array
        optional cross channel windowshape, default is square

    Returns
    -------
    stacked : 2d np.array
        data array containing the mean of n adjacent channels 
    
    Note: when using a window other than 1, if n does not divide into the number of X, the final group  average will be returend without the window applied, 
    and it contains fewerthan requested channels.
     """
     rows,cols = X.shape
     trimval = np.mod(cols,n)

     if wind == 1:
        wind = np.ones(n)
     elif len(wind) != n:
         raise ValueError("window not same size as stack number.")
         

     if trimval!=0:
          trimmedmean = X[:,-trimval:].average(axis = 1)
          X = X[:,:-trimval]
          stacked = X.reshape(-1,n).average(axis = 1, weights = wind).reshape(rows,-1)
          stacked = np.c_[stacked,trimmedmean]
     else:
          stacked = X.reshape(-1,n).mean(axis = 1).reshape(rows,-1)

     return stacked

def loadparams(X):
    config = configparser.ConfigParser()
    config.read(filenames=X)
    #perform some io logic to get stuff set up for the run then return the io logic

    
    return config
