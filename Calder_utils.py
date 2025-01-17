#this script contains the helper functions developed by calder robinson

import numpy as np
import configparser


def faststack(X,n):
     """ fast adjacent channel stack through time
     Inputs:
     Parameters
    ----------
    X : 2d np.array 
        2d data array with dimension (T,X).
    n : int
        number of channels to stack together

    Returns
    -------
    stacked : 2d np.array
        data array containing the mean of n adjacent channels 
     """
     rows,cols = X.shape
     trimval = np.mod(cols,n)

     if trimval!=0:
          trimmedmean = X[:,-trimval:].mean(axis = 1)
          X = X[:,:-trimval]
          stacked = X.reshape(-1,n).mean(axis = 1).reshape(rows,-1)
          stacked = np.c_[stacked,trimmedmean]
     else:
          stacked = X.reshape(-1,n).mean(axis = 1).reshape(rows,-1)

     return stacked

def loadparams(X):
    config = configparser.ConfigParser()
    config.read(filenames=X)
    #perform some io logic to get stuff set up for the run then return the io logic

    
    return config
