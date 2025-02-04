#this script is used to make figures for the seasounds poster and presentation

#%%
import numpy as np
import matplotlib.pyplot as plt
from Calder_utils import loadFTX

directory = 'C:/Users/Calder/Outputs/DASdata4/'
data,freqs, times, channels = loadFTX(directory=directory, nworkers = 10)

data = data[10:61,-7200:,0:120]
data = 10**(data/10)
TX = np.mean(data, axis = 0)
TX = 10*(np.log10(TX))
times = times/60
channels = channels[0:120]/1000
del(data)



#%%
# mtx = np.mean(TX, axis=0)
# TX = TX - mtx[None,:]
# stx = np.std(TX, axis=0)
# TX =TX/stx[None,:]
plt.figure(figsize=(15,5))
plt.imshow(np.fliplr(TX) ,origin = 'lower', extent = (np.max(channels),np.min(channels),np.max(times),np.min(times)), aspect = 'auto', cmap = 'turbo')
plt.colorbar(label = 'strain: dB re 1 pE')
plt.clim(10,50)
plt.xlabel('Fiber distance (km)')
plt.ylabel('Relative time (min)')
plt.title('2022-08-21 16:00 UTC, 10-30 Hz mean strain')
#plt.savefig('C:/Users/Calder/Outputs/SEASOUNDS plts1/tx1.png')



 # %%
