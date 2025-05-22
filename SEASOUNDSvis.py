#this script is used to make figures for the seasounds poster and presentation

#%%
import numpy as np
import matplotlib.pyplot as plt
from Calder_utils import loadFTX

directory = 'C:/Users/Calder/Outputs/DASdata4/'
data,freqs, times, channels = loadFTX(directory=directory, 
                                      nworkers = 10,
                                      start_f= 10,
                                      stop_f=61,
                                      start_cidx=0,
                                      stop_cidx=120)

#data = data[10:61,-7200:,0:120]
#data = data[10:61,:,0:120]

data = 10**(data/10)
TX = np.mean(data, axis = 0)
TX = 10*(np.log10(TX))
times = times/60
channels = channels[0:120]/1000
del(data)



#%%
low = np.percentile(TX,25)
high = np.percentile(TX,99)
plt.figure()
plt.imshow(np.flipud(TX.T) ,origin = 'upper', extent = (np.min(times),np.max(times),np.min(channels),np.max(channels)), aspect = 'auto', cmap = 'turbo')
plt.colorbar(label = 'strain: dB re 1 pE')
plt.clim(low,high)
plt.xlabel('Relative time (min)')
plt.ylabel('Fiber distance (km)')
plt.title('2022-08-21 16:00 UTC, 10-30 Hz mean strain')
plt.savefig('C:\\Users\\Calder\\OneDrive - NTNU\\Documents\\Presentations\\ASA May 2025\\Figures\\TX.png', dpi = 500)
plt.close

TX_small = TX[-14400:,0:45]
Tsmall = times[-14400:,]
chansmall = channels[0:45]
# low = np.percentile(TX_small,25)
# high = np.percentile(TX_small,99)

plt.figure()
plt.imshow(np.flipud(TX_small.T) ,origin = 'upper', extent = (np.min(Tsmall),np.max(Tsmall),np.min(chansmall),np.max(chansmall)), aspect = 'auto', cmap = 'turbo')
#plt.colorbar(label = 'strain: dB re 1 pE')
plt.clim(low,high)
plt.xlabel('Relative time (min)')
plt.ylabel('Fiber distance (km)')
plt.tight_layout()
#plt.title('2022-08-21 18:00 UTC, 10-30 Hz mean strain')
plt.savefig('C:\\Users\\Calder\\OneDrive - NTNU\\Documents\\Presentations\\ASA May 2025\\Figures\\TX_zoom.png', dpi = 500)
plt.close
 # %%
