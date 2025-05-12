#this script loads and plots the tx representations from cleaning data
#%%
import numpy as np
import matplotlib.pyplot as plt
import glob,os

#%%
directory = 'C:\\Users\\Calder\\Outputs\\E7_02_cleaning\\NORSAR01v2_cleaning_outer\\20220821CRfull_20250414T173836\\5Hz_30Hz'

start = '180007'
stop = '190007'
files = glob.glob(os.path.join(directory, '*.npy'))

start_idx = [i for i, s in enumerate(files) if start in s][0]
stop_idx = [i for i, s in enumerate(files) if stop in s][0]

files = files[int(start_idx):int(stop_idx)]



# %%
data = []
for f in files:
    data.append(np.load(f))

stacked = np.concatenate( data, axis=0 )
#%%

cmean = np.mean(stacked, axis = 0)
adj = stacked - cmean[None,:]
cstd = np.std(stacked, axis = 0)
adj /=cstd[None,:]
adj -=np.min(adj,axis = 0)[None,:]

plt.figure(figsize=(15,10))
plt.imshow(adj, origin = 'lower', cmap = 'magma',aspect = 'auto')
plt.colorbar()
plt.clim(0,20)
plt.xlim(0,150)
#%%
