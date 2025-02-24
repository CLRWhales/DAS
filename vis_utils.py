# this script is data visualization utilities
#%%
import numpy as np
import matplotlib.pyplot as plt
import glob

def PostProcessVis(directory):
    """
    This function loads a direcotry of processed das data and uses them to make basic spectrograms
    """
    #directory = 'C:/Users/Calder/Outputs/DASdata2/'


    clist = directory + '/Dim_Channel.txt'
    flist = directory + '/Dim_Frequency.txt'
    tlist = directory + '/Dim_Time.txt'

    filepaths = sorted( glob.glob(directory + '*.npy') )[-10:-3]
    
    channels = np.loadtxt(clist)
    freqs = np.loadtxt(flist)
    times = np.loadtxt(tlist)

    data = np.empty((len(freqs),len(times)*len(filepaths)+1,len(channels)))

    i = 0

    for ll in filepaths:
        print(ll)
        t = i*60 + times
        tidx = i*len(times) + np.arange(len(times),dtype= 'int')
        #print(tidx)
        data[:,tidx,:] = np.load(ll)
        i = i+1

    data = data[:101,:,0:40]
    times = np.arange(start = 0, stop = data.shape[1])*0.25
    fullmins = times/60
    freqs = freqs[:101]

    #channels = channels[::4]
    for chan in range(data.shape[2]):
        fname = 'C:/Users/Calder/Outputs/DASplots4/' + 'fig_'+str(channels[chan])+'.png'
        plt.figure(figsize=(10,15))
        plt.imshow(np.fliplr(data[:,:,chan]),cmap = 'turbo', origin = 'lower', aspect = 'auto', extent = (np.min(fullmins),np.max(fullmins),np.min(freqs),np.max(freqs)))
        plt.colorbar(label = 'strain: dB re 1 pE')
        plt.title(str(channels[chan])+' (m)')
        plt.clim(10,40)
        plt.xlabel('Time (min)')
        plt.ylabel('Frequency (Hz)')
        plt.savefig(fname, dpi=500, bbox_inches='tight')
        plt.clf()
        plt.close()
    del(data)




directory = 'C:/Users/Calder/Outputs/DASdata4/'
tmp = PostProcessVis(directory)


  
# %%
