# this script is data visualization utilities
#%%
import numpy as np
import matplotlib.pyplot as plt
import os
from Calder_utils import loadFTX

def PostProcessVis(src, dst, norm = True):
    """
    This function loads a direcotry of processed das data and uses them to make basic spectrograms

    inputs: src, string, file path
    """
    #directory = 'C:/Users/Calder/Outputs/DASdata2/'
    data, freqs,times,channels = loadFTX(src)

    if norm:
        #data -= np.mean(data, axis = 1)[:,None,:]
        data /= np.std(data,axis = 1)[:,None,:]
        data -= np.min(data, axis = 1)[:,None,:]
        data /= np.max(data,axis = 1)[:,None,:]

    data = data[10:100,:,120:180]

    #channels = channels[::4]
    for chan in range(data.shape[2]):
        fname = os.path.join(dst, 'fig_'+str(channels[chan])+'.png')
        plt.figure(figsize=(15,10))
        plt.imshow(np.fliplr(data[:,:,chan]),cmap = 'turbo', origin = 'lower', aspect = 'auto', extent = (np.min(times),np.max(times),np.min(freqs[10:100]),np.max(freqs[10:100])))
        plt.colorbar(label = 'strain: dB re 1 pE')
        plt.title(str(channels[chan])+' (m)')
        plt.clim(15,35)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.savefig(fname, dpi=500, bbox_inches='tight')
        plt.clf()
        plt.close()
    del(data)




# src = 'C:\\Users\\Calder\\Outputs\\3D_cornellposter20250428T155428\\Magnitude'
# dst = 'C:\\Users\\Calder\\Outputs\\CornellPosterplots'
# tmp = PostProcessVis(src,dst, norm = False)

def joyplot(data, axis = 0, lcolor = 'white',offset=2):
    '''
    This plot make a joy division plot

    inputs:
    a, 2d np array of data to visualize
    axis, int, along which axis ar ethe data to be split along
    lcolor: str, line color, black on white, or white on black?
    '''

    fig, ax = plt.subplots(figsize=(10, 6))
    offset = 2  # Vertical spacing between lines

    for i, row in enumerate(data):
        y = row + i * offset
        x = np.arange(len(row))

        # Fill white to hide lower lines
        ax.fill_between(x, y - offset, y + offset, color='white', zorder=i)

        # Plot the line in black
        ax.plot(x, y, color='black', linewidth=1, zorder=i+1)

    # Set white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, data.shape[1] - 1)
    ax.set_ylim(-offset, data.shape[0] * offset)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.show()
  
# %%
