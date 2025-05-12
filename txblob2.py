# %% detection?
from scipy.ndimage import convolve1d, binary_closing, generate_binary_structure, binary_dilation
import numpy as np
import matplotlib.pyplot as plt
from simpleDASreader4 import load_DAS_file
from Calder_utils import faststack
from scipy.signal import detrend, resample, butter, sosfiltfilt
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing
import matplotlib.patches as mpatches
import scipy.optimize as optimize
import glob, os
import pickle
from collections import deque
from scipy import ndimage as ndi


def cfar(array, guardsize,sample_size):
    kernalsize = 2*(sample_size + guardsize)+1
    kernal = np.zeros(kernalsize)
    kernal[:sample_size] = 1
    kernal[-sample_size:]=1
    kernal = kernal/np.sum(kernal)
    thresh = convolve1d(array,kernal,axis = 0,mode = 'reflect')
    return thresh

def hysterysis(mask_low,mask_high):
    labels_low, num_labels = ndi.label(mask_low)
    # Check which connected components contain pixels from mask_high
    sums = ndi.sum(mask_high, labels_low, np.arange(num_labels + 1))
    connected_to_high = sums > 0
    thresholded = connected_to_high[labels_low]
    return thresholded

def hyperbola_equation(x, a, b,c):
  return np.sqrt(b**2 *(((x-c)**2 / a**2) + 1))

def hyperbola_2(x,b,c,d,v):
    return(np.sqrt(b**2 *(((x-c)**2 / (v*b)**2) + 1))-d)


def error_function(params, x, y):
  a, b, c = params
  predicted_y = hyperbola_equation(x, a, b, c)
  return np.sum((y - predicted_y)**2)

def error_function2(params, x, y):
  b,c,d,v = params
  predicted_y = hyperbola_2(x, b, c, d,v)
  return np.sum((y - predicted_y)**2)

directory = 'E:\\NORSAR01v2\\20220821\\dphi'
files=  glob.glob(os.path.join(directory, '*.hdf5'))

start = '183007'
stop = '183037'
nchans = 15000
dst = 'C:\\Users\\Calder\\Outputs\\annotatedtx2'
nstack = 5
low = 25, 
high = 99

#rms windowsize
windowsize = 125

#cfar params
nguard = 625
nsample = 300
maskthresh = 30 #in percent false alarm rate
truethresh = 1 #in percent false alarm rate

#blobparam
minsize = 10000
maxsize = 1000000
aspect_thresh = 8
buffer = 0

#parabola fitting params 
v = 1500 #vwater
b = 0.5 #curve param
a = v/b
c = 2000 #apex location in box lateral
d = 0.5 #apex location vertical 
#accumulators
a_out= []
b_out = []
c_out = []
d_out = []
v_out = []
scatters = []
channels = []
fnames = []
minchan = []
maxchan = []
mintime= []
maxtime = []
aspect = []


#####start the routine####

start_idx = [i for i, s in enumerate(files) if start in s][0]
stop_idx = [i for i, s in enumerate(files) if stop in s][0]

files = files[int(start_idx):int(stop_idx)]

ns = 2*nsample
lowthresh = ns*((maskthresh/100)**(-1/ns)-1)
highthresh = ns*((truethresh/100)**(-1/ns)-1)
print(lowthresh)
print(highthresh)

data = deque(maxlen=2)
name = deque(maxlen=2)
for f in files:
    name.append(os.path.basename(f).split('.')[0])
    print(name[0])
    chans = range(nchans)
    sig,meta = load_DAS_file(f, chIndex=chans, unwr = True,integrate=False)
    data.append(sig)
    dt = meta['header']['dt']
    signal= np.vstack(data)
    signal=np.cumsum(signal,axis=0)*dt/1e-9
    signal = faststack(signal,nstack)
    
    chIDX = np.arange(start = 0,stop = len(chans), step = nstack)
    true_chans = meta['header']['channels'][chIDX]
    dx = true_chans[1]-true_chans[0]

    sost = butter(N = 6,
            Wn = (5,30),
            btype = 'bandpass',
            fs = 1/dt,
            output = 'sos')

    signal = sosfiltfilt(sos = sost,
                        x = signal,
                        axis = 0)


    #sliding RMS
    kernal = np.ones(windowsize)/windowsize
    sig2 = np.square(signal)
    vis = np.sqrt(convolve1d(sig2,kernal,axis = 0,mode = 'reflect'))

    #getting cfar thresholds and hysteresis filtering
    #struc = generate_binary_structure(2,2)
    
    tmp = cfar(vis, nguard,nsample)
    mask = np.greater(vis,tmp*lowthresh)
    #mask = binary_dilation(input = mask,structure=struc, iterations=1)
    output = np.greater(vis,tmp*highthresh)

    #mask_low = binary_closing(input = mask, iterations=1)
    #mask_high = binary_closing(input = output,iterations=1)

    map = hysterysis(mask,output)

    output2 = binary_closing(input = map, iterations=5)#
    # output2 = binary_closing(input = output2,structure=struc, iterations=1)#smoothing?
    time = np.arange(output2.shape[0])*dt
    space = np.arange(output2.shape[1])*dx/1000
    plt.imshow(output2, aspect = 'auto', cmap = 'grey',extent=(np.min(space),np.max(space),np.max(time),np.min(time)))
    plt.xlabel('Fiber Distance (km)')
    plt.ylabel('Relative Time (s)')
    plotname = os.path.join(dst,name[0] + 'mask.png')
    #print(plotname)
    plt.savefig(plotname)
    plt.close()

#extent=(np.min(space),np.max(space),np.max(time),np.min(time))

    # #labelling blobs
    cb = clear_border(output2)
    label_image = label(output2)

    min = np.percentile(vis,low)#for good color scale
    max = np.percentile(vis,high)

    fig, ax = plt.subplots()
    img = ax.imshow(vis, cmap= 'turbo', aspect = 'auto',extent=(np.min(space),np.max(space),np.max(time),np.min(time)))
    fig.colorbar(img, ax=ax,label = 'RMS (nε)')
    img.set_clim(min,max)
    plt.xlabel('Fiber Distance (km)')
    plt.ylabel('Relative Time (s)')
    mag = vis*output2
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= minsize and region.area <= maxsize:
            minr, minc, maxr, maxc = region.bbox
            rrange = maxr-minr
            crange = maxc-minc
            ap = np.max([rrange,crange])/np.min([rrange,crange])
            
            rect = mpatches.Rectangle(
                (minc*dx/1000, minr*dt),
                crange*dx/1000,
                rrange*dt,
                fill=False,
                edgecolor='white',
                linewidth=2,
            )
            ax.add_patch(rect)


            #fitting hyperbolas
            c = crange/2 * dx
            initial_guess = [b,c,d,v]
            wind  = np.ones(7)
            #get the scatter values for fitting
            scat = np.argmax(mag[minr-buffer:maxr+buffer,minc-buffer:maxc+buffer], axis = 0)
            y = scat*dt
            y = np.convolve(y, wind/np.sum(wind), mode = 'valid')
            x = np.arange(start=0,stop = len(y), step = 1)*dx
            
            results= optimize.minimize(error_function2,x0=initial_guess,args = (x,y))

            #Prepping data for export
            scatters.append(scat*dt)
            b_out.append(results.x[0])
            c_out.append(results.x[1])
            d_out.append(results.x[2])
            v_out.append(results.x[3])
            fnames.append(name[0])
            minchan.append(true_chans[minc])
            maxchan.append(true_chans[maxc])
            mintime.append(minr*dt)
            maxtime.append(maxr*dt)
            aspect.append(ap)
            

    #plt.tight_layout()
    plotname = os.path.join(dst,name[0] + 'Bbox.png')
    plt.savefig(plotname)
    plt.close()
    del fig,ax

tmp_struct = list(zip(b_out,c_out,d_out,v_out,fnames,minchan,maxchan,mintime,maxtime,aspect,scatters))

data_name = os.path.join(dst,'data_out.pkl')
with open(data_name, "wb") as file:
    pickle.dump(tmp_struct, file)


for i,s in enumerate(scatters):
    #ename = os.path.basename(fnames[i]).split('.')[0]
    y = s
    #y = np.convolve(y, wind/np.sum(wind), mode = 'valid')
    x = np.arange(len(y))*dx
    plt.scatter(x,y)
    ypred = hyperbola_2(x,b_out[i],c_out[i],d_out[i],v_out[i])
    plt.plot(x,ypred, 'red')
    pname = os.path.join(dst,fnames[i] + '_'+ str(i)+'.png')
    plt.savefig(pname)
    plt.close()


# %% playing around with CFAR
import numpy as np
n = 1250
pfa = 0.018

scale = n*(pfa**(-1/n)-1)
print(scale)
# %%
