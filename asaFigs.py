#%% this is plots for the ASA presnetation
import numpy as np
import matplotlib.pyplot as plt
from simpleDASreader4 import load_DAS_file
from Calder_utils import faststack
from scipy.signal import detrend, resample, butter, sosfiltfilt
from scipy.ndimage import convolve1d, binary_closing, generate_binary_structure, binary_dilation
from matplotlib import cm

file = "E:\\NORSAR01v2\\20220821\\dphi\\183137.hdf5"
#file = "E:\\NORSAR01v2\\20220821\\dphi\\182847.hdf5"
nchans = 15000
nstack = 5

chans = range(nchans)
signal,meta = load_DAS_file(file, chIndex=chans)
signal = signal/1e-9 #moving to microstrain
signal = faststack(signal,nstack)
dt = meta['header']['dt']
waveform = signal[:,760]
x = np.arange(signal.shape[0])*dt
plt.figure(figsize=(10,5))
plt.plot(x,waveform)
plt.ylabel('Strain (nε)')
plt.xlabel('Time (s)')
fname = 'C:\\Users\\Calder\\OneDrive - NTNU\\Documents\\Presentations\\ASA May 2025\\Figures\\raw_waveform.png'
plt.tight_layout()
plt.savefig(fname)
plt.close()

colors = cm.Blues(np.linspace(0.3, 1, signal.shape[1]))
# %%
sost = butter(N = 6,
        Wn = (5,30),
        btype = 'bandpass',
        fs = 1/dt,
        output = 'sos')

waveform_filt = sosfiltfilt(sos = sost,
                    x = signal,
                    axis = 0)

plt.figure(figsize=(10,5))
plt.plot(x,waveform_filt[:,760])
plt.xlabel('Time (s)')
plt.ylabel('Strain (nε)')
fname = 'C:\\Users\\Calder\\OneDrive - NTNU\\Documents\\Presentations\\ASA May 2025\\Figures\\filt_waveform.png'
plt.tight_layout()
plt.savefig(fname)
plt.close()
# %%
windowsize=125
kernal = np.ones(windowsize)/windowsize
sig2 = np.square(waveform_filt)
vis = np.sqrt(convolve1d(sig2,kernal,axis = 0,mode = 'reflect'))
plt.figure(figsize=(6,4))
plt.plot(x,vis[:,760])
plt.xlabel('Time (s)')
plt.ylabel('RMS strain (nε)')
fname = 'C:\\Users\\Calder\\OneDrive - NTNU\\Documents\\Presentations\\ASA May 2025\\Figures\\RMS_waveform.png'
plt.ylim(0,2)
plt.savefig(fname, dpi = 500)
plt.close()

#%% three by one plotting of prepro
slice = 760
fig,ax = plt.subplots(3,1, figsize = (5,5), tight_layout = True)
ax[0].plot(x,signal[:,slice])
ax[0].set_ylabel('Strain (nε)')
ax[0].set_xticks([])
ax[1].plot(x,waveform_filt[:,slice])
ax[1].set_ylabel('Strain (nε)')
ax[1].set_xticks([])
ax[2].plot(x,vis[:,slice])
ax[2].set_ylabel('RMS Strain (nε)')
ax[2].set_xlabel('Time (s)')
fname = 'C:\\Users\\Calder\\OneDrive - NTNU\\Documents\\Presentations\\ASA May 2025\\Figures\\prepro.png'
plt.savefig(fname, dpi = 500)
plt.close()
# %%
plt.figure(figsize = (6,4))
plt.plot(x,vis)
plt.ylim(0,2)
plt.xlabel('Time (s)')
plt.ylabel('RMS strain (nε)')
fname = 'C:\\Users\\Calder\\OneDrive - NTNU\\Documents\\Presentations\\ASA May 2025\\Figures\\alltrace.png'
plt.savefig(fname, dpi = 500)
plt.close()
 # %%

def cfar(array, guardsize,sample_size):
    kernalsize = 2*(sample_size + guardsize)+1
    kernal = np.zeros(kernalsize)
    kernal[:sample_size] = 1
    kernal[-sample_size:]=1
    kernal = kernal/np.sum(kernal)
    thresh = convolve1d(array,kernal,axis = 0,mode = 'reflect')
    return thresh

nguard = 625
nsample = 300
maskthresh = 1.2
truethresh = 5

struc = generate_binary_structure(2,2)

tmp = cfar(vis, nguard,nsample)
mask = np.greater(vis,tmp*maskthresh)
#mask = binary_dilation(input = mask,structure=struc, iterations=1)
output = np.greater(vis,tmp*truethresh)



# %%
tmp2 = np.where(output,vis,np.nan)
plt.figure(figsize=(6,4))
plt.plot(x,tmp2)
plt.ylim(0,2)
plt.xlabel('Time (s)')
plt.ylabel('RMS strain (nε)')
fname = 'C:\\Users\\Calder\\OneDrive - NTNU\\Documents\\Presentations\\ASA May 2025\\Figures\\alltracethresh.png'
plt.savefig(fname, dpi = 500)
plt.close()
# %%

# colors = cm.viridis(np.linspace(0, 1, signal.shape[1]))

# fig, ax = plt.subplots(figsize=(10, 6))
# lines = ax.plot(x,vis)

# # Apply the colors to the lines
# for line, color in zip(lines, colors):
#     line.set_color(color)

# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Sensor Reading')
# ax.set_title('Sensor Data from Multiple Positions')
# plt.ylim(0,4)
# plt.tight_layout()
# plt.show()
 # %% 

ns = 2*nsample
mt = 30
tt=1
lowthresh = ns*((mt/100)**(-1/ns)-1)
highthresh = ns*((tt/100)**(-1/ns)-1)

plt.figure(figsize = (6,4))
plt.plot(x,vis[:,760])
plt.plot(x,tmp[:,760]*highthresh,linestyle='dashed')
plt.plot(x,tmp[:,760],linestyle='dashed')
#plt.plot(x,tmp[:,760]*lowthresh)
plt.xlabel('Time (s)')
plt.ylabel('RMS strain (nε)')
plt.ylim(0,2 )
fname = 'C:\\Users\\Calder\\OneDrive - NTNU\\Documents\\Presentations\\ASA May 2025\\Figures\\trace thresh example.png'
plt.savefig(fname, dpi = 500)
plt.close()
# %%

n = 125
idx = np.arange(start = 0,stop = n)* 3 + 625
data = vis[:,idx]/2


fig, ax = plt.subplots(figsize=(10, 10))
offset = 0.05  # Vertical spacing between lines
spacing = np.arange(start = 0, stop = n) * offset
data = data + spacing[None:,]
data = np.fliplr(data)
x = np.arange(data.shape[0])
for i in range(data.shape[1]):
    ax.fill_between(x,y1 = data[:,i],y2 = 0, color='white',zorder=i)
    ax.plot(data[:,i], color = 'black',zorder=i+1)
    if i == 65:
        ax.fill_between(x,y1 = data[:,i],y2 = 0, color='firebrick',zorder=i)
        ax.plot(data[:,i], color = 'firebrick',zorder=i+1)

# Set white background
ax.set_facecolor('white')
fig.patch.set_facecolor('white')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
plt.tight_layout()
plt.close()
#plt.show()
#fname = 'C:\\Users\\Calder\\OneDrive - NTNU\\Documents\\Presentations\\ASA May 2025\\Figures\\JOYColor.png'
#plt.savefig(fname, dpi = 500)


# %%
from scipy import ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
#from simpleDASreader4 import load_DAS_file
#from Calder_utils import faststack
from scipy.signal import detrend, resample, butter, sosfiltfilt
from scipy.ndimage import convolve1d, binary_closing, generate_binary_structure, binary_dilation
from matplotlib import cm

def hysterysis(mask_low,mask_high):
    labels_low, num_labels = ndi.label(mask_low)
    # Check which connected components contain pixels from mask_high
    sums = ndi.sum(mask_high, labels_low, np.arange(num_labels + 1))
    connected_to_high = sums > 0
    thresholded = connected_to_high[labels_low]
    return thresholded

low = np.percentile(vis,25)
high = np.percentile(vis,99)
dist = np.arange(vis.shape[1])*20/1000
time = np.arange(vis.shape[0])*dt
plt.figure(figsize = (6,4))
plt.imshow(vis.T, cmap = 'turbo',origin = 'lower', aspect = 'auto', extent = (np.min(time),np.max(time),np.min(dist),np.max(dist)))
# plt.colorbar(label = 'RMS strain (nε)')
plt.clim(low,high)
plt.ylabel('Fiber distance (km)')
plt.xlabel('Time (s)')
fname = 'C:\\Users\\Calder\\OneDrive - NTNU\\Documents\\Presentations\\ASA May 2025\\Figures\\TX.png'
plt.savefig(fname, dpi = 500)
plt.close()

output = np.greater(vis,tmp*highthresh)

plt.figure(figsize = (6,4))
plt.imshow(output.T, cmap = 'gray',origin = 'lower', aspect = 'auto', extent = (np.min(time),np.max(time),np.min(dist),np.max(dist)))
# plt.colorbar(label = 'RMS strain (nε)')
plt.clim(low,high)
plt.ylabel('Fiber distance (km)')
plt.xlabel('Time (s)')
fname = 'C:\\Users\\Calder\\OneDrive - NTNU\\Documents\\Presentations\\ASA May 2025\\Figures\\TX_high.png'
plt.savefig(fname, dpi = 500)
plt.close()

mask = np.greater(vis,tmp*lowthresh)
plt.figure(figsize = (6,4))
plt.imshow(mask.T, cmap = 'gray',origin = 'lower', aspect = 'auto', extent = (np.min(time),np.max(time),np.min(dist),np.max(dist)))
# plt.colorbar(label = 'RMS strain (nε)')
# plt.clim(low,high)
plt.ylabel('Fiber distance (km)')
plt.xlabel('Time (s)')
fname = 'C:\\Users\\Calder\\OneDrive - NTNU\\Documents\\Presentations\\ASA May 2025\\Figures\\TX_low.png'
plt.savefig(fname, dpi = 500)
plt.close()

together = hysterysis(mask,output)
plt.figure(figsize = (6,4))
plt.imshow(together.T, cmap = 'gray',origin = 'lower', aspect = 'auto', extent = (np.min(time),np.max(time),np.min(dist),np.max(dist)))
# plt.colorbar(label = 'RMS strain (nε)')
# plt.clim(low,high)
plt.ylabel('Fiber distance (km)')
plt.xlabel('Time (s)')
fname = 'C:\\Users\\Calder\\OneDrive - NTNU\\Documents\\Presentations\\ASA May 2025\\Figures\\TX_combined.png'
plt.savefig(fname, dpi = 500)
plt.close()

# %% star plotting hyperbolas
import pickle
import datetime

def hyperbola_2(x,b,c,d,v):
    return(np.sqrt(b**2 *(((x-c)**2 / (v*b)**2) + 1))-d)

def error_function2(params, x, y,w):
  b,c,d,v = params
  predicted_y = hyperbola_2(x, b, c, d,v)
  return np.sum(w*(y - predicted_y)**2)

# Replace 'filename.pkl' with your actual file path
data = []
with open("C:\\Users\\Calder\\Outputs\\annotatedtx3\\data_out.pkl", 'rb') as file:
    data = pickle.load(file)

id = 401

fit_data = data[id]
y = fit_data[10]
x2 = np.arange(len(y))*20

plt.figure(figsize=(6,4))
plt.scatter(x2,y)
plt.xlabel('Relative distance (m)')
plt.ylabel('Relative time (s)')
plt.ylim(0,3)


wind  = np.hamming(7)
y = np.convolve(y, wind/np.sum(wind), mode = 'valid')
x3 = np.arange(start=0,stop = len(y), step = 1)*20
yfit = hyperbola_2(x2,fit_data[0],fit_data[1],fit_data[2],fit_data[3])
plt.plot(x2,yfit, color = 'red')
fname = 'C:\\Users\\Calder\\OneDrive - NTNU\\Documents\\Presentations\\ASA May 2025\\Figures\\hyperscatter.png'
plt.savefig(fname, dpi = 500)
plt.close()

apex = []
minchan = []
cwater = []
toff = []
aspect = []
name = []
mtime = []
error = []
n = []
rsq = []
for call in data:
    apex.append(call[1])
    minchan.append(call[5])
    toff.append(call[0])
    cwater.append(call[3])
    aspect.append(call[9])
    name.append(call[4])
    mtime.append(call[7])
    params = [call[0],call[1],call[2],call[3]]
    y = call[10]
    x = np.arange(start=0,stop = len(y), step = 1)*20
    tmp = error_function2(params,x,y,call[11])
    error.append(tmp)
    ypred= hyperbola_2(x,call[0],call[1],call[2],call[3])
    t2 = 1-(np.sum((y-ypred)**2)/np.sum((y-np.mean(y))**2))
    rsq.append(t2)

apex = np.stack(apex)
minchan = np.stack(minchan)
cwater = np.stack(cwater)
toff = np.stack(toff)
aspect = np.stack(aspect)
rsq = np.stack(rsq)
thresh = 10 
amask = np.where(aspect<thresh)[0]
whaleonfiber = minchan + apex
x3 = np.arange(len(whaleonfiber))

tsteps = [datetime.datetime.strptime('2022-08-21 ' + n, '%Y-%m-%d %H%M%S') for n in name]
tsteps_adj = [(t+datetime.timedelta(seconds =toff[i] + mtime[i])).timestamp() for i,t in enumerate(tsteps)]
tsa = np.vstack(tsteps_adj)

plt.figure()
plt.scatter(tsteps,rsq)
plt.ylim(0,1)

mask2 = np.where(rsq>0.91)
# plt.figure()
# plt.scatter(x3,aspect)
# plt.ylim(0,100)
# plt.hlines(11,0,1000, colors='red')

# plt.figure()
# plt.scatter(tsa[amask], whaleonfiber[amask])
# plt.ylim(15000,17000)

# plt.figure()
# plt.scatter(x3[amask],abs(toff[amask]*cwater[amask]))
# plt.ylim(-1000,1000)

range1 = abs(toff*cwater)
maxrange = 800

r = range1[mask2]#[range[mask2]<maxrange]
w = whaleonfiber[mask2]#[range[mask2]<maxrange]
t = tsa[mask2]#[range[mask2]<maxrange]
plt.figure(figsize = (6,4))
plt.scatter(t,w, c =r , cmap = 'viridis')
plt.ylim(15250,16750)
plt.ylabel('Along fiber distance (m)')
plt.xlabel('Time')
plt.colorbar(label = 'Crossline distance (m)')
plt.clim(0,maxrange)
fname = 'C:\\Users\\Calder\\OneDrive - NTNU\\Documents\\Presentations\\ASA May 2025\\Figures\\track.png'
plt.savefig(fname, dpi = 500)
#plt.close()

plt.figure(figsize = (6,4))
plt.scatter(w,r, c = t, cmap = 'viridis')
plt.xlabel('Along fiber distance (m)')
plt.ylabel('Crossline distance (m)')
plt.colorbar(label = 'Time')
plt.xlim(15400,16600)
plt.ylim(0,1200)
fname = 'C:\\Users\\Calder\\OneDrive - NTNU\\Documents\\Presentations\\ASA May 2025\\Figures\\track2.png'
plt.savefig(fname, dpi = 500)
#plt.close()


 # %%
#look into inter call interval?
tmp = np.diff(t, axis = 0)
plt.scatter(np.squeeze(t[1:]),np.squeeze(tmp))

# %% getting the whale positions
import numpy as np
from geopy.distance import geodesic
from geopy import Point

# Example: Your track data (distance [m], lat, lon, depth)
track_data = np.loadtxt("D:\DAS\DASsourceLOC\Svalbard_DAS_latlondepth_outer.txt",delimiter = ',')

# Example offset data (you will replace with your real data)
offset_dists = w  # distances along track
perpendicular_offsets = r     # offsets in meters

def compute_bearing(lat1, lon1, lat2, lon2):
    """Compute bearing from (lat1, lon1) to (lat2, lon2) in degrees"""
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    x = np.sin(dLon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def interpolate_track_point(d, track_data):
    """Find two track points surrounding d and interpolate position"""
    for i in range(track_data.shape[0] - 1):
        d1, lat1, lon1, _ = track_data[i]
        d2, lat2, lon2, _ = track_data[i+1]
        if d1 <= d <= d2:
            frac = (d - d1) / (d2 - d1)
            lat = lat1 + frac * (lat2 - lat1)
            lon = lon1 + frac * (lon2 - lon1)
            bearing = compute_bearing(lat1, lon1, lat2, lon2)
            return lat, lon, bearing
    return None, None, None  # If outside track range

def offset_point(lat, lon, bearing, offset_m):
    """Apply perpendicular offset to a point using geodesic projection"""
    offset_bearing = (bearing + 90) % 360 if offset_m >= 0 else (bearing - 90) % 360
    point = Point(lat, lon)
    new_point = geodesic(meters=abs(offset_m)).destination(point, offset_bearing)
    return new_point.latitude, new_point.longitude

# Process each offset point
results = []
for d, offset in zip(offset_dists, perpendicular_offsets):
    lat, lon, bearing = interpolate_track_point(d, track_data)
    if lat is not None:
        lat_offset, lon_offset = offset_point(lat, lon, bearing, offset)
        results.append((d, lat_offset, lon_offset, offset))
    else:
        results.append((d, None, None, offset))

results2 = []
perpendicular_offsets = -1*r
for d, offset in zip(offset_dists, perpendicular_offsets):
    lat, lon, bearing = interpolate_track_point(d, track_data)
    if lat is not None:
        lat_offset, lon_offset = offset_point(lat, lon, bearing, offset)
        results2.append((d, lat_offset, lon_offset, offset))
    else:
        results2.append((d, None, None, offset))

r1 = np.vstack(results)
r2 = np.vstack(results2)
np.save(file = 'D:\\DAS\\DASsourceLOC\\r1',arr = r1)
np.save(file = 'D:\\DAS\\DASsourceLOC\\r2',arr = r2)
np.save(file = 'D:\\DAS\\DASsourceLOC\\t',arr = t)
# %%

import numpy as np
from geopy.distance import geodesic
from geopy import Point
#compute ship pos

perpendicular_offsets = np.load('D:\\DAS\\DASsourceLOC\\shiptrack.npy')
offset_dists = np.load('D:\\DAS\\DASsourceLOC\\shipfiber.npy')*1000
plt.scatter(offset_dists,perpendicular_offsets)
#%%
track_data = np.loadtxt("D:\\DAS\\DASsourceLOC\\Svalbard_DAS_latlondepth_outer.txt",delimiter = ',')


def compute_bearing(lat1, lon1, lat2, lon2):
    """Compute bearing from (lat1, lon1) to (lat2, lon2) in degrees"""
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    x = np.sin(dLon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def interpolate_track_point(d, track_data):
    """Find two track points surrounding d and interpolate position"""
    for i in range(track_data.shape[0] - 1):
        d1, lat1, lon1, _ = track_data[i]
        d2, lat2, lon2, _ = track_data[i+1]
        if d1 <= d <= d2:
            frac = (d - d1) / (d2 - d1)
            lat = lat1 + frac * (lat2 - lat1)
            lon = lon1 + frac * (lon2 - lon1)
            bearing = compute_bearing(lat1, lon1, lat2, lon2)
            return lat, lon, bearing
    return None, None, None  # If outside track range

def offset_point(lat, lon, bearing, offset_m):
    """Apply perpendicular offset to a point using geodesic projection"""
    offset_bearing = (bearing + 90) % 360 if offset_m >= 0 else (bearing - 90) % 360
    point = Point(lat, lon)
    new_point = geodesic(meters=abs(offset_m)).destination(point, offset_bearing)
    return new_point.latitude, new_point.longitude

results = []
for d, offset in zip(offset_dists, perpendicular_offsets):
    lat, lon, bearing = interpolate_track_point(d, track_data)
    if lat is not None:
        lat_offset, lon_offset = offset_point(lat, lon, bearing, offset)
        results.append((d, lat_offset, lon_offset, offset))
    else:
        results.append((d, None, None, offset))

results2 = []
perpendicular_offsets = -1*perpendicular_offsets
for d, offset in zip(offset_dists, perpendicular_offsets):
    lat, lon, bearing = interpolate_track_point(d, track_data)
    if lat is not None:
        lat_offset, lon_offset = offset_point(lat, lon, bearing, offset)
        results2.append((d, lat_offset, lon_offset, offset))
    else:
        results2.append((d, None, None, offset))

r1 = np.vstack(results)
r2 = np.vstack(results2)
np.save(file = 'D:\\DAS\\DASsourceLOC\\ship1',arr = r1)
np.save(file = 'D:\\DAS\\DASsourceLOC\\ship2',arr = r2)
#%%plt.figure()
plt.scatter(r1[:,2],r1[:,3], marker = '+', color = 'pink')
plt.scatter(r2[:,2],r2[:,3],  marker = '+', color = 'pink')
# %%
