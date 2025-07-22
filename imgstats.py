#this script looks into various filtering/smoothing schemes of images.
#%%
import imageio
import numpy as np
import matplotlib.pyplot as plt
from  skimage.filters import difference_of_gaussians
from skimage.restoration import denoise_tv_chambolle
import scipy.ndimage as NDI

path = "D:\\DAS\\FK\\full2d20250618T160735\\FK\\20220821T183537Z\\T0_X512_H135521.0_20220821T183537Z.png"


img = imageio.imread(path)
#img2 = imageio.imread(path2)

print(np.std(img))
#print(np.std(img2))

output = NDI.gaussian_filter(img,sigma=(1,2))
#output = NDI.median_filter(output, size=(5))

#output = NDI.uniform_filter(img,size = 5)
#output = difference_of_gaussians(img,2,10)
#output = denoise_tv_chambolle(img)#

fig, ax = plt.subplots(2,1, figsize = (10,10))

ax[0].imshow(img, cmap = 'gray')
ax[1].imshow(output, cmap = 'gray')

plt.show()

plt.figure()
plt.hist(img.ravel(),bins = np.arange(255)-0.5)
plt.ylim(0,4000)




plt.figure()
plt.imshow(img)
# plt.figure()
# plt.imshow(img2)
# %%
tmp = np.mean(img,axis =1)
tmp2 = np.mean(output,axis = 1)
plt.figure()
plt.plot(tmp)
plt.plot(tmp2)
plt.hlines(y = 128, xmin = 0, xmax = 255, color = 'red')
plt.ylim(0,255)
# plt.figure()
# plt.plot(tmp-tmp2)
# %%

ks = np.fft.fftshift(np.fft.fftfreq(512,12))
freqs = np.fft.rfftfreq(512,1/256)[1:]
peak_lock = np.unravel_index(np.argmax(output), output.shape)
print(peak_lock)
cm = NDI.center_of_mass(output)
plt.figure()
plt.imshow(output, origin = 'lower', extent = (np.min(ks),np.max(ks),np.min(freqs),np.max(freqs)), aspect = 'auto')
plt.scatter(ks[peak_lock[1]],freqs[peak_lock[0]], color = 'red') 

#plt.plot([0,ks[peak_lock[1]]],[0,freqs[peak_lock[0]]],color = 'white')

slope = freqs[peak_lock[0]]/ks[peak_lock[1]]
print(slope)
# %%
from Calder_utils import compute_FK_speed,average_values_by_slope_bins


slope,angle= compute_FK_speed(ks,freqs)

slopebins = np.linspace(0,10000,num = 101)
centers = 0.5* (slopebins[:-1] + slopebins[1:])
slopebins[0] = -np.inf
slopebins[-1] = np.inf

averages = average_values_by_slope_bins(slope,output,slopebins)

plt.figure()
plt.plot(centers, averages)
#plt.vlines(x = [-1500,1500],ymin = 120, ymax = 150, color = 'red')
plt.xlabel('Velocity (m/s)')
plt.ylabel('energy')

wind = np.hanning(7)
wind = wind/np.sum(wind)
averages= np.convolve(averages,wind,mode='same')
plt.plot(centers,averages)
vel = centers[np.argmax(averages)]
plt.scatter(vel,np.max(averages), color = 'red')
print(vel)

# %%
import imageio
import numpy as np
import matplotlib.pyplot as plt



path = "D:\\DAS\\FK\\peaktest20250626T142743\\FK\\20220821T183537Z\\T0_X1536_F94_K150.0_V1946.0_20220821T183537Z.png"
freqs = np.fft.rfftfreq(n = 512,d = 1/256)[1:]
ks = np.fft.rfftfreq(n = 512, d = 12)[1:]
img = imageio.imread(path)
plt.figure()
plt.imshow(img[:,:,0], origin = 'lower', extent = (np.min(ks),np.max(ks),np.min(freqs), np.max(freqs)), aspect = 'auto')

plt.figure()
plt.imshow(img[:,:,1], origin = 'lower', extent = (np.min(ks),np.max(ks),np.min(freqs), np.max(freqs)), aspect = 'auto')


height, width,n = img.shape


# Circle parameters
radius = 256
cx= 0
cy = 0 
Y, X = np.ogrid[:height, :width]
mask = ((X - cy) ** 2 + (Y - cx) ** 2 <= radius ** 2) #& (Y >= cx)

c_min = 1400
c_max = 4000
f = np.fft.rfftfreq(512, d = 1/256)[1:]
k = np.fft.rfftfreq(512, d = 12)[1:]
ff,kk = np.meshgrid(f,k)

g = 1.0*((ff < kk*c_min))
g2 = 1.0*((ff < kk*c_max))

# g = g + np.fliplr(g)
# g2 = g2 + np.fliplr(g2)
g = (g2-g).T*mask

img = img*g[:,:,None]
plt.figure()
plt.imshow(mask)

plt.figure()
plt.imshow(img[:,:,0], origin = 'lower', extent = (np.min(ks),np.max(ks),np.min(freqs), np.max(freqs)), aspect = 'auto')

plt.figure()
plt.imshow(img[:,:,1], origin = 'lower', extent = (np.min(ks),np.max(ks),np.min(freqs), np.max(freqs)), aspect = 'auto')


# %%
