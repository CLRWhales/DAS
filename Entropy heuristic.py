#this script looks into entropy based FK detectors
#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as NDI
from scipy.stats import entropy
from Calder_utils import KL_div

k = [20,30,70]
img = []

for i in range(120):
  tmp = np.random.randn(256,512)+10
  if i in k:
    t2 = np.ones(shape = (20,300))*1.1
    tmp[100:120,100:400] = tmp[100:120,100:400] + t2
  img.append(tmp)

#%%

ents= KL_div(img)
med = np.median(ents)
sigma = np.std(ents)
plt.figure()
plt.plot(ents)
plt.hlines([med,med+2.5*sigma],0,len(ents), color = 'red')
print(sigma)

# %%
img_mean = np.mean(img,axis = 0)
img_std = np.std(img)
# plt.figure()
# plt.imshow(img_mean)
# plt.colorbar()
# plt.clim(5,15)


img2 = [(im/img_mean) for im in img]
img2 = [np.where(im<1,1,im) for im in img2]

ents= KL_div(img2)
med = np.median(ents)
sigma = np.std(ents)
plt.figure()
plt.plot(ents)
plt.hlines([med,med+2.5*sigma],0,len(ents), color = 'red')
print(sigma)

#%%

#%%
c_min = 1000
c_max = 5000
f = np.fft.rfftfreq(512, d = 1/256)[1:]
k1 = np.fft.fftshift(np.fft.fftfreq(512, d = 12))
kk,ff = np.meshgrid(k1,f)
slow = ff<np.abs(kk*c_min)
fast = ff > np.abs(kk*c_max)

full = slow+fast
plt.figure()
plt.imshow(full)
# %%
img3 = [np.where(full,1,im) for im in img2]

plt.figure()
plt.imshow(img3[k[1]])
plt.colorbar()

ents= KL_div(img3)
med = np.median(ents)
sigma = np.std(ents)
plt.figure()
plt.plot(ents)
plt.hlines([med,med+2.5*sigma],0,len(ents), color = 'red')
print(sigma)

# %%
