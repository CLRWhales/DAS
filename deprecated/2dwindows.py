#%%
import numpy as np
import matplotlib.pyplot as plt
spaceweight = np.hanning(512)
timeweight = np.hanning(512)

wind = np.outer(timeweight,spaceweight)

print(wind.shape)

plt.imshow(wind)
plt.colorbar()
# %%
a = 0.5
print(np.log10(a))

# %% tryign to make 2d fk fan filters
c_min = 1450
c_max = 8000
f = np.fft.fftshift(np.fft.fftfreq(512, d = 1/256))
k = np.fft.fftshift(np.fft.fftfreq(512, d = 12))
ff,kk = np.meshgrid(f,k)

g = 1.0*((ff < kk*c_min) & (ff < -kk*c_min))
g2 = 1.0*((ff < kk*c_max) & (ff < -kk*c_max))

g = g + np.fliplr(g)
g2 = g2 + np.fliplr(g2)
g = g-g2
plt.figure
plt.imshow(g, extent = (np.min(f),np.max(f),np.min(k),np.max(k)), origin='lower', aspect = 'auto')
# %%
