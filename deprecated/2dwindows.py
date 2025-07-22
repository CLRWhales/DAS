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
c_min = 1000
c_max = 4000
f = np.fft.rfftfreq(512, d = 1/256)[1:]
k = np.fft.rfftfreq(512, d = 12)[1:]
kk,ff = np.meshgrid(k,f)

g = 1.0*((ff < kk*c_min))
g2 = 1.0*((ff < kk*c_max))

# g = g + np.fliplr(g)
# g2 = g2 + np.fliplr(g2)
g = g2-g
plt.figure
plt.imshow(g)
# %%
#circle windows:
height, width = 256, 256

# Create empty array
img = np.zeros((height, width), dtype=np.uint8)

# Circle parameters
radius = 256
cx= 0
cy = 0 # Top middle

# Generate grid of coordinates
Y, X = np.ogrid[:height, :width]

# Compute mask for filled lower half circle
mask = ((X - cy) ** 2 + (Y - cx) ** 2 <= radius ** 2) #& (Y >= cx)

# Apply mask
img[mask] = 1


img = g*mask
# Display the result
plt.imshow(img)
#plt.axis('off')
plt.ylabel('F')
plt.xlabel('K')
plt.show()

# %%
