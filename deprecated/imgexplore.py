#imgexplore:
#%%
import numpy as np
import matplotlib
import os, glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

path = 'D:\\DAS\\FK\\filttest_20250630T150440\\FK\\20220821T183537Z'
files = glob.glob(os.path.join(path,'*.png'))
pat = '_'
t0 = [x for x in files if pat in x]
# %%

hlist = []
xoff = []
V = []
for f in t0:
    tmp = np.asarray(Image.open(f))
    xoff.append(float(os.path.basename(f).split('_')[2].split('X')[1]))
    V.append(float(os.path.basename(f).split('_')[5].split('V')[1]))
    hist,edge= np.histogram(tmp.flatten(), bins = 256, range = (0,256),density=True)
    hist += 1e-10
    hist /=hist.sum()
    hlist.append(hist)


# norm = colors.Normalize(vmin=min(V), vmax=max(V))
# cmap = cm.get_cmap('viridis')
# sm = cm.ScalarMappable(norm=norm, cmap=cmap)

# fig, ax = plt.subplots(figsize = (8,8))  # Get explicit axes
# for i,vals in enumerate(hlist):
#     color = cmap(norm(xoff[i]))
#     plt.plot(edge[:-1],vals, color = color)

# plt.colorbar(sm, ax = ax,label='V')
V = np.asarray(V)
xoff = np.asarray(xoff)
plt.hist(V,bins = 50)

v1 = 1400
v2 = 4000

plt.figure()
plt.scatter(xoff,V)
plt.xlabel('Xoff')
plt.ylabel('V')
plt.hlines(y = [v1,v2],xmin = 0, xmax = 10000, color = 'red')

vsub = V[np.where(np.logical_and(V>v1,V<v2))]
xo_sub = xoff[np.where(np.logical_and(V>v1,V<v2))]
plt.scatter(xo_sub,vsub, color = 'red')
plt.figure()
plt.hist(vsub,bins = 50)

# %%
