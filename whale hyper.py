#%% this script plots hyperbola of whales at different ranges

import numpy as np
import matplotlib.pyplot as plt

def hyperbola_2(x,b,c,d,v):
    return(np.sqrt(b**2 *(((x-c)**2 / (v*b)**2) + 1))-d)

crossrange = 1000
c = 0
v = 1500
d = 0
b = crossrange/v

x = np.arange(-5000,5000,1)

crossrange = np.arange(0,10,1)*-1000 +1
print(crossrange)
for cross in crossrange:
    b = cross/v
    y = hyperbola_2(x,b,c,d,v)
    plt.plot(x,y)
    plt.xlabel('fiber')
    plt.ylabel('time')


# %%
