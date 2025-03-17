#%% this script looks to model amplitude variations by frequency for depth and range along the fiber

import numpy as np
import matplotlib.pyplot as plt


whale_x = 0 #should always be zero
whale_y = 10000 #cross line distance (m)
whale_z = 25 #depth (m)
whale_SL = 100 #db
whale_f = 20 #hz
v_water = 1500 


fiber_x = np.arange(start = -50000, stop = 50000)
fiber_y = np.zeros(shape = fiber_x.shape)
fiber_z = np.ones(shape = fiber_x.shape)*200


rel_fiber_sens = abs(np.tanh(fiber_x/whale_y))
whale_r_horiz = np.sqrt(whale_y**2 + fiber_x**2)
whale_r_slope = np.sqrt(whale_y**2 + fiber_x**2 + (fiber_z-whale_z)**2)
TL = -20*np.log10(whale_r_slope)
fiber_RL = (whale_SL + TL)*rel_fiber_sens

theta = np.tanh(fiber_z/whale_r_horiz)



#llyods mirror estimation
top = 2*np.pi*whale_f*whale_z*np.cos(theta)
inner = top/v_water

h= 2*np.sin((2*np.pi*whale_f*whale_z*np.cos(theta))/v_water)


#plt.plot(fiber_x,fiber_y)
#plt.plot(whale_x,whale_y)
#lt.plot(fiber_x,rel_fiber_sens)
#plt.plot(fiber_x,TL)
plt.plot(fiber_x,fiber_RL)
#plt.plot(fiber_x, whale_r_slope)
#plt.plot(fiber_x,theta)
# %%
