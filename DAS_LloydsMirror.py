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
import numpy as np
import matplotlib.pyplot as plt

x= np.arange(start = -10000, stop = 10000)
y= 1000.0
z = 300.0
zs = 30.0
fs = 125
c = 1500.0

#Distance from whale to cable: 
y = 0.0
R = np.sqrt(x*x + y*y + z*z)
ap1 = x*x/(R*R*R)
as1 = 2*np.sqrt(y*y+z*z)*x/(R*R*R)
#Include source ghost effect: 
c2 = z/R
arg2 = 2*np.pi*fs*zs*c2/c
app1 = ap1*abs(np.sin(arg2))


#Distance from whale to cable: 
y = 5000
R = np.sqrt(x*x + y*y + z*z)
ap2 = x*x/(R*R*R)
as2 = 2*np.sqrt(y*y+z*z)*x/(R*R*R)
#Include source ghost effect: 
c2 = z/R
arg2 = 2*np.pi*fs*zs*c2/c
app2 = ap2*abs(np.sin(arg2))

#Distance from whale to cable: 
y = 0.0
zs = 10

R = np.sqrt(x*x + y*y + z*z)
ap3 = x*x/(R*R*R)
as3 = 2*np.sqrt(y*y+z*z)*x/(R*R*R)
#Include source ghost effect: 
c2 = z/R
arg2 = 2*np.pi*fs*zs*c2/c
app3 = ap3*abs(np.sin(arg2))


plt.plot(x,ap1)
plt.plot(x,as1)
plt.plot(x,ap2)
plt.plot(x,as2)
plt.close()


plt.plot(x,app1)
plt.plot(x,app2)
plt.plot(x,app3)

# %%

import numpy as np
import matplotlib.pyplot as plt

x= np.arange(start = -10000, stop = 10000)
y= 500
z = 300.0
zs = 10
fs = 125
c = 1500.0

freqs = np.arange(start=10, stop=150, step= 1)
ranges = np.arange(start=0, stop=5000, step = 50)
depths = np.arange(start = 1, stop = 299)
out = np.zeros((depths.size,x.size))

for i in range(out.shape[0]):
    #y = ranges[i]
    zs = depths[i]
    #fs= freqs[i]
    R = np.sqrt(x*x + y*y + z*z)
    ap3 = x*x/(R*R*R)
    #as3 = 2*np.sqrt(y*y+z*z)*x/(R*R*R)
    #Include source ghost effect: 
    c2 = z/R
    arg2 = 2*np.pi*fs*zs*c2/c
    out[i,] = ap3*abs(np.sin(arg2))

plt.imshow(out, aspect='auto', extent = (np.min(x),np.max(x), np.min(depths),np.max(depths)), origin = 'lower')

# %%
