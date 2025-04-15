#this script looks to replicate the mvdr beamformer approach in pysdr

#%% generate some functions to do the testing

import numpy as np

d = 0.5
Nr = 10
theta = 40

s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta))

s = s.reshape(-1,1)
tmp = np.outer(s, np.conj(s))

tmpinv = np.linalg.pinv(tmp)

print(s)
print(tmp)
#for Each frequency bin 
print(2*40*40/11)



# %%
import matplotlib.pyplot as plt


f = np.arange(512)
v = 1470
lamda = v/f
D = 20
Frndist = (2*D*D)/lamda
spacing = 4
thresh = v/(2*spacing)

plt.plot(f,lamda)
plt.plot(f,Frndist)
plt.axvline(x = thresh)


# %%
#running throug hthe example pythonsdr code 
import numpy as np
import matplotlib.pyplot as plt
import time

sample_rate = 1e6
N = 10000 # number of samples to simulate

# Create a tone to act as the transmitter signal
t = np.arange(N)/sample_rate # time vector
f_tone = 0.02e6
tx = np.exp(2j * np.pi * f_tone * t)

d = 0.5 # half wavelength spacing
Nr = 10
theta_degrees = 20 # direction of arrival (feel free to change this, it's arbitrary)
theta = theta_degrees / 180 * np.pi # convert to radians
s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # Steering Vector
print(s) # note that it's 3 elements long, it's complex, and the first element is 1+0j

s = s.reshape(-1,1) # make s a column vector
print(s.shape) # 3x1
tx = tx.reshape(1,-1) # make tx a row vector
print(tx.shape) # 1x10000

X = s @ tx # Simulate the received signal X through a matrix multiply
print(X.shape) # 3x10000.  X is now going to be a 2D array, 1D is time and 1D is the spatial dimension



n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
X = X + 0.5*n # X and n are both 3x10000

plt.plot(np.asarray(X[0,:]).squeeze().real[0:200]) # the asarray and squeeze are just annoyances we have to do because we came from a matrix
plt.plot(np.asarray(X[1,:]).squeeze().real[0:200])
plt.plot(np.asarray(X[2,:]).squeeze().real[0:200])
plt.show()

theta_scan = np.linspace(-1*np.pi/2, np.pi/2, 1000) # 1000 different thetas between -180 and +180 degrees
results = []
tapering = np.hamming(Nr) # Hamming window function

t_ex_start=time.perf_counter()  

for theta_i in theta_scan:
    w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)) # Conventional, aka delay-and-sum, beamformer  
    w *= tapering
    X_weighted = w.conj().T @ X # apply our weights. remember X is 3x10000
    results.append(10*np.log10(np.var(X_weighted))) # power in signal, in dB so its easier to see small and large lobes at the same time
results -= np.max(results) # normalize (optional)

t_ex_end=time.perf_counter(); print(f'Delay and sum time: {t_ex_end-t_ex_start}s');

# print angle that gave us the max value
print(theta_scan[np.argmax(results)] * 180 / np.pi) # 19.99999999999998

# plt.plot(theta_scan*180/np.pi, results) # lets plot angle in degrees
# plt.xlabel("Theta [Degrees]")
# plt.ylabel("DOA Metric")
# plt.grid()
# plt.show()

# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
# ax.set_theta_zero_location('N') # make 0 degrees point up
# ax.set_theta_direction(-1) # increase clockwise
# ax.set_rlabel_position(55)  # Move grid labels away from other labels
# plt.show()


# %%
#trying the same definition this time with MVDR method
from numba import njit

def w_mvdr(theta, Nr, Rinv):
   s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # steering vector in the desired direction theta
   s = s.reshape(-1,1) # make into a column vector (size 3x1)
   w = (Rinv @ s)/(s.conj().T @ Rinv @ s) # MVDR/Capon equation! numerator is 3x3 * 3x1, denominator is 1x3 * 3x3 * 3x1, resulting in a 3x1 weights vector
   return w

def InverseCorelation(X):
    R = (X @ X.conj().T)/X.shape[1] # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
    Rinv = np.linalg.pinv(R) # 3x3. pseudo-inverse tends to work better/faster than a true inverse
    return Rinv

theta_scan = np.linspace(-1*np.pi/2, np.pi/2, 1000) # 1000 different thetas between -180 and +180 degrees
results = []

t_ex_start=time.perf_counter()  

Rinv = InverseCorelation(X)

for theta_i in theta_scan:
   w = w_mvdr(theta_i, Nr,Rinv) # 3x1
   X_weighted = w.conj().T @ X # apply weights
   power_dB = 10*np.log10(np.var(X_weighted)) # power in signal, in dB so its easier to see small and large lobes at the same time
   results.append(power_dB)
results -= np.max(results) # normalize

t_ex_end=time.perf_counter(); print(f'MVDR time: {t_ex_end-t_ex_start}s'); 

print(theta_scan[np.argmax(results)] * 180 / np.pi) # 19.99999999999998

# plt.plot(theta_scan*180/np.pi, results) # lets plot angle in degrees
# plt.xlabel("Theta [Degrees]")
# plt.ylabel("DOA Metric")
# plt.grid()
# plt.show()

# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
# ax.set_theta_zero_location('N') # make 0 degrees point up
# ax.set_theta_direction(-1) # increase clockwise
# ax.set_rlabel_position(55)  # Move grid labels away from other labels
# plt.show()



# %%
Nr = 8 # 8 elements
theta1 = 20 / 180 * np.pi # convert to radians
theta2 = 25 / 180 * np.pi
theta3 = -40 / 180 * np.pi
s1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1)).reshape(-1,1) # 8x1
s2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
s3 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta3)).reshape(-1,1)
# we'll use 3 different frequencies.  1xN
tone1 = np.exp(2j*np.pi*0.01e6*t).reshape(1,-1)
tone2 = np.exp(2j*np.pi*0.02e6*t).reshape(1,-1)
tone3 = np.exp(2j*np.pi*0.03e6*t).reshape(1,-1)
X = s1 @ tone1 + s2 @ tone2 + 0.1 * s3 @ tone3 # note the last one is 1/10th the power
n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
X = X + 0.5*n # 8xN


theta_scan = np.linspace(-1*np.pi/2, np.pi/2, 1000) # 1000 different thetas between -180 and +180 degrees
results = []

t_ex_start=time.perf_counter()  

Rinv = InverseCorelation(X)

for theta_i in theta_scan:
   w = w_mvdr(theta_i, Nr,Rinv) # 3x1
   X_weighted = w.conj().T @ X # apply weights
   power_dB = 10*np.log10(np.var(X_weighted)) # power in signal, in dB so its easier to see small and large lobes at the same time
   results.append(power_dB)
results -= np.max(results) # normalize

t_ex_end=time.perf_counter(); print(f'Multi time: {t_ex_end-t_ex_start}s'); 

print(theta_scan[np.argmax(results)] * 180 / np.pi) # 19.99999999999998

plt.plot(theta_scan*180/np.pi, results) # lets plot angle in degrees
plt.xlabel("Theta [Degrees]")
plt.ylabel("DOA Metric")
plt.grid()
plt.show()

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlabel_position(55)  # Move grid labels away from other labels
plt.show()

# %%

