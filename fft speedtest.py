#%% this is a np.fft speed test

import numpy as np
import cProfile
from pstats import Stats
import timeit
import matplotlib.pyplot as plt
# test = np.random.randn(1024,1024)
# t1 = timeit.repeat("np.fft.rfft(test,axis = 0)","import numpy as np; test = np.random.randn(512,512)",number=1000,repeat = 50)
# t2 = timeit.repeat("np.fft.rfft(test,axis = 1)","import numpy as np; test = np.random.randn(512,512)",number=1000,repeat = 50)
# t3 = timeit.repeat("np.fft.rfft(test.T,axis = 1)","import numpy as np; test = np.random.randn(512,512)",number=1000,repeat = 50)

# m1 = np.mean(t1)/1000
# m2 = np.mean(t2)/1000
# m3 = np.mean(t3)/1000
# print(m1)
# print(m2)
# print(m3)

    
#
# %%
fs = 0.0001
N = 10000
time = np.arange(N)*fs
f = 30 #hz
val = np.sin(2*np.pi*f*time)
plt.figure()
plt.plot(time,val)
pspec = np.abs(np.fft.rfft(val))**2/N**2
freqs = np.fft.rfftfreq(n = N,d = fs)

plt.figure()
plt.plot(freqs,pspec)

rms = np.sqrt(np.mean(val**2))
print(rms)
# %%
import numpy as np
import matplotlib.pyplot as plt

# Signal parameters
fs = 0.0001  # sample spacing (not sample rate!)
N = 10000
time = np.arange(N) * fs
f = 30  # Hz
val = np.sin(2 * np.pi * f * time)

# Time-domain RMS
rms = np.sqrt(np.mean(val**2))
print("Time-domain RMS:", rms)
print("Time-domain RMS power:", rms**2)

# FFT and power spectrum
X = np.fft.rfft(val)
pspec = np.abs(X)**2 / N**2
pspec[1:-1] *= 2  # Double non-DC and non-Nyquist bins

# Frequency axis
freqs = np.fft.rfftfreq(n=N, d=fs)

# Frequency-domain RMS power
rms_power_freq = np.sum(pspec)
print("Freq-domain RMS power:", rms_power_freq)
print("Freq-domain RMS:", np.sqrt(rms_power_freq))

# Plot time-domain signal
plt.figure()
plt.plot(time, val)
plt.title("Time-domain Signal")

# Plot power spectrum
plt.figure()
plt.plot(freqs, pspec)
plt.title("Normalized Power Spectrum (RMS-Equivalent)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power")
plt.xlim(0, 100)  # Zoom in near 30 Hz
plt.grid()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Sampling parameters
fs = 0.001  # sample spacing
N = 2056
time = np.arange(N) * fs

# Create a signal with multiple sine waves
frequencies = [10, 30, 60]  # Hz
amplitudes = [1.0, 0.5, 0.8]

val = sum(A * np.sin(2 * np.pi * f * time) for A, f in zip(amplitudes, frequencies))

# Time-domain RMS and power
rms = np.sqrt(np.mean(val**2))
print("Time-domain RMS:", rms)
print("Time-domain RMS power:", rms**2)

# FFT and power spectrum
X = np.fft.rfft(val)
pspec = np.abs(X)**2 / N**2
pspec[1:-1] *= 2  # Compensate for symmetry (except DC/Nyquist)

# Frequency axis
freqs = np.fft.rfftfreq(n=N, d=fs)

# Frequency-domain RMS power
rms_power_freq = np.sum(pspec)
print("Freq-domain RMS power:", rms_power_freq)
print("Freq-domain RMS:", np.sqrt(rms_power_freq))

# Plot time-domain signal
plt.figure()
plt.plot(time, val)
plt.title("Time-domain Signal")

# Plot power spectrum
plt.figure()
plt.plot(freqs, pspec)
plt.title("Normalized Power Spectrum (RMS-Equivalent)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power")
plt.xlim(0, 100)
plt.grid()
plt.show()

# %%
