
#%%
#kindly provided by tora


from  utils_tmp import butter_bandpass
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, filtfilt, butter

fs = 512
b,a = butter(N = 6,Wn=0.5,fs = fs,btype = 'highpass' )


# Impulse response (in time domain)
impulse = np.zeros(100)  # Create an impulse signal
impulse[50] = 1  # Set the impulse at the center

filtered_impulse = filtfilt(b, a, impulse)  # Zero-phase filtering

# Frequency response
w, h = freqz(b, a, worN=8000)

# Plotting
plt.figure(figsize=(14, 6))

# Time domain plot (impulse response)
plt.subplot(1, 2, 1)
plt.plot(impulse, label='Original Impulse')
plt.plot(filtered_impulse, label='Filtered Impulse')
plt.title('Impulse Response (Time Domain)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

# Frequency domain plot (frequency response)
plt.subplot(1, 2, 2)
plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
plt.title('Frequency Response (Frequency Domain)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.xlim(0, 10)
plt.grid()

plt.tight_layout()
plt.show()

#return



# %%
