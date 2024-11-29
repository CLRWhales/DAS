#this script seeks to perform MVDR summation on frames of spectrogram.
import numpy as np
from scipy.linalg import solve

def estimate_covariance_matrix_per_frame(X):
    """
    Estimate the covariance matrix for each frequency and each time frame.

    Args:
        X: (F, T, M) ndarray, where F is the number of frequency bins,
           M is the number of microphones, and T is the number of time frames.
    
    Returns:
        R: (F, T, M, M) ndarray, spatial covariance matrix for each frequency and time frame.
    """
    F, T, M = X.shape
    R = np.zeros((F, T, M, M), dtype=np.complex64)
    for f in range(F):
        for t in range(T):
            # Outer product for each time frame
            R[f, t] = np.outer(X[f, t, :], np.conj(X[f, t, :]))
            #R[f,t] = np.cov(X[f, t, :])
    return R

def compute_steering_vector_per_frame(R, target_index):
    """
    Compute the steering vector for each frequency and each time frame.
    
    Args:
        R: (F, T, M, M) ndarray, spatial covariance matrix for each frequency and time frame.
        target_index: int, index of the microphone to use as the reference.

    Returns:
        d: (F, T, M) ndarray, steering vector for each frequency and time frame.
    """
    F, T, M, _ = R.shape
    d = np.zeros((F, T, M), dtype=np.complex64)
    for f in range(F):
        for t in range(T):
            # Use the eigenvector corresponding to the maximum eigenvalue as the steering vector
            eigvals, eigvecs = np.linalg.eigh(R[f, t])
            d[f, t] = eigvecs[:, -1]  # Largest eigenvalue's eigenvector
            # Normalize the steering vector so that it equals 1 at the reference microphone
            d[f, t] /= d[f, t, target_index]
    return d

def compute_mvdr_weights_per_frame(R, d):
    """
    Compute the MVDR weights for each frequency and each time frame.

    Args:
        R: (F, T, M, M) ndarray, spatial covariance matrix for each frequency and time frame.
        d: (F, T, M) ndarray, steering vector for each frequency and time frame.

    Returns:
        w: (F, T, M) ndarray, MVDR weights for each frequency and time frame.
    """
    F, T, M = d.shape
    w = np.zeros((F, T, M), dtype=np.complex64)
    for f in range(F):
        for t in range(T):
            # Solve R * w = d for MVDR beamforming weights
            w[f, t] = solve(R[f, t], d[f, t]) / (np.conj(d[f, t]) @ solve(R[f, t], d[f, t]))
    return w

def mvdr_beamform_per_frame(X, target_index=0):
    """
    Apply frame-by-frame MVDR beamforming to multichannel spectrogram data.

    Args:
        X: (F, M, T) ndarray, multichannel spectrogram (frequency, mic, time).
        target_index: int, index of the reference microphone.

    Returns:
        Y: (F, T) ndarray, beamformed output spectrogram.
    """
    F, T, M = X.shape
    Y = np.zeros((F, T), dtype=np.complex64)

    # Estimate the spatial covariance matrix per frame
    R = estimate_covariance_matrix_per_frame(X)
    
    # Compute the steering vector per frame
    d = compute_steering_vector_per_frame(R, target_index)
    
    # Compute MVDR weights per frame
    w = compute_mvdr_weights_per_frame(R, d)
    
    # Apply the weights to the multichannel spectrogram
    for f in range(F):
        for t in range(T):
            Y[f, t] = np.conj(w[f, t]) @ X[f, :, t]
    
    return Y

