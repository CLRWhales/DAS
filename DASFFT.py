# this function seeks to process DAS FFT dat a s quickly as possible. GPU from cupy should be a drop in replacement
import numpy as np; import time
#from numba import njit

#build in a flag that toggles the zero padding on and off.
def reshape_array_with_overlap(arr, N, N_overlap):
    rows, cols = arr.shape
    
    # Validate that overlap doesn't exceed block size
    if N_overlap >= N:
        raise ValueError("Overlap cannot be greater than or equal to the block size.")
    
    # Split the array into blocks with the specified overlap
    reshaped = []
    start_idx = 0
    while start_idx + N <= rows:
        reshaped.append(arr[start_idx:start_idx + N, :])
        start_idx += N - N_overlap  # Move the starting index to create overlap
    
    # Check if there's a remaining block that is less than N rows
    if start_idx < rows:
        # Calculate how much padding is needed
        remaining_rows = rows - start_idx
        padded_block = np.vstack([
            arr[start_idx:rows, :],  # Remaining rows
            np.zeros((N - remaining_rows, cols))  # Pad the block to size N
        ])
        reshaped.append(padded_block)


    # Concatenate the blocks along the columns axis (axis=1)
    reshaped_arr = np.concatenate(reshaped, axis=1)
    
    return reshaped_arr


#X: preprocessed np array of Das Data in the TX domain. must be of dimentsion time, channels


def sneakyfft(X,N_samp,N_overlap,N_FFT, window):
    if N_overlap >= N_samp:
        raise ValueError("Overlap cannot be greater than or equal to the block size.")
    
    if len(window) != N_samp:
        raise ValueError("Window and block size must be the same.")
    
    reshaped = reshape_array_with_overlap(X,N_samp,N_overlap)
    reshaped = reshaped * window[:,None]
    fft_out = np.fft.rfft(reshaped, n = N_FFT, axis =0)
   
    nt_slices = fft_out.shape[1]//X.shape[1]
    output = fft_out.reshape(fft_out.shape[0],nt_slices,X.shape[1])

    return output 




