# this function seeks to process DAS FFT dat a s quickly as possible. GPU from cupy should be a drop in replacement

def DASFFT(x,hop,n,window):
    #restack the array into fast fft shape
    #multiply by window
    #compute fft with padding 
    #unwrap back into 3d for output

