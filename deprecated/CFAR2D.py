
# %% setup for a 2d spectrogram cfar detector by double convolution
import scipy.signal as sig
import numpy as np

def cfar_2D(X, nf_guard, nt_guard, nf_ref, nt_ref):
    """ this function performs a 2d cfar pass by using the difference between 2 convolutions of unity to get the sum to compute the average for the threshold. 
    this can then be scaled to used in single or double threshold techniques.

        Parameters
        ----------
        X : 2d np.array 
            2d data array with dimension(F,T).
        nf_guard : int
            number of guard samples in the F direction (note it will be f sample before and after)
        nt_guard : int
            number of guard samples in the T direction (note it will be T sample before and after)

        Returns
        -------
        thresh : 2d np.array
            data array containing the cfar(bias1) threshold for each pixel location, should be the same dimension of X
    """
    f_dim = 2*(nf_guard+nf_ref) + 1
    t_dim = 2*(nt_guard+nt_ref) + 1
    outer_kern = np.ones((f_dim,t_dim))
    inner_kern = np.ones((nf_guard+1, nt_guard+1))
    npoints = outer_kern.size - inner_kern.size

    inner_kern = np.zeros((nf_guard+1, nt_guard+1))
    full = np.pad(inner_kern,((nf_ref,nf_ref),(nt_ref,nt_ref)),mode = 'constant', constant_values=((1,1),(1,1)))

    outer_conv = sig.convolve2d(X,full, 'same','wrap')
    #inner_conv = sig.convolve2d(X,inner_kern,'same','wrap')

    #thresh= (outer_conv - inner_conv)/npoints
    thresh= outer_conv/npoints

    return thresh

# example 1
# tmp = abs(spec[:,:,255])
# bias = 1
# thresh = cfar_2D(tmp, 4,8,4,8)
# thresh = thresh * bias
# targets = np.copy(tmp)
# targets[np.where(tmp<thresh)] = np.ma.masked

# cmap = plt.cm.turbo
# cmap.set_bad('white',1.)
# plt.imshow(targets, origin='lower', cmap=cmap)


# example using with hysteretic thresholding using adaptive bias?
# from skimage import filters
# #this is a slow elementwise hysteresys threshold method. perhaps there is a faster one?

# b1=2
# b2 = 5
# map = np.zeros(shape = (tmp.shape))

# map[np.where(tmp>=thresh*b1)] = 1
# map[np.where(tmp>=thresh*b2)] = 3

# #plt.imshow(map, origin='lower', cmap=cmap)

# new = filters.apply_hysteresis_threshold(map,0.5,2)

# plt.imshow(new, origin='lower', cmap = 'Greys')

# %%
