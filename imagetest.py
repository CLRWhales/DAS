#this script looks to try and supress horizontal lines in an image

#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import interp1d

def cfar(array, guardsize,sample_size):
    kernalsize = 2*(sample_size + guardsize)+1
    kernal = np.zeros(kernalsize)
    kernal[:sample_size] = 1
    kernal[-sample_size:]=1
    kernal = kernal/np.sum(kernal)
    thresh = np.convolve(array,kernal,mode = 'same')
    return thresh

def detoneImg(arr,guardsize,samplesize,mult = 2,sigmascale = 1):
    arrout = np.copy(arr)
    rmeans = np.mean(arr,axis = (1,2))
    plt.figure()
    plt.plot(rmeans)
    plt.ylim(0,256)
    rdiff = np.diff(rmeans,axis = 0)
    thresh = cfar(np.abs(rdiff),guardsize,samplesize)*mult
    
    b = np.where(rdiff>thresh)[0]
    missing_indices = np.unique([b+1,b+2])
    missing_indices = missing_indices[missing_indices<arr.shape[1]]
    missing_indices = list(missing_indices)
    kept_indices = [i for i in range(arrout.shape[0]) if i not in missing_indices]
    interpfunc = interp1d(kept_indices,arrout[kept_indices,:,:],kind = 'linear', fill_value='extrapolate', axis = 0)
    interp_rows = interpfunc(missing_indices)
    #add noise
    col_std = arr[kept_indices,:,:].std(axis = 0)
    noise = np.random.normal(loc=0.0, scale=col_std*sigmascale, size=interp_rows.shape)
    interp_rows_noisy = interp_rows + noise

    arrout[missing_indices] = np.clip(interp_rows_noisy,0,255)
    return arrout

def generate_gaussian_bar_mask(
    image_shape,
    max_num_bars=10,
    directions=('horizontal', 'vertical'),
    max_amplitude=1.0,
    max_sigma=2,
    max_coverage=0.5,
    seed=None
):
    """
    Generate a structured noise mask with Gaussian-profiled bars.

    Args:
        image_shape: (H, W)
        num_bars: Number of bars to insert
        directions: Tuple of 'horizontal', 'vertical', or both
        max_amplitude: Max intensity of the bars
        max_sigma: Max std-dev for Gaussian cross-section
        max_coverage: Max % of image area to occlude
        seed: Random seed

    Returns:
        mask: np.ndarray of shape (H, W)
    """
    if seed is not None:
        np.random.seed(seed)

    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.float32)
    total_pixels = H * W
    covered_pixels = 0
    max_pixels = int(max_coverage * total_pixels)

    num_bars = np.random.randint(1, max_num_bars + 1)


    for _ in range(num_bars):
        direction = np.random.choice(directions)
        amp = np.random.uniform(0.2, max_amplitude)
        sigma = np.random.uniform(1.0, max_sigma)

        if direction == 'horizontal':
            center_y = np.random.randint(0, H)
            y = np.arange(H)
            profile = amp * np.exp(-0.5 * ((y - center_y) / sigma) ** 2)
            bar = np.tile(profile[:, np.newaxis], (1, W))
        elif direction == 'vertical':
            center_x = np.random.randint(0, W)
            x = np.arange(W)
            profile = amp * np.exp(-0.5 * ((x - center_x) / sigma) ** 2)
            bar = np.tile(profile[np.newaxis, :], (H, 1))
        else:
            continue

        bar_pixels = (bar > 0.01).sum()  # Only count non-trivial intensities
        if covered_pixels + bar_pixels <= max_pixels:
            mask += bar
            covered_pixels += bar_pixels
        else:
            break

    return np.clip(mask, 0.0, 1.0)

def generate_structured_bar_noise(
    image_shape,
    max_num_bars=10,
    max_bar_thickness=5,
    max_amplitude=1.0,
    directions=('horizontal', 'vertical'),
    max_coverage=0.5,
    seed=None
):
    """
    Generate a 2D structured noise mask with horizontal/vertical bars.

    Parameters:
        image_shape: tuple (H, W)
        max_num_bars: int, maximum number of bars
        max_bar_thickness: int, maximum thickness (in pixels)
        max_amplitude: float, max pixel value for noise [0, 1]
        directions: tuple, 'horizontal', 'vertical', or both
        max_coverage: float, max fraction of image pixels that can be masked
        seed: int or None, for reproducibility

    Returns:
        noise_mask: np.ndarray of shape (H, W), float32
    """
    if seed is not None:
        np.random.seed(seed)

    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.float32)
    total_pixels = H * W
    covered_pixels = 0
    max_pixels = int(max_coverage * total_pixels)

    num_bars = np.random.randint(1, max_num_bars + 1)

    for _ in range(num_bars):
        dir = np.random.choice(directions)
        amp = np.random.uniform(0.2, max_amplitude)
        thickness = np.random.randint(1, max_bar_thickness + 1)

        if dir == 'horizontal':
            y = np.random.randint(0, H - thickness)
            new_bar = np.zeros_like(mask)
            new_bar[y:y+thickness, :] = amp
        elif dir == 'vertical':
            x = np.random.randint(0, W - thickness)
            new_bar = np.zeros_like(mask)
            new_bar[:, x:x+thickness] = amp
        else:
            continue

        bar_pixels = (new_bar > 0).sum()
        if covered_pixels + bar_pixels <= max_pixels:
            mask += new_bar
            covered_pixels += bar_pixels
        else:
            break

    return np.clip(mask, 0.0, 1.0)

file = "C:\\Users\\Calder\\Models\\FKv5_speedsubNoMask_20250704T131434\\groups\\5_15_unknownFW\\FS256_T3584_X3328_F49_K69.0_V2226.0_L1_20220821T232607Z.png"
f2 = "C:\\Users\\Calder\\Models\\FKv5_speedsubNoMask_20250704T131434\\groups\\0_6_multitonal\\FS256_T1024_X6400_F143_K246.0_V1798.0_L1_20220821T213007Z.png"
f3 = "C:\\Users\\Calder\\Models\\FKv5_speedsubNoMask_20250704T131434\\groups\\-6_9_ FW\\FS256_T0_X512_F39_K67.0_V1834.0_L0_20220821T185207Z.png"
f4 = "C:\\Users\\Calder\\Models\\FKv5_speedsubNoMask_20250704T131434\\groups\\9_4_unknownDS+noise\\FS256_T1792_X256_F92_K145.0_V1970.0_L0_20220821T103807Z.png"
f5 = "C:\\Users\\Calder\\Models\\FKv5_speedsubNoMask_20250704T131434\\groups\\6_9_LF\\FS256_T2048_X4096_F18_K18.0_V3243.0_L0_20220821T151507Z.png"

f6 = "D:\\DAS\\FK\\test_threshDM20250716T103835\\FK\\20220821T183537Z\\FS256_T0_X1536_F94_K150.0_V1946.0_L1_20220821T183537Z.png"
f7 = "D:\\DAS\\FK\\test_threshDM20250716T103835\\FK\\20220821T183537Z\\FS256_T0_X10240_F17_K171.0_V323.0_L1_20220821T183537Z.png"
img = Image.open(f7)
mult = 2

img = np.array(img)

plt.figure(figsize = (10,15))
plt.imshow(img[:,:,0])
plt.clim(0,255)

detoned = detoneImg(img,9,9,2,0.5)


plt.figure(figsize = (10,10))
plt.imshow(detoned[:,:,0])
plt.clim(0,255)

image_shape = (256,256)

mask = generate_structured_bar_noise(image_shape)*np.std(img)*2
filled_data = img+mask[:,:,None]
plt.figure(figsize=(10,10))
plt.imshow(filled_data[:,:,0])
plt.clim(0,255)

mask = generate_gaussian_bar_mask(image_shape)*np.std(img)*2
filled_data = img+mask[:,:,None]
plt.figure(figsize=(10,10))
plt.imshow(filled_data[:,:,0])
plt.clim(0,255)
# %%
