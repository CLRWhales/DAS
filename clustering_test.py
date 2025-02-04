#this script explores clustering on minute x minute datasets
import numpy as np

import glob
import pyvista as pv

import numpy as np
from skimage import exposure, util


directory = 'C:/Users/Calder/Outputs/DASdata4/'
filepaths = sorted( glob.glob(directory + '*.npy') )#[0:10]



# Example 3D array (replace with your actual data)
data = np.load(filepaths[50])

img = (data - data.min()) / (data.max() - data.min())

kernel_size = (data.shape[0] // 5, data.shape[1] // 5, data.shape[2] // 2)
kernel_size = np.array(kernel_size)
clip_limit = 0.9

data_ahe = exposure.equalize_adapthist(img,
                                       kernel_size= kernel_size,
                                       clip_limit= clip_limit
                                       )

data_ahe = np.rot90(data, 1, (0,2))
grid = pv.ImageData()
grid.dimensions = np.array(data_ahe.shape)+1
grid.origin = (0,0,0)
grid.spacing = (1,0.5,0.5)
grid.cell_data['spec'] = data_ahe.flatten(order = 'F')
threshed = grid.threshold(25)

p = pv.Plotter()
p.add_mesh(threshed)
p.show_bounds()
p.show()

