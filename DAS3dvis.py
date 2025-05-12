#this script works on the visualization of the processed data from DASprocess
#%%
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from Calder_utils import loadFTX
import time


directory = 'C:\\Users\\Calder\\Outputs\\DASdata2\\'
t_ex_start=time.perf_counter()  
spec,freqs,times,channels = loadFTX(directory= directory,
                                    nworkers= 10)
t_ex_end=time.perf_counter(); print(f'duration: {t_ex_end-t_ex_start}s'); 

#%%%


kernel = np.array(((1,2,1),(2,4,2),(1,2,1)))
smoother = kernel[:,:,None]
spec = sig.convolve(spec,smoother,mode = 'same')
spec /=np.sum(kernel)
spec[spec<=0.1] = 0.1
spec = spec - np.mean(spec,axis = (0,1))[None,None,:]
spec = spec / np.std(spec, axis = (0,1))[None,None,:]


# for i in range(spec.shape[2]):
#     fname = 'C:/Users/Calder/Outputs/DASplots1/' + 'fig_'+str(channels[i])+'.png'
#     plt.figure(figsize=(15,5))
#     plt.imshow(X = spec[:,:,i], cmap = 'turbo',origin = 'lower', extent=(min(times2),max(times2),min(freqs),max(freqs)))
#     plt.colorbar()
#     plt.clim(0,10)
#     plt.savefig(fname, dpi=300, bbox_inches='tight')
#     plt.close()
    
# plt.imshow(spec[30,:,:])
# plt.colorbar()

spec = np.rot90(spec, 1, (0,2))
grid = pv.ImageData()
grid.dimensions = np.array(spec.shape)+1
grid.origin = (0,0,0)
grid.spacing = (1,0.5,0.5)
grid.cell_data['spec'] = spec.flatten(order = 'F')
outline = grid.outline
threshed = grid.threshold(2.1)
# labels =  dict(ztitle='channel', xtitle='frequency', ytitle='time')

# new = grid.cells_to_points()
# surf = new.contour(8)
# bodies = threshed.split_bodies()

# for key in bodies.keys():
#     b = bodies[key]
#     vol = b.volume
#     if vol < 30:
#         del bodies[key]
#         continue 
#     b.cell_data["TOTAL VOLUME"] = np.full(b.n_cells, vol)



p = pv.Plotter()
p.add_mesh(threshed, lighting= False)
p.camera.zoom(1.5)
p.show(auto_close=False)
# viewup = [0.5, 0.5, 1]
# path = p.generate_orbital_path(n_points=40, shift=threshed.length, viewup= viewup)
# p.open_gif('C:/Users/Calder/Outputs/DASplots2/unknownUNclipped.gif')
# p.orbit_on_path(path, write_frames=True)
# p.close()


# %%
