#this script explores the entropy of an FK image using the del entropy shifts
#%%
import numpy as np
import  imageio 
import matplotlib.pyplot as plt


whale1 = "D:\\DAS\\FK\\EntTest_dbchange20250611T140106\\FK\\20220821T183537Z\\FK256_T0_X768_E0.989032378690515_M407.0_m0.020220821T183537Z.png"
whale2 = "D:\\DAS\\FK\\EntTest_dbchange20250611T140106\\FK\\20220821T183537Z\\FK256_T0_X1536_E0.9886171781734272_M369.0_m0.020220821T183537Z.png"
white1 = "D:\\DAS\\FK\\EntTest_dbchange20250611T140106\\FK\\20220821T183537Z\\FK256_T0_X2048_E0.9890418998982868_M812.0_m0.020220821T183537Z.png"
white2 = "D:\\DAS\\FK\\EntTest_dbchange20250611T140106\\FK\\20220821T183537Z\\FK256_T0_X8448_E0.9892970304461952_M3644.0_m0.020220821T183537Z.png"

flist = [whale1,whale2,white1,white2]
n = 50
for f in flist:
    img = imageio.imread(f)
    dy, dx = np.gradient(img)
    hist, x_edges, y_edges = np.histogram2d(dx.ravel(),dy.ravel(),density=True, bins = 64)

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(img, cmap = 'gray')
    ax[1].imshow(hist, cmap = 'binary', extent = (-100,100,-100,100), origin = 'lower')
    ax[1].hlines(y = 0, xmin= -100, xmax = 100, color = 'red')
    ax[1].vlines(x = 0, ymin = -100, ymax = 100, color = 'red')
    plt.show()

    # ent = -1*np.sum(hist+1e-7 * np.log2(hist+1e-7))/np.log2(n*n)
    # print(ent)



# %%
