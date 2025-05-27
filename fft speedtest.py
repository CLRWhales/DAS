#%% this is a np.fft speed test

import numpy as np
import cProfile
from pstats import Stats
import timeit
test = np.random.randn(1024,1024)
t1 = timeit.repeat("np.fft.rfft(test,axis = 0)","import numpy as np; test = np.random.randn(512,512)",number=1000,repeat = 50)
t2 = timeit.repeat("np.fft.rfft(test,axis = 1)","import numpy as np; test = np.random.randn(512,512)",number=1000,repeat = 50)

m1 = np.mean(t1)/1000
m2 = np.mean(t2)/1000
print(m1)
print(m2)
# %%
# with cProfile.Profile() as pr:
#     tmp = np.fft.rfft(test,axis = 0)
#     pr.print_stats(sort='cumtime')

# with cProfile.Profile() as pr:
#     tmp = np.fft.rfft(test,axis = 1)
#     pr.print_stats(sort = 'cumtime')

    
# %%

# %%
