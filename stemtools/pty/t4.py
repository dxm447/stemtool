import time
import numpy as np
import acc_image_utils as acc

original_size = 50000
new_size = (20000,15000)
sigma = original_size/3
n1y,n1x = np.mgrid[0:original_size,0:original_size]
n1 = (((n1y - (0.5*original_size))/sigma)**2) + (((n1x - (0.5*original_size))/sigma)**2)
n1 = np.exp(-n1)
loop_tester = 2
t1 = time.time()
for ii in range(loop_tester):
    n2 = acc.cupy_jit_resizer2D(n1,new_size)
t2 = time.time()
print((t2 - t1)/loop_tester)
