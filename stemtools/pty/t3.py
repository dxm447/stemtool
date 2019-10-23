import time
import numpy as np
import acc_image_utils as acc

original_size = 50000
new_size = 20000
sigma = original_size/3
n1 = np.exp(-(((np.arange(original_size) - (0.5*original_size))/sigma)**2))
loop_tester = 25
t1 = time.time()
for ii in range(loop_tester):
    n2 = acc.cupy_resizer(n1,new_size)
t2 = time.time()
print((t2 - t1)/loop_tester)
