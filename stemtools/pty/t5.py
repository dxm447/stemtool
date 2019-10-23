import time
import numpy as np
import acc_image_utils as acc

ori_size = 64
new_size = 32
n1 = np.zeros((ori_size,ori_size,ori_size,ori_size),dtype=np.float32)
loop_tester = 3
t1 = time.time()
for ii in range(loop_tester):
    n2 = acc.cupy_jit_resizer4D(n1,(new_size,new_size))
t2 = time.time()
print((t2 - t1)/loop_tester)
