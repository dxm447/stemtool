import time
import numpy as np
import acc_image_utils_piotr as acc

ori_size = 64
new_size = 32
n1 = np.zeros((ori_size,ori_size,ori_size,ori_size),dtype=np.float32)
loop_tester = 3
tsum = 0.0
for ii in range(loop_tester):
    t1 = -time.time()
    n2 = acc.cupy_jit_resizer4D(n1,(new_size,new_size))
    t1 += time.time()
    tsum += t1
    print(ori_size,"->",new_size,"t=",t1,"tavg=",tsum/(ii+1))
