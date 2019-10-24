import time
import numpy as np
import cupy as cp
import acc_no_numba as acc
import ssb2 as ssb

ori_size = 64
new_size = 32
n1 = np.random.rand(ori_size,ori_size,ori_size,ori_size)
loop_tester = 1
t1 = time.time()
for _ in range(loop_tester):
    #real in 0,1 Q in 2,3
    n2 = acc.gpu_rot4D(n1,53)
    n2 = acc.cupy_resizer4D(n2,(new_size,new_size))
    n2 = acc.cupy_pad4D(n2,(ori_size,ori_size))
    n2 = cp.transpose(n2,(2,3,0,1)) #real in 2,3 
    n2 = cp.fft.fftshift(cp.fft.fft2(n2,axes=(-2,-1)),axes=(-2,-1)) #now real is Q' which is 2,3
    li,ri = ssb.ssb_kernel(n2,0.2,32,60)
t2 = time.time()
print((t2 - t1)/loop_tester)
