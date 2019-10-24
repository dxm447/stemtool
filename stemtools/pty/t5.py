import time
import numpy as np
import cupy as cp
import acc_image_utils as acc

ori_size = 64
new_size = 32
#n1 = np.zeros((ori_size,ori_size,ori_size,ori_size),dtype=np.float32)
n1 = np.random.rand(ori_size,ori_size,ori_size,ori_size)
loop_tester = 3
t1 = time.time()
for ii in range(loop_tester):
    n2 = acc.gpu_rot4D(n1,53)
    n2 = acc.cupy_jit_resizer4D(n2,(new_size,new_size))
    n2 = acc.cupy_pad(n2,(ori_size,ori_size))
    n2 = cp.fft.fftshift(cp.fft.fft2(n2,axes=(0,1)),axes=(0,1))
    n2 = cp.transpose(n2,(2,3,0,1))
    n2 = cp.fft.ifftshift(cp.fft.ifft2(n2,axes=(0,1)),axes=(0,1)) 
    n2 = cp.transpose(n2,(2,3,0,1))
t2 = time.time()
print((t2 - t1)/loop_tester)
