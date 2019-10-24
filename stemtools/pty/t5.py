import time
import numpy as np
import cupy as cp
import acc_image_utils as acc

ori_size = 96
new_size = 32
#n1 = np.zeros((ori_size,ori_size,ori_size,ori_size),dtype=np.float32)
n1 = np.random.rand(ori_size,ori_size,ori_size,ori_size)
loop_tester = 4
t1 = time.time()
for ii in range(loop_tester):
    #real in 0,1 Q in 2,3
    n2 = acc.gpu_rot4D(n1,53)
    n2 = acc.cupy_jit_resizer4D(n2,(new_size,new_size))
    n2 = acc.cupy_pad(n2,(ori_size,ori_size))
    n2 = cp.transpose(n2,(2,3,0,1)) #real in 2,3 
    n2 = cp.fft.fftshift(cp.fft.fft2(n2,axes=(-2,-1)),axes=(-2,-1)) #now real is Q' which is 2,3
    G = cp.asnumpy(n2)
    n2 = cp.transpose(n2,(2,3,0,1)) #move original Q to 2,3 and Q' to 0,1
    n2 = cp.fft.ifftshift(cp.fft.ifft2(n2,axes=(-2,-1)),axes=(-2,-1)) #now 0,1 is Q' and 2,3  is R' 
    n2 = cp.transpose(n2,(2,3,0,1)) #0,1 is R' and 2,3 is Q'
    H = cp.asnumpy(n2)
t2 = time.time()
print((t2 - t1)/loop_tester)
