import numba
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as csnd
from numba import cuda
import numba.cuda
numba.cuda.select_device(3)

@numba.cuda.jit
def cuda_resizer(data,res):   
    M = data.size
    N = res.size
    carry=0
    m=0
    for n in range(int(N)):
        data_sum = carry
        while m*N - n*M < M :
            data_sum += data[m]
            m += 1
        carry = (m-(n+1)*M/N)*data[m-1]
        data_sum -= carry
        res[n] = data_sum*N/M

def cu_rot(arr,angle):
    cu_arr = cp.asarray(arr)
    cu_rot = csnd.rotate(cu_arr,angle,reshape=False)
    rot_arr = cp.asnumpy(cu_rot)
    return rot_arr

def gpu_rot4D(data4D,rotangle,flip=True):
    cu4D = cp.asarray(data4D,dtype=np.float16)
    if flip:
       cu4D = cp.flip(cu4D,axis=-1)
    data_shape = np.shape(data4D)
    for itm in cu4D.reshape(-1, data_shape[-2], data_shape[-1]):
            csnd.rotate(itm, rotangle,reshape=False)
    rot4D = cp.asnumpy(cu4D)
    return rot4D
