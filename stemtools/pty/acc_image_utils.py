import numba.cuda
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as csnd

def resizer(data,
            N):
    M = data.size
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
    return res

def cu_rot(arr,angle):
    cu_arr = cp.asarray(arr)
    cu_rot = csnd.rotate(cu_arr,angle,reshape=False)
    rot_arr = cp.asnumpy(cu_rot)
    return rot_arr

def gpu_rot4D(data4D,rotangle):
    cu4D = cp.asarray(data4D)
    data_shape = np.shape(data4D)
    rotcu4D = cp.zeros_like(cu4D)
    for ii in arange(data_shape[0]):
        for jj in arange(data_shape[1]):
            rotcu4D[ii,jj,:,:] = csnd.rotate(cu4D[ii,jj,:,:],rotangel,reshape=False)
    rot4D = cp.asnumpy(rotcu4D)
    return rot4D

def gpu_rot4D(data4D,rotangle):
    cu4D = cp.asarray(data4D)
    data_shape = np.shape(data4D)
    #rotcu4D = cp.zeros_like(cu4D)
    for itm in cu4D.reshape(-1, data_shape[-2], data_shape[-1]):
            csnd.rotate(itm, rotangle,reshape=False)
    rot4D = cp.asnumpy(cu4D)
    return rot4D
