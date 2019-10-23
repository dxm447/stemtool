import math
import numba
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as csnd
import operator
import functools
from numba import cuda
import numba.cuda
numba.cuda.select_device(3)


def cupy_resizer(data,N):   
    cudat = cp.asarray(data)
    M = cudat.size
    cures = cp.zeros(int(N),dtype=data.dtype)
    carry=cp.zeros(1,dtype=data.dtype)
    m_start = 0
    #m_array = cp.floor((M/N)*cp.arange(N+1)) 
    data_sum = cp.zeros(1,dtype=data.dtype)
    for n in range(int(N)):
        data_sum[0] = carry[0]
        m_stop = np.floor((n + 1)*(M/N))
        data_sum[0] = data_sum[0] + cp.sum(cudat[m_start:m_stop])
        m_start = m_stop
        carry[0] = (m_stop-(n+1)*M/N)*cudat[m_stop-1]
        data_sum[0] = data_sum[0] - carry[0]
        cures[n] = data_sum[0]*N/M
    res = cp.asnumpy(cures)
    return res

def cupy_jit_resizer(data,N):   
    cudat = cp.asarray(data)
    cures = cp.zeros(int(N),dtype=data.dtype)
    carry=cp.zeros(1,dtype=data.dtype)
    data_sum = cp.zeros(1,dtype=data.dtype)
    cupy_jit_resizer_gpu(cudat,N, cures, carry, data_sum)
    res = cp.asnumpy(cures)
    return res

@numba.vectorize([numba.int64(numba.float64)])
def floor_jit(x):
    rx = round(x)
    if rx > x:
       fx = rx - 1
    else:
       fx = rx
    return fx

@numba.vectorize([numba.int64(numba.float64)])
def sum_jit(x):
    rx = round(x)
    if rx > x:
       fx = rx - 1
    else:
       fx = rx
    return fx

@numba.cuda.jit
def cupy_jit_resizer_gpu(cudat,N,cures, carry, data_sum):
    M = cudat.size
    m_start = 0
    #m_array = cp.floor((M/N)*cp.arange(N+1)) 
    for n in range(int(N)):
        data_sum[0] = carry[0]
        m_stop = int(math.floor((n + 1)*(M/N)))
        for ii in range(m_start,m_stop):
            data_sum[0] += cudat[ii]
        #data_sum[0] = data_sum[0] + functools.reduce(operator.add,list(cudat[m_start:m_stop]),0,0)
        m_start = m_stop
        carry[0] = (m_stop-(n+1)*M/N)*cudat[m_stop-1]
        data_sum[0] = data_sum[0] - carry[0]
        cures[n] = data_sum[0]*N/M


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
    # for itm in cu4D.reshape(-1, data_shape[-2], data_shape[-1]):
           # csnd.rotate(itm, rotangle,reshape=False)
    #rot4D = cp.asnumpy(csnd.rotate(cu4D.reshape(-1, data_shape[-2], data_shape[-1]), rotangle, reshape=False))
    rot4D = cp.asnumpy(csnd.rotate(cu4D.reshape(-1, data_shape[-2], data_shape[-1]), rotangle, axes=(1,2), reshape=False))        
    return rot4D

