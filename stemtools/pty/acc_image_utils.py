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
    cupy_jit_resizer_gpu(cudat,N, cures)
    res = cp.asnumpy(cures)
    return res

def cupy_jit_resizer2D(data2D,new_size):
    size_y = int(new_size[0])
    size_x = int(new_size[1])
    cudat2D = cp.asarray(data2D)
    cures_y = cp.zeros((size_y,data2D.shape[1]),dtype=data2D.dtype) #first resize along y dim
    cures_f = cp.zeros((size_y,size_x),dtype=data2D.dtype) #now avlong both
    cupy_jit_2D_ydim(cudat2D,size_y,cures_y,data2D.shape[1])
    cupy_jit_2D_xdim(cures_y,size_x,cures_f,size_y)
    res2D = cp.asnumpy(cures_f)
    return res2D

def cupy_jit_resizer4D(data4D,resized_size,return_numpy=False):
    data4D = cp.asarray(data4D) 
    data_size = data4D.shape
    flattened_shape = (data_size[0]*data_size[1],data_size[2]*data_size[3])
    data4D_flatten = cp.reshape(data4D,flattened_shape)
    flat_res_shape = (data_size[0]*data_size[1],resized_size[0]*resized_size[1])
    flatres4D = cp.zeros(flat_res_shape,dtype=data4D.dtype)
    cupy_jit_2D_xdim(data4D_flatten,flat_res_shape[1],flatres4D,flat_res_shape[0])
    res4D = cp.reshape(flatres4D,(data_size[0],data_size[1],resized_size[0],resized_size[1]))
    if return_numpy:
       res4D = cp.asnumpy(res4D)
    return res4D

def cupy_pad(data4D,padded_size):
    data4D = cp.asarray(data4D)
    data_size = data4D.shape
    no_pixels = int(data_size[0]*data_size[1])
    reshaped_data4D_size = (no_pixels,data_size[2],data_size[3])
    padded_4D_size = (no_pixels,padded_size[0],padded_size[1])
    data4D = cp.reshape(data4D,reshaped_data4D_size)
    pad_4D = cp.zeros(padded_4D_size,dtype=data4D.dtype)
    raw_size = data_size[2:]
    pad_width = (0.5*(np.asarray(padded_size) - np.asarray(raw_size))).astype(int)
    cupy_jit_gpu_pad4D(data4D,pad_4D,pad_width,no_pixels)
    final_pad_size = (data_size[0],data_size[1],padded_size[0],padded_size[1])
    pad_4D = cp.reshape(pad_4D,final_pad_size)
    if return_numpy:
       pad_4D = cp.asnumpy(pad_4D)
    return pad_4D

def cupy_jit_gpu_pad4D(cudata4D_flat,cupad4D_flat,pad_width,no_pixels):
    for ii in range(no_pixels):
        cupad4D_flat[ii,:,:] = cp.pad(cudata4D_flat,((pad_width[0], pad_width[0]), (pad_width[1], pad_width[1])),mode='constant')
    
@numba.cuda.jit(device=True)
def cupy_jit_resizer_gpu(cudat,N,cures):
    M = cudat.size
    m_start = 0
    carry = 0
    data_sum = 0 
    for n in range(int(N)):
        data_sum = carry
        m_stop = int(math.floor((n + 1)*(M/N)))
        for ii in range(m_start,m_stop):
            data_sum += cudat[ii]
        m_start = m_stop
        carry = (m_stop-(n+1)*M/N)*cudat[m_stop-1]
        data_sum -= carry
        cures[n] = data_sum*(N/M)

@numba.cuda.jit
def cupy_jit_2D_ydim(cudat2D,size_y,cures_y,Nx):
    for ii in range(Nx):
        cupy_jit_resizer_gpu(cudat2D[:,ii],size_y,cures_y[:,ii])
    
@numba.cuda.jit
def cupy_jit_2D_xdim(cures_y,size_x,cures_f,size_y):
    for ii in range(size_y):
        cupy_jit_resizer_gpu(cures_y[ii,:],size_x,cures_f[ii,:])

def cu_rot(arr,angle):
    cu_arr = cp.asarray(arr)
    cu_rot = csnd.rotate(cu_arr,angle,reshape=False)
    rot_arr = cp.asnumpy(cu_rot)
    return rot_arr

def gpu_rot4D(data4D,rotangle,flip=True,return_numpy=False):
    data4D = cp.asarray(data4D,dtype=np.float16)
    if flip:
       data4D = cp.flip(data4D,axis=-1)
    data_shape = data4D.shape
    data4D = csnd.rotate(data4D.reshape(-1, data_shape[-2], data_shape[-1]), rotangle, axes=(1,2), reshape=False)
    if return_numpy:
       data4D = cp.asnumpy(data4D)        
    return data4D

