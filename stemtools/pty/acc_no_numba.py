import math
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as csnd
import operator
import functools
import warnings

def cupy_resizer_gpu(cudat,cures):
    M = cudat.size
    N = cures.size
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

def cupy_resizer1D(data,N,return_numpy=True):   
    data = cp.asarray(data)
    cures = cp.zeros(int(N),dtype=data.dtype)
    cures = cupy_resizer_gpu(data,cures)
    if return_numpy:
       cures = cp.asnumpy(cures)
    return cures

def cupy_resizer2D(data2D,new_size,return_numpy=True):
    size_y = int(new_size[0])
    size_x = int(new_size[1])
    data2D = cp.asarray(data2D)
    cures_y = cp.zeros((size_y,data2D.shape[1]),dtype=data2D.dtype) #first resize along y dim
    cures_f = cp.zeros((size_y,size_x),dtype=data2D.dtype) #now along both
    cures_y = cupy_ydim_res_loop(data2D,cures_y,data2D.shape[1])
    cures_f = cupy_xdim_res_loop(cures_y,cures_f,size_y)
    if return_numpy:
       cures_f = cp.asnumpy(cures_f)
    return cures_f

def cupy_ydim_res_loop(cudat2D,cures_y,Nx):
    for ii in range(Nx):
        cupy_resizer_gpu(cudat2D[:,ii],cures_y[:,ii])
    
def cupy_xdim_res_loop(cures_y,cures_f,size_y):
    for ii in range(size_y):
        cupy_resizer_gpu(cures_y[ii,:],cures_f[ii,:])


def cupy_resizer4D(data4D,resized_size,return_numpy=False):
    data4D = cp.asarray(data4D) 
    data_size = data4D.shape
    flattened_shape = (data_size[0]*data_size[1],data_size[2]*data_size[3])
    data4D = cp.reshape(data4D,flattened_shape)
    flat_res_shape = (data_size[0]*data_size[1],resized_size[0]*resized_size[1])
    res4D = cp.zeros(flat_res_shape,dtype=data4D.dtype)
    res4D = cupy_xdim_res_loop(data4D,res4D,flat_res_shape[0])
    res4D = cp.reshape(res4D,(data_size[0],data_size[1],resized_size[0],resized_size[1]))
    if return_numpy:
       res4D = cp.asnumpy(res4D)
    return res4D

def cupy_pad4D(data4D,padded_size,return_numpy=False):
    data4D = cp.asarray(data4D)
    data_size = data4D.shape
    no_pixels = int(data_size[0]*data_size[1])
    reshaped_data4D_size = (no_pixels,data_size[2],data_size[3])
    padded_4D_size = (no_pixels,padded_size[0],padded_size[1])
    data4D = cp.reshape(data4D,reshaped_data4D_size)
    pad_4D = cp.zeros(padded_4D_size,dtype=data4D.dtype)
    pad_width = (0.5*(np.asarray(padded_size) - np.asarray(raw_size))).astype(int)
    for ii in range(no_pixels):
        cupad4D_flat[ii,:,:] = cp.pad(cudata4D_flat[ii,:,:],((pad_width[0], pad_width[0]), (pad_width[1], pad_width[1])),mode='constant')
    final_pad_size = (data_size[0],data_size[1],padded_size[0],padded_size[1])
    pad_4D = cp.reshape(pad_4D,final_pad_size)
    if return_numpy:
       pad_4D = cp.asnumpy(pad_4D)
    return pad_4D

def cu_rot(arr,angle):
    cu_arr = cp.asarray(arr)
    cu_rot = csnd.rotate(cu_arr,angle,reshape=False)
    rot_arr = cp.asnumpy(cu_rot)
    return rot_arr

def gpu_rot4D(data4D,rotangle,flip=True,return_numpy=False):
    warnings.filterwarnings('ignore')
    data4D = cp.asarray(data4D,dtype=data4D.dtype)
    if flip:
       data4D = cp.flip(data4D,axis=-1)
    data_shape = data4D.shape
    data4D = csnd.rotate(data4D.reshape(-1, data_shape[-2], data_shape[-1]), rotangle, axes=(1,2), reshape=False)
    data4D = cp.reshape(data4D,data_shape)
    if return_numpy:
       data4D = cp.asnumpy(data4D)        
    return data4D

