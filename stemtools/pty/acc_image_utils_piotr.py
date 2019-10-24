import math
import numpy as np
import cupy as cp
import numba.cuda

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
def cupy_jit_2D_xdim(cures_y,size_x,cures_f,size_y):
    for ii in range(numba.cuda.grid(1), size_y, numba.cuda.gridsize(1)):
        cupy_jit_resizer_gpu(cures_y[ii,:],size_x,cures_f[ii,:])

def cupy_jit_resizer4D(data4D,resized_size,return_numpy=False):
    data_size = np.shape(data4D)
    flattened_shape = (data_size[0]*data_size[1],data_size[2]*data_size[3])
    data4D_flatten = cp.reshape(cp.asarray(data4D),flattened_shape)
    flat_res_shape = (data_size[0]*data_size[1],resized_size[0]*resized_size[1])
    flatres4D = cp.zeros(flat_res_shape,dtype=data4D.dtype)
    cupy_jit_2D_xdim[1,32](data4D_flatten,flat_res_shape[1],flatres4D,flat_res_shape[0]) # call numba.cuda kernel with 1 block and 32 threads in that block
    res4D = cp.reshape(flatres4D,(data_size[0],data_size[1],resized_size[0],resized_size[1]))
    if return_numpy:
       res4D = cp.asnumpy(res4D)
    return res4D
