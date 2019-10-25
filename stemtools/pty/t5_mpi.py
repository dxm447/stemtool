import os
import time
import numpy as np

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

# manually initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
rank = comm.Get_rank()

# manually initialize GPU
import cupy as cp
deviceID = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
cp.cuda.Device(deviceID).use()
import numba.cuda
numba.cuda.select_device(deviceID)

print(rank, deviceID)
assert rank == deviceID

import acc_image_utils as acc


ori_size = 64
new_size = 32
#n1 = np.zeros((ori_size,ori_size,ori_size,ori_size),dtype=np.float32)
n1 = np.random.rand(ori_size,ori_size,ori_size,ori_size)
n1_flat = np.reshape(n1, (ori_size*ori_size, ori_size, ori_size))
partition = n1_flat.shape[0]//comm_size
istart = rank*partition
if rank != comm_size - 1:
  iend = rank*(partition+1)
else:
  iend = n1_flat.shape[0]
n1_rank = n1_flat[istart:iend,:,:]
n1_rank = np.expand_dims(n1_rank,axis=0)

loop_tester = 3
t1 = time.time()

#with cp.cuda.Device(deviceID):
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
for ii in range(loop_tester):
  n2 = acc.gpu_rot4D(n1_rank,53)
  n2 = acc.cupy_jit_resizer4D(n2,(new_size,new_size))
  n2 = acc.cupy_pad(n2,(ori_size,ori_size))
  # n2 = cp.fft.fftshift(cp.fft.fft2(n2,axes=(0,1)),axes=(0,1))
  # n2 = cp.transpose(n2,(2,3,0,1))
  # n2 = cp.fft.ifftshift(cp.fft.ifft2(n2,axes=(0,1)),axes=(0,1)) 
  # n2 = cp.transpose(n2,(2,3,0,1))
end.record()
end.synchronize()
t_cuda = cp.cuda.get_elapsed_time(start, end)

t2 = time.time()
t_total = MPI.COMM_WORLD.allreduce((t2-t1), op=MPI.MAX)/loop_tester
t_gpu = MPI.COMM_WORLD.allreduce(t_cuda, op=MPI.MAX)/loop_tester/1000
if rank == 0: 
  print("overall time: %f\ncuda time: %f"%(t_total, t_gpu))
