import time
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
rank = comm.Get_rank()
import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

import numba.cuda
import cupy as cp
import acc_image_utils as acc
from cupy import cuda

try: 
  #deviceID = numba.cuda.get_current_device().id
  #print(rank, deviceID)
  numba.cuda.get_current_device().reset()
  print(rank, "device reset")
except:
  deviceID = rank

comm.barrier()

numba.cuda.select_device(rank%6)
deviceID = numba.cuda.get_current_device().id
print(rank, deviceID)

#for r in range(comm_size):
#  print(comm_size, rank, r, file=open(f"out{rank}.txt", "w"))
#  numba.cuda.select_device(r)

#numba.cuda.select_device(rank%6)

#raise SystemExit

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
start = cuda.Event()
end = cuda.Event()
start.record()
with cp.cuda.Device(0):
  for ii in range(loop_tester):
    n2 = acc.gpu_rot4D(n1_rank,53)
    n2 = acc.cupy_jit_resizer4D(n2,(new_size,new_size))
    n2 = acc.cupy_pad(n2,(ori_size,ori_size))
#    n2 = cp.fft.fftshift(cp.fft.fft2(n2,axes=(0,1)),axes=(0,1))
#    n2 = cp.transpose(n2,(2,3,0,1))
#    n2 = cp.fft.ifftshift(cp.fft.ifft2(n2,axes=(0,1)),axes=(0,1)) 
#    n2 = cp.transpose(n2,(2,3,0,1))
end.record()
end.synchronize()
t_cuda = cuda.get_elapsed_time(start, end)
t2 = time.time()
t_total = MPI.COMM_WORLD.allreduce((t2-t1), op=MPI.MAX)/loop_tester
t_gpu = MPI.COMM_WORLD.allreduce(t_cuda, op=MPI.MAX)/loop_tester/1000
if rank == 0: 
  print("overall time: %f\ncuda time: %f"%(t_total, t_gpu))

