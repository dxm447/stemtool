import numpy as np
import acc_image_utils as acc
import time
from mpi4py import MPI
import cupy as cp 

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
rank = comm.Get_rank()

t1 = time.time()
size=128
if rank != comm_size - 1:
  partition = size//comm_size 
else:
  partition = size//comm_size + (size%comm_size) 
shape = (partition,size,size,size)
n1 = np.zeros(shape, dtype=np.float32)
with cp.cuda.Device(rank%6): 
  n2 = acc.gpu_rot4D(n1,55)
t2 = time.time()
t = MPI.COMM_WORLD.allreduce((t2-t1), op=MPI.MAX)
if rank == 0:
  print("rank %d time: %f"%(rank, t))

