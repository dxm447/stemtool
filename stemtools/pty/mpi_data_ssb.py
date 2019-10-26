import numpy as np
import os, sys
import cupy as cp
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()
gpu_id = comm_rank # 1 mpi rank <--> 1 gpu, single process, single device

def get_chunk(data_shape, comm_size):
    """Returns list of slice objects used to read parts of the data
    
    Arguments:
        data_shape {tuple} -- shape of data
        comm_size {int} -- MPI communicator size
    
    Returns:
        list -- list of slice objects
    """
    chunks = []
    chunk_size = data_shape[0] // comm_size
    for itm in range(comm_size):
        partition = slice(itm * chunk_size, None) if itm == comm_size - 1 else slice(itm * chunk_size, (itm + 1) * chunk_size)
        chunks.append(partition)
    return chunks

def get_gatherv_args(data_shape, chunks, xy_offset=1, mpi_dtype=MPI.C_FLOAT_COMPLEX, div=1, debug=False):
    """Constructs args to an MPI-Gatherv call
    
    Arguments:
        data_shape {tuple} -- shape of data 
        chunks {list} -- list of slice objects defining the data partitions 
    
    Keyword Arguments:
        xy_offset {int} -- value to offset the displacement when sending 3D arrays (default: {1})
        mpi_dtype {MPI.Type} -- type of the MPI message (default: {MPI.C_FLOAT_COMPLEX})
    
    Returns:
        [count, displ, mpi_dtype] -- count, displacement, and MPI type for the recv buff argument. 
        ex: MPI.COMM_WORLD.Gatherv(sendbuff, [recvbuff, count, displ, mpi_dtype], root=0)
    """
    count = []
    for chunk in chunks:
        start = chunk.start if chunk.start is not None else 0
        stop = chunk.stop if chunk.stop is not None else data_shape[0] * data_shape[1]
        count.append(stop - start)
    count = np.array(count)//div
    displ = np.cumsum(count)
    displ = np.append(displ[::-1], 0)[::-1]
    xy_offset = np.prod(xy_offset)
    count *= xy_offset
    displ *= xy_offset 
    count = count.astype(np.int)
    displ = displ.astype(np.int)
    if debug:
        print_rank("Count:{}, Displ:{}".format(count, displ))
    return [count, displ[:-1], mpi_dtype]

def print_rank(*args, **kwargs):
    """print_rank [summary]
    """
    if comm_rank == 0:
        print(*args, **kwargs)

def get_data(data_path, data_range=32):
    """get_data [loads memory-mapped 4-D numpy array or generates new one]
    
    Args:
        data_path ([type]): [description]
        data_range (int, optional): [description]. Defaults to 32.
    
    Returns:
        [type]: [description]
    """
    if not os.path.exists(data_path):
        print_rank("data path does not exist... will return dummy array (32,32,32,32)!")
        data_shape = (data_range,) * 4 
        data = np.zeros(data_shape, dtype=np.float32)
    else:
        data = np.load(data_path, mmap_mode='r')
    return data

def send_call(data_shape, chunks, mpi_comm, send_buff=None, recv_buff=None, root=0, mpi_dtype=MPI.C_FLOAT_COMPLEX):
    recv_buff_args = get_gatherv_args(data_shape, chunks, xy_offset=(data_shape[-1], data_shape[-2]), mpi_dtype=mpi_dtype)
    count = recv_buff_args[0][0]
    if count > 2**30:
        # make custom MPI_TYPE
        TYPE_2BINT = MPI.Datatype(MPI.FLOAT)
        TYPE_2BINT.Create_contiguous(min(count, 2**30))
        TYPE_2BINT.Commit()
        recv_buff_args[-1] = TYPE_2BINT
        recv_buff_args[1] /= count // 2**30
        send_buff_args = [send_buff, count // 2**30, TYPE_2BINT]
        mpi_comm.Gatherv(send_buff, recv_buff_args, root=root)

def process_data(data_path, debug=True):
    """
    Main function to process data.
    """
    #0. load data [x, ,y, qx, qy]
    data = get_data(data_path)
    data_shape = data.shape
    if debug:
        print_rank("Total data shape:", data.shape)
    comm.Barrier()

    #1. partition data per mpi rank over x-axis
    chunks = get_chunk(data.shape, comm_size)
    data = data[chunks[comm_rank]]
    if debug:
        print("rank=%d, read data shape=" %comm_rank, data.shape)

    #2. each mpi-rank (gpu) works on a partition [x', y, qx, qy] 
    with cp.cuda.Device(gpu_id):
        if isinstance(data, np.ndarray):
            data = cp.asarray(data)
        # i.  Do preprocessing on [x', y, qx, qy]
        data = data
        # ii. Transpose to [qx, qy, x' , y]
        data = data.transpose((2,3,0,1))
        # iii.Reshape to [qx * qy, x', y]
        data = data.reshape(-1, data.shape[-2], data.shape[-1])
        # iv. batched 2D-FFT over (x',y)
        data = cp.fft.fft2(data)
        # v. kernel element-wise mult 
        kernel = cp.ones_like(data)
        data *= kernel
        # vi. Reduce-sum over (qx, qy) to get a 2-d tile [x', y]
        data = data.sum(axis=0)

    # going back to host memory
    send_proc_data = cp.asnumpy(data)
    if debug:
        print("rank=%d, processed data shape=" %comm_rank, send_proc_data.shape)

    # 3. Gatherv all results
    recv_proc_data = np.empty((data_shape[0], data_shape[1]), dtype=np.complex64) if comm_rank == 0 else None 
    recv_buff_args = get_gatherv_args(data_shape, chunks, xy_offset=data_shape[0], mpi_dtype=MPI.C_FLOAT_COMPLEX)
    recv_buff_args = [recv_proc_data] + recv_buff_args
    # send_call(data_shape, chunks, comm, send_buff=chunk_proc_data, recv_buff=recv_proc_data, root=0)
    comm.Gatherv(send_proc_data, recv_buff_args, root=0)

    #4. reshape
    comm.Barrier()
    if comm_rank == 0:
        print_rank("Final data shape:", recv_proc_data.shape)
        if debug:
            pass
            print_rank("results=", recv_proc_data.sum())

    MPI.Finalize()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_data(sys.argv[-1], debug=True)
    else:
        process_data('dummy_path', debug=True)