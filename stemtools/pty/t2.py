import numpy as np
import acc_image_utils as acc
import time

t1 = time.time()
size=128
shape = (size,size,size,size)
n1 = np.zeros(shape, dtype=np.float32)
n2 = acc.gpu_rot4D(n1,55)
t2 = time.time()
print(t2 - t1)

