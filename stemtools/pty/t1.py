import time
import numpy as np
import acc_image_utils as acc

original_size = 50000
new_size = 20000
sigma = original_size/5
n1 = np.exp(-(((np.arange(original_size) - (0.5*original_size))/sigma)**2))
n2 = np.zeros(new_size)
t1 = time.time()
n2 = acc.cuda_resizer(n1,n2)
t2 = time.time()
print(t2 - t1)
