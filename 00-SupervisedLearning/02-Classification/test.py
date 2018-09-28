import numpy as np
import numdifftools as nd 
from math import log

a = log(0+0.00000001)
print(a)
arr = np.array([[0.6], [0.8], [0.3]])
print(arr[1])
arr = arr > 0.5
print(arr)