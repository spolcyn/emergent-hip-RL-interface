import numpy as np
import random
import time

from ctypes import *

import numpy.ctypeslib as npct

lib = CDLL("iwhipmodel.so")
print(lib)

t = time.monotonic()
sum = 0
for i in range(1000000):
    sum += lib.add(random.randint(0,10), random.randint(0,10))

print("Elapsed time", time.monotonic() - t)

#getattr(lib, "ParseTensorFromJSON")

