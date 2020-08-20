# Copyright (c) 2020, Stephen Polcyn. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# hip_util.py
# Provides various Python utility methods used by other files

import numpy as np

import tensor_pb2

def make_tensor_from_numpy(array):
    assert isinstance(array, np.ndarray), "Argument must be a numpy ndarray"

    tensor = tensor_pb2.Tensor()
    tensor.version = 1
    tensor.dimensions.extend(list(array.shape))
    tensor.data.extend(array.flatten())

    return tensor
