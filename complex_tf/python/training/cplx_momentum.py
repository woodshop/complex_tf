# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Momentum for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.training.momentum import MomentumOptimizer
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.framework import dtypes


class CplxMomentumOptimizer(MomentumOptimizer):
  """Overrides tf.train.MomentumOptimizer to support complex values.
  """

  def __init__(self, learning_rate, momentum,
               use_locking=False, name="CplxMomentum", use_nesterov=False):
    super(CplxMomentumOptimizer, self).__init__(learning_rate, momentum,
                                                use_locking, name, use_nesterov)
    
  def _apply_sparse(self, grad, var):
    raise NotImplementedError

  def _valid_dtypes(self):
    """Valid types for loss, variables and gradients.

    Subclasses should override to allow other float types.

    Returns:
      Valid types for loss, variables and gradients.
    """
    valid_dtypes = super(CplxMomentumOptimizer, self)._valid_dtypes()
    valid_dtypes.add(dtypes.complex64)
    return valid_dtypes
