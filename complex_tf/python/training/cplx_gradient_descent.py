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

"""GradientDescent for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow.python.framework import dtypes

class CplxGradientDescentOptimizer(GradientDescentOptimizer):
  """Overrides tf.train.GradientDescentOptimizer to support complex values.
  """

  def _apply_sparse(self, grad, var):
    raise NotImplementedError

  def _valid_dtypes(self):
    """Valid types for loss, variables and gradients.

    Subclasses should override to allow other float types.

    Returns:
      Valid types for loss, variables and gradients.
    """
    valid_dtypes = super(CplxGradientDescentOptimizer, self)._valid_dtypes()
    valid_dtypes.add(dtypes.complex64)
    return valid_dtypes
              
