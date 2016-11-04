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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.rnn_cell import BasicRNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.util import nest
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

class CplxBasicRNNCell(BasicRNNCell):

  def __init__(self, num_units, input_size=None, activation=tanh,
               matrix_initializer=None, bias_initializer=None):
    super(CplxBasicRNNCell, self).__init__(num_units, input_size, activation)
    self._matrix_init = matrix_initializer
    self._bias_init = bias_initializer

  def __call__(self, inputs, state, scope=None):
    """Basic RNN: output = new_state = activation(W * input + U * state + B).
    """
    with vs.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
      output = self._activation(_linear([inputs, state], self._num_units, True,
                                        matrix_init=self._matrix_init,
                                        bias_init=self._bias_init))
    return output, output


def _linear(args, output_size, bias, bias_start=0.0, scope=None,
            matrix_init=None, bias_init=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with vs.variable_scope(scope or "Linear", dtype=dtype):
    shape = [total_arg_size, output_size] if matrix_init is None else None
    matrix = vs.get_variable(
        "Matrix", shape=shape, dtype=dtype,
      initializer=matrix_init)
    if len(args) == 1:
      res = math_ops.matmul(args[0], matrix)
    else:
      res = math_ops.matmul(array_ops.concat(1, args), matrix)
    if not bias:
      return res
    if bias_init is None:
      shape = [output_size]
      bias_init=init_ops.constant_initializer(bias_start, dtype=dtype)
    else:
      shape = None
    bias_term = vs.get_variable(
        "Bias", shape=shape,
        dtype=dtype,
        initializer=bias_init)
  return res + bias_term