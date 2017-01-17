from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import complex_tf as ctf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


@ops.RegisterGradient("CplxMatMul")
def _CplxMatMulGrad(op, grad):
  inp0 = tf.conj(op.inputs[0])
  inp1 = tf.conj(op.inputs[1])
  t_a = op.get_attr("transpose_a")
  t_b = op.get_attr("transpose_b")
  if not t_a and not t_b:
    return (math_ops.matmul(
        grad, inp1, transpose_b=True), math_ops.matmul(
            inp0, grad, transpose_a=True))
  elif not t_a and t_b:
    return (math_ops.matmul(grad, inp1), math_ops.matmul(
        grad, inp0, transpose_a=True))
  elif t_a and not t_b:
    return (math_ops.matmul(
        inp1, grad, transpose_b=True),
            math_ops.matmul(inp0, grad))
  elif t_a and t_b:
    return (math_ops.matmul(
        inp1, grad, transpose_a=True, transpose_b=True),
            math_ops.matmul(
                grad, inp0, transpose_a=True, transpose_b=True))

