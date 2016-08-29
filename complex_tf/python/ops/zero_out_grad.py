from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
import tensorflow as tf

@ops.RegisterGradient("ZeroOut")
def _zero_out_grad(op, grad):
      """The gradients for `zero_out`.

  Args:
    op: The `zero_out` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `zero_out` op.

  Returns:
    Gradients with respect to the input of `zero_out`.
  """
      to_zero = op.inputs[0]
      shape = array_ops.shape(to_zero)
      n = tf.size(to_zero)
      first_grad = array_ops.reshape(grad, [-1])[0]
      index = array_ops.zeros_like(shape)
      to_zero_grad = sparse_ops.sparse_to_dense(0, [tf.size(to_zero),],
                                                first_grad, 0,
                                                validate_indices=True)
      to_zero_grad = array_ops.reshape(to_zero_grad, shape)
      return [to_zero_grad]  # List of one Tensor, since we have one input
