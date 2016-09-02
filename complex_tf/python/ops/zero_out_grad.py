from tensorflow.python.framework import ops
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import sparse_ops
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
      shape = tf.shape(to_zero)
      n = tf.size(to_zero)
      first_grad = tf.reshape(grad, [-1])[0]
      grad = tf.zeros([n-1], dtype=tf.float32)
      grad = tf.concat(0, [[first_grad], grad])
      grad = tf.reshape(grad, shape)
      return [grad]
