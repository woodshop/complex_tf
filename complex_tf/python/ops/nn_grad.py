from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

@ops.RegisterGradient("CplxL2Loss")
def _CplxL2LossGrad(op, grad):
      """Return the gradients for L2Loss.

  Args:
    op: The L2LossOp for which we need to generate gradients.
    grad: Tensor containing a single number.

  Returns:
    The gradient, which is (x * grad).
  """
      a = op.inputs[0]
      a_conj = math_ops.conj(op.inputs[0]) 
      return (0.5 * a * grad) + math_ops.conj(0.5 * a_conj * grad)
    
