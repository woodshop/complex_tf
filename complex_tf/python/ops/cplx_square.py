from tensorflow.python.framework import ops
import tensorflow as tf
import complex_tf as ctf
_complex_tf_lib = ctf._complex_tf_lib

def cplx_square(x, name="CplxSquare"):
    """
    Squaresingle-precision complex numbers.
    """
    with ops.name_scope(name) as name:
        return _complex_tf_lib.cplx_square(x, name=name)

# @ops.RegisterGradient("CplxTanh")
# def _cplx_tanh_grad(op, grad):
#     """The gradients for `cplx_tanh`.

#   Args:
#     op: The `cplx_tanh` `Operation` that we are differentiating.
#     grad: Gradient with respect to the output of the `cplx_tanh` op.

#   Returns:
#     Gradients with respect to the input of `cplx_tanh`.
#   """
#     return 1. - tf.square(tf.conj(op.outputs[0]))
    
# @tf.RegisterShape("CplxTanh")
# def _zero_out_shape(op):
#     """
#     Shape function for the CplxTanh op.    
#     """
#     return [op.inputs[0].get_shape()]
