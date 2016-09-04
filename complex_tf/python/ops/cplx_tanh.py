from tensorflow.python.framework import ops
import tensorflow as tf
import complex_tf as ctf
_complex_tf_lib = ctf._complex_tf_lib

def cplx_tanh(x, name="CplxTanh"):
    """
    Hyperbolic tangent for single-preceision cmplex numbers.
    """
    with ops.name_scope(name) as name:
        return _complex_tf_lib.cplx_tanh(x, name=name)

@tf.RegisterShape("CplxTanh")
def _zero_out_shape(op):
    """
    Shape function for the CplxTanh op.    
    """
    return [op.inputs[0].get_shape()]
