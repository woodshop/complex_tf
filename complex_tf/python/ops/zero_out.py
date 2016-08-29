from tensorflow.python.framework import ops
import tensorflow as tf
import complex_tf as ctf
_complex_tf_lib = ctf._complex_tf_lib

def zero_out(x, name="ZeroOut"):
    """
    Zero out op for testing tensorflow op customization.b
    """
    with ops.name_scope(name) as name:
        return _complex_tf_lib.zero_out(x, name=name)

@tf.RegisterShape("ZeroOut")
def _zero_out_shape(op):
    """
    Shape function for the ZeroOut op.
    
    This is the unconstrained version of ZeroOut, which produces an output
    with the same shape as its input.
    """
    return [op.inputs[0].get_shape()]
