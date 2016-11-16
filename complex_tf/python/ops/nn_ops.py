from tensorflow.python.framework import ops
from tensorflow.python.framework import common_shapes

ops.RegisterShape("CplxL2Loss")(common_shapes.call_cpp_shape_fn)
