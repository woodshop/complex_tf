# Complex Tensorflow
This library extends functionality to several Tensorflow ops. Most Tensorflow
ops do not support processing complex-valued dtypes on the GPU. This library
attempts to rectify this by registering GPU ops for the `complex64` dtype. The
library relies on the header files of [Pycuda](https://github.com/inducer/pycuda)
for basic complex-valued GPU ops.

The code should be compiled against Tensorflow's bleeding edge source. As of
this writing, the newest official TF release has hidden bugs affecting the
proper computation of gradients in the complex domain. A [recent commit](https://github.com/tensorflow/tensorflow/commit/821063df9f0e6a0eec8cb78cb0ddc5c5b2b91b9f)
addresses these problems.

## Notes
Since this project is currently private, I'll keep notes here related to building.

### ZerosLike
The bleeding-edge Tensorflow distrobution appears to support, but not register 
a "ZerosLike" op for `complex64`. I've registered the op in my TF branch. I haven't written any
unit tests for this dtype. "ZerosLike" is necessary for computing the gradient of "Pow".
