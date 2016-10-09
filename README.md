# Complex Tensorflow
This library extends functionality to several core Tensorflow ops. Many Tensorflow
ops do not support computing complex-valued dtypes on the GPU. In a few cases the
ops support complex-valued variables but they have not been registered and/or 
tested for complex dtypes. This is the case with ops such as "Sub" and "ZerosLike".

In other cases a kernel needs to be explicitly written for complex-valued computation.
The goal of this project is build up the capabilities of TensorFlow for complex dtypes.
When the repository has been sufficiently developed and tested, I'll likely port it and create a
a TF pull request.

## Building
This repository provides Makefiles that should compile a shared library named `complextf.so`.
You may find that some of the settings in `complex_tf\core\kernels\Makefile` need to be adjusted.
This repository was wiritten as a private repository for my particular environment:
  - Ubuntu 16.04
  - Cuda 8.0rc
  - gcc 5.4
  - bazel 0.3.1
  - Python 2

In order to make this library it is currently required to have the TensorFlow source files **and**
the python tensorflow python package installed. The Makefile looks for TF's python package `include` directory
by calling `tf.sysconfig.get_include()`. It relies on many of the headers in the source ditribution. 

## Contributions
Contributions are welcome! If you decide to add an op, please follow these steps:
  - Make sure that a complex-valued versin of yhe op is not already registered in the TF master branch
  - Investigate whether the op can just be "turned on", i.e. whether simply registering the op works
  - If not, write a kernel for the op.
  - Please add a few test cases for the op (and its gradient)
  
In adition, please feel free to make pull requests that will help configure the library more generally 
for other peoples' environments.

## Using
After building the library `pip install` the package. Then in python:
```python
import tensorflow as tf
import complex_tf as ctf
```
The shared new ops will be loaded as a plugin and should work when using Tensorflow.
