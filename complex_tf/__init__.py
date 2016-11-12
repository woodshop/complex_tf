import tensorflow as tf
import os

path = (os.path.dirname(os.path.realpath(__file__)) +
        '/core/kernels/complextf.so')
_mod = tf.load_op_library(path)
from .python import training as train
from .python.ops import nn
