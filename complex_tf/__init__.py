import tensorflow as tf
import os

path = (os.path.dirname(os.path.realpath(__file__)) +
        '/core/kernels/complextf.so')
tf.load_op_library(path)
import python.training as train
import python.ops.nn as nn
