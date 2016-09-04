import tensorflow as tf
import os

path = os.path.dirname(os.path.realpath(__file__)) + '/core/ops/complextf.so'
_complex_tf_lib = tf.load_op_library(path)

from python.ops import *
