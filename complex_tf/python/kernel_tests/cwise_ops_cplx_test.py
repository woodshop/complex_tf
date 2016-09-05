"""Functional tests for coefficient-wise operations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class UnaryOpTest(tf.test.TestCase):

    def _compareCpu(self, x, np_func, tf_func):
        np_ans = np_func(x)
        with self.test_session(use_gpu=False):
            inx = tf.convert_to_tensor(x)
            y = tf_func(inx)
            tf_cpu = y.eval()
            self.assertShapeEqual(np_ans, y)
            self.assertAllClose(np_ans, tf_cpu)
            s = list(np.shape(x))
            jacob_t, jacob_n = tf.test.compute_gradient(inx,
                                                        s,
                                                        y,
                                                        s,
                                                        x_init_value=x)
            self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
          
    def _compareGpu(self, x, np_func, tf_func):
        np_ans = np_func(x)
        with self.test_session(use_gpu=True):
            inx = tf.convert_to_tensor(x)
            y = tf_func(inx)
            tf_gpu = y.eval()
            self.assertAllClose(np_ans, tf_gpu)
            s = list(np.shape(x))
            jacob_t, jacob_n = tf.test.compute_gradient(inx,
                                                        s,
                                                        y,
                                                        s,
                                                        x_init_value=x)
            self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
                
    def _compareBoth(self, x, np_func, tf_func):
        self._compareCpu(x, np_func, tf_func)
        self._compareGpu(x, np_func, tf_func)
                                          
        
    def testComplex64Basic(self):
        x = np.complex(1, 1) * np.arange(-3, 3).reshape(1, 3, 2).astype(
            np.complex64)
        y = x + 0.5  # no zeros
        self._compareBoth(x, np.tanh, tf.tanh)
    
if __name__ == "__main__":
    tf.test.main()
