"""Functional tests for coefficient-wise operations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import complex_tf as ctf

class UnaryOpTest(tf.test.TestCase):

    def _compareGpu(self, x, np_func, tf_func):
        np_ans = np_func(x)
        inx = tf.Variable(x)
        with self.test_session(use_gpu=True, force_gpu=True):
            tf.initialize_all_variables().run()
            y = tf_func(inx)
            tf_gpu = y.eval()
            self.assertAllClose(np_ans, tf_gpu)
                
    def _compareGpuGrad(self, x, np_func, tf_func):
        np_ans = np_func(x)
        inx = tf.Variable(x)
        with self.test_session(use_gpu=True, force_gpu=True):
            tf.initialize_all_variables().run()
            y = tf_func(inx)
            s = list(np.shape(x))
            jacob_t, jacob_n = tf.test.compute_gradient(inx,
                                                        s,
                                                        y,
                                                        s,
                                                        x_init_value=x)
            self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
                
    def testCplxTanhGPU(self):
        shapes = [(5,4,3), (5,4), (5,), (1,)]
        for sh in shapes:
            x = ((np.random.randn(*sh) +
                  1j*np.random.randn(*sh)).astype(np.complex64))
            self._compareGpu(x, np.tanh, tf.tanh)
                  
    def testCplxTanhGradGPU(self):
        shapes = [(5,4,3), (5,4), (5,), (1,)]
        for sh in shapes:
            x = ((np.random.randn(*sh) +
                  1j*np.random.randn(*sh)).astype(np.complex64))
            self._compareGpuGrad(x, np.tanh, tf.tanh)
                      
class BinaryOpTest(tf.test.TestCase):

    def _compareGpu(self, x, y, np_func, tf_func):
        np_ans = np_func(x, y)
        inx = tf.Variable(x)
        iny = tf.Variable(y)
        with self.test_session(use_gpu=True, force_gpu=True):
            tf.initialize_all_variables().run()
            z = tf_func(inx, iny)
            tf_gpu = z.eval()
            self.assertAllClose(np_ans, tf_gpu)
                
    def _compareGpuGrad(self, x, y, np_func, tf_func):
        np_ans = np_func(x, y)
        inx = tf.Variable(x)
        iny = tf.Variable(y)
        with self.test_session(use_gpu=True, force_gpu=True):
            tf.initialize_all_variables().run()
            z = tf_func(inx, iny)
            s1 = list(np.shape(x))
            s2 = list(np.shape(np_ans))
            jacob_t, jacob_n = tf.test.compute_gradient(inx,
                                                        s1,
                                                        z,
                                                        s2,
                                                        x_init_value=x)
            self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
                
    def testCplxAddGPU(self):
        shapes1 = [(5,4,3), (5,4), (5,), (5,4), (1,)]
        shapes2 = [(5,4,3), (5,4), (5,), (1,), (5,4)]
        for [sh0, sh1] in zip(shapes1, shapes2):
            x = ((np.random.randn(*sh0) +
                  1j*np.random.randn(*sh0)).astype(np.complex64))
            y = ((np.random.randn(*sh1) +
                  1j*np.random.randn(*sh1)).astype(np.complex64))
            self._compareGpu(x, y, np.add, tf.add)
                  
    def testCplxAddGradGPU(self):
        shapes1 = [(5,4,3), (5,4), (5,), (5,4), (1,)]
        shapes2 = [(5,4,3), (5,4), (5,), (1,), (5,4)]
        for [sh0, sh1] in zip(shapes1, shapes2):
            x = ((np.random.randn(*sh0) +
                  1j*np.random.randn(*sh0)).astype(np.complex64))
            y = ((np.random.randn(*sh1) +
                  1j*np.random.randn(*sh1)).astype(np.complex64))
            self._compareGpuGrad(x, y, np.add, tf.add)
                      
    def testCplxSubGPU(self):
        shapes1 = [(5,4,3), (5,4), (1,), (5,)]
        shapes2 = [(5,4,3), (1,), (5,4), (5,)]
        for [sh0, sh1] in zip(shapes1, shapes2):
            x = ((np.random.randn(*sh0) +
                  1j*np.random.randn(*sh0)).astype(np.complex64))
            y = ((np.random.randn(*sh1) +
                  1j*np.random.randn(*sh1)).astype(np.complex64))
            self._compareGpu(x, y, np.subtract, tf.sub)
                  
    #### Depends on Neg
    # def testCplxSubGradGPU(self):
    #     shapes1 = [(5,4,3), (5,4), (1,), (5,)]
    #     shapes2 = [(5,4,3), (1,), (5,4), (5,)]
    #     for [sh0, sh1] in zip(shapes1, shapes2):
    #         x = ((np.random.randn(*sh0) +
    #               1j*np.random.randn(*sh0)).astype(np.complex64))
    #         y = ((np.random.randn(*sh1) +
    #               1j*np.random.randn(*sh1)).astype(np.complex64))
    #         self._compareGpuGrad(x, y, np.subtract, tf.sub)
                      
    def testCplxMulGPU(self):
        shapes1 = [(5,4,3), (5,4), (1,), (5,)]
        shapes2 = [(5,4,3), (1,), (5,4), (5,)]
        for [sh0, sh1] in zip(shapes1, shapes2):
            x = ((np.random.randn(*sh0) +
                  1j*np.random.randn(*sh0)).astype(np.complex64))
            y = ((np.random.randn(*sh1) +
                  1j*np.random.randn(*sh1)).astype(np.complex64))
            self._compareGpu(x, y, np.multiply, tf.mul)
                  
    def testCplxMulGradGPU(self):
        shapes1 = [(5,4,3), (5,4), (1,), (5,)]
        shapes2 = [(5,4,3), (1,), (5,4), (5,)]
        for [sh0, sh1] in zip(shapes1, shapes2):
            x = ((np.random.randn(*sh0) +
                  1j*np.random.randn(*sh0)).astype(np.complex64))
            y = ((np.random.randn(*sh1) +
                  1j*np.random.randn(*sh1)).astype(np.complex64))
            self._compareGpuGrad(x, y, np.multiply, tf.mul)
                      
    def testCplxDivGPU(self):
        shapes1 = [(5,4,3), (5,4), (1,), (5,)]
        shapes2 = [(5,4,3), (1,), (5,4), (5,)]
        for [sh0, sh1] in zip(shapes1, shapes2):
            x = ((np.random.randn(*sh0) +
                  1j*np.random.randn(*sh0)).astype(np.complex64))
            y = ((np.random.randn(*sh1) +
                  1j*np.random.randn(*sh1)).astype(np.complex64))
            self._compareGpu(x, y, np.divide, tf.div)
                  
    #### Depends on Square
    # def testCplxDivGradGPU(self):
    #     shapes1 = [(5,4,3), (5,4), (1,), (5,)]
    #     shapes2 = [(5,4,3), (1,), (5,4), (5,)]
    #     for [sh0, sh1] in zip(shapes1, shapes2):
    #         x = ((np.random.randn(*sh0) +
    #               1j*np.random.randn(*sh0)).astype(np.complex64))
    #         y = ((np.random.randn(*sh1) +
    #               1j*np.random.randn(*sh1)).astype(np.complex64))
    #         self._compareGpuGrad(x, y, np.divide, tf.div)
                      
    def testCplxPowGPU(self):
        shapes1 = [(5,4,3), (5,4), (1,), (5,)]
        shapes2 = [(5,4,3), (1,), (5,4), (5,)]
        for [sh0, sh1] in zip(shapes1, shapes2):
            x = ((np.random.randn(*sh0) +
                  1j*np.random.randn(*sh0)).astype(np.complex64))
            y = ((np.random.randn(*sh1) +
                  1j*np.random.randn(*sh1)).astype(np.complex64))
            self._compareGpu(x, y, np.power, tf.pow)
                  
    #### Depends on ZerosLike
    # def testCplxPowGradGPU(self):
    #     shapes1 = [(5,4,3), (5,4), (1,), (5,)]
    #     shapes2 = [(5,4,3), (1,), (5,4), (5,)]
    #     for [sh0, sh1] in zip(shapes1, shapes2):
    #         x = ((np.random.randn(*sh0) +
    #               1j*np.random.randn(*sh0)).astype(np.complex64))
    #         y = ((np.random.randn(*sh1) +
    #               1j*np.random.randn(*sh1)).astype(np.complex64))
    #         self._compareGpuGrad(x, y, np.power, tf.pow)
                      
if __name__ == "__main__":
    tf.test.main()
