# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functional test for GradientDescent."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import complex_tf as ctf

import numpy as np

class CplxGradientDescentOptimizerTest(tf.test.TestCase):

  def testBasic(self):
    for dtype in [tf.complex64]:
      with self.test_session(force_gpu=True):
        v0 = [1.0+2.0j, 2.0+1.0j]
        v1 = [3.0-4.0j, 4.0-3.0j]
        g0 = [0.1+0.1j, 0.1-0.1j]
        g1 = [0.01-0.01j, 0.01+0.01j]
        lr = 3.0-1.5j 
        var0 = tf.Variable(v0, dtype=dtype)
        var1 = tf.Variable(v1, dtype=dtype)
        grads0 = tf.constant(g0, dtype=dtype)
        grads1 = tf.constant(g1, dtype=dtype)
        sgd_op = ctf.train.CplxGradientDescentOptimizer(
          lr).apply_gradients(zip([grads0, grads1], [var0, var1]))
        tf.initialize_all_variables().run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType(v0, var0.eval())
        self.assertAllCloseAccordingToType(v1, var1.eval())
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType(
            [v0[0] - lr * g0[0],
             v0[1] - lr * g0[1]], var0.eval())
        self.assertAllCloseAccordingToType(
            [v1[0] - lr * g1[0],
             v1[1] - lr * g1[1]], var1.eval())

  def testTensorLearningRate(self):
    for dtype in [tf.complex64]:
      with self.test_session(force_gpu=True):
        v0 = [1.0+2.0j, 2.0+1.0j]
        v1 = [3.0-4.0j, 4.0-3.0j]
        g0 = [0.1+0.1j, 0.1-0.1j]
        g1 = [0.01-0.01j, 0.01+0.01j]
        lr = 3.0-1.5j 
        var0 = tf.Variable(v0, dtype=dtype)
        var1 = tf.Variable(v1, dtype=dtype)
        grads0 = tf.constant(g0, dtype=dtype)
        grads1 = tf.constant(g1, dtype=dtype)
        lrate = tf.constant(lr)
        sgd_op = ctf.train.CplxGradientDescentOptimizer(
          lrate).apply_gradients(zip([grads0, grads1], [var0, var1]))
        tf.initialize_all_variables().run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType(v0, var0.eval())
        self.assertAllCloseAccordingToType(v1, var1.eval())
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType(
            [v0[0] - lr * g0[0],
             v0[1] - lr * g0[1]], var0.eval())
        self.assertAllCloseAccordingToType(
            [v1[0] - lr * g1[0],
             v1[1] - lr * g1[1]], var1.eval())

  def testGradWrtRef(self):
    for dtype in [tf.complex64]:
      with self.test_session(force_gpu=True):
        values = [1.0+2.0j, 2.0+1.0j]
        lr = 3.0-1.5j 
        opt = ctf.train.CplxGradientDescentOptimizer(lr)
        values = [1.0, 3.0]
        vars_ = [tf.Variable([v], dtype=dtype) for v in values]
        grads_and_vars = opt.compute_gradients(vars_[0].ref() + vars_[1], vars_)
        tf.initialize_all_variables().run()
        for grad, _ in grads_and_vars:
          self.assertAllCloseAccordingToType([1.0], grad.eval())

  def testWithGlobalStep(self):
    for dtype in [tf.complex64]:
      with self.test_session(force_gpu=True):
        with tf.device('/cpu'):
          global_step = tf.Variable(0, trainable=False)
        v0 = [1.0+2.0j, 2.0+1.0j]
        v1 = [3.0-4.0j, 4.0-3.0j]
        g0 = [0.1+0.1j, 0.1-0.1j]
        g1 = [0.01-0.01j, 0.01+0.01j]
        lr = 3.0-1.5j 
        var0 = tf.Variable(v0, dtype=dtype)
        var1 = tf.Variable(v1, dtype=dtype)
        grads0 = tf.constant(g0, dtype=dtype)
        grads1 = tf.constant(g1, dtype=dtype)
        sgd_op = ctf.train.CplxGradientDescentOptimizer(lr).apply_gradients(
            zip([grads0, grads1], [var0, var1]),
            global_step=global_step)
        tf.initialize_all_variables().run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType(v0, var0.eval())
        self.assertAllCloseAccordingToType(v1, var1.eval())
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params and global_step
        self.assertAllCloseAccordingToType(
            [v0[0] - lr * g0[0],
             v0[1] - lr * g0[1]], var0.eval())
        self.assertAllCloseAccordingToType(
            [v1[0] - lr * g1[0],
             v1[1] - lr * g1[1]], var1.eval())
        self.assertAllCloseAccordingToType(1, global_step.eval())

  ### Currently no support for sparse complex tensors
  # def testSparseBasic(self):
  #   for dtype in [tf.complex64]:
  #     with self.test_session(force_gpu=True):
  #       var0 = tf.Variable([[1.0], [2.0]], dtype=dtype)
  #       var1 = tf.Variable([[3.0], [4.0]], dtype=dtype)
  #       grads0 = tf.IndexedSlices(
  #           tf.constant([0.1], shape=[1, 1], dtype=dtype),
  #           tf.constant([0]),
  #           tf.constant([2, 1]))
  #       grads1 = tf.IndexedSlices(
  #           tf.constant([0.01], shape=[1, 1], dtype=dtype),
  #           tf.constant([1]),
  #           tf.constant([2, 1]))
  #       sgd_op = ctf.train.CplxGradientDescentOptimizer(3.0).apply_gradients(
  #           zip([grads0, grads1], [var0, var1]))
  #       tf.initialize_all_variables().run()
  #       # Fetch params to validate initial values
  #       self.assertAllCloseAccordingToType([[1.0], [2.0]], var0.eval())
  #       self.assertAllCloseAccordingToType([[3.0], [4.0]], var1.eval())
  #       # Run 1 step of sgd
  #       sgd_op.run()
  #       # Validate updated params
  #       self.assertAllCloseAccordingToType(
  #           [[1.0 - 3.0 * 0.1], [2.0]], var0.eval())
  #       self.assertAllCloseAccordingToType(
  #           [[3.0], [4.0 - 3.0 * 0.01]], var1.eval())


if __name__ == "__main__":
  tf.test.main()
