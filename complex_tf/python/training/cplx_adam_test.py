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
#
# Modified by Andy Sarroff for complex_tf 2018
# =============================================================================

"""Tests for Adam."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from complex_tf.python.training import cplx_adam

def adam_update_numpy(param,
                      g_t,
                      t,
                      m,
                      v,
                      alpha=0.001,
                      beta1=0.9,
                      beta2=0.999,
                      epsilon=1e-8):
  alpha_t = alpha * np.sqrt(1 - beta2**t) / (1 - beta1**t)

  m_t = beta1 * m + (1 - beta1) * g_t
  v_t = beta2 * v + (1 - beta2) * g_t * np.conj(g_t)

  param_t = param - alpha_t * m_t / (np.sqrt(v_t) + epsilon)
  return param_t, m_t, v_t


class CplxAdamOptimizerTest(test.TestCase):

  def doTestBasic(self, use_resource=False):
    for i, dtype in enumerate([dtypes.complex64]):
      with self.test_session(graph=ops.Graph(), use_gpu=False):
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0 + 1j*1.0, 2.0 + 1j*2.0],
                           dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1 - 1j*0.1, 0.1 + 1j*0.1],
                             dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0 - 1j*3.0, 4.0 + 1j*4.0],
                           dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01 + 1j*0.1, 0.01 + 1j*0.1],
                             dtype=dtype.as_numpy_dtype)
        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(
              var0_np, name="var0_%d" % i)
          var1 = resource_variable_ops.ResourceVariable(
              var1_np, name="var1_%d" % i)
        else:
          var0 = variables.Variable(var0_np)
          var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)

        opt = cplx_adam.CplxAdamOptimizer()
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        opt_variables = opt.variables()
        self.assertIn(opt._beta1_power, opt_variables)
        self.assertIn(opt._beta2_power, opt_variables)

        with ops.Graph().as_default():
          # Shouldn't return non-slot variables from other graphs.
          self.assertEqual(0, len(opt.variables()))

        if context.in_graph_mode():
          self.evaluate(variables.global_variables_initializer())
          # Fetch params to validate initial values
          self.assertAllClose([1.0 + 1j*1.0, 2.0 + 1j*2.0],
                              self.evaluate(var0))
          self.assertAllClose([3.0 - 1j*3.0, 4.0 + 1j*4.0],
                              self.evaluate(var1))

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Adam
        for t in range(1, 4):
          if context.in_graph_mode():
            self.evaluate(update)
          elif t > 1:
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

          self.assertAllCloseAccordingToType(0.9**(t + 1),
                                             self.evaluate(beta1_power))
          self.assertAllCloseAccordingToType(0.999**(t + 1),
                                             self.evaluate(beta2_power))

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))
          if use_resource:
            self.assertEqual("var0_%d/CplxAdam:0" % (i,),
                             opt.get_slot(var=var0, name="m").name)

  def testBasic(self):
    with self.test_session(use_gpu=False):
      self.doTestBasic(use_resource=False)

  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testResourceBasic(self):
    self.doTestBasic(use_resource=True)

  def testTensorLearningRate(self):
    for dtype in [dtypes.complex64]:
      with self.test_session(use_gpu=False):
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0 + 1j*1.0, 2.0 + 1j*2.0],
                           dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1 - 1j*0.1, 0.1 + 1j*0.1],
                             dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0 - 1j*3.0, 4.0 + 1j*4.0],
                           dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01 + 1j*0.1, 0.01 + 1j*0.1],
                             dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)
        opt = cplx_adam.CplxAdamOptimizer(constant_op.constant(0.001))
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        # Fetch params to validate initial values
        self.assertAllClose([1.0 + 1j*1.0, 2.0 + 1j*2.0],
                            var0.eval())
        self.assertAllClose([3.0 - 1j*3.0, 4.0 + 1j*4.0],
                            var1.eval())

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Adam
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9**t, beta1_power.eval())
          self.assertAllCloseAccordingToType(0.999**t, beta2_power.eval())
          update.run()

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, var0.eval())
          self.assertAllCloseAccordingToType(var1_np, var1.eval())

  def testSharing(self):
    for dtype in [dtypes.complex64]:
      with self.test_session():
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0 + 1j*1.0, 2.0 + 1j*2.0],
                           dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1 - 1j*0.1, 0.1 + 1j*0.1],
                             dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0 - 1j*3.0, 4.0 + 1j*4.0],
                           dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01 + 1j*0.1, 0.01 + 1j*0.1],
                             dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)
        opt = cplx_adam.CplxAdamOptimizer()
        update1 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        update2 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType(var0_np, var0.eval())
        self.assertAllCloseAccordingToType(var1_np, var1.eval())

        # Run 3 steps of intertwined Adam1 and Adam2.
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9**t, beta1_power.eval())
          self.assertAllCloseAccordingToType(0.999**t, beta2_power.eval())
          if t % 2 == 0:
            update1.run()
          else:
            update2.run()

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, var0.eval())
          self.assertAllCloseAccordingToType(var1_np, var1.eval())

  def testTwoSessions(self):
    optimizer = cplx_adam.CplxAdamOptimizer()
    g = ops.Graph()
    with g.as_default():
      with session.Session():
        var0 = variables.Variable(np.array([1.0 + 1j*1.0, 2.0 + 1j*2.0],
                           dtype=dtypes.complex64.as_numpy_dtype))
        grads0 = constant_op.constant(
            np.array([0.1 - 1j*0.1, 0.1 + 1j*0.1],
                     dtype=dtypes.complex64.as_numpy_dtype))
        optimizer.apply_gradients([(grads0, var0)])

    gg = ops.Graph()
    with gg.as_default():
      with session.Session():
        var0 = variables.Variable(np.array([1.0 + 1j*1.0, 2.0 + 1j*2.0],
                           dtype=dtypes.complex64.as_numpy_dtype))
        grads0 = constant_op.constant(
            np.array([0.1 - 1j*0.1, 0.1 + 1j*0.1],
                     dtype=dtypes.complex64.as_numpy_dtype))

        # If the optimizer saves any state not keyed by graph the following line
        # fails.
        optimizer.apply_gradients([(grads0, var0)])


if __name__ == "__main__":
  test.main()
