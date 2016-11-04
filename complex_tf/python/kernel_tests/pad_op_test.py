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

"""Tests for tensorflow.ops.nn_ops.Pad."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import complex_tf as ctf

class PadOpTest(tf.test.TestCase):

  def _npPad(self, inp, paddings, mode):
    return np.pad(inp, paddings, mode=mode.lower())

  def _testPad(self, np_inputs, paddings, mode):
    np_val = self._npPad(np_inputs, paddings, mode=mode)
    with self.test_session(force_gpu=True):
      tf_val = tf.pad(np_inputs, paddings, mode=mode)
      out = tf_val.eval()
    self.assertAllEqual(np_val, out)
    self.assertShapeEqual(np_val, tf_val)

  def _testGradient(self, x, a, mode):
    with self.test_session(force_gpu=True):
      inx = tf.convert_to_tensor(x)
      xs = list(x.shape)
      ina = tf.convert_to_tensor(a)
      y = tf.pad(inx, ina, mode=mode)
      # Expected y's shape to be:
      ys = list(np.array(x.shape) + np.sum(np.array(a), axis=1))
      jacob_t, jacob_n = tf.test.compute_gradient(inx,
                                                  xs,
                                                  y,
                                                  ys,
                                                  x_init_value=x)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _testAll(self, np_inputs, paddings):
    for mode in ("CONSTANT", "REFLECT", "SYMMETRIC"):
    #for mode in ("CONSTANT",):
      # Zero-sized input is not allowed for REFLECT mode, but we still want
      # zero-sized input test cases for the other modes.
      if np_inputs.size or mode != "REFLECT":
        self._testPad(np_inputs, paddings, mode=mode)
        self._testGradient(np_inputs, paddings, mode=mode)

  def testComplexTypes(self):
    for t in [np.complex64]:
      x = np.random.rand(2, 5).astype(t)
      self._testAll(x + 1j * x, [[1, 0], [2, 0]])
      x = np.random.rand(3, 2, 1, 1).astype(t)
      self._testAll(x + 1j * x, [[0, 0], [0, 0], [0, 0], [0, 0]])

if __name__ == "__main__":
  tf.test.main()
