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

"""Tests for ConstantOp."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import complex_tf

class ZerosLikeTest(tf.test.TestCase):

  def _compareZeros(self, dtype, force_gpu):
    with self.test_session(force_gpu=force_gpu):
      # Creates a tensor of non-zero values with shape 2 x 3.
      numpy_dtype = dtype.as_numpy_dtype
      d = tf.constant(np.ones((2, 3), dtype=numpy_dtype), dtype=dtype)
      # Constructs a tensor of zeros of the same dimensions and type as "d".
      z_var = tf.zeros_like(d)
      # Test that the type is correct
      self.assertEqual(z_var.dtype, dtype)
      z_value = z_var.eval()

      # Test that the value is correct
      self.assertTrue(np.array_equal(z_value, np.array([[0] * 3] * 2)))
      self.assertEqual([2, 3], z_var.get_shape())

  def testZerosLikeGPU(self):
    for dtype in [tf.complex64]:
      self._compareZeros(dtype, True)

  def testZerosLikePartialShape(self):
    d = tf.placeholder(tf.complex64, shape=[None, 4, None])
    z = tf.zeros_like(d)
    self.assertEqual(d.get_shape().as_list(), z.get_shape().as_list())

  def testZerosLikeDtype(self):
    # Make sure zeros_like works even for dtypes that cannot be cast between
    with self.test_session():
      shape = (3, 5)
      dtypes = np.float32, np.complex64
      for in_type in dtypes:
        x = np.arange(15).astype(in_type).reshape(*shape)
        for out_type in dtypes:
          y = tf.zeros_like(x, dtype=out_type).eval()
          self.assertEqual(y.dtype, out_type)
          self.assertEqual(y.shape, shape)
          self.assertAllEqual(y, np.zeros(shape, dtype=out_type))


class FillTest(tf.test.TestCase):

  def _compare(self, dims, val, np_ans, force_gpu):
    with self.test_session(force_gpu=force_gpu):
      tf_ans = tf.fill(dims, val, name="fill")
      out = tf_ans.eval()
    self.assertAllClose(np_ans, out)
    # Fill does not set the shape.
    # self.assertShapeEqual(np_ans, tf_ans)

  def testFillComplex64(self):
    np_ans = np.array([[0.15] * 3] * 2).astype(np.complex64)
    self._compare([2, 3], np_ans[0][0], np_ans, force_gpu=True)

  def testGradient(self):
    with self.test_session():
      in_v = tf.constant(5.0)
      out_shape = [3, 2]
      out_filled = tf.fill(out_shape, in_v)
      err = tf.test.compute_gradient_error(in_v, [],
                                           out_filled, out_shape)
    self.assertLess(err, 1e-3)


if __name__ == "__main__":
  tf.test.main()
