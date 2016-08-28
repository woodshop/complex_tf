import tensorflow as tf
import complex_tf as ctf
import numpy as np


def to_cplx(x):
    return x[0] + 1j*x[1]


def to_real(x):
    return np.stack([x.real, x.imag])
    

class CplxMatMulTest(tf.test.TestCase):

    def testCplxMatMul(self):
        a = (np.random.randn(5, 3) + 
             1j*np.random.randn(5, 3)).astype(np.complex64)
        b = (np.random.randn(3, 2) + 
             1j*np.random.randn(3, 2)).astype(np.complex64)
        c = a.dot(b)
        with self.test_session():
            a_tf = tf.constant(to_real(a))
            b_tf = tf.constant(to_real(b))
            c_tf = ctf.ops.cplx_math_ops.cplx_matmul(a_tf, b_tf).eval()
            self.assertAllClose(c, to_cplx(c_tf))


class CplxDivTest(tf.test.TestCase):

    def testCplxDiv(self):
        a = (np.random.randn(5, 3) + 
             1j*np.random.randn(5, 3)).astype(np.complex64)
        b = (np.random.randn(5, 3) + 
             1j*np.random.randn(5, 3)).astype(np.complex64)
        c = a / b
        with self.test_session():
            a_tf = tf.constant(to_real(a))
            b_tf = tf.constant(to_real(b))
            c_tf = ctf.ops.cplx_math_ops.cplx_div(a_tf, b_tf).eval()
            self.assertAllClose(c, to_cplx(c_tf))


class CplxTanhTest(tf.test.TestCase):

    def testCplxTanh(self):
        a = (np.random.randn(5, 3) + 
             1j*np.random.randn(5, 3)).astype(np.complex64)
        c = np.tanh(a)
        with self.test_session():
            a_tf = tf.constant(to_real(a))
            c_tf = ctf.ops.cplx_math_ops.cplx_tanh(a_tf).eval()
            self.assertAllClose(c, to_cplx(c_tf))


if __name__ == '__main__':
    tf.test.main()
