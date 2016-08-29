import tensorflow as tf
import complex_tf as ctf

class ZeroOutTest(tf.test.TestCase):
    def _test_1(self, use_gpu):
        with self.test_session(use_gpu=use_gpu):
            result = ctf.zero_out([5., 4., 3., 2., 1.])
            self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])

    def test_1(self):
        for use_gpu in [True, False]:
            self._test_1(use_gpu)
            
    def _test_grad_1(self, use_gpu):
        with self.test_session(use_gpu=use_gpu):
            shape = (5,)
            x = tf.constant([5, 4, 3, 2, 1], dtype=tf.float32)
            y = ctf.zero_out(x)
            err = tf.test.compute_gradient_error(x, shape, y, shape)
            self.assertLess(err, 1e-4)

    def test_grad_1(self):
        for use_gpu in [True, False]:
            self._test_grad_1(use_gpu)
            
    def _test_2(self, use_gpu):
        with self.test_session(use_gpu=use_gpu):
            result = ctf.zero_out([[4., 3.], [2., 1.]])
            self.assertAllEqual(result.eval(), [[4, 0], [0, 0]])

    def test_2(self):
        for use_gpu in [True, False]:
            self._test_2(use_gpu)
            
    def _test_grad_2(self, use_gpu):
        with self.test_session(use_gpu=use_gpu):
            shape = (2,2)
            x = tf.constant([[4., 3.], [2., 1.]], dtype=tf.float32)
            y = ctf.zero_out(x)
            err = tf.test.compute_gradient_error(x, shape, y, shape)
            self.assertLess(err, 1e-4)

    def test_grad_2(self):
        for use_gpu in [True, False]:
            self._test_grad_2(use_gpu)
            
                                                       
if __name__ == "__main__":
      tf.test.main()
