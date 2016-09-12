#ifndef CTF_CWISE_CPLX_OPS_GRADIENTS_H_
#define CTF_CWISE_CPLX_OPS_GRADIENTS_H_
#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
  
  namespace functor {
    
    template <typename T>
      struct cplx_inverse_grad {
    	typedef std::complex<T> result_type;
    	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	result_type operator()(std::complex<T> output,
			       std::complex<T> output_grad) const {
	  T ret_r, ret_i;

	  // conj = conj(output)
	  const T conj_r = output.real();
	  const T conj_i = -output.imag();

	  // conj_2 = conj^2
	  T conj_2_r, conj_2_i;
	  conj_2_r = conj_r * conj_r - conj_i * conj_i;
	  conj_2_i = 2 * conj_r * conj_i;

	  // ret = -output_grad * conj_2
	  ret_r = -output_grad.real() * conj_2_r +
	    output_grad.imag() * conj_2_i;
	  ret_i = -output_grad.real() * conj_2_i -
	    output_grad.imag() * conj_2_r;

	  return std::complex<T>(ret_r, ret_i);
    	}
      };

    template <typename T>
      struct cplx_tanh_grad {
    	typedef std::complex<T> result_type;
    	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  	result_type operator()(std::complex<T> output,
  			       std::complex<T> output_grad) const {

  	  // ret = output_grad * (1 - output^2);
  	  T ret_r, ret_i;
	  
  	  // rhs = 1 - output^2
  	  T rhs_r, rhs_i;
  	  rhs_r = 1. - (output.real() * output.real() -
			output.imag() * output.imag());
  	  rhs_i = -2. * output.real() * output.imag();

  	  // ret = output_grad * rhs
  	  ret_r = output_grad.real() * rhs_r - output_grad.imag() * rhs_i;
  	  ret_i = output_grad.real() * rhs_i + output_grad.imag() * rhs_r;

  	  return std::complex<T>(ret_r, ret_i);
    	}
      };

    template <>
      struct inverse_grad<std::complex<float> > : base<std::complex<float>,
      cplx_inverse_grad<float> > { };
     
    template <>
      struct tanh_grad<std::complex<float> > : base<std::complex<float>,
      cplx_tanh_grad<float> > { };

  }  // namespace functor
  
}  // namespace tensorflow
#endif  // CTF_CWISE_CPLX_OPS_GRADIENTS_H_
