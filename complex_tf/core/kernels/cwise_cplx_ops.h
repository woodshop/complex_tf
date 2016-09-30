/*
Functors for complex functions. Many of these were adapted from
Pycuda header files.
 */

#include "tensorflow/core/kernels/cwise_ops_common.h"

#ifndef CTF_CWISE_CPLX_OPS_H_
#define CTF_CWISE_CPLX_OPS_H_

#ifndef FLT_MAX
#define FLT_MAX 3.402823466E+38F
#endif

#ifndef DBL_MAX
#define DBL_MAX 1.7976931348623158e+308
#endif

#define float_limit ::log(FLT_MAX)
#define double_limit ::log(DBL_MAX)

namespace tensorflow {
  
  namespace functor {
    
    template <typename T>
      struct cplx_inverse {
    	typedef std::complex<T> result_type;
    	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	result_type operator()(std::complex<T> a) const {
	  std::complex<T> res;
	  
	  T ar = a.real() >= 0 ? a.real() : -a.real();
	  T ai = a.imag() >= 0 ? a.imag() : -a.imag();

	  if (ar <= ai) {
	    T ratio = a.real() / a.imag();
	    T denom = a.imag() * (1 + ratio * ratio);
	    res.real(ratio / denom);
	    res.imag(- 1. / denom);
	  } else {
	    T ratio = a.imag() / a.real();
	    T denom = a.real() * (1 + ratio * ratio);
	    res.real(1. / denom);
	    res.imag(-ratio / denom);
	  }
	  return res;
    	}
      };

    template <typename T>
      struct cplx_square {
    	typedef std::complex<T> result_type;
    	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	result_type operator()(std::complex<T> a) const {
	  return std::complex<T>(a.real() * a.real() - a.imag() * a.imag(),
				 2 * a.real() * a.imag());
    	}
      };

    template <typename T>
      struct cplx_log {
    	typedef std::complex<T> result_type;
    	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	result_type operator()(std::complex<T> a) const {
    	  std::complex<T> r;

    	  r.imag(::atan2(a.imag(), a.real()));
    	  r.real(::log(::hypot(a.real(), a.imag())));
    	  return r;
    	}
      };

    template <typename T>
      struct cplx_tanh {
    	typedef std::complex<T> result_type;
    	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	result_type operator()(std::complex<T> a) const {
	  T re2 = 2.f * a.real();
	  T im2 = 2.f * a.imag();
	  if (::abs(re2) > float_limit)
	    return std::complex<T>((re2 > 0 ? 1.f : -1.f), 0.f);
	  else {
	    T den = ::cosh(re2) + ::cos(im2);
	    return std::complex<T>(::sinh(re2) / den, ::sin(im2) / den);
	  }
    	}
      };

    template <typename T>
      struct cplx_pow {
    	typedef std::complex<T> result_type;
    	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	result_type operator()(std::complex<T> a, std::complex<T> b) const {
	  T logr = ::log(::hypot(a.real(), a.imag()));
	  T logi = ::atan2(a.imag(), a.real());
	  T x = ::exp(logr * b.real() - logi * b.imag());
	  T y = logr * b.imag() + logi * b.real();

	  return std::complex<T>(x * ::cos(y), x * ::sin(y));
    	}
      };


    ////////////////////////////////////////////////////////////////////////////
    // Unary functors
    ////////////////////////////////////////////////////////////////////////////

    template <>
      struct inverse<std::complex<float> > : base<std::complex<float>,
      cplx_inverse<float> > {};

    template <>
      struct square<std::complex<float> > : base<std::complex<float>,
      cplx_square<float> > {};

    template <>
      struct log<std::complex<float> > : base<std::complex<float>,
      cplx_log<float> > {};

    template <>
      struct tanh<std::complex<float> > : base<std::complex<float>,
      cplx_tanh<float> > {};

    ////////////////////////////////////////////////////////////////////////////
    // Binary functors
    ////////////////////////////////////////////////////////////////////////////

    template <>
      struct pow<std::complex<float> > : base<std::complex<float>,
      cplx_pow<float> > {};

  }  // namespace functor
  
}  // namespace tensorflow
#endif  // CTF_CWISE_CPLX_OPS_H_
