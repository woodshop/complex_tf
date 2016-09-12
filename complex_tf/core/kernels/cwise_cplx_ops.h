#ifndef CTF_CWISE_CPLX_OPS_H_
#define CTF_CWISE_CPLX_OPS_H_

namespace tensorflow {
  
  namespace functor {
    
    template <typename T>
      struct cplx_log {
    	typedef std::complex<T> result_type;
    	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type operator()(std::complex<T> a) const {
    	  std::complex<T> r;

    	  r.imag(::atan2(a.imag(), a.real()));
    	  r.real(::log(::hypot(a.real(), a.imag())));
    	  return r;
    	}
      };


    ////////////////////////////////////////////////////////////////////////////
    // Unary functors
    ////////////////////////////////////////////////////////////////////////////

    struct CplxSquareKernelLauncher;
    template <typename T>
      struct cplx_square : base<T, CplxSquareKernelLauncher> { };    

    template <>
      struct log<std::complex<float> > : base<std::complex<float>,
      cplx_log<float> > {};

    struct CplxTanhKernelLauncher;
    template <typename T>
      struct cplx_tanh : base<T, CplxTanhKernelLauncher> { };    

    struct CplxInvKernelLauncher;
    template <typename T>
      struct cplx_inv : base<T, CplxInvKernelLauncher> { };    

    ////////////////////////////////////////////////////////////////////////////
    // Binary functors
    ////////////////////////////////////////////////////////////////////////////

    struct CplxAddKernelLauncher;
    template <typename T>
      struct cplx_add : base<T, CplxAddKernelLauncher> { };    

    struct CplxSubKernelLauncher;
    template <typename T>
      struct cplx_sub : base<T, CplxSubKernelLauncher> { };    
    
    struct CplxMulKernelLauncher;
    template <typename T>
      struct cplx_mul : base<T, CplxMulKernelLauncher> { };    

    struct CplxDivKernelLauncher;
    template <typename T>
      struct cplx_div : base<T, CplxDivKernelLauncher> { };    

    struct CplxPowKernelLauncher;
    template <typename T>
      struct cplx_pow : base<T, CplxPowKernelLauncher> { };    

    struct CplxNotEqualKernelLauncher;
    template <typename T>
      struct cplx_not_equal : base<T, CplxNotEqualKernelLauncher, bool> { };    

  }  // namespace functor
  
}  // namespace tensorflow
#endif  // CTF_CWISE_CPLX_OPS_H_
