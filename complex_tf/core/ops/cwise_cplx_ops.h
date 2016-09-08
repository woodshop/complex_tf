#ifndef CTF_CWISE_CPLX_OPS_H_
#define CTF_CWISE_CPLX_OPS_H_
#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
  
  namespace functor {
    
    ////////////////////////////////////////////////////////////////////////////
    // Unary functors
    ////////////////////////////////////////////////////////////////////////////
    struct CplxNegKernelLauncher;
    template <typename T>
      struct cplx_neg : base<T, CplxNegKernelLauncher> { };    

    struct CplxTanhKernelLauncher;
    template <typename T>
      struct cplx_tanh : base<T, CplxTanhKernelLauncher> { };    

    struct CplxSquareKernelLauncher;
    template <typename T>
      struct cplx_square : base<T, CplxSquareKernelLauncher> { };    

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

  }  // namespace functor
  
}  // namespace tensorflow
#endif  // CTF_CWISE_CPLX_OPS_H_
