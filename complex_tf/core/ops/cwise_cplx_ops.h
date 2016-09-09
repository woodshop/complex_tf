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

    struct CplxSquareKernelLauncher;
    template <typename T>
      struct cplx_square : base<T, CplxSquareKernelLauncher> { };    

    struct CplxLogKernelLauncher;
    template <typename T>
      struct cplx_log : base<T, CplxLogKernelLauncher> { };    

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
