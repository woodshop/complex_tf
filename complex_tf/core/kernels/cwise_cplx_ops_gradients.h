#ifndef CTF_CWISE_CPLX_OPS_GRADIENTS_H_
#define CTF_CWISE_CPLX_OPS_GRADIENTS_H_
#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
  
  namespace functor {
    
    struct CplxTanhGradKernelLauncher;
    template <typename T>
      struct cplx_tanh_grad : base<T, CplxTanhGradKernelLauncher> { };    

    struct CplxInvGradKernelLauncher;
    template <typename T>
      struct cplx_inv_grad : base<T, CplxInvGradKernelLauncher> { };    

  }  // namespace functor
  
}  // namespace tensorflow
#endif  // CTF_CWISE_CPLX_OPS_GRADIENTS_H_
