#ifndef CTF_CWISE_CPLX_OPS_H_
#define CTF_CWISE_CPLX_OPS_H_
#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
  
  namespace functor {
    
    struct CplxTanhKernelLauncher;
    template <typename T>
      struct cplx_tanh : base<T, CplxTanhKernelLauncher> { };    

    struct CplxSquareKernelLauncher;
    template <typename T>
      struct cplx_square : base<T, CplxSquareKernelLauncher> { };    

  }  // namespace functor
  
}  // namespace tensorflow
#endif  // CTF_CWISE_CPLX_OPS_H_
