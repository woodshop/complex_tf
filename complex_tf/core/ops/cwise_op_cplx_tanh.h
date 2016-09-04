#ifndef TENSORFLOW_KERNELS_CPLX_TANH_OP_H_
#define TENSORFLOW_KERNELS_CPLX_TANH_OP_H_

#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
  
  typedef Eigen::GpuDevice GPUDevice;
  
  namespace functor {
    struct CplxTanhFunctor {
      void operator()(const GPUDevice& d,
    		      typename TTypes<complex64>::ConstFlat input,
    		      typename TTypes<complex64>::Flat output,
    		      const int N);
    };
    
  }  // namespace functor
  
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CPLX_TANH_OP_H_
