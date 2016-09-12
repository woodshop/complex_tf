#if GOOGLE_CUDA
#include "tensorflow/core/kernels/cwise_ops_gpu_common.cu.h"
#include "cwise_cplx_ops.h"
namespace tensorflow {

  namespace functor {

    DEFINE_UNARY1(log, complex64);
    
  } // namespace functor
  
} // namespace tensorflow
#endif  // GOOGLE_CUDA
