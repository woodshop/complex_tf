#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "cwise_cplx_ops.h"

namespace tensorflow {
  
#if GOOGLE_CUDA
  REGISTER(BinaryOp, GPU, "Sub", functor::cplx_sub, complex64);
#endif // GOOGLE_CUDA

} // namespace tensorflow 
