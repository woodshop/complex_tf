#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "cwise_cplx_ops.h"
#include "cwise_cplx_ops_gradients.h"

namespace tensorflow {
  
#if GOOGLE_CUDA
  REGISTER(UnaryOp, GPU, "Log", functor::cplx_log, complex64);
#endif // GOOGLE_CUDA

} // namespace tensorflow 
