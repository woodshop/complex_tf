#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "cwise_cplx_ops.h"

namespace tensorflow {
  
#if GOOGLE_CUDA
  REGISTER(BinaryOp, GPU, "Mul", functor::cplx_mul, complex64);
#endif // GOOGLE_CUDA

} // namespace tensorflow 
