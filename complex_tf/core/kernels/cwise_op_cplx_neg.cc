#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
  
#if GOOGLE_CUDA
  REGISTER(UnaryOp, GPU, "Neg", functor::neg, complex64);
#endif // GOOGLE_CUDA

} // namespace tensorflow 
