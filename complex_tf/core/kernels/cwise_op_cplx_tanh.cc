#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "cwise_cplx_ops.h"
#include "cwise_cplx_ops_gradients.h"

namespace tensorflow {
  
#if GOOGLE_CUDA
  REGISTER(UnaryOp, GPU, "Tanh", functor::cplx_tanh, complex64);
  REGISTER(SimpleBinaryOp, GPU, "TanhGrad", functor::cplx_tanh_grad, complex64);
#endif // GOOGLE_CUDA

} // namespace tensorflow 
