#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "cwise_cplx_ops.h"
#include "cwise_cplx_ops_gradients.h"

namespace tensorflow {
  
#if GOOGLE_CUDA
  REGISTER(UnaryOp, GPU, "Inv", functor::cplx_inv, complex64);
  REGISTER(SimpleBinaryOp, GPU, "InvGrad", functor::cplx_inv_grad, complex64);
#endif // GOOGLE_CUDA

} // namespace tensorflow 
