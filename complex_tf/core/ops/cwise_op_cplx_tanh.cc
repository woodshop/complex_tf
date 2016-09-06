#include "cwise_op_cplx_tanh.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
  typedef Eigen::GpuDevice GPUDevice;
    
  #if GOOGLE_CUDA  
  class CplxTanhOp : public OpKernel {
  public:
    explicit CplxTanhOp(OpKernelConstruction* context) : OpKernel(context) {}
  
    void Compute(OpKernelContext* ctx) override {
      const Tensor& input_tensor = ctx->input(0);
      auto input = input_tensor.flat<complex64>();
      
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor.shape(),
					       &output_tensor));    
      auto output = output_tensor->flat<complex64>();
      const int N = input.size();
      functor::CplxTanhFunctor()(ctx->eigen_device<GPUDevice>(),
				 input, output, N);
    }
  };

  REGISTER_KERNEL_BUILDER(Name("Tanh").Device(DEVICE_GPU),
			  CplxTanhOp);
  #endif // GOOGLE_CUDA

} // namespace tensorflow 
