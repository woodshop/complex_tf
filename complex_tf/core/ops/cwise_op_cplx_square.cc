#include "cwise_op_cplx_square.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
  typedef Eigen::ThreadPoolDevice CPUDevice;
  typedef Eigen::GpuDevice GPUDevice;
  
  // exact copy of the TF Square function, restricted to complex64
  REGISTER_OP("CplxSquare")
  .Input("x: T")
  .Output("y: T")
  .Attr("T: {complex64}")
  .SetShapeFn(shape_inference::UnchangedShape)
  .Doc(R"doc(
Computes hyperbolic tangent of `x` element-wise.
)doc");
  
  REGISTER(UnaryOp, CPU, "CplxSquare", functor::square, complex64);
  
  #if GOOGLE_CUDA  
  class CplxSquareOp : public OpKernel {
  public:
    explicit CplxSquareOp(OpKernelConstruction* context) : OpKernel(context) {}
  
    void Compute(OpKernelContext* ctx) override {
      const Tensor& input_tensor = ctx->input(0);
      auto input = input_tensor.flat<complex64>();
      
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor.shape(),
					       &output_tensor));    
      auto output = output_tensor->flat<complex64>();
      const int N = input.size();
      functor::CplxSquareFunctor()(ctx->eigen_device<GPUDevice>(),
				 input, output, N);
    }
  };

  REGISTER_KERNEL_BUILDER(Name("CplxSquare").Device(DEVICE_GPU),
			  CplxSquareOp);
  #endif // GOOGLE_CUDA

  
} // namespace tensorflow 
