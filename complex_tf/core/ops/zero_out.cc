#define EIGEN_USE_THREADS
#include "zero_out.h"

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

  REGISTER_OP("ZeroOut")
  .Input("to_zero: float")
  .Output("zeroed: float")
  .Doc(R"doc(
Zeros all elements of the tensor except the first.
zeroed: A Tensor.
  output[0] = input[0]
  output[1:N] = 0
)doc");;
  
  typedef Eigen::ThreadPoolDevice CPUDevice;
  typedef Eigen::GpuDevice GPUDevice;
 
  namespace functor {

    template <typename T>
    struct ZeroOutFunctor<CPUDevice, T> {
      void operator()(const CPUDevice& d,
		      typename TTypes<T>::ConstFlat input,
		      typename TTypes<T>::Flat output,
		      const int N) {
	for (int i = 1; i < N; i++) {
	  output(i) = 0;
	}
	if (N > 0) output(0) = input(0);
      }
    };
  } // namespace functor    

  template <typename Device, typename T>
  class ZeroOutOp : public OpKernel {
  public:
    explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}
  
    void Compute(OpKernelContext* ctx) override {
      const Tensor& input_tensor = ctx->input(0);
      auto input = input_tensor.flat<T>();
      
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor.shape(),
					       &output_tensor));
    
      auto output = output_tensor->flat<T>();
      const int N = input.size();
      functor::ZeroOutFunctor<Device, T>()(ctx->eigen_device<Device>(),
					input, output, N);
    }
  };
  
  REGISTER_KERNEL_BUILDER(Name("ZeroOut")		\
			  .Device(DEVICE_CPU),		\
			  ZeroOutOp<CPUDevice, float>)
 
#if GOOGLE_CUDA
  REGISTER_KERNEL_BUILDER(Name("ZeroOut")		\
			  .Device(DEVICE_GPU),		\
			  ZeroOutOp<GPUDevice, float>)
#endif // GOOGLE_CUDA
} // namespace tensoroflow
