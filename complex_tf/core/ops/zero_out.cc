#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

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

template <typename Device>
class ZeroOutOp : public OpKernel {
public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
						     &output_tensor));

    auto output = output_tensor->flat<float>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output(0) = input(0);
  }
};

REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU),
			ZeroOutOp<CPUDevice>);

#if GOOGLE_CUDA
typedef Eigen::GpuDevice GPUDevice;
void ZeroOutKernelLauncher(const GPUDevice& d, const int* in, const int N,
			   int* out);

template <typename Device>
class ZeroOutOp : public OpKernel {
public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
						     &output_tensor));

    auto output = output_tensor->flat<float>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    // Call the cuda kernel launcher
    ZeroOutKernelLauncher(context->eigen_gpu_device(), input.data(), N,
			  output.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_GPU),
			ZeroOutOp);
#endif
