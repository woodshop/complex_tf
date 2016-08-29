#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
// The following isn't included in the binary installation of TF :/
// #include "tensorflow/core/util/cuda_kernel_helper.h"

typedef Eigen::GpuDevice GPUDevice;

__global__ void ZeroOutKernel(const int* in, const int N, int* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    if (i == 0) {
      out[i] = in[i];
    } else {
      out[i] = 0;
    }
  }
}
void ZeroOutKernelLauncher(const GPUDevice& d, const int* in, const int N,
			   int* out) {
  // CudaLaunchConfig config = GetCudaLaunchConfig(N, d);
  // ZeroOutKernel<<<config.block_count, config.thread_per_block, 0,
  //   d.stream()>>>(in, N, out);
  ZeroOutKernel<<<32, 256, 0, d.stream()>>>(in, N, out);
}

#endif
