#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "zero_out.h"

#include <assert.h>
#include <stdio.h>

#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

  namespace functor {

    using GPUDevice = Eigen::GpuDevice;
  
    __global__ void ZeroOutKernel(const float* in, float* out, const int N) {
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
	   i += blockDim.x * gridDim.x) {
	if (i == 0) {
	  out[i] = in[i];
	} else {
	  out[i] = 0;
	}
      }
    }

    template<typename T>
    struct ZeroOutFunctor<GPUDevice, T> {
      void operator()(const GPUDevice& d,
		      typename TTypes<T>::ConstFlat input,
		      typename TTypes<T>::Flat output,
		      const int N) {
	
	CudaLaunchConfig config = GetCudaLaunchConfig(N, d);
	ZeroOutKernel<<<config.block_count, config.thread_per_block, 0,
	  d.stream()>>>(input.data(),
			output.data(),
			N);
      }
    };
    template struct ZeroOutFunctor<GPUDevice, float>;
  } // namespace functor 
} // namespace tensorflow
#endif // GOOGLE_CUDA
