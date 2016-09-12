#if GOOGLE_CUDA
#include "cwise_cplx_ops.h"
#include "cwise_cplx_ops_gradients.h"
#include "cwise_cplx_ops_gpu_common.cu.h"
#include "cwise_cplx_ops_gpu_gradients.cu.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include <pycuda-complex.hpp>

namespace pycuda {
  
  __global__ void CplxSquareKernel(const complex<float> *in,
				 complex<float> *out,
				 unsigned long n) {
    unsigned tid = threadIdx.x;
    unsigned total_threads = gridDim.x*blockDim.x;
    unsigned cta_start = blockDim.x*blockIdx.x;
    unsigned i;
    for (i = cta_start + tid; i < n; i += total_threads) {
      out[i] = pow(in[i], 2);
    }
  }

} // namespace pycuda


namespace tensorflow {

  namespace functor {
    struct CplxSquareKernelLauncher {
      void operator()(const GPUDevice& d,
		      typename TTypes<complex64>::Flat output,
		      typename TTypes<complex64>::ConstFlat input) {
	const int N =input.size();
	CudaLaunchConfig config = GetCudaLaunchConfig(N, d);
	pycuda::CplxSquareKernel<<<config.block_count, config.thread_per_block, 0,
	  d.stream()>>>((pycuda::complex<float> *)input.data(),
			(pycuda::complex<float> *)output.data(),
			N);
      }
    };
    
    DEFINE_UNARY1(cplx_square, complex64);
    
  } // namespace functor
  
} // namespace tensorflow
#endif  // GOOGLE_CUDA
