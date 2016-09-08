#if GOOGLE_CUDA
#include "cwise_cplx_ops.h"
#include "cwise_cplx_ops_gpu_common.cu.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include <pycuda-complex.hpp>

namespace pycuda {
  
  __global__ void CplxDivKernel(const complex<float> *in1,
				const complex<float> *in2,
				complex<float> *out,
				unsigned long n,
				tensorflow::functor::scalar_side side) {
    unsigned tid = threadIdx.x;
    unsigned total_threads = gridDim.x*blockDim.x;
    unsigned cta_start = blockDim.x*blockIdx.x;
    unsigned i;
    if (side == tensorflow::functor::none) {
      for (i = cta_start + tid; i < n; i += total_threads) {
	out[i] = in1[i] / in2[i];
      }
    } else if (side == tensorflow::functor::left) {
      for (i = cta_start + tid; i < n; i += total_threads) {
	out[i] = in2[0] / in1[i];
      }
    } else if (side == tensorflow::functor::right) {
      for (i = cta_start + tid; i < n; i += total_threads) {
	out[i] = in1[i] / in2[0];
      }
    }
  }
} // namespace pycuda


namespace tensorflow {

  namespace functor {
    struct CplxDivKernelLauncher {
      void operator()(const GPUDevice& d,
		      typename TTypes<complex64>::Flat output,
		      typename TTypes<complex64>::ConstFlat input1,
		      typename TTypes<complex64>::ConstFlat input2) {
	const int N = input1.size();
	CudaLaunchConfig config = GetCudaLaunchConfig(N, d);
	pycuda::CplxDivKernel<<<config.block_count, config.thread_per_block, 0,
	  d.stream()>>>((pycuda::complex<float> *)input1.data(),
			(pycuda::complex<float> *)input2.data(),
			(pycuda::complex<float> *)output.data(),
			N, none);
      }

      void operator()(const GPUDevice& d,
		      typename TTypes<complex64>::Flat output,
		      typename TTypes<complex64>::ConstFlat input1,
		      typename TTypes<complex64>::ConstScalar input2,
		      scalar_side side) {
	const int N = input1.size();
	CudaLaunchConfig config = GetCudaLaunchConfig(N, d);
	pycuda::CplxDivKernel<<<config.block_count, config.thread_per_block, 0,
	  d.stream()>>>((pycuda::complex<float> *)input1.data(),
			(pycuda::complex<float> *)input2.data(),
			(pycuda::complex<float> *)output.data(),
			N, side);
      }
    };
    

    DEFINE_BINARY1(cplx_div, complex64);
    
  } // namespace functor
  
} // namespace tensorflow
#endif  // GOOGLE_CUDA
