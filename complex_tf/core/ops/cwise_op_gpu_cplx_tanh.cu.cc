#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "cwise_op_cplx_tanh.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include <pycuda-complex.hpp>

namespace pycuda {
  
  __global__ void CplxTanhKernel(const complex<float> *in,
				 complex<float> *out,
				 unsigned long n) {
    unsigned tid = threadIdx.x;
    unsigned total_threads = gridDim.x*blockDim.x;
    unsigned cta_start = blockDim.x*blockIdx.x;
    unsigned i;
    for (i = cta_start + tid; i < n; i += total_threads) {
      out[i] = tanh(in[i]);
    }
  }
} // namespace pycuda


 namespace tensorflow {

  typedef Eigen::GpuDevice GPUDevice;

   namespace functor {

    void CplxTanhFunctor::operator()(const GPUDevice& d,
    		      typename TTypes<complex64>::ConstFlat input,
    		      typename TTypes<complex64>::Flat output,
    		      const int N) {
      printf("\t\tCalling GPU kernel for CplxTanh.\n");
      CudaLaunchConfig config = GetCudaLaunchConfig(N, d);
      pycuda::CplxTanhKernel<<<config.block_count, config.thread_per_block, 0,
	d.stream()>>>((pycuda::complex<float> *)input.data(),
		      (pycuda::complex<float> *)output.data(),
		      N);
    }
    
  } // namespace functor
} // namespace tensorflow
#endif  // GOOGLE_CUDA
