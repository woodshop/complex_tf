/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/training_ops.h"
#include <algorithm>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/bounds_check.h"

namespace tensorflow {

  using GPUDevice = Eigen::GpuDevice;

  // MaybeLockMutexesInOrder is a helper function to acquire mutexes in address
  // order to mitigate deadlock.  Returns a vector of acquired mutexes.
  // Safe to pass duplicates - will only lock each distinct mutex once.
  // If do_lock is false, returns immediately.
  std::vector<mutex_lock>
  MaybeLockMutexesInOrder(OpKernelContext* ctx, bool do_lock,
			  const std::vector<int>& input_ids) {
    std::vector<mutex_lock> locks;
    if (!do_lock) {
      return locks;
    }
    std::vector<mutex*> mutexes;
    std::vector<int> acquire_order;
    for (auto input : input_ids) {
      auto* mutex = ctx->input_ref_mutex(input);
      // Only lock each mutex once if duplicates exist (n^2 but n is 2 or 3).
      if (std::find(mutexes.begin(), mutexes.end(), mutex) == mutexes.end()) {
	acquire_order.push_back(input);
	mutexes.push_back(mutex);
      }
    }
    std::sort(acquire_order.begin(), acquire_order.end(),
	      [&mutexes](int a, int b) { return mutexes[a] < mutexes[b]; });
    
    for (auto input : acquire_order) {
      locks.emplace_back(*ctx->input_ref_mutex(input));
    }
    return locks;
  }
  
  template <typename Device, typename T>
  class ApplyGradientDescentOp : public OpKernel {
  public:
    explicit ApplyGradientDescentOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    }
    
    void Compute(OpKernelContext* ctx) override {
      auto locks = MaybeLockMutexesInOrder(ctx, use_exclusive_lock_, {0});
      Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
      
      OP_REQUIRES(ctx, var.IsInitialized(),
		  errors::FailedPrecondition
		  ("Attempting to use uninitialized variables: ",
		   def().input(0)));
      const Tensor& alpha = ctx->input(1);
      OP_REQUIRES(ctx, IsLegacyScalar(alpha.shape()),
		  errors::InvalidArgument("alpha is not a scalar: ",
					  alpha.shape().DebugString()));
      const Tensor& delta = ctx->input(2);
      OP_REQUIRES(
		  ctx, var.shape().IsSameSize(delta.shape()),
		  errors::InvalidArgument
		  ("var and delta do not have the same shape",
		   var.shape().DebugString(), " ",
		   delta.shape().DebugString()));
      
      const Device& device = ctx->template eigen_device<Device>();
      functor::ApplyGradientDescent<Device, T>()
	(device, var.flat<T>(), alpha.scalar<T>(), delta.flat<T>());
      
      ctx->forward_ref_input_to_ref_output(0, 0);
    }
    
  private:
    bool use_exclusive_lock_;
  };
  
#define REGISTER_KERNELS(D, T)						\
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ApplyGradientDescent").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyGradientDescentOp<D##Device, T>);

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                             \
  template <>                                           \
  void ApplyGradientDescent<GPUDevice, T>::operator()(  \
      const GPUDevice& d, typename TTypes<T>::Flat var, \
      typename TTypes<T>::ConstScalar alpha,            \
      typename TTypes<T>::ConstFlat delta);             \
  extern template struct ApplyGradientDescent<GPUDevice, T>;
DECLARE_GPU_SPEC(complex64);
#undef DECLARE_GPU_SPEC
}  // namespace functor
  
REGISTER_KERNELS(GPU, complex64);
#endif
#undef REGISTER_KERNELS

}  // namespace tensorflow
