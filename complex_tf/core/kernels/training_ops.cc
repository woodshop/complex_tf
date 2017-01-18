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


template <typename Device, typename T>
class ApplyMomentumOp : public OpKernel {
 public:
  explicit ApplyMomentumOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto locks = MaybeLockMutexesInOrder(ctx, use_exclusive_lock_, {0, 1});

    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor accum = ctx->mutable_input(1, use_exclusive_lock_);
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& grad = ctx->input(3);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Tensor& momentum = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyMomentum<Device, T>()(device, var.flat<T>(), accum.flat<T>(),
                                        lr.scalar<T>(), grad.flat<T>(),
                                        momentum.scalar<T>(), use_nesterov_);
    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
};


#define REGISTER_KERNELS(D, T)                                         \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("ApplyMomentum").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyMomentumOp<D##Device, T>);

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                               \
  template <>                                                             \
  void ApplyMomentum<GPUDevice, T>::operator()(                           \
      const GPUDevice& d, typename TTypes<T>::Flat var,                   \
      typename TTypes<T>::Flat accum, typename TTypes<T>::ConstScalar lr, \
      typename TTypes<T>::ConstFlat grad,                                 \
      typename TTypes<T>::ConstScalar momentum, bool use_nesterov);       \
  extern template struct ApplyMomentum<GPUDevice, T>;
DECLARE_GPU_SPEC(complex64);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, complex64);
#endif
#undef REGISTER_KERNELS


template <typename Device, typename T>
class ApplyAdamOp : public OpKernel {
 public:
  explicit ApplyAdamOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto locks = MaybeLockMutexesInOrder(ctx, use_exclusive_lock_, {0, 1, 2});

    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor m = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor v = ctx->mutable_input(2, use_exclusive_lock_);
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(2)));

    const Tensor& beta1_power = ctx->input(3);
    const Tensor& beta2_power = ctx->input(4);
    const Tensor& lr = ctx->input(5);
    const Tensor& beta1 = ctx->input(6);
    const Tensor& beta2 = ctx->input(7);
    const Tensor& epsilon = ctx->input(8);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    const Tensor& grad = ctx->input(9);
    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument("var and m do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        m.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()),
                errors::InvalidArgument("var and v do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        v.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyAdam<Device, T>()(device, var.flat<T>(), m.flat<T>(),
                                    v.flat<T>(), beta1_power.scalar<T>(),
                                    beta2_power.scalar<T>(), lr.scalar<T>(),
                                    beta1.scalar<T>(), beta2.scalar<T>(),
                                    epsilon.scalar<T>(), grad.flat<T>());

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                     \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("ApplyAdam").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyAdamOp<D##Device, T>);

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                   \
  template <>                                                 \
  void ApplyAdam<GPUDevice, T>::operator()(                   \
      const GPUDevice& d, typename TTypes<T>::Flat var,       \
      typename TTypes<T>::Flat m, typename TTypes<T>::Flat v, \
      typename TTypes<T>::ConstScalar beta1_power,            \
      typename TTypes<T>::ConstScalar beta2_power,            \
      typename TTypes<T>::ConstScalar lr,                     \
      typename TTypes<T>::ConstScalar beta1,                  \
      typename TTypes<T>::ConstScalar beta2,                  \
      typename TTypes<T>::ConstScalar epsilon,                \
      typename TTypes<T>::ConstFlat grad);                    \
  extern template struct ApplyAdam<GPUDevice, T>;
DECLARE_GPU_SPEC(complex64);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, complex64);
#endif
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyRMSPropOp : public OpKernel {
 public:
  explicit ApplyRMSPropOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto locks = MaybeLockMutexesInOrder(ctx, use_exclusive_lock_, {0, 1, 2});

    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor ms = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor mom = ctx->mutable_input(2, use_exclusive_lock_);

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, ms.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    OP_REQUIRES(
        ctx, mom.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(2)));

    const Tensor& lr = ctx->input(3);
    const Tensor& rho = ctx->input(4);
    const Tensor& momentum = ctx->input(5);
    const Tensor& epsilon = ctx->input(6);
    const Tensor& grad = ctx->input(7);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho.shape()),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(ms.shape()),
                errors::InvalidArgument("var and ms do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        ms.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(mom.shape()),
                errors::InvalidArgument(
                    "var and mom do not have the same shape",
                    var.shape().DebugString(), " ", mom.shape().DebugString()));

    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyRMSProp<Device, T>()(device, var.flat<T>(), ms.flat<T>(),
                                       mom.flat<T>(), lr.scalar<T>(),
                                       rho.scalar<T>(), momentum.scalar<T>(),
                                       epsilon.scalar<T>(), grad.flat<T>());

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
};


using GPUDevice = Eigen::GpuDevice;

#define REGISTER_KERNELS(D, T)                                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ApplyRMSProp").Device(DEVICE_##D).TypeConstraint<T>("T"),         \
      ApplyRMSPropOp<D##Device, T>);

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                    \
  template <>                                                                  \
  void ApplyRMSProp<GPUDevice, T>::operator()(                                 \
      const GPUDevice& d, typename TTypes<T>::Flat var,                        \
      typename TTypes<T>::Flat ms, typename TTypes<T>::Flat mom,               \
      typename TTypes<T>::ConstScalar lr, typename TTypes<T>::ConstScalar rho, \
      typename TTypes<T>::ConstScalar momentum,                                \
      typename TTypes<T>::ConstScalar epsilon,                                 \
      typename TTypes<T>::ConstFlat grad);                                     \
  extern template struct ApplyRMSProp<GPUDevice, T>;                           \
DECLARE_GPU_SPEC(complex64);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, complex64);
#endif
#undef REGISTER_KERNELS

}  // namespace tensorflow
