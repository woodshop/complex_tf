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

Modified by Andy Sarroff for complex_tf 2018
==============================================================================*/

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/training_ops.h"
#include <algorithm>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/kernels/variable_ops.h"

namespace tensorflow {

  // using GPUDevice = Eigen::GpuDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
template <typename Device, typename T>
struct ApplyAdamNonCuda {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad, bool use_nesterov) {
    const T alpha = lr() * Eigen::numext::sqrt(T(1) - beta2_power()) /
                    (T(1) - beta1_power());
    // beta1 == μ
    // beta2 == ν
    // v     == n
    // var   == θ

    m.device(d) += (grad - m) * (T(1) - beta1());
    v.device(d) += (grad * grad.conjugate() - v) * (T(1) - beta2());
    if (use_nesterov) {
      var.device(d) -= ((grad * (T(1) - beta1()) + beta1() * m) * alpha) /
                       (v.sqrt() + epsilon());
    } else {
      var.device(d) -= (m * alpha) / (v.sqrt() + epsilon());
    }
  }
};

template <typename T>
struct ApplyAdam<CPUDevice, T> : ApplyAdamNonCuda<CPUDevice, T> {};
}  // namespace functor

template <typename Device, typename T>
class ApplyAdamOp : public OpKernel {
 public:
  explicit ApplyAdamOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto locks = MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_,
                                                      {0, 1, 2});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, false, &var));
    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, false, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, false, &v));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));

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
    functor::ApplyAdam<Device, T>()(
        device, var.flat<T>(), m.flat<T>(), v.flat<T>(),
        beta1_power.scalar<T>(), beta2_power.scalar<T>(), lr.scalar<T>(),
        beta1.scalar<T>(), beta2.scalar<T>(), epsilon.scalar<T>(),
        grad.flat<T>(), use_nesterov_);

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
};

#define REGISTER_KERNELS(D, T)                                     \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("ApplyAdam").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyAdamOp<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdam")                \
                              .HostMemory("var")                   \
                              .HostMemory("m")                     \
                              .HostMemory("v")                     \
                              .Device(DEVICE_##D)                  \
                              .TypeConstraint<T>("T"),             \
                          ApplyAdamOp<D##Device, T>);

#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

REGISTER_CPU_KERNELS(complex64);


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
      typename TTypes<T>::ConstFlat grad, bool use_nesterov); \
  extern template struct ApplyAdam<GPUDevice, T>;

DECLARE_GPU_SPEC(complex64);
#undef DECLARE_GPU_SPEC
}  // namespace functor

  // Haven't gotten this to work because it depends on
  // dense_update_ops and variable_ops. Haven't gotten GPU
  // versions of those to work.
  // REGISTER_KERNELS(GPU, complex64);

#endif
#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNELS(T) REGISTER_KERNELS(SYCL, T);

TF_CALL_float(REGISTER_SYCL_KERNELS);
TF_CALL_double(REGISTER_SYCL_KERNELS);
#endif
#undef REGISTER_KERNELS
}  // namespace tensorflow
