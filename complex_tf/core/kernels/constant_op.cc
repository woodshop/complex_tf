/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/constant_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class FillOp : public OpKernel {
 public:
  explicit FillOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& Tdims = context->input(0);
    OP_REQUIRES(
        context, IsLegacyVector(Tdims.shape()),
        errors::InvalidArgument("dims must be a vector of int32, got shape ",
                                Tdims.shape().DebugString()));
    const Tensor& Tvalue = context->input(1);
    OP_REQUIRES(context, IsLegacyScalar(Tvalue.shape()),
                errors::InvalidArgument("value must be a scalar, got shape ",
                                        Tvalue.shape().DebugString()));
    auto dims = Tdims.flat<int32>();
    TensorShape shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                reinterpret_cast<const int32*>(dims.data()),
                                dims.size(), &shape));
    Tensor* out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &out));
    functor::FillFunctor<Device, T> functor;
    functor(context->eigen_device<Device>(), out->flat<T>(),
            Tvalue.scalar<T>());
  }
};

#define REGISTER_KERNEL(D, TYPE)                         \
  REGISTER_KERNEL_BUILDER(Name("Fill")                   \
                              .Device(DEVICE_##D)        \
                              .TypeConstraint<TYPE>("T") \
                              .HostMemory("dims"),       \
                          FillOp<D##Device, TYPE>);

#if GOOGLE_CUDA
REGISTER_KERNEL(GPU, complex64);
#endif

#undef REGISTER_KERNEL

template <typename Device, typename T>
class ZerosLikeOp : public OpKernel {
 public:
  explicit ZerosLikeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &out));
    functor::SetZeroFunctor<Device, T> f;
    f(ctx->eigen_device<Device>(), out->flat<T>());
  }
};

#define REGISTER_KERNEL(type, dev)                                      \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ZerosLike").Device(DEVICE_##dev).TypeConstraint<type>("T"), \
      ZerosLikeOp<dev##Device, type>)

#if GOOGLE_CUDA
REGISTER_KERNEL(complex64, GPU);
#endif  // GOOGLE_CUDA

#undef REGISTER_KERNEL

}  // namespace tensorflow
