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

#include <limits>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
#if GOOGLE_CUDA
typedef Eigen::GpuDevice GPUDevice;
#endif  // GOOGLE_CUDA

enum AxisArgumentName { NAME_IS_AXIS, NAME_IS_CONCAT_DIM };

// --------------------------------------------------------------------------
template <typename Device, typename T, AxisArgumentName AxisArgName>
class ConcatBaseOp : public OpKernel {
 public:
  typedef std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>
      ConstMatrixVector;

  explicit ConcatBaseOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    const Tensor* concat_dim_tensor;
    const char* axis_attribute_name =
        AxisArgName == NAME_IS_AXIS
            ? "axis"
            : AxisArgName == NAME_IS_CONCAT_DIM ? "concat_dim" : "<invalid>";
    OP_REQUIRES_OK(c, c->input(axis_attribute_name, &concat_dim_tensor));
    OP_REQUIRES(c, IsLegacyScalar(concat_dim_tensor->shape()),
                errors::InvalidArgument(
                    axis_attribute_name,
                    " tensor should be a scalar integer, but got shape ",
                    concat_dim_tensor->shape().DebugString()));
    const int32 concat_dim =
        internal::SubtleMustCopy(concat_dim_tensor->scalar<int32>()());
    OpInputList values;
    OP_REQUIRES_OK(c, c->input_list("values", &values));
    const int N = values.size();
    const int input_dims = values[0].dims();
    const TensorShape& input_shape = values[0].shape();

    int32 axis = concat_dim < 0 ? concat_dim + input_dims : concat_dim;
    OP_REQUIRES(c, (0 <= axis && axis < input_dims) ||
                       (allow_legacy_scalars() && concat_dim == 0),
                errors::InvalidArgument(
                    "ConcatOp : Expected concatenating dimensions in the range "
                    "[",
                    -input_dims, ", ", input_dims, "), but got ", concat_dim));
    // Note that we reduce the concat of n-dimensional tensors into a two
    // dimensional concat. Assuming the dimensions of any input/output
    // tensor are {x0, x1,...,xn-1, y0, y1,...,ym-1}, where the concat is along
    // the dimension indicated with size y0, we flatten it to {x, y}, where y =
    // Prod_i(yi) and x = ((n > 0) ? Prod_i(xi) : 1).
    ConstMatrixVector inputs_flat;
    inputs_flat.reserve(N);
    int64 inputs_flat_dim0 = 1;
    for (int d = 0; d < axis; ++d) {
      inputs_flat_dim0 *= input_shape.dim_size(d);
    }
    int64 output_concat_dim = 0;
    const bool input_is_scalar = IsLegacyScalar(input_shape);
    for (int i = 0; i < N; ++i) {
      const auto in = values[i];
      const bool in_is_scalar = IsLegacyScalar(in.shape());
      OP_REQUIRES(
          c, in.dims() == input_dims || (input_is_scalar && in_is_scalar),
          errors::InvalidArgument(
              "ConcatOp : Ranks of all input tensors should match: shape[0] = ",
              input_shape.DebugString(), " vs. shape[", i, "] = ",
              in.shape().DebugString()));
      for (int j = 0; j < input_dims; ++j) {
        if (j == axis) {
          continue;
        }
        OP_REQUIRES(
            c, in.dim_size(j) == input_shape.dim_size(j),
            errors::InvalidArgument(
                "ConcatOp : Dimensions of inputs should match: shape[0] = ",
                input_shape.DebugString(), " vs. shape[", i, "] = ",
                in.shape().DebugString()));
      }
      if (in.NumElements() > 0) {
        int64 inputs_flat_dim1 = in.NumElements() / inputs_flat_dim0;
        inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
            in.shaped<T, 2>({inputs_flat_dim0, inputs_flat_dim1})));
      }
      // TODO(irving): Remove check once !allow_legacy_scalars().
      output_concat_dim += in.dims() > 0 ? in.dim_size(axis) : 1;
    }

    TensorShape output_shape(input_shape);
    // TODO(irving): Remove rank 0 case once !allow_legacy_scalars().
    if (output_shape.dims() == 0) {
      output_shape.AddDim(output_concat_dim);
    } else {
      output_shape.set_dim(axis, output_concat_dim);
    }
    Tensor* output = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output));
    if (output->NumElements() > 0) {
      int64 output_dim1 = output->NumElements() / inputs_flat_dim0;
      auto output_flat = output->shaped<T, 2>({inputs_flat_dim0, output_dim1});
#if GOOGLE_CUDA
      if (std::is_same<Device, GPUDevice>::value) {
        ConcatGPU<T>(c, inputs_flat, output, &output_flat);
        return;
      }
#endif  // GOOGLE_CUDA
      ConcatCPU<T>(c->device(), inputs_flat, &output_flat);
    }
  }
};

template <typename Device, typename T>
using ConcatOp = ConcatBaseOp<Device, T, NAME_IS_CONCAT_DIM>;
template <typename Device, typename T>
using ConcatV2Op = ConcatBaseOp<Device, T, NAME_IS_AXIS>;

#if GOOGLE_CUDA

#define REGISTER_GPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("Concat")                     \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<type>("T")     \
                              .HostMemory("concat_dim"),     \
                          ConcatOp<GPUDevice, type>)         \
  REGISTER_KERNEL_BUILDER(Name("ConcatV2")                   \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<type>("T")     \
                              .TypeConstraint<int32>("Tidx") \
                              .HostMemory("axis"),           \
                          ConcatV2Op<GPUDevice, type>)

REGISTER_GPU(complex64);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
