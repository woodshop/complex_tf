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

#define EIGEN_USE_THREADS
#include "tensorflow/core/kernels/variable_ops.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Variable").Device(DEVICE_GPU).TypeConstraint<type>("dtype"),   \
      VariableOp);                                                         \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("VariableV2").Device(DEVICE_GPU).TypeConstraint<type>("dtype"), \
      VariableOp);                                                         \
  REGISTER_KERNEL_BUILDER(Name("TemporaryVariable")                      \
                              .Device(DEVICE_GPU)                        \
                              .TypeConstraint<type>("dtype"),            \
                          TemporaryVariableOp);                          \
  REGISTER_KERNEL_BUILDER(Name("DestroyTemporaryVariable")               \
                              .Device(DEVICE_GPU)                        \
                              .TypeConstraint<type>("T"),                \
                          DestroyTemporaryVariableOp);                   \
  REGISTER_KERNEL_BUILDER(Name("IsVariableInitialized")                  \
                              .Device(DEVICE_GPU)                        \
                              .TypeConstraint<type>("dtype")             \
                              .HostMemory("is_initialized"),             \
                          IsVariableInitializedOp);

  REGISTER_GPU_KERNELS(complex64);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
