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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/l2loss_op.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// Functor used by L2LossOp to do the computations.
template <typename T>
struct L2Loss<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::ConstTensor input,
                  typename TTypes<T>::Scalar output) {
    // We flatten the input tensor and reduce on dimension 0, producing
    // a single number which is Mul(Sum(x^2), 0.5).
    output.device(d) = (input * input.conjugate() * static_cast<T>(0.5)).sum();
  }
};

}  // namespace functor

template struct functor::L2Loss<GPUDevice, complex64>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
