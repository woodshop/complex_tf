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

#include "tensorflow/core/kernels/training_ops.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

  typedef Eigen::GpuDevice GPUDevice;

  namespace functor {

    template <typename T>
    struct ApplyGradientDescent<GPUDevice, T> {
      void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
		      typename TTypes<T>::ConstScalar lr,
		      typename TTypes<T>::ConstFlat grad) {
	Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
	bcast[0] = grad.dimension(0);
	Eigen::Sizes<1> single;
	
	var.device(d) -=
	  grad.binaryExpr(lr.reshape(single).broadcast(bcast),
			  Eigen::internal::scalar_product_op<T>());
      }
    };
  }  // namespace functor

  template struct functor::ApplyGradientDescent<GPUDevice, complex64>;
  
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
