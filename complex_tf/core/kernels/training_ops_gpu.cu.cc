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

template <typename T>
struct ApplyMomentum<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar momentum, bool use_nesterov) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    accum.device(d) = accum * momentum.reshape(single).broadcast(bcast) + grad;
    if (use_nesterov) {
      var.device(d) -= grad * lr.reshape(single).broadcast(bcast) +
                       accum * momentum.reshape(single).broadcast(bcast) *
                           lr.reshape(single).broadcast(bcast);
    } else {
      var.device(d) -= lr.reshape(single).broadcast(bcast) * accum;
    }
  }
};
    // This breaks because of sqrt, which is not provided by Eigen
template <typename T>
struct ApplyAdam<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    const auto one = static_cast<T>(1.0);
    m.device(d) =
        m +
        (beta1.constant(one) - beta1).reshape(single).broadcast(bcast) *
            (grad - m);
    v.device(d) =
        v +
        (beta2.constant(one) - beta2).reshape(single).broadcast(bcast) *
      (grad * grad.conjugate() - v);
    var.device(d) -= (lr * (beta2_power.constant(one) - beta2_power).sqrt() /
                      (beta1_power.constant(one) - beta1_power))
                         .reshape(single)
                         .broadcast(bcast) *
                     m / (epsilon.reshape(single).broadcast(bcast) + v.sqrt());
  }
};

 }  // namespace functor

  template struct functor::ApplyGradientDescent<GPUDevice, complex64>;
  template struct functor::ApplyMomentum<GPUDevice, complex64>;
  template struct functor::ApplyAdam<GPUDevice, complex64>;
  
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
