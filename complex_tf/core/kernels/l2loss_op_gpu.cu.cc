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
#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;
  
namespace functor {

// template <typename T>
// struct L2Loss<GPUDevice, T> {
//   void operator()(const GPUDevice& d,
// 		  typename TTypes<T>::ConstTensor input,
//                   typename TTypes<T>::Scalar output) {
    
//     auto mult = input * input.conjugate();
//     auto mult_r = mult.real().sum();
//     auto mult_i = mult.imag().sum();

//     Tensor tmp_t;
//       //TTypes<float>::Scalar *mult_rr;
//     //auto buf_r = mult_rr.flat<float>();
//     OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value,
// 						   TensorShape({}), &tmp_t));
//     typename TTypes<float>::Scalar buf_r = tmp_t.scalar<T>();
    
//     buf_r = mult_r;

//     Tensor mult_ii(DT_FLOAT, TensorShape({1}));
//     auto buf_i = mult_ii.flat<float>();
//     buf_i = mult_i;

//     output.device(d) = buf_r.binaryExpr(buf_i, make_complex_func<float>());
//     //TTypes<float>::Scalar mult_i = mult.imag().sum();
//     // auto cplx_mult = mult_r.binaryExpr(mult_i, make_complex_func<float>());
//     //output.device(d) = cplx_mult;
//     //output.real() = mult_r;
//     //  mult_r.binaryExpr(mult_i, make_complex_func<float>());
//   }
// };

  template <typename T>
  struct L2Loss<GPUDevice, T> {
    void operator()(const GPUDevice& d,
		    typename TTypes<T>::ConstTensor input,
		    typename TTypes<T>::Scalar output) {
      output.device(d) =
	(input * input.conjugate()).sum() * static_cast<complex64>(0.5);
  }
};

}  // namespace functor

template struct functor::L2Loss<GPUDevice, complex64>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
