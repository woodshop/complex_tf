#ifndef TENSORFLOW_KERNELS_ZERO_OUT_OP_H_
#define TENSORFLOW_KERNELS_ZERO_OUT_OP_H_

#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

  namespace functor {
   
    template <typename Device, typename T>
      struct ZeroOutFunctor;

    template <typename Device, typename T>
    struct ZeroOutFunctor {
      void operator()(const Device& d,
		      typename TTypes<T>::ConstFlat input,
		      typename TTypes<T>::Flat output,
		      const int N);
    };
    
  }  // namespace functor
  
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_ZERO_OUT_OP_H_
