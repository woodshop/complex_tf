INCLUDE += -I /home/sarroff/repo/tensorflow
INCLUDE += -I $(shell python -c \
	'import tensorflow as tf; print(tf.sysconfig.get_include())')
CXX = gcc -std=c++11

CXXFLAGS = -DGOOGLE_CUDA=1          \
	-D_MWAITXINTRIN_H_INCLUDED  \
	-D_FORCE_INLINES            \
	$(INCLUDE) -fPIC -lcudart  

NVCC = nvcc -std=c++11
NVCCFLAGS = -DGOOGLE_CUDA=1         \
	-D_MWAITXINTRIN_H_INCLUDED  \
	-D_FORCE_INLINES            \
	$(INCLUDE) -c -x cu -Xcompiler -fPIC --expt-relaxed-constexpr

LDFLAGS = -shared

SRCS =                          \
	cwise_op_neg.cc    \
	cwise_op_abs.cc    \
	cwise_op_inverse.cc   \
	cwise_op_square.cc \
	cwise_op_log.cc    \
	cwise_op_tanh.cc   \
	# cwise_op_cplx_add.cc    \
	# cwise_op_cplx_sub.cc    \
	# cwise_op_cplx_mul.cc    \
	# cwise_op_cplx_div.cc    \
	# cwise_op_cplx_pow.cc    \
	# cwise_op_cplx_not_equal.cc    \

CUDA_SRCS =                            \
	cwise_op_gpu_neg.cu.cc    \
	cwise_op_gpu_abs.cu.cc    \
	cwise_op_gpu_inverse.cu.cc   \
	cwise_op_gpu_square.cu.cc \
	cwise_op_gpu_log.cu.cc    \
	cwise_op_gpu_tanh.cu.cc   \
	# cwise_op_gpu_cplx_add.cu.cc    \
	# cwise_op_gpu_cplx_sub.cu.cc    \
	# cwise_op_gpu_cplx_mul.cu.cc    \
	# cwise_op_gpu_cplx_div.cu.cc    \
	# cwise_op_gpu_cplx_pow.cu.cc    \
	# cwise_op_gpu_cplx_not_equal.cu.cc    \

RM = rm -f
TARGET_LIB = complextf.so

all: $(TARGET_LIB)

$(TARGET_LIB): $(CUDA_SRCS:.cc=.o) $(SRCS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(CXXFLAGS)

$(CUDA_SRCS:.cc=.o): $(CUDA_SRCS) 
	$(NVCC) $^ $(NVCCFLAGS)

.PHONY: clean
clean:
	-$(RM) $(TARGET_LIB)
	-$(RM) *~
	-$(RM) *.o
	-$(RM) *.so
