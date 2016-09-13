
INCLUDE += -I $(TF_SRC)
INCLUDE += -I $(shell python -c \
	'import tensorflow as tf; print(tf.sysconfig.get_include())')
CXX = gcc -std=c++11

CXXFLAGS = -DGOOGLE_CUDA=1          \
	-D_MWAITXINTRIN_H_INCLUDED  \
	-D_FORCE_INLINES            \
	-DEIGEN_AVOID_STL_ARRAY     \
	$(INCLUDE) -fPIC -lcudart  

NVCC = nvcc -std=c++11
NVCCFLAGS = -DGOOGLE_CUDA=1         \
	-D_MWAITXINTRIN_H_INCLUDED  \
	-D_FORCE_INLINES            \
	-DEIGEN_AVOID_STL_ARRAY     \
	$(INCLUDE) -c -x cu -Xcompiler -fPIC --expt-relaxed-constexpr

LDFLAGS = -shared

SRCS =                           \
	cwise_op_neg.cc          \
	cwise_op_abs.cc          \
	cwise_op_inverse.cc      \
	cwise_op_square.cc       \
	cwise_op_log.cc          \
	cwise_op_tanh.cc         \
	cwise_op_sub.cc          \
	cwise_op_mul.cc          \
	cwise_op_div.cc          \
	cwise_op_pow.cc          \
	cwise_op_not_equal_to.cc \

CUDA_SRCS =                             \
	cwise_op_gpu_neg.cu.cc          \
	cwise_op_gpu_abs.cu.cc          \
	cwise_op_gpu_inverse.cu.cc      \
	cwise_op_gpu_square.cu.cc       \
	cwise_op_gpu_log.cu.cc          \
	cwise_op_gpu_tanh.cu.cc         \
	cwise_op_gpu_sub.cu.cc          \
	cwise_op_gpu_mul.cu.cc          \
	cwise_op_gpu_div.cu.cc          \
	cwise_op_gpu_pow.cu.cc          \
	cwise_op_gpu_not_equal_to.cu.cc \

RM = rm -f
TARGET_LIB = complextf.so

all: check-env $(TARGET_LIB)

check-env:
ifndef TF_SRC
    $(error TF_SRC is undefined)
endif

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
