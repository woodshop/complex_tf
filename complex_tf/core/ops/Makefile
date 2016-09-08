INCLUDE += -I /home/sarroff/repo/tensorflow
INCLUDE += -I /home/sarroff/repo/pycuda/pycuda/cuda
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

# muldefs fixes linking problem created by including pycuda in each object
LDFLAGS = -shared -z muldefs

SRCS =                         \
	cwise_op_cplx_neg.cc   \
	cwise_op_cplx_tanh.cc  \
	cwise_op_cplx_add.cc   \
	cwise_op_cplx_sub.cc   \
	cwise_op_cplx_mul.cc   \
	cwise_op_cplx_div.cc   \
	cwise_op_cplx_pow.cc   \
#	cwise_op_cplx_square.cc \

CUDA_SRCS =                           \
	cwise_op_gpu_cplx_neg.cu.cc   \
	cwise_op_gpu_cplx_tanh.cu.cc  \
	cwise_op_gpu_cplx_add.cu.cc   \
	cwise_op_gpu_cplx_sub.cu.cc   \
	cwise_op_gpu_cplx_mul.cu.cc   \
	cwise_op_gpu_cplx_div.cu.cc   \
	cwise_op_gpu_cplx_pow.cu.cc   \
#	cwise_op_gpu_cplx_square.cu.cc \

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
