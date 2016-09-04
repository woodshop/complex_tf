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

LDFLAGS = -shared
CUDA_SRCS = zero_out_gpu.cu.cc cwise_op_gpu_cplx_tanh.cu.cc

SRCS = zero_out.cc cwise_op_cplx_tanh.cc

RM = rm -f
TARGET_LIB = complextf.so

all: $(TARGET_LIB)

$(TARGET_LIB): $(CUDA_SRCS:.cc=.o) $(SRCS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(CXXFLAGS)

$(CUDA_SRCS:.cc=.o): $(CUDA_SRCS) 
	$(NVCC) $^ $(NVCCFLAGS)

$(CUDA_SRCS): zero_out.h

$(SRCS): zero_out.h

.PHONY: clean
clean:
	-$(RM) $(TARGET_LIB)
	-$(RM) *~
	-$(RM) *.o
	-$(RM) *.so
