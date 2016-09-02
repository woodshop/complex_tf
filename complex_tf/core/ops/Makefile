INCLUDE += -I /home/sarroff/repo/tensorflow
INCLUDE += -I $(shell python -c \
	'import tensorflow as tf; print(tf.sysconfig.get_include())')

CXX = gcc -std=c++11
CXXFLAGS = -DGOOGLE_CUDA=1          \
	-D_MWAITXINTRIN_H_INCLUDED  \
	-D_FORCE_INLINES            \
	$(INCLUDE) -fPIC -lcudart   \

NVCC = nvcc -std=c++11 -c
NVCCFLAGS = -DGOOGLE_CUDA=1         \
	-D_MWAITXINTRIN_H_INCLUDED  \
	-D_FORCE_INLINES            \
	$(INCLUDE) -x cu -Xcompiler -fPIC

LDFLAGS = -shared
CUDA_SRCS = zero_out_gpu.cu.cc
SRCS = zero_out.cc
RM = rm -f
TARGET_LIB = zero_out.so
CUDA_OBJ = zero_out.cu.o

all: $(TARGET_LIB)

$(TARGET_LIB): $(CUDA_OBJ) $(SRCS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(CXXFLAGS)

$(CUDA_OBJ): $(CUDA_SRCS) 
	$(NVCC) -o $@ $^ $(NVCCFLAGS)

$(CUDA_SRCS): zero_out.h

$(SRCS): zero_out.h

.PHONY: clean
clean:
	-$(RM) $(TARGET_LIB)
	-$(RM) *~
	-$(RM) *.o
	-$(RM) *.so
