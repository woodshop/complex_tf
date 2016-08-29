INCLUDE = 
CXX = gcc -std=c++11 
CXXFLAGS = -I $(INCLUDE) -fPIC -lcudart -D_GLIBCXX_USE_CXX11_ABI=0
NVCC = nvcc -std=c++11 -c
NVCCFLAGS = -I $(INCLUDE) -D GOOGLE_CUDA=1 -D_MWAITXINTRIN_H_INCLUDED \
	-D_FORCE_INLINES -x cu -Xcompiler -fPIC
LDFLAGS = -shared
CUDA_SRCS = zero_out_gpu.cu.cc
SRCS = zero_out.cc
RM = rm -f
TARGET_LIB = libcomplextf.so
CUDA_OBJ = complex_tf.o

all: INCLUDE += $(shell python -c \
	'import tensorflow as tf; print(tf.sysconfig.get_include())')
# all: INCLUDE += /home/sarroff/repo/tensorflow_dnu/tensorflow/core/util/
all: $(TARGET_LIB)

# nvcc -std=c++11 -c -o cuda_op_kernel.cu.o cuda_op_kernel.cu.cc \
# -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# g++ -std=c++11 -shared -o cuda_op_kernel.so cuda_op_kernel.cc \
# cuda_op_kernel.cu.o -I $TF_INC -fPIC -lcudart

$(TARGET_LIB): $(SRCS) $(CUDA_OBJ) 
	$(CXX) $(LDFLAGS) -o $@ $^ $(CXXFLAGS)

$(CUDA_OBJ): $(CUDA_SRCS)
	$(NVCC) -o $@ $^ $(NVCCFLAGS) 

.PHONY: clean
clean:
	-$(RM) $(TARGET_LIB)
	-$(RM) *~
	-$(RM) *.o
