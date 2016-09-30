INCLUDE += -I $(TF_SRC)

INCLUDE += -I $(shell python -c \
	'import tensorflow as tf; print(tf.sysconfig.get_include())')

CXX = gcc -std=c++11 -L$(CUDA_HOME)/lib64

CXXFLAGS = -DGOOGLE_CUDA=1          \
	-D_MWAITXINTRIN_H_INCLUDED  \
	-D_FORCE_INLINES            \
	-DEIGEN_AVOID_STL_ARRAY     \
	$(INCLUDE) -fPIC            \

# Using libprotobuf_lite to resolve runtime linking errors. Much
# simpler than figuring out how to compile and link protobuf with
# the project, but shoudl find a better solution.
PROTOBUF_LIB = $(TF_SRC)/bazel-bin/external/protobuf/libprotobuf_lite.pic.a

NVCC = nvcc -std=c++11
NVCCFLAGS = -DGOOGLE_CUDA=1         \
	-D_MWAITXINTRIN_H_INCLUDED  \
	-D_FORCE_INLINES            \
	-DEIGEN_AVOID_STL_ARRAY     \
	$(INCLUDE) -c -x cu -Xcompiler -fPIC --expt-relaxed-constexpr \

LDFLAGS = -shared -lprotobuf -lcudart

SRCS =                           	\
	cwise_op_neg.cc          	\
	cwise_op_inverse.cc      	\
	cwise_op_square.cc       	\
	cwise_op_log.cc          	\
	cwise_op_tanh.cc         	\
	cwise_op_pow.cc          	\
	cwise_op_not_equal_to.cc 	\
	constant_op.cc			\
	training_ops.cc			\
	variable_ops.cc			\
	dense_update_ops.cc		\
	training_ops.cc			\
	$(PROTOBUF_LIB)

CUDA_SRCS =                             \
	cwise_op_gpu_neg.cu.cc          \
	cwise_op_gpu_inverse.cu.cc      \
	cwise_op_gpu_square.cu.cc       \
	cwise_op_gpu_log.cu.cc          \
	cwise_op_gpu_tanh.cu.cc         \
	cwise_op_gpu_pow.cu.cc          \
	cwise_op_gpu_not_equal_to.cu.cc \
	constant_op_gpu.cu.cc		\
	dense_update_ops_gpu.cu.cc	\
	training_ops_gpu.cu.cc		\

RM = rm -f
TARGET_LIB = complextf.so

all: check-env $(TARGET_LIB)

check-env:
ifndef TF_SRC
    $(error TF_SRC is undefined)
endif

$(TARGET_LIB): $(CUDA_SRCS:.cc=.o) $(SRCS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

$(CUDA_SRCS:.cc=.o): $(CUDA_SRCS) 
	$(NVCC) $^ $(NVCCFLAGS)

.PHONY: clean
clean:
	-$(RM) $(TARGET_LIB)
	-$(RM) *~
	-$(RM) *.o
	-$(RM) *.so
