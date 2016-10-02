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

CUDA_SRCS = $(wildcard *.cu.cc)
SRCS = $(filter-out $(CUDA_SRCS),$(wildcard *.cc)) $(PROTOBUF_LIB)

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
