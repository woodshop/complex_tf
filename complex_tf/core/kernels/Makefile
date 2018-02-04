TF_CFLAGS = $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS = $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
TF_CFLAGS += -I$(TF_SRC)
TF_LFLAGS += -ltensorflow_framework

CXX = g++ -std=c++11 -shared
NVCC = nvcc -std=c++11 -c
CUDA_SRCS = $(wildcard *.cu.cc)
SRCS = $(filter-out $(CUDA_SRCS),$(wildcard *.cc)) # $(PROTOBUF_LIB)

RM = rm -f
TARGET_LIB = complextf.so

all: check-env $(TARGET_LIB)

check-env:
ifndef TF_SRC
    $(error TF_SRC is undefined)
endif

$(TARGET_LIB): $(CUDA_SRCS:.cc=.o) $(SRCS)
	$(CXX) -o $@ $^ $(TF_CFLAGS) -D GOOGLE_CUDA=1 -fPIC -lcudart $(TF_LFLAGS)

$(CUDA_SRCS:.cc=.o): $(CUDA_SRCS)
	$(NVCC) $^ $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC  --expt-relaxed-constexpr $(TF_LFLAGS)

.PHONY: clean
clean:
	-$(RM) $(TARGET_LIB)
	-$(RM) *~
	-$(RM) *.o
	-$(RM) *.so
