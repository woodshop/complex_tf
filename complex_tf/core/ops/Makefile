INCLUDE = 
CXX = gcc -std=c++11 
CXXFLAGS = -I$(INCLUDE) -fPIC -D_GLIBCXX_USE_CXX11_ABI=0
LDFLAGS = -shared
SRCS = zero_out.cc
RM = rm -f
TARGET_LIB = libcomplextf.so

all: $(TARGET_LIB)

$(TARGET_LIB): INCLUDE += $(shell python -c \
	'import tensorflow as tf; print(tf.sysconfig.get_include())')
$(TARGET_LIB): $(SRCS)
	$(CXX) $(LDFLAGS) $^ -o $@ $(CXXFLAGS)

.PHONY: clean
clean:
	-$(RM) $(TARGET_LIB)
	-$(RM) *~
