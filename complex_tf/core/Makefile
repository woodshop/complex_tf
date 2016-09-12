MAKE = make
RM = rm -f

.PHONY: all_libs
.PHONY: clean

all_libs:
	$(MAKE) -C kernels

clean:
	-$(RM) *~
	$(MAKE) -C kernels clean
