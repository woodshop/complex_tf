MAKE = make
RM = rm -f

.PHONY: all_libs
.PHONY: clean

all_libs:
	$(MAKE) -C complex_tf

clean:
	-$(RM) *~
	$(MAKE) -C complex_tf clean

