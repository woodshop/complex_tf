MAKE = make
RM = rm -f

.PHONY: all_libs
.PHONY: clean
.PHONY: test

all_libs:
	$(MAKE) -C complex_tf

test:
	$(MAKE) -C complex_tf test

clean:
	-$(RM) *~
	$(MAKE) -C complex_tf clean
