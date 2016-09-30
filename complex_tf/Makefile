MAKE = make
RM = rm -f

.PHONY: all_libs
.PHONY: clean
.PHONY: test

all_libs:
	$(MAKE) -C core

test:
	$(MAKE) -C python test

clean:
	-$(RM) *~
	-$(RM) *.pyc
	$(MAKE) -C core clean
	$(MAKE) -C python clean
