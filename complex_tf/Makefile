MAKE = make
RM = rm -f

.PHONY: all_libs
.PHONY: clean

all_libs:
	$(MAKE) -C core

clean:
	-$(RM) *~
	-$(RM) *.pyc
	$(MAKE) -C core clean
	$(MAKE) -C python clean
