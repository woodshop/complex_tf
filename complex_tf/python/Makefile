MAKE = make
RM = rm -f

.PHONY: clean

clean:
	-$(RM) *~
	-$(RM) *.pyc
	$(MAKE) -C ops clean
	$(MAKE) -C kernel_tests clean
