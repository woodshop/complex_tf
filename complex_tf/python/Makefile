MAKE = make
RM = rm -f

.PHONY: clean
.PHONY: test

test:
	$(MAKE) -C kernel_tests test
	$(MAKE) -C training test
	$(MAKE) -C ops test

clean:
	-$(RM) *~
	-$(RM) *.pyc
	$(MAKE) -C kernel_tests clean
	$(MAKE) -C training clean
	$(MAKE) -C ops clean
