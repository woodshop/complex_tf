MAKE = make
RM = rm -f

.PHONY: clean
.PHONY: test

test:
	$(MAKE) -C training test

clean:
	-$(RM) *~
	-$(RM) *.pyc
	$(MAKE) -C training clean
