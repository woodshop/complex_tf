MAKE = make
RM = rm -f

PY_TESTS = $(wildcard *_test.py)

.PHONY: clean
.PHONY: test

test: $(PY_TESTS)
	$(foreach f,$^,python $(f);)

clean:
	-$(RM) *~
	-$(RM) *.pyc
