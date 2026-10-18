PYTHON ?= python3
FIGURES := $(wildcard figures/fig*.py)

.PHONY: help test figures check-figures check clean

help:
	@echo "make test            run the test suite"
	@echo "make figures         regenerate every figure into figures/"
	@echo "make check-figures   fail if any committed figure differs from what the code produces"
	@echo "make check           test, then check-figures"
	@echo "make clean           remove caches"

test:
	$(PYTHON) -m pytest -q

# Each script writes its own PNG next to itself, named in its save() call.
figures:
	@for f in $(FIGURES); do \
	  printf '%-34s ' "$$(basename $$f .py)"; \
	  ( cd figures && $(PYTHON) "$$(basename $$f)" >/dev/null 2>&1 ) \
	    && echo OK || { echo FAILED; exit 1; }; \
	done

# Nothing otherwise guarantees that a committed PNG was produced by the
# committed code, so a parameter change silently stales every figure. This
# regenerates into a scratch copy and diffs, leaving the working tree alone.
check-figures:
	@tmp=$$(mktemp -d) && trap 'rm -rf "$$tmp"' EXIT && \
	git ls-files -z | xargs -0 tar cf - | ( cd "$$tmp" && tar xf - ) && \
	for f in $(FIGURES); do \
	  ( cd "$$tmp/figures" && $(PYTHON) "$$(basename $$f)" >/dev/null 2>&1 ) \
	    || { echo "FAILED to run $$f"; exit 1; }; \
	done && \
	stale=0; \
	for png in figures/*.png; do \
	  if ! cmp -s "$$png" "$$tmp/$$png"; then echo "STALE: $$png"; stale=1; fi; \
	done; \
	if [ "$$stale" -eq 0 ]; then echo "all figures match the committed code"; \
	else echo; echo "run 'make figures' and commit the result"; exit 1; fi

check: test check-figures

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache
