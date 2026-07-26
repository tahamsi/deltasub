PYTHON ?= python
export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)

.PHONY: bootstrap doctor smoke diagnostic tables

bootstrap:
	$(PYTHON) -m pip install -e '.[dev]'

doctor:
	$(PYTHON) -m deltasub.cli doctor

smoke:
	bash scripts/run_local_smoke.sh

diagnostic:
	bash scripts/run_a100_diagnostic.sh

tables:
	bash scripts/collect_results.sh
