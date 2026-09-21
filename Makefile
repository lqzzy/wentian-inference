.PHONY: help test verify model-check format lint

PYTHON ?= python3
RUFF ?= ruff
export PYTHONPATH := $(CURDIR)/src

help:
	@$(PYTHON) -m wentian --help

test:
	@$(PYTHON) -m unittest discover -s tests -v

verify: test
	@$(PYTHON) scripts/verify_artifacts.py

model-check:
	@$(PYTHON) scripts/ensure_checkpoint.py
	@$(PYTHON) scripts/check_model.py

format:
	@$(RUFF) check --fix src scripts tests
	@$(RUFF) format src scripts tests

lint:
	@$(RUFF) format --check src scripts tests
	@$(RUFF) check src scripts tests
