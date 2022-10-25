SHELL := /bin/bash
CONDAENV := environment.yaml

install: $(CONDAENV)
	conda env create -f $(CONDAENV)

install_ci: requirements.txt
	pip install --upgrade pip &&\
		pip install -r requirements.txt

build:
	python -m build

test:
	pytest -vv --cov --disable-warnings

format:
	black reconstruction tests
	isort reconstruction tests
	mypy reconstruction tests

lint:
	pylint -j 4 reconstruction tests

clean:
	rm -r __pycache__ .coverage .mypy_cache .pytest_cache *.log .ipynb_checkpoints dist

all: install lint test

.PHONY: lint format clean all