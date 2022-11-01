SHELL := /bin/bash
CONDAENV := environment.yml
REQ := requirements.txt

install: $(CONDAENV)
	conda env create -f $(CONDAENV)

install_ci: $(REQ)
	pip install --upgrade pip &&\
		pip install -r $(REQ)

build:
	python -m build

test:
	pytest -vv --cov --disable-warnings --cov-report=xml

format:
	black src tests
	isort src tests
	mypy src tests

lint:
	pylint -j 4 src tests

clean:
	rm -r .coverage .mypy_cache .pytest_cache dist src/*.egg-info
	find . -name "__pycache__" -exec rm -r {} +

all: install lint test

.PHONY: lint format clean all