SHELL := /bin/bash
CONDAENV := environment.yml

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
	black src tests
	isort src tests
	mypy src tests

lint:
	pylint -j 4 src tests

clean:
	rm -r .coverage .mypy_cache .pytest_cache dist src/*.egg-info

all: install lint test

.PHONY: lint format clean all