autoformat:
	set -e
	isort .
	black --config pyproject.toml .
	flake8

lint:
	set -e
	isort -c .
	black --check --config pyproject.toml .
	flake8

test:
	set -e
	coverage run -m pytest tests/

test-cov:
	set -e
	pytest tests/ --cov=./ --cov-report=xml

dev:
	pip install --upgrade -e '.[alldev]'
	pre-commit install

dev-lint:
	pip install --upgrade black==22.3.0 coverage isort flake8 flake8-bugbear flake8-comprehensions pre-commit pooch

build-docs:
	rm -rf docs/build
	rm -rf docs/source/apidocs/generated
	rm -rf docs/assets/temp
	python docs/source/autogen.py
	sphinx-build -b html docs/source/ docs/build/html/

all: autoformat test build-docs