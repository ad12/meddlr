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
	MEDDLR_TEST_MODEL_ZOOS=True pytest tests/ --cov=./ --cov-report=xml

dev:
	pip install --upgrade black==21.10b0 coverage isort flake8 flake8-bugbear flake8-comprehensions pre-commit pooch
	pre-commit install

build-docs:
	rm -rf docs/build
	rm -rf docs/source/apidocs/generated
	sphinx-build -b html docs/source/ docs/build/html/

all: autoformat test build-docs