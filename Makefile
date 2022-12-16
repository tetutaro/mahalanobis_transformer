PACKAGE = mahalanobis_transformer

.PHONY: clean
clean:
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -fr {} +
	@find . -name '.DS_Store' -exec rm -f {} +
	@rm -rf dist
	@rm -rf build
	@rm -rf $(PACKAGE).egg-info
	@rm -rf .pytest_cache
	@rm -f .coverage
	@rm -rf htmlcov/

.PHONY: build-package
build-package:
	$(eval VERSION := $(shell poetry version -s))
	poetry build
	@tar zxf dist/$(PACKAGE)-$(VERSION).tar.gz -C ./dist
	@cp dist/$(PACKAGE)-$(VERSION)/setup.py setup.py
	@rm -rf dist

.PHONY: install
install:
	python3 setup.py install

.PHONY: uninstall
uninstall:
	pip3 uninstall -y $(PACKAGE)

.PHONY: test
test:
#	python3 -u -m pytest -v --cov --cov-report=html
	poetry run test

.PHONY: doc
doc:
	cd docs && make html

.PHONY: exclude-export
exclude-export:
	tar zcvf exclude_export.tgz .git .ipynb_checkpoints .python-version .virtual_documents poetry.lock
	@rm -rf .git
	@rm -rf .ipynb_checkpoints
	@rm -f .python-version
	@rm -rf .virtual_documents
	@rm -f poetry.lock
