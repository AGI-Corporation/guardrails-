.PHONY: install install-dev test test-cov lint clean help

help:          ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:       ## Install runtime dependencies
	pip install -r requirements.txt

install-dev:   ## Install development + test dependencies
	pip install -r requirements-dev.txt

test:          ## Run the full test suite
	pytest tests/

test-cov:      ## Run tests and show a coverage report
	pytest tests/ --cov=. --cov-report=term-missing

clean:         ## Remove pytest / coverage artifacts
	rm -rf .pytest_cache htmlcov coverage.xml .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
