.PHONY: validate-config python-tests go-tests yaml-validate reproduction-dry-run artifact-consistency crp-dry-run record-versions docs-check license-check softwarex-%

PYTHON ?= python3
SOFTWAREX_CONFIG ?= experiments/config/softwarex_reproduction.yaml
SOFTWAREX_RUN_ID ?= dry-run

validate-config:
	$(PYTHON) experiments/validate_configs.py
	$(PYTHON) -c "from experiments.softwarex_reproduction import load_config; load_config('$(SOFTWAREX_CONFIG)')"
	$(PYTHON) -c "from experiments.validate_crp_trace import load_config; load_config('experiments/config/crp_validation.yaml')"

python-tests:
	$(PYTHON) -m unittest discover -s experiments/tests
	$(PYTHON) -m unittest discover -s ai-module/tests

go-tests:
	@if command -v go >/dev/null 2>&1; then \
		(cd actions-handler && go test ./...); \
		(cd controlled-replica-propagation && make test); \
	else \
		echo "go not available; skipping Go tests"; \
	fi

yaml-validate:
	$(PYTHON) experiments/validate_yaml_files.py

reproduction-dry-run:
	$(PYTHON) experiments/softwarex_reproduction.py --config $(SOFTWAREX_CONFIG) --stage all --dry-run --run-id $(SOFTWAREX_RUN_ID)

artifact-consistency:
	$(PYTHON) experiments/validate_artifacts.py --results-root results/softwarex

crp-dry-run:
	$(PYTHON) experiments/validate_crp_trace.py --dry-run --output results/softwarex-reproduction/$(SOFTWAREX_RUN_ID)/crp_validation_plan.csv

record-versions:
	$(PYTHON) experiments/record_versions.py --config $(SOFTWAREX_CONFIG) --output results/softwarex-reproduction/versions.json

docs-check:
	$(PYTHON) scripts/check_docs.py

license-check:
	@test -f LICENSE
	@if command -v pip-licenses >/dev/null 2>&1; then pip-licenses --format=plain; else echo "pip-licenses not installed; install requirements-dev.txt for dependency license report"; fi

softwarex-%:
	$(PYTHON) experiments/softwarex_reproduction.py --config $(SOFTWAREX_CONFIG) --stage $* --dry-run --run-id $(SOFTWAREX_RUN_ID)
