# Results Directory Classification

`results/` is intentionally ignored except for this classification metadata.

Retained local result files in this checkout are not automatically validated by
Git. Use `experiments/validate_artifacts.py` before generating measured tables
from a complete measured matrix:

- `results/calibration/`: measured or partially measured calibration/debug
  material, depending on the subdirectory. Use the bundle metadata and raw logs
  to determine provenance before citing it.
- `results/calibration/headlight-bundle/` and `results/calibration/lstm-bundle/`:
  frozen model bundles if their `metadata.json` hash validation passes.
- `results/softwarex/`: default configured location for validated measured
  comparison bundles. Generated CSVs are valid only when accompanied by
  `artifact_validation_manifest.json`.
- Dry-run outputs under `results/softwarex-reproduction/dry-run/` are
  illustrative contract checks and sample output format.

Large artifacts that should be deposited in Zenodo for release validation:

- raw request traces;
- Prometheus range exports;
- Kubernetes event and CRP transition snapshots;
- model bundles and training metrics;
- generated measured artifact CSVs and plots;
- version-capture JSON files.
