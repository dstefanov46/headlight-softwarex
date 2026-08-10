# Artifact Policy

The repository snapshot contains ignored local files under `results/`. They are
retained for provenance and debugging. Measured tables use bundles from the
configured measured results root after the artifact validator reports
`validation_status=validated`.

Use `experiments/validate_artifacts.py` before generating measured tables.
Dry-run output demonstrates the output format and keeps its provenance visible:

```bash
python3 experiments/validate_artifacts.py --results-root results/softwarex
python3 experiments/generate_paper_artifacts.py --replace-stale-artifacts
```
