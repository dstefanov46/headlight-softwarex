# Reproduction

Run commands from the repository root. The measured workflow is intended for a
dedicated measurement host with Docker, Kind, Karmada, `kubectl`, Helm,
`istioctl`, Python, and Go available. Dry-run and sample outputs are contract
checks that demonstrate output format. Measured performance tables and plots use
the validation and artifact-generation workflow.

## 1. Repository, Config, And Test Validation

These checks do not start the capacity campaign, calibration trace, grid search,
tuning campaign, or measured comparison matrix:

```bash
make validate-config
make python-tests
make yaml-validate
make reproduction-dry-run
make go-tests
```

The reproduction dry run is equivalent to:

```bash
python3 experiments/softwarex_reproduction.py \
  --config experiments/config/softwarex_reproduction.yaml \
  --stage all --dry-run --run-id dry-run
```

Important output: `results/softwarex-reproduction/dry-run/`. Its status is
`dry-run-unvalidated`; it validates planning and file contracts only. The
illustrative compact table is:

```text
results/softwarex-reproduction/dry-run/compact_results.csv
```

That CSV is produced by the dry-run command above and includes a
`validation_status` column set to `dry-run-unvalidated`. It demonstrates the
sample output format.

## 2. Testbed Bootstrap

Set the kubeconfig path used by the workflow:

```bash
export HEADLIGHT_KUBECONFIG="$PWD/.kube/headlight-testbed.config"
export KUBECONFIG="$HEADLIGHT_KUBECONFIG"
```

For a clean rebuild of the local Karmada/Kind measurement testbed, run:

```bash
./experiments/rebuild_local_testbed.sh
```

Important inputs: Docker, Kind, `karmadactl`, Helm, `istioctl`, and
`experiments/testbed/kind/single-node.yaml`.

Important outputs: `.kube/headlight-testbed.config`,
`.kube/headlight-members-internal.config`, ready contexts
`karmada-apiserver`, `karmada-host`, `cluster1`, `cluster2`, and `cluster3`,
installed CRP/actions-handler control components, Istio on member clusters, and
metrics-server. The helper also deploys the default minimal Train Ticket slice;
the next step redeploys the measured-workflow slice explicitly.

## 3. Train Ticket Deployment

Deploy the advanced-ticket slice used by the measured workflow:

```bash
HEADLIGHT_TRAINTICKET_PATH=advanced-ticket \
HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE=capacity-vm \
HEADLIGHT_TRAINTICKET_DEPLOY_MODE=karmada \
HEADLIGHT_TRAINTICKET_ACTIVE_CLUSTERS="cluster2 cluster3" \
./experiments/deploy_minimal_trainticket.sh
```

Important input: `ai-module/configs/service_profiles/advanced-ticket.json`.
Important output: `experiments/testbed/trainticket-minimal/rendered-advanced-ticket.yaml`
unless `HEADLIGHT_MINIMAL_TRAINTICKET_MANIFEST` overrides it. `capacity-vm` is
the existing Train Ticket resource profile name, not a host-class requirement.

## 4. Capacity Qualification

Run the capacity pilots and seed-101 controller qualification:

```bash
capacity_dir="results/capacity-qualification/softwarex-advanced-ticket"

./experiments/run_capacity_qualification.sh \
  --output-dir "$capacity_dir" \
  --trainticket-path advanced-ticket \
  --resource-profile capacity-vm
```

Important inputs: the bootstrapped testbed, Prometheus in the monitored member
cluster, the advanced-ticket service profile, and any frozen bundles required by
the selected controllers. For the default controller set, Headlight and LSTM
bundle preflight uses `results/calibration/headlight-bundle` and
`results/calibration/lstm-bundle` unless `HEADLIGHT_MODEL_BUNDLE_DIR` or
`HEADLIGHT_LSTM_BUNDLE_DIR` is set.

Important outputs: `$capacity_dir/campaign_status.json`,
`$capacity_dir/capacity_summary.json`, pilot bundles under
`$capacity_dir/pilots/`, and qualification bundles under
`$capacity_dir/qualification/`.

## 5. Read Selected Capacity

Extract the selected capacity from the generated capacity summary:

```bash
selected_capacity="$(
  python3 - "$capacity_dir/capacity_summary.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["selected_capacity"])
PY
)"
printf 'selected_capacity=%s\n' "$selected_capacity"
```

Important input: `$capacity_dir/capacity_summary.json`. Important output:
the shell variable `selected_capacity`, which is used by calibration and tuning.

To summarize reduced-slice capacity rows without changing the selected-capacity
source, run:

```bash
python3 experiments/summarize_capacity_results.py \
  "$capacity_dir" \
  --candidate-output "$capacity_dir/capacity_rows.csv"
```

## 6. Full Calibration Trace

Run the release-candidate calibration trace:

```bash
HEADLIGHT_CALIBRATION_PHASE=paper \
HEADLIGHT_CALIBRATION_TRAINTICKET_PATH=advanced-ticket \
HEADLIGHT_CALIBRATION_MAX_USERS="$selected_capacity" \
./experiments/run_calibration_trace.sh
```

Important inputs: the advanced-ticket deployment, Prometheus, and
`selected_capacity`. Important outputs: `results/calibration/train-ticket-calibration.csv`,
`results/calibration/static-adjacency.json`,
`results/calibration/calibration_manifest.json`, raw telemetry under
`results/calibration/raw/`, logs under `results/calibration/logs/`, and feature
histories under `results/calibration/feature_history/`.

## 7. Headlight Grid Search

Train and rank the fixed 48-candidate offline Headlight grid before any measured
seeds are run:

```bash
python3 experiments/run_headlight_grid.py
```

Important inputs: `results/calibration/train-ticket-calibration.csv`,
`results/calibration/static-adjacency.json`, and
`ai-module/configs/service_profiles/advanced-ticket.json`. Important outputs:
`results/headlight-grid/grid-plan.json`,
`results/headlight-grid/grid-results.csv`, candidate bundles under
`results/headlight-grid/h*/bundle/`, and `results/headlight-grid/best-three.json`.

## 8. Verify `best-three.json`

Verify that the grid produced exactly three tuning candidates:

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("results/headlight-grid/best-three.json")
payload = json.loads(path.read_text(encoding="utf-8"))
assert len(payload.get("candidates", [])) == 3, path
print(path)
PY
```

Important input and output: `results/headlight-grid/best-three.json`.

## 9. Tuning With Selected Capacity

Run tuning on the best three candidates using the capacity selected before
calibration:

```bash
python3 experiments/run_headlight_tuning.py \
  --selected-capacity "$selected_capacity"
```

Important inputs: `results/headlight-grid/best-three.json`, the three candidate
bundles referenced by it, and `selected_capacity`. Important output:
`results/headlight-tuning/winner.json`. Tuning uses seeds 41 and 43 and occurs
before the measured comparison seeds.

## 10. Verify Frozen Winner

Verify the winner metadata and frozen model bundle before running the measured
matrix:

```bash
python3 - <<'PY'
import json
from pathlib import Path

winner_path = Path("results/headlight-tuning/winner.json")
payload = json.loads(winner_path.read_text(encoding="utf-8"))
assert payload.get("frozen_before_paper_seeds") is True, winner_path
bundle = Path(payload["winner"]["bundle"])
metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
assert metadata.get("frozen") is True, bundle
print(bundle)
PY
```

Important inputs: `results/headlight-tuning/winner.json` and the winner bundle's
`metadata.json`. The frozen winner is required by `experiments/run_paper_matrix.sh`;
the matrix script fails before running any measured seed if the winner is missing,
not marked `frozen_before_paper_seeds`, or the bundle metadata is not frozen.

## 11. Measured Comparison Matrix

Run the measured comparison matrix after model selection is frozen:

```bash
./experiments/run_paper_matrix.sh
```

Important inputs: `results/headlight-tuning/winner.json`, the frozen winner
bundle, measured run configs under `experiments/configs/runs/`, and workload
profiles under `experiments/configs/workloads/`. By default the matrix uses
Headlight, HPA, KEDA, LSTM, static, and threshold controllers; diurnal,
flash-crowd, and gradual-ramp workloads; and seeds 101, 102, 103, 104, and 105.

Important outputs: measured seed bundles under the configured results root
(`results/softwarex/` by default) and progress state in
`results/softwarex/matrix-progress.tsv`. Use `HEADLIGHT_RESUME=1` only to
resume already validated completed matrix entries.

## 12. Artifact Validation

Validate the measured matrix before interpreting it:

```bash
python3 experiments/validate_artifacts.py --results-root results/softwarex
```

Important input: `results/softwarex/`. Important output: JSON on stdout with
`validation_status=validated`, `validated_rows`, and the validated matrix. Any
failure identifies measured bundle fields that need correction before artifact
generation.

## 13. Measured Artifact Generation

Generate measured CSV artifacts only after validation succeeds:

```bash
python3 experiments/generate_paper_artifacts.py \
  --results-root results/softwarex \
  --output-dir results/softwarex/artifacts
```

Important inputs: validated measured bundles under `results/softwarex/`. Important
outputs: `results/softwarex/artifacts/main_comparison.csv`,
`load_profiles.csv`, `ablation.csv`, `migration_offload.csv`, `overhead.csv`,
and `artifact_validation_manifest.json`.

Use validated measured artifacts for performance tables and plots. Keep dry-run
outputs, sample calibration files, incomplete matrix bundles, and other local
files separate from measured artifacts.
