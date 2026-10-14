# Validated evaluation

The evaluation contains 60 valid measured runs: four controllers, three workloads
and five seeds.

| Controller | Workloads | Seeds | Runs |
| --- | --- | --- | ---: |
| Headlight | gradual-ramp, flash-crowd, diurnal | 101–105 | 15 |
| HPA | gradual-ramp, flash-crowd, diurnal | 101–105 | 15 |
| KEDA | gradual-ramp, flash-crowd, diurnal | 101–105 | 15 |
| Threshold | gradual-ramp, flash-crowd, diurnal | 101–105 | 15 |

[run-index.csv](run-index.csv) and [run-index.json](run-index.json) identify the
selected runs, raw traces, measurement windows, configuration and provenance.
Use this selection when reproducing the evaluation aggregates.

```bash
python experiments/validate_softwarex_evaluation.py --output results/controller-archive/validation.json
```

The validator requires exactly these 60 cells, valid summaries, consistent run
identities and intact artifact checksums. It returns `validation_status=validated`
and the twelve controller/workload aggregates. [validation.json](validation.json)
records the checked artifacts and mean ± sample SD. `sla_violation_rate` is a
fraction; multiply by 100 for the percentage.

Raw request latency and resource measurements are preserved. A change to metadata
wording does not change the observations. Original artifact hashes are retained
as `source_sha256` for metadata files whose descriptive labels were normalized.

The evaluation treats the five seeds as comparable. Results describe one
Train Ticket service slice on a shared-host Kind federation. Its interpretation
limits are documented in [evaluation scope](../../docs/evaluation.md).
Dry-run records are separate from these measured runs.
