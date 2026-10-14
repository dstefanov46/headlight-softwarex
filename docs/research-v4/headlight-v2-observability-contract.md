# Headlight observability and profiling dataset contract

Dry-run validation can check record schemas and serialization; it does not
provide measured controller outcomes. Performance claims require measured
results from completed runs.

Stage 2 introduces versioned, allocation-sensitive records without changing
the HeadLight-v1 controller, its policy decisions, or research-v3 artifacts.
The contract lives in `headlight.observability` and is deliberately opt-in:
research-v3 action logs remain unversioned JSONL and are read as legacy rows,
not converted into v2 records.

## Decision records

`headlight.decision-record/v2` is one complete decision for one service. It
requires a stable record/run/seed identity, timestamp, service, observed demand
and CPU/latency/error values, per-field missing/stale masks, and separate ready,
desired, and pending replicas for every cluster. It also records action recency,
per-horizon q50/q90 demand forecasts, every valid candidate placement and total,
candidate p95/SLA predictions, resource/SLA-risk/churn terms, the selected
candidate, deterministic reason/rule, action lifecycle timestamps, readiness
delay, policy identifiers/versions, and feature-schema and adjacency hashes.

Unavailable candidate predictions must be represented explicitly with
`prediction_available=false` and null prediction values. A no-action decision is
represented explicitly as `action_timestamps.status=not_requested`; timestamps
and readiness delay must then be null. The validator never substitutes desired
replicas for ready replicas, computes a missing action, or fills an outcome.

Action lifecycle states are ordered: `requested` has only `requested_at`;
`acknowledged` adds `acknowledged_at`; `ready` requires both plus `ready_at` and
the exact readiness delay. A `failed` action may retain an acknowledgement but
must never claim Ready or a readiness delay.

Candidate placement totals must match their cluster allocation, q50 must not
exceed q90, SLA probability is bounded to `[0, 1]`, and Ready delay must equal
`ready_at - requested_at`. Reactive emergency records must set
`action_origin=reactive_emergency` and `out_of_distribution=true`.

## Profiling records and datasets

`headlight.profiling-record/v2` wraps one validated decision with complete,
per-horizon/per-endpoint end-to-end outcomes: offered/completed requests,
end-to-end p95 latency, and SLA violations. It carries an explicit
`clean_action_label` and exclusion reasons. Out-of-distribution emergency
records cannot be clean labels.

`headlight.profiling-dataset/v2` is a non-empty, self-hashed manifest produced
only from complete profiling records. Every accepted decision's feature-schema
and adjacency hashes must equal the input schemas' content hashes. It stores:

- the feature-schema and adjacency content hashes;
- separate fitting, calibration, holdout, and evaluation seed partitions;
- accepted samples and their canonical source-record hashes;
- explicitly excluded samples and reasons; and
- its own `dataset_hash`.

The builder rejects duplicate record IDs, duplicate canonical samples,
incompatible schema/adjacency provenance, incomplete records, overlapping
training/evaluation partitions, profiling samples from evaluation seeds, and
frozen research-v3 evaluation seeds 101–105. It retains explicitly unclean
records in `excluded_records`; each contains a record ID, source-record hash,
and at least one reason. Clean labels cannot carry exclusion reasons. The
builder never turns excluded records into clean training labels by inference.
The research-v3 denylist is immutable: caller-supplied exclusions are added to,
not substituted for, the frozen 101–105 prohibition.

## Usage

Write one v2 decision row with `write_decision_record(path, record)`. Read mixed
action logs with `load_action_records(path)`: v2 rows are validated, while
unversioned research-v3 rows are returned unchanged.

Build a profiling manifest with:

```text
python3 experiments/scripts/build_headlight_v2_profiling_dataset.py \
  --records profiling-run.jsonl \
  --feature-schema feature-schema-v2.json \
  --adjacency adjacency-v2.json \
  --fitting-seeds 301,302 \
  --calibration-seeds 303 \
  --holdout-seeds 304 \
  --evaluation-seeds 501,502,503,504,505,506,507,508 \
  --output profiling-dataset-v2.json
```

The command is an offline file transformation. It does not contact Kubernetes,
modify research-v3 protocol/artifacts, or write generated artifacts unless the
caller explicitly supplies an output path.

## Compatibility boundary

This stage does not wire v2 writing into `ControllerRuntime`, `ForecastPlacementPolicy`,
or the Threshold controller. That keeps existing decision behavior and action-log
format stable. A later Headlight runtime can opt into this contract only after
it has the required candidate enumeration, action conditioning, and readiness
state; it must not retrofit those values into historic v3 logs.
