# Evaluation

The evaluation consists of Headlight, native HPA, KEDA and Threshold
under gradual-ramp, flash-crowd and diurnal load, repeated with seeds 101–105.
There are 60 selected measured runs and five runs per controller/workload pair.
Every selected summary records `technical_validity=valid`.

The [evaluation archive](../results/controller-archive/README.md) provides the
selected run index, artifact hashes and summary aggregates. Run:

```bash
python experiments/validate_softwarex_evaluation.py --output results/controller-archive/validation.json
```

The report returns `validation_status=validated` after verifying all 60 cells and
all indexed artifacts. Aggregates use the arithmetic mean and sample standard
deviation of the five run-level summaries. Latency is run-level p95, SLA breaches
are successful requests exceeding 1000 ms, and resources are ready pod-seconds.

Workload warmup is 480 s, measurement is 1200 s for gradual-ramp and flash-crowd
and 2400 s for diurnal, sampling is 15 s, and cooldown is 60 s. A bursty profile
is available for additional tests but is outside the reported matrix.

HeadLight uses a frozen synthetic-trained model and policy. Model seed 601 was
selected on calibration loss and policy development used seeds 711–713. Reported
evaluation seeds 101–105 are separate from fitting, calibration and tuning.
Reactive emergency overrides are disabled in the reported HeadLight runs.

HeadLight and Threshold use the Actions Handler/CRP path. HPA and KEDA use their
native controllers. Preserve request traces, telemetry, measurement windows,
controller actions and model/environment provenance when reproducing a run.

The comparison assumes the five seeds are comparable and describes results from
one shared-host Kind federation. See [scope](evaluation.md).
The reproduction dry-run checks configuration and planning only.
