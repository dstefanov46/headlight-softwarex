# Evaluation artifacts

The [controller archive](controller-archive/README.md) indexes the exact 60 valid
measured runs. It covers Headlight, HPA, KEDA and Threshold, three workloads and
seeds 101–105.

```bash
python experiments/validate_softwarex_evaluation.py --output results/controller-archive/validation.json
```

Successful validation returns `validation_status=validated` and reproduces the
controller/workload aggregates from the selected summaries.

Frozen HeadLight bundles, calibration inputs, capacity records and source/image
provenance support reproduction. Dry-run outputs describe execution plans and
are separate from measured artifacts. See [evaluation scope](../docs/evaluation.md).
