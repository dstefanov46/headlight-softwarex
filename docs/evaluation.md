# Evaluation scope and validation

The evaluation contains 60 valid measured runs from Headlight, HPA, KEDA and
Threshold across three workloads and seeds 101–105. The validator checks matrix
completeness, artifact hashes, summary identity and status, and finite metric
values before returning `validation_status=validated`.

```bash
python experiments/validate_softwarex_evaluation.py
```

The evaluation treats the five seeds as comparable. The shared-host Kind testbed
evaluates one Train Ticket service slice and supports descriptive comparisons.
It does not establish geographic edge–cloud performance, causal controller
effects, production scalability or an availability guarantee.

Source, deployment configuration, raw measurements, frozen model files and
provenance are supplied with the repository. Dry-run output exercises contracts
and planning. It is distinct from the measured evaluation.
