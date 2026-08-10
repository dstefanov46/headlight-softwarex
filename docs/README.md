# Headlight documentation

This is the canonical documentation index for users, reviewers, and
reproducers.

- [Architecture](architecture.md): components, boundaries, and data flow.
- [Reproduction](reproduction.md): deterministic dry run and measured-run procedure.
- [Experiments](experiments.md): configs, baselines, workloads, seeds, and artifacts.
- [SoftwareX readiness](softwarex.md): availability, limitations, and evidence policy.
- [Release](release.md): versioning, Zenodo, citation, and release checks.
- [Components](components/): Actions Handler, CRP, and AI Module references.

The dry-run path validates configuration and orchestration contracts and
demonstrates sample output format. Measured tables use result bundles accepted by
`experiments/validate_artifacts.py`.
