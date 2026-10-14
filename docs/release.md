# Release and citation checklist

Before a release:

1. Run `make docs-check`, `make validate-config`, `make python-tests`,
   `make yaml-validate`, `make reproduction-dry-run`, and `make go-tests`.
2. Run the measured matrix and confirm that the artifact validator reports
   `validation_status=validated`.
3. Record commit, dependency, cluster, image, seed, workload, and model bundle
   metadata; archive the validated data and figures.
4. Keep the source version in `VERSION`, `CITATION.cff`, and `.zenodo.json`
   aligned. This branch identifies code version 2.0.0. Add a release date and
   DOI only after creating the corresponding tag and Zenodo record.

The metadata identifies version 2.0.0 without claiming that a Zenodo record or
DOI exists. Cite the repository using `CITATION.cff`; cite a Zenodo record only
after it has been created.

The reproduction dry run remains a documentation and contract check. It is not
measured release evidence.
