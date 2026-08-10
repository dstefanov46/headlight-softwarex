# Release and citation checklist

Before a release:

1. Run `make docs-check`, `make validate-config`, `make python-tests`,
   `make yaml-validate`, `make reproduction-dry-run`, and `make go-tests`.
2. Run the measured matrix and confirm that the artifact validator reports
   `validation_status=validated`.
3. Record commit, dependency, cluster, image, seed, workload, and model bundle
   metadata; archive the validated data and figures.
4. Update `CITATION.cff`, version metadata, and the release notes with the real
   version and DOI only after the tag and Zenodo deposit exist.

Keep `CITATION.cff` and `.zenodo.json` explicitly `unreleased`, with no DOI or
publication claim, until the release tag and Zenodo deposit exist. Cite the
repository using `CITATION.cff`; use the eventual Zenodo record only after it
exists.

The reproduction dry run remains a documentation and contract check. It is not
part of the measured artifact workflow.
