# Frozen Headlight image inputs

`source-1c1d8ea3fe6d.tar.gz` contains 66 hashed source files. It is a Docker
build context capsule, not a container image. Its checksum and file hashes are
recorded in `source-capsule.json`.

Six files were recovered from the recorded Git commit. The other 60 are
verified against the per-file hashes in `source-capsule.json`. The helper checks
every file before staging an isolated build context. It does not modify the
current checkout or copy untracked working files.

The original Dockerfile is preserved, including its pinned Python packages.
Its base-image tag and apt dependencies were not historically digest locked.
A rebuild records its actual Docker image ID, layers and retained image
identity separately; source equivalence does not establish historical image
or VM equivalence. The frozen model checkpoint is versioned separately in
`results/headlight-v2-live/bundle-full-seed-601/` and verified before building.
