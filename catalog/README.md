# Catalogs

`benchmarks.json` is the default benchmark catalog. Catalog entries may use:

- `${PLATFORM_ROOT}` for the HarnessEval checkout.
- `${ORCH_ROOT}` for datasets and source snapshots supplied with
  `harnesseval --orch-root`.
- `${HOME}` for explicitly user-owned files.

Keep machine-specific paths and credentials out of committed catalogs. Create
an organization-specific catalog and select it with `--catalog` when additional
benchmarks, images, mounts, or allowed environment variable names are needed.

An adapter must state whether its score is official, an official subset, a
proxy, or an infrastructure-only oracle smoke. Registering a dataset does not
by itself establish end-to-end benchmark support.
