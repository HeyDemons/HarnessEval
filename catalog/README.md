# Catalogs

`benchmarks.json` is the default benchmark catalog. Catalog entries may use:

- `${PLATFORM_ROOT}` for the HarnessEval checkout.
- `${ORCH_ROOT}` for datasets and source snapshots supplied with
  `harnesseval --orch-root`.
- `${CATALOG_DIR}` for files shipped beside a custom catalog. This keeps an
  external product adapter independent of its checkout name and location.
- `${HOME}` for explicitly user-owned files.

Keep machine-specific paths and credentials out of committed catalogs. Create
an organization-specific catalog and select it with `--catalog` when additional
benchmarks, images, mounts, or allowed environment variable names are needed.

An adapter must state whether its score is official, an official subset, a
proxy, or an infrastructure-only oracle smoke. Registering a dataset does not
by itself establish end-to-end benchmark support.

`suites.json` defines the available `light` and `full` experiment scales.
Materialized light manifests live under `suites/light/`; they contain only case
identity and public selection metadata, never prompts, answers, model outcomes,
or credentials. Full suites leave enumeration with the benchmark release.
