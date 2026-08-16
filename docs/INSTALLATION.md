# macOS Installation And Host Compatibility

## Supported Host Path

HarnessEval requires Python 3.11 or newer, Git, and a running Linux-container
Docker daemon. The reproduction target is macOS:

| Mac | Status | Execution mode |
| --- | --- | --- |
| macOS Intel | Supported | Docker Desktop or Colima, Linux amd64 VM |
| macOS Apple silicon | Supported with benchmark exceptions | Native ARM64 images; explicitly x86-only tasks use emulation |

No GPU is required. CPU, memory, disk, and network requirements depend on the
selected benchmark; SWE-bench and Terminal-Bench task images are substantially
heavier than the portable harness and API benchmarks.

Docker Desktop is the simplest supported path. Colima is also supported and
its Docker socket is handled explicitly. For the full adapter matrix on Apple
silicon, a practical Colima profile is six CPUs, 12 GiB memory, and 100 GiB
disk. This is a reproducibility recommendation rather than a controller limit:

```bash
brew install python@3.12 git docker colima docker-buildx
colima start --runtime docker --vm-type vz --cpu 6 --memory 12 --disk 100
docker info
docker buildx version
```

Keep the HarnessEval checkout, external harness checkout, mounted datasets, and
run directory under a macOS directory shared with the Docker VM. A directory
under `$HOME` is the portable default. In particular, Colima does not expose
every macOS `/tmp` mount with the same writable ownership semantics as Home
directories.

## Docker Download Behavior

HarnessEval does not bundle benchmark images. It automatically pulls declared
base images and builds versioned benchmark or product images locally. Images
with source/fingerprint labels are reused until their inputs change.

The Docker daemon, not the HarnessEval Python process, performs registry pulls.
On a restricted network, configure the proxy or registry mirror in Docker
Desktop, Colima, or `dockerd`. `BENCHMARK_BUILD_PROXY` controls HTTP(S) access
inside Dockerfile build steps; it does not configure the daemon's own registry
connection. Its URL must also be reachable from inside the Docker VM; do not
assume a proxy bound to macOS localhost is container-reachable.
`BENCHMARK_PIP_INDEX_URL` selects an alternate Python package index for
supported image builds.

Dockerfiles continue to use the official Debian source by default. In regions
where that CDN route is unstable, `BENCHMARK_APT_MIRROR` can select a trusted
mirror without editing the image definition:

```bash
BENCHMARK_APT_MIRROR=https://mirrors.example.org \
  harnesseval build gaia gdpval
```

The mirror base must provide both `/debian` and `/debian-security`. Signed
Debian metadata and benchmark source-revision checks remain active.

Modern Docker with BuildKit is required because the portable images use cache
mounts and external build contexts. Run these checks before downloading a full
benchmark:

```bash
docker info
docker buildx version
harnesseval doctor gaia vitabench tau2
```

`doctor` distinguishes a missing but buildable image from a missing dataset or
unsupported environment. `harnesseval build BENCHMARK --pull` refreshes base
images explicitly.

The BFCL image pins Torch 2.8.0 before installing the official evaluator. This
keeps the published evaluator dependency contract while avoiding accidental
resolution to future CUDA-heavy Torch wheels on Linux ARM64. The tau2 image is
installed from its pinned `uv.lock`; real `banking_knowledge` episodes also
need an embedding provider when the official embeddings cache is absent.

## Architecture Exceptions

The bundled Debian, Python, Node, uv, and Docker CLI base images publish both
`linux/amd64` and `linux/arm64` variants. Docker selects the native image for
Intel or Apple silicon unless a benchmark explicitly declares another target.

- The registered Terminal-Bench sample declares `linux/amd64`. ARM hosts need
  QEMU/binfmt support and should expect slower execution.
- SWE-bench uses its official x86_64 task images on x86 hosts. On ARM64, the
  pinned smoke uses a digest-verified ARM64 task image and records that
  provenance in its result.
- Benchmark-specific images can still be single-architecture even when the
  HarnessEval control plane is portable. Check the benchmark compatibility
  document before comparing wall-clock time across architectures.

Docker Desktop includes cross-architecture emulation. Colima enables foreign
architecture binfmt support by default; its VZ/Rosetta option can also be used
for amd64 workloads on Apple silicon. Emulated and native wall-clock results
must not be mixed in one performance table.
