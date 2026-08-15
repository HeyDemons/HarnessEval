# Security

Report security issues privately to the repository owner rather than opening a
public issue containing credentials or sensitive benchmark data.

HarnessEval records complete model and tool content by design. Treat run
directories as sensitive. The control plane does not redact task data after the
fact; keep secrets out of prompts and tool output, and pass credentials only by
allow-listed environment variable name.

Mounting the Docker socket grants the controller broad host-container authority.
It is reserved for benchmark controllers whose official harness creates child
containers. Review custom catalogs before running them.
