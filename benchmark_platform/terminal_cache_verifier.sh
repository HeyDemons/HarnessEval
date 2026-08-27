#!/usr/bin/env bash
# Grading step for the separate-container verifier used by the PERSEUS integration.
#
# This mirrors benchmark_platform/terminal_runtime.py::run_shared_verifier: clear any
# stale reward, run `bash test.sh`, and leave the reward the test wrote in
# /logs/verifier for the caller to read. Two deltas are forced by the separate
# container and are documented rather than hidden:
#
#   * The caller bind-mounts the hidden tests read-only. run_shared_verifier uploads
#     them instead, precisely because several TB2 tests compile native extensions
#     inside their own directory. Grade from a writable copy so those tasks still run.
#   * The caller supplies `--network none`, so run_shared_verifier's transient-network
#     retry loop cannot apply and is deliberately not reproduced here.
set -uo pipefail

mkdir -p /logs/verifier
# A zero written by a failed dependency install must not survive as a valid verdict.
rm -f /logs/verifier/reward.txt /logs/verifier/reward.json

tests_dir=/opt/harnesseval/tests
rm -rf "${tests_dir}"
mkdir -p "${tests_dir}"
cp -a /tests/. "${tests_dir}/"

if [[ ! -f "${tests_dir}/test.sh" ]]; then
	echo "terminal-cache verifier: tests/test.sh is missing" >&2
	exit 1
fi

bash "${tests_dir}/test.sh"
exit $?
