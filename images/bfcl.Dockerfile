FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/gorilla/berkeley-function-call-leaderboard

COPY --from=source . /opt/gorilla
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_VERSION=2.8.0
# Keep Linux ARM on the CPU wheel while satisfying sentence-transformers' published range.
RUN pip install --no-cache-dir --retries 12 --timeout 60 "torch==${TORCH_VERSION}" \
    && pip install --no-cache-dir --retries 12 --timeout 60 /opt/gorilla/berkeley-function-call-leaderboard
COPY drivers/bfcl_smoke.py /opt/platform/bfcl_smoke.py
COPY drivers/toolset_probe.py /opt/platform/toolset_probe.py
ARG SOURCE_REV
LABEL org.orch.benchmark.source-revision="${SOURCE_REV}"
ENV BFCL_PROJECT_ROOT=/opt/gorilla/berkeley-function-call-leaderboard
WORKDIR /opt/gorilla/berkeley-function-call-leaderboard
