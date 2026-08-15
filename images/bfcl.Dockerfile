FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/gorilla/berkeley-function-call-leaderboard

COPY --from=source . /opt/gorilla
COPY drivers/bfcl_smoke.py /opt/platform/bfcl_smoke.py
ARG SOURCE_REV
LABEL org.orch.benchmark.source-revision="${SOURCE_REV}"
ENV BFCL_PROJECT_ROOT=/opt/gorilla/berkeley-function-call-leaderboard
WORKDIR /opt/gorilla/berkeley-function-call-leaderboard
