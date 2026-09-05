FROM docker:28-cli AS docker-cli

FROM python:3.11-slim

ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=source . /opt/swebench
RUN pip install --no-cache-dir --retries 12 --timeout 60 /opt/swebench
COPY drivers/swebench_smoke.py /opt/platform/swebench_smoke.py
COPY drivers/swebench_bridge.py /opt/platform/swebench_bridge.py
ARG SOURCE_REV
LABEL org.orch.benchmark.source-revision="${SOURCE_REV}"
WORKDIR /job
RUN pip install --no-cache-dir --retries 12 --timeout 60 sacrebleu==2.3.1
