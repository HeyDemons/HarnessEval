FROM python:3.12-slim

ARG SOURCE_URL
ARG SOURCE_REV
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

RUN printf '%s\n' 'Acquire::Retries "12";' 'Acquire::http::Pipeline-Depth "0";' 'Acquire::Queue-Mode "host";' \
        > /etc/apt/apt.conf.d/99benchmark-network \
    && apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none "${SOURCE_URL}" /opt/vitabench \
    && git -C /opt/vitabench fetch --depth 1 origin "${SOURCE_REV}" \
    && git -C /opt/vitabench checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/vitabench rev-parse HEAD)" = "${SOURCE_REV}"
RUN pip install --no-cache-dir --retries 12 --timeout 60 /opt/vitabench
RUN chmod -R a+rX /opt/vitabench && chmod -R a+rwX /opt/vitabench/data
COPY drivers/vitabench_smoke.py /opt/platform/vitabench_smoke.py
LABEL org.orch.benchmark.source-revision="${SOURCE_REV}"
WORKDIR /opt/vitabench
