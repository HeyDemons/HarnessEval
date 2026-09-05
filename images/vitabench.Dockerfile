FROM python:3.12-slim

ARG SOURCE_URL
ARG SOURCE_REV
ARG APT_MIRROR
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/opt/vitabench/src

RUN if [ -n "${APT_MIRROR}" ]; then \
        sed -i "s|http://deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; \
    fi; \
    printf '%s\n' 'Acquire::Retries "12";' 'Acquire::http::Pipeline-Depth "0";' 'Acquire::Queue-Mode "host";' \
        > /etc/apt/apt.conf.d/99benchmark-network \
    && apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none "${SOURCE_URL}" /opt/vitabench \
    && git -C /opt/vitabench fetch --depth 1 origin "${SOURCE_REV}" \
    && git -C /opt/vitabench checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/vitabench rev-parse HEAD)" = "${SOURCE_REV}"
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir --retries 12 --timeout 60 /opt/vitabench
RUN chmod -R a+rX /opt/vitabench && chmod -R a+rwX /opt/vitabench/data
COPY drivers/vitabench_smoke.py /opt/platform/vitabench_smoke.py
COPY drivers/toolset_probe.py /opt/platform/toolset_probe.py
LABEL org.orch.benchmark.source-revision="${SOURCE_REV}"
WORKDIR /opt/vitabench
RUN pip install --no-cache-dir --retries 12 --timeout 60 sacrebleu==2.3.1
