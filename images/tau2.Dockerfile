FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ARG SOURCE_URL
ARG SOURCE_REV
ARG APT_MIRROR
ENV UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

RUN if [ -n "${APT_MIRROR}" ]; then \
        sed -i "s|http://deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; \
    fi; \
    printf '%s\n' 'Acquire::Retries "12";' 'Acquire::http::Pipeline-Depth "0";' 'Acquire::Queue-Mode "host";' \
        > /etc/apt/apt.conf.d/99benchmark-network \
    && apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none "${SOURCE_URL}" /opt/tau2 \
    && git -C /opt/tau2 fetch --depth 1 origin "${SOURCE_REV}" \
    && git -C /opt/tau2 checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/tau2 rev-parse HEAD)" = "${SOURCE_REV}"
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN cd /opt/tau2 && UV_INDEX_URL="${PIP_INDEX_URL}" uv sync --frozen --no-dev --extra knowledge
RUN chmod -R a+rX /opt/tau2 && chmod -R a+rwX /opt/tau2/data
COPY drivers/cli_package_smoke.py /opt/platform/cli_package_smoke.py
COPY drivers/toolset_probe.py /opt/platform/toolset_probe.py
ENV PATH="/opt/tau2/.venv/bin:${PATH}"
LABEL org.orch.benchmark.source-revision="${SOURCE_REV}"
WORKDIR /opt/tau2
