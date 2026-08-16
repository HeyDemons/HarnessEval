FROM python:3.12-slim

ARG SOURCE_URL
ARG SOURCE_REV
ARG APT_MIRROR
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN if [ -n "${APT_MIRROR}" ]; then \
        sed -i "s|http://deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; \
    fi; \
    printf '%s\n' 'Acquire::Retries "12";' 'Acquire::http::Pipeline-Depth "0";' 'Acquire::Queue-Mode "host";' \
        > /etc/apt/apt.conf.d/99benchmark-network \
    && apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none "${SOURCE_URL}" /opt/trajectory \
    && git -C /opt/trajectory fetch --depth 1 origin "${SOURCE_REV}" \
    && git -C /opt/trajectory checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/trajectory rev-parse HEAD)" = "${SOURCE_REV}"
# The published freeze contains local file:// URLs and CUDA-only wheels. This
# curated API path covers case loading, remote tool execution, and the official
# non-retrieval evaluator without changing source. The optional retriever needs
# external model assets and belongs in a separately versioned image.
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir --retries 12 --timeout 60 \
    boto3==1.40.1 \
    google-genai==1.29.0 \
    numpy==2.1.2 \
    openai==1.93.0 \
    python-dotenv==1.1.1 \
    requests==2.32.4 \
    scikit-learn==1.7.0
COPY drivers/portable_smoke.py /opt/platform/portable_smoke.py
COPY drivers/toolset_probe.py /opt/platform/toolset_probe.py
LABEL org.orch.benchmark.source-revision="${SOURCE_REV}"
WORKDIR /opt/trajectory
