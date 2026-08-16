FROM python:3.12-slim AS core

ARG APT_MIRROR

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN if [ -n "${APT_MIRROR}" ]; then \
        sed -i "s|http://deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; \
    fi; \
    printf '%s\n' \
        'Acquire::Retries "12";' \
        'Acquire::http::Pipeline-Depth "0";' \
        'Acquire::https::Pipeline-Depth "0";' \
        'Acquire::Queue-Mode "host";' \
        > /etc/apt/apt.conf.d/99benchmark-network
RUN set -eu; \
    packages="bash ca-certificates curl file ffmpeg git jq poppler-utils ripgrep sqlite3 tesseract-ocr unzip"; \
    apt-get update; \
    attempt=1; \
    until apt-get install -y --no-install-recommends --fix-missing ${packages}; do \
        if [ "${attempt}" -ge 3 ]; then exit 1; fi; \
        attempt=$((attempt + 1)); \
        sleep $((attempt * 2)); \
        apt-get update; \
    done; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir --retries 12 --timeout 60 \
    beautifulsoup4==4.13.4 \
    ddgs==9.5.4 \
    openpyxl==3.1.5 \
    pandas==2.3.1 \
    pdfplumber==0.11.7 \
    pillow==11.3.0 \
    pyarrow==21.0.0 \
    pypdf==5.9.0 \
    python-docx==1.2.0 \
    python-pptx==1.0.2 \
    requests==2.32.4
COPY drivers/portable_smoke.py /opt/platform/portable_smoke.py
COPY drivers/toolset_probe.py /opt/platform/toolset_probe.py
COPY benchmark_platform/scorers/gaia.py /opt/platform/gaia_scorer.py
COPY drivers/web_search.py /usr/local/bin/web_search
RUN chmod 0755 /usr/local/bin/web_search
WORKDIR /work

FROM core AS office
RUN set -eu; \
    apt-get update; \
    attempt=1; \
    until apt-get install -y --no-install-recommends --fix-missing libreoffice; do \
        if [ "${attempt}" -ge 3 ]; then exit 1; fi; \
        attempt=$((attempt + 1)); \
        sleep $((attempt * 2)); \
        apt-get update; \
    done; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*
