FROM python:3.12-slim AS core

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN printf '%s\n' \
        'Acquire::Retries "12";' \
        'Acquire::http::Pipeline-Depth "0";' \
        'Acquire::https::Pipeline-Depth "0";' \
        'Acquire::Queue-Mode "host";' \
        > /etc/apt/apt.conf.d/99benchmark-network
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        bash ca-certificates curl file ffmpeg git jq \
        poppler-utils ripgrep sqlite3 tesseract-ocr unzip \
    && apt-get clean
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
COPY benchmark_platform /opt/platform/benchmark_platform
COPY drivers/portable_smoke.py /opt/platform/portable_smoke.py
COPY drivers/web_search.py /usr/local/bin/web_search
RUN chmod 0755 /usr/local/bin/web_search
WORKDIR /work

FROM core AS office
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends libreoffice \
    && apt-get clean
