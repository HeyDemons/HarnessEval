FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
ENV UV_LINK_MODE=copy PYTHONDONTWRITEBYTECODE=1
COPY --from=source . /opt/automationbench
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN cd /opt/automationbench && UV_INDEX_URL="${PIP_INDEX_URL}" uv sync --frozen --no-dev
RUN UV_INDEX_URL="${PIP_INDEX_URL}" uv pip install --python /opt/automationbench/.venv/bin/python sacrebleu==2.3.1
ENV PATH="/opt/automationbench/.venv/bin:${PATH}"
ARG SOURCE_REV
LABEL org.orch.benchmark.source-revision="${SOURCE_REV}"
WORKDIR /opt/automationbench
