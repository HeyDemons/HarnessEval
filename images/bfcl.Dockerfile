FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/gorilla/berkeley-function-call-leaderboard

COPY --from=source . /opt/gorilla
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_VERSION=2.8.0
# BFCL scoring uses sentence-transformers on CPU. PyPI's Linux wheel pulls the
# complete CUDA 12 stack (>4 GB) even though the image never exposes a GPU; the
# matching official PyTorch CPU wheel preserves the package/API version and
# keeps the benchmark image small enough to unpack on the shared runner.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --retries 12 --timeout 60 \
        --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}" \
    && pip install --no-cache-dir --retries 12 --timeout 60 /opt/gorilla/berkeley-function-call-leaderboard
# The pinned package's own metadata misses this: importing the official checker pulls in
# MODEL_CONFIG_MAPPING, which imports every model handler, one of which reaches qwen_agent
# -> soundfile. Its own layer so a fix here never re-downloads the 4 GB torch layer above.
RUN pip install --no-cache-dir --retries 12 --timeout 60 soundfile
COPY drivers/bfcl_smoke.py /opt/platform/bfcl_smoke.py
COPY drivers/bfcl_score.py /opt/platform/bfcl_score.py
COPY drivers/toolset_probe.py /opt/platform/toolset_probe.py
ARG SOURCE_REV
LABEL org.orch.benchmark.source-revision="${SOURCE_REV}"
ENV BFCL_PROJECT_ROOT=/opt/gorilla/berkeley-function-call-leaderboard
WORKDIR /opt/gorilla/berkeley-function-call-leaderboard
RUN pip install --no-cache-dir --retries 12 --timeout 60 sacrebleu==2.3.1
