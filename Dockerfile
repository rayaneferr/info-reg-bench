# CPU image — for a quick smoke test or CI-like runs anywhere.
# GPU users: start from pytorch/pytorch:*-cuda* instead and drop --torch-backend cpu.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
    HF_HOME=/data/hf

WORKDIR /app
COPY pyproject.toml README.md ./
COPY inforeg ./inforeg
# `uv sync` would follow uv.lock and pull the CUDA wheels; `uv pip` honours --torch-backend.
RUN uv pip install --system --no-cache --torch-backend cpu .
COPY main.py ./
COPY scripts ./scripts
COPY docs ./docs

VOLUME ["/data", "/app/results"]
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
