# CPU image — for a quick smoke test or CI-like runs anywhere.
# GPU users: start from pytorch/pytorch:*-cuda* instead and drop UV_TORCH_BACKEND.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_TORCH_BACKEND=cpu \
    UV_LINK_MODE=copy \
    HF_HOME=/data/hf

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY inforeg ./inforeg
RUN uv sync --frozen --no-dev
COPY main.py ./
COPY scripts ./scripts
COPY docs ./docs

VOLUME ["/data", "/app/results"]
ENTRYPOINT ["uv", "run", "main.py"]
CMD ["--help"]
