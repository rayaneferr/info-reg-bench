.PHONY: install smoke grid figures test lint docker

install:        ## create the env and install everything
	uv sync --extra dev

smoke:          ## 1-minute end-to-end check
	uv run main.py smoke

grid:           ## full V1 grid at n_train=1000 (~1h on an M4 Pro)
	uv run main.py grid --n_train 1000

figures:        ## summary.csv + figures/*.png from results/
	uv run main.py figures

test:           ## unit tests (no model download)
	uv run pytest -q

lint:
	uv run ruff check .

docker:         ## CPU image; run: docker run --rm -v $$PWD/results:/app/results info-reg-bench smoke --model_name HuggingFaceTB/SmolLM2-135M
	docker build -t info-reg-bench .
