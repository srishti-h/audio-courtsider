.PHONY: setup data train eval bench test serve web lint

setup:            ## install Python + web dependencies
	uv sync
	cd web && npm install

data:             ## download SoccerNet labels, Echoes transcripts, football-data odds; build tables
	uv run python -m courtsider.data.build --download

train:            ## keyword / TF-IDF / DistilRoBERTa event extractors (~25 min on an M-series GPU)
	uv run python -m courtsider.text.train

eval:             ## all experiments -> results/*.json and results/figures/
	uv run python -m courtsider.eval.run_pricing
	uv run python -m courtsider.eval.run_audio
	uv run python -m courtsider.eval.run_text
	uv run python -m courtsider.eval.run_market
	uv run python -m courtsider.eval.run_bench

test:
	uv run pytest -q

lint:
	uv run ruff check src tests && uv run ruff format --check src tests

serve:            ## API on :8000 (also serves web/dist if built)
	uv run uvicorn courtsider.api.server:app --port 8000

web:              ## dev server on :5173 proxying to the API
	cd web && npm run dev
