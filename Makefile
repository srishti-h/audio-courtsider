.PHONY: setup data train eval bench test serve web lint screenshot demo gif live-report refresh

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

screenshot:       ## README dashboard image (needs serve + web running, Google Chrome)
	uv run python scripts/screenshot.py

demo:             ## static GitHub Pages build: pre-computed replays -> web/public/demo, then web/dist
	uv run python scripts/export_demo.py
	cd web && npm run build:pages

gif:              ## README GIF (needs `cd web && npx vite preview --base /audio-courtsider/` on :4173, Google Chrome)
	uv run python scripts/record_gif.py

live-report:      ## lead time of live commentary detections over the live-score feed (data/live -> results/live.json)
	uv run python -m courtsider.live.report

refresh:          ## pull recent data (football-data, ESPN 2025/26+, recorded radio) and retrain on it; rerun after match weekends
	uv run python -c "from courtsider.data import footballdata; footballdata.download()"
	uv run python -m courtsider.data.build
	uv run python -m courtsider.data.espn
	uv run python -m courtsider.data.radio
	uv run python -m courtsider.text.refresh
	uv run python -m courtsider.acoustic.goal_model --radio
	uv run python -m courtsider.eval.run_pricing
	uv run python -m courtsider.eval.run_text
	uv run python -m courtsider.eval.run_market
	uv run python -m courtsider.eval.run_market_audio
	uv run python -m courtsider.eval.run_fast_text
	uv run python -m courtsider.live.report
	uv run python scripts/export_demo.py
	git add results web/public/demo
	git diff --cached --quiet || (git commit -m "Refresh results $$(date +%F)" && git push)  # publishes the site
