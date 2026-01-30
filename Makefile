.PHONY: setup lint test api dashboard

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

lint:
	python -m compileall src

test:
	pytest -q

api:
	uvicorn services.api.main:app --reload --port 8000

dashboard:
	streamlit run services/dashboard/app.py
