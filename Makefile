.PHONY: install run api ui test

install:
	pip install -r requirements.txt

run:
	bash scripts/run_all.sh

api:
	uvicorn src.api:app --reload --port 8000

ui:
	streamlit run ui/streamlit_app.py

test:
	pytest
