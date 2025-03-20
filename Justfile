set shell := ["bash", "-cu"]
set dotenv-load

default:
  just --list

setup:
	uv sync

run-fastapi:
	@-pkill -f server.py
	uv run fastapi run "app/api/server.py"

run-gui:
	@echo "Stopping existing Streamlit instance..."
	@-pkill -f "streamlit run app/gui/main_window.py"
	@echo "Starting new Streamlit instance..."
	cd app/gui && uv run streamlit run main_window.py

serve-bm:
	@-pkill -f service.py 
	cd app/api && uv run bentoml serve

run-mkdocs:
	@-pkill -f mkdocs.yml
	uv run mkdocs serve --config-file project-docs/mkdocs.yml

run-ruff:
	uv run ruff check

