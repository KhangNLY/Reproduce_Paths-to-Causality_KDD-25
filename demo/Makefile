.PHONY: up down logs model data load app
up:
	docker compose up -d --build
down:
	docker compose down
logs:
	docker compose logs -f --tail=100
model:
	docker compose exec ollama ollama pull qwen2.5:7b
data:
	bash scripts/download_hetionet.sh
load:
	docker compose exec app python scripts/load_hetionet.py
app:
	docker compose up -d app
