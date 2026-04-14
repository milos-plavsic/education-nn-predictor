.PHONY: install run api test docker-cli docker-api

install:
	python3 -m pip install --upgrade pip
	python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	python3 -m pip install -r requirements.txt

run:
	python3 app/main.py

api:
	uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

test:
	NN_EPOCHS=5 python3 -m pytest -q

docker-cli:
	docker compose run --rm app

docker-api:
	docker compose --profile api up --build api
