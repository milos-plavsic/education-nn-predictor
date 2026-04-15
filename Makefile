.PHONY: install run api test finetune report docker-cli docker-api

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

finetune:
	FT_PHASE1_EPOCHS=40 FT_PHASE2_EPOCHS=30 python3 -m finetune

report:
	rm -rf reports
	@if [ -x .venv/bin/python ]; then \
	  NN_EPOCHS=60 FT_PHASE1_EPOCHS=50 FT_PHASE2_EPOCHS=40 .venv/bin/python -m analysis; \
	else \
	  NN_EPOCHS=60 FT_PHASE1_EPOCHS=50 FT_PHASE2_EPOCHS=40 python3 -m analysis; \
	fi

docker-cli:
	docker compose run --rm app

docker-api:
	docker compose --profile api up --build api
