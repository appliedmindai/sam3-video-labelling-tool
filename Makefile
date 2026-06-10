.PHONY: dev docker-build

## Local development
dev:
	@echo "Starting backend and frontend..."
	@trap 'kill 0' EXIT; \
	cd backend && PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -m flask --app app:create_app run --port 5555 & \
	cd frontend && npm run dev & \
	wait

## Docker
docker-build:
	docker build -t sam3-annotator .
