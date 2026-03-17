# ORIUS Docker Configuration

Docker containerization for the ORIUS platform.

## Files

| File | Purpose |
|------|---------|
| `Dockerfile.api` | API service container |
| `Dockerfile.app` | Full application container |
| `docker-compose.yml` | Multi-container orchestration |
| `docker-compose.streaming.yml` | Kafka streaming stack |

## Quick Start

```bash
# Build and run API service
docker-compose up -d api

# Run full stack (API + Kafka + monitoring)
docker-compose -f docker-compose.yml -f docker-compose.streaming.yml up -d

# View logs
docker-compose logs -f api
```

## Images

### API Service (`Dockerfile.api`)

Lightweight FastAPI service for predictions:

```bash
docker build -f Dockerfile.api -t orius-api:latest .
docker run -p 8000:8000 -e ORIUS_API_KEY=secret orius-api:latest
```

### Full Application (`Dockerfile.app`)

Complete environment with all dependencies:

```bash
docker build -f Dockerfile.app -t orius:latest .
docker run -it orius:latest python scripts/train_multi_dataset.py
```

## Volumes

| Volume | Purpose |
|--------|---------|
| `./data:/app/data` | Training data |
| `./artifacts:/app/artifacts` | Model artifacts |
| `./configs:/app/configs` | Configuration files |

## Networks

Services communicate over a shared `orius-net` bridge network.

## Resource Limits

Production deployments should set:

```yaml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2'
```
