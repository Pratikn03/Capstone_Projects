# GridPulse Deployment

Infrastructure-as-code and deployment configurations.

## Directory Structure

```
deploy/
├── aws/              # AWS-specific deployment
│   ├── README.md     # AWS setup guide
│   └── observability.md
├── k8s/              # Kubernetes manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   └── configmap.yaml
└── systemd/          # Linux service files
    └── gridpulse-api.service
```

## Deployment Options

### 1. Docker Compose (Development/Staging)

```bash
cd docker/
docker-compose up -d
```

### 2. Kubernetes (Production)

```bash
kubectl apply -f deploy/k8s/
```

### 3. Systemd (Bare Metal)

```bash
sudo cp deploy/systemd/gridpulse-api.service /etc/systemd/system/
sudo systemctl enable gridpulse-api
sudo systemctl start gridpulse-api
```

### 4. AWS (Cloud)

See `deploy/aws/README.md` for:
- ECS Fargate deployment
- Load balancer configuration
- Auto-scaling policies

## Environment Requirements

| Variable | Description | Required |
|----------|-------------|----------|
| `GRIDPULSE_API_KEY` | API authentication key | Yes |
| `MODEL_PATH` | Path to trained models | Yes |
| `LOG_LEVEL` | Logging verbosity | No |
| `PROMETHEUS_PORT` | Metrics port | No |

## Health Checks

All deployments include health check endpoints:
- `/health` - Basic liveness
- `/health/ready` - Readiness with dependency checks

## Monitoring

Prometheus metrics exposed at `/metrics` for:
- Request latency
- Prediction accuracy
- Model drift scores
