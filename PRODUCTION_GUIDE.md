# GridPulse Production Deployment Guide

## Quick Start Checklist

```
✅ Models trained (12 models in registry)
✅ Dashboard data extracted
✅ Release gate passed
✅ All artifacts verified
```

## Deployment Options

### Option 1: Docker Compose (Recommended for Small Scale)

```bash
# Start all services
docker compose -f docker/docker-compose.yml up -d

# Check health
curl http://localhost:8000/ready
curl http://localhost:3000
```

### Option 2: AWS ECS Fargate (Production Scale)

1. **Configure secrets in AWS Secrets Manager:**
   ```bash
   aws secretsmanager create-secret --name gridpulse/prod \
     --secret-string '{"GRIDPULSE_ENV":"production"}'
   ```

2. **Deploy ECS services:**
   ```bash
   # API Service
   aws ecs register-task-definition --cli-input-json file://deploy/aws/ecs-task-def-api.json
   
   # Dashboard Service  
   aws ecs register-task-definition --cli-input-json file://deploy/aws/ecs-task-def-dashboard.json
   
   # Scheduled Refresh
   aws events put-rule --cli-input-json file://deploy/aws/eventbridge-refresh.json
   ```

3. **Configure Load Balancer** with:
   - API: port 8000 → /api/*
   - Dashboard: port 3000 → /*

### Option 3: Kubernetes (Enterprise)

```bash
# Apply manifests
kubectl apply -f deploy/k8s/

# Verify
kubectl get pods -l app=gridpulse
kubectl port-forward svc/gridpulse-api 8000:8000
```

### Option 4: systemd (On-Premise)

```bash
# Install services
sudo cp deploy/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start services
sudo systemctl enable --now gridpulse-api
sudo systemctl enable --now gridpulse-dashboard
sudo systemctl enable --now gridpulse-refresh.timer
```

## Environment Variables

Create `/etc/gridpulse/gridpulse.env`:

```bash
GRIDPULSE_ENV=production
GRIDPULSE_LOG_FORMAT=json
GRIDPULSE_MODELS_DIR=/opt/gridpulse/artifacts/models
GRIDPULSE_DATA_DIR=/opt/gridpulse/data
GRIDPULSE_ALERT_WEBHOOK=https://hooks.slack.com/xxx
```

## Production Monitoring

### SLO Targets
| Metric | Target |
|--------|--------|
| Uptime | 99.9% |
| API p95 latency | <500ms |
| Forecast refresh | <5 min |
| Model retraining | Weekly |

### Health Endpoints
- `/health` - Basic liveness check
- `/ready` - Full readiness with model count
- `/metrics` - Prometheus metrics

### Alerting Rules
Configure in `configs/monitoring.yaml`:
- **Drift alert**: KS-test p-value < 0.05 on any feature
- **Performance alert**: RMSE increase > 10%
- **System alert**: API response time > 1s

## Model Registry

Models are tracked in `artifacts/registry/models.json`:
```json
{
  "latest": {
    "generated_at": "2026-02-09T06:43:17Z",
    "models_dir": "artifacts/models",
    "model_count": 12
  }
}
```

### Rollback Procedure
```bash
# List available models
ls -la artifacts/models/

# Pin specific model in configs/forecast.yaml:
models:
  load_mw: artifacts/models/gbm_lightgbm_load_mw_20260208.pkl
```

## Scheduled Tasks

| Task | Frequency | Command |
|------|-----------|---------|
| Data refresh | Hourly | `make refresh` |
| Dispatch optimization | Every 15 min | `make dispatch` |
| Model monitoring | Daily | `make monitoring` |
| Retraining (if drift) | Weekly | `make retrain-if-drift` |

## Security Checklist

- [ ] API behind reverse proxy (nginx/ALB)
- [ ] HTTPS only
- [ ] Rate limiting enabled
- [ ] API keys for external access
- [ ] Secrets in environment (never in code)
- [ ] Audit logging enabled

## Troubleshooting

### API won't start
```bash
# Check logs
journalctl -u gridpulse-api -f

# Verify models exist
ls -la artifacts/models/*.pkl

# Test manually
python -m uvicorn services.api.main:app --port 8000
```

### Dashboard shows no data
```bash
# Re-extract dashboard data
python scripts/extract_dashboard_data.py

# Verify JSON files
ls -la data/dashboard/*.json
```

### Model drift detected
```bash
# Check monitoring report
cat reports/monitoring_summary.json | jq '.data_drift'

# Trigger retraining
make train
```

---

## Support

- **Docs**: [docs/RUNBOOK.md](docs/RUNBOOK.md)
- **Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Evaluation**: [reports/formal_evaluation_report.md](reports/formal_evaluation_report.md)
