# AWS ECS Fargate Deployment (Production)

This folder contains templates for ECS services, scheduled tasks, and observability.  
You selected: **ECS Fargate**, **weekly refresh + retrain**, **SLO 99.9% uptime**, **p95 < 500ms**.

## Required AWS Resources
- ECR repositories for API and dashboard images  
- ECS cluster (Fargate)  
- Secrets Manager secret for tokens  
- CloudWatch logs + alarms  
- Persistent storage (EFS or S3 sync) for `data/`, `artifacts/`, `reports/`  

## Secrets Manager
Create one secret (JSON) named `gridpulse/prod`:
```json
{
  "ELECTRICITYMAPS_TOKEN": "xxx",
  "WATTTIME_USERNAME": "xxx",
  "WATTTIME_PASSWORD": "xxx",
  "GRIDPULSE_ALERT_WEBHOOK": "https://..."
}
```

Use the secret ARN in the task definitions.

## Task Definitions
- `ecs-task-def-api.json`  
- `ecs-task-def-dashboard.json`  
- `ecs-task-def-refresh.json` (weekly data refresh)  
- `ecs-task-def-runner.json` (weekly retrain)

Replace placeholders:
- `<AWS_REGION>`
- `<ACCOUNT_ID>`
- `<ECR_REPO_API>`
- `<ECR_REPO_DASHBOARD>`
- `<SECRET_ARN>`
- `<LOG_GROUP>`

If you use EFS, add a `volumes` block and mount `/app/data`, `/app/artifacts`, `/app/reports`.

## Schedules (Weekly)
Use EventBridge to trigger ECS tasks:
- `eventbridge-refresh.json`  
- `eventbridge-retrain.json`  

These are set to **Sunday 03:00 UTC**. Change as needed.
Retrain runs at **Sunday 04:00 UTC** by default.

## Deployment Flow (CI)
GitHub Actions workflow: `.github/workflows/deploy.yml`  
Required repo secrets:
- `AWS_REGION`
- `ECR_REPOSITORY_API`
- `ECR_REPOSITORY_DASHBOARD`
- `ECS_CLUSTER`
- `ECS_SERVICE_API`
- `ECS_SERVICE_DASHBOARD`
- `AWS_ROLE_ARN` (OIDC role)

## Observability
See `observability.md` for recommended CloudWatch alarms and SLOs.
