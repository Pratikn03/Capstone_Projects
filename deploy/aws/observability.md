# Observability and SLOs (AWS)

Target SLOs:
- **Uptime**: 99.9%  
- **p95 latency**: 500ms  

Recommended CloudWatch metrics:
- `AWS/ApplicationELB` `TargetResponseTime` (p95)  
- `AWS/ApplicationELB` `HTTPCode_Target_5XX_Count`  
- `AWS/ApplicationELB` `HealthyHostCount`  
- `AWS/ECS` `CPUUtilization`, `MemoryUtilization`  

Suggested alarms:
- p95 latency > 0.5s for 5 minutes  
- 5xx errors > 1 for 5 minutes  
- HealthyHostCount < 1  

Connect alerts to SNS and forward to Slack via `GRIDPULSE_ALERT_WEBHOOK` if desired.

