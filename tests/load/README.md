# Load Testing Guide

## Overview

GridPulse includes comprehensive load testing infrastructure using both **k6** (JavaScript) and **Locust** (Python) for API performance validation.

## Quick Start

### k6 (Recommended for CI/CD)

```bash
# Install k6
brew install k6  # macOS
# or
sudo apt install k6  # Ubuntu

# Run smoke test (10 VUs, 30s)
k6 run tests/load/k6_load_test.js

# Full load test (50 VUs, 5 min)
k6 run --vus 50 --duration 5m tests/load/k6_load_test.js

# Stress test (100 VUs, 10 min)
k6 run --vus 100 --duration 10m tests/load/k6_load_test.js

# Custom target
k6 run -e BASE_URL=https://api.gridpulse.example.com tests/load/k6_load_test.js
```

### Locust (For Interactive Testing)

```bash
# Install locust
pip install locust

# Start web UI (http://localhost:8089)
locust -f tests/load/locustfile.py --host http://localhost:8000

# Headless mode
locust -f tests/load/locustfile.py \
    --host http://localhost:8000 \
    --users 50 \
    --spawn-rate 10 \
    --run-time 5m \
    --headless \
    --html reports/locust_report.html
```

## Test Scenarios

### Smoke Test
- **VUs**: 10
- **Duration**: 30s
- **Purpose**: Validate basic functionality

### Load Test
- **VUs**: 25 → 50 (ramping)
- **Duration**: 5 minutes
- **Purpose**: Normal operational load

### Stress Test
- **VUs**: 50 → 100
- **Duration**: 10 minutes
- **Purpose**: Peak capacity testing

### Spike Test
- **VUs**: 10 → 100 → 10
- **Duration**: 3 minutes
- **Purpose**: Sudden traffic surge handling

## Endpoints Tested

| Endpoint | Weight | Timeout | Threshold (p95) |
|----------|--------|---------|-----------------|
| `/health` | 30% | 100ms | < 100ms |
| `/api/v1/forecast` | 50% | 2s | < 800ms |
| `/api/v1/optimize` | 10% | 10s | < 2000ms |
| `/api/v1/models` | 5% | 1s | < 500ms |
| `/metrics` | 5% | 1s | < 500ms |

## Performance Thresholds

```javascript
thresholds: {
  http_req_duration: ['p(95)<500', 'p(99)<1000'],
  'forecast_latency': ['p(95)<800', 'p(99)<2000'],
  'optimize_latency': ['p(95)<2000', 'p(99)<5000'],
  'errors': ['rate<0.01'],
}
```

## Reports

### k6 Output
```bash
# JSON summary
reports/load_test_summary.json

# Console output with custom metrics
```

### Locust Output
```bash
# HTML report
reports/locust_report.html

# CSV export
reports/locust_stats.csv
reports/locust_failures.csv
```

## CI/CD Integration

### GitHub Actions

```yaml
load-test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    
    - name: Install k6
      run: |
        sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg \
          --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
        echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | \
          sudo tee /etc/apt/sources.list.d/k6.list
        sudo apt-get update && sudo apt-get install k6
    
    - name: Start API
      run: make serve &
    
    - name: Run Load Test
      run: k6 run --vus 20 --duration 2m tests/load/k6_load_test.js
```

## Tips

1. **Always warm up** - Let the API process a few requests before measuring
2. **Test with realistic data** - Use production-like payloads
3. **Monitor resources** - Watch CPU, memory, and DB connections
4. **Use think time** - Real users don't send requests instantly
5. **Test in isolation** - Ensure no other processes compete for resources

## Troubleshooting

### High Error Rate
- Check API logs for exceptions
- Verify database connection pool is adequate
- Ensure timeout values are appropriate

### High Latency
- Profile the slowest endpoints
- Check for N+1 query patterns
- Verify model caching is working

### Connection Refused
- Ensure API is running and accessible
- Check firewall rules
- Verify correct host/port
