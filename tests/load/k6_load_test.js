/**
 * k6 Load Testing Script for GridPulse API
 * 
 * This script tests the GridPulse API under various load conditions
 * to ensure performance meets production requirements.
 * 
 * Run with: k6 run tests/load/k6_load_test.js
 * 
 * Options:
 *   k6 run --vus 10 --duration 30s tests/load/k6_load_test.js
 *   k6 run --vus 100 --duration 5m tests/load/k6_load_test.js
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

// =============================================================================
// CONFIGURATION
// =============================================================================

const BASE_URL = __ENV.API_URL || 'http://localhost:8000';
const API_KEY = __ENV.API_KEY || 'test-api-key';

// Custom metrics
const forecastLatency = new Trend('forecast_latency', true);
const optimizeLatency = new Trend('optimize_latency', true);
const anomalyLatency = new Trend('anomaly_latency', true);
const errorRate = new Rate('error_rate');
const successfulRequests = new Counter('successful_requests');

// Load test options
export const options = {
  stages: [
    { duration: '30s', target: 10 },   // Ramp up to 10 users
    { duration: '1m', target: 50 },    // Ramp up to 50 users
    { duration: '2m', target: 100 },   // Ramp up to 100 users
    { duration: '1m', target: 100 },   // Stay at 100 users
    { duration: '30s', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'],  // 95% of requests < 2s
    http_req_failed: ['rate<0.05'],     // Error rate < 5%
    forecast_latency: ['p(95)<1500'],   // Forecast p95 < 1.5s
    optimize_latency: ['p(95)<3000'],   // Optimize p95 < 3s
    error_rate: ['rate<0.01'],          // Custom error rate < 1%
  },
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

const headers = {
  'Content-Type': 'application/json',
  'X-API-Key': API_KEY,
};

function generateForecastRequest() {
  return JSON.stringify({
    region: Math.random() > 0.5 ? 'DE' : 'US',
    target: ['load_mw', 'wind_mw', 'solar_mw'][Math.floor(Math.random() * 3)],
    horizon: 24,
    include_intervals: true,
  });
}

function generateOptimizeRequest() {
  const horizon = 24;
  const load = Array.from({ length: horizon }, () => 40000 + Math.random() * 10000);
  const wind = Array.from({ length: horizon }, () => Math.random() * 5000);
  const solar = Array.from({ length: horizon }, () => Math.random() * 3000);
  const price = Array.from({ length: horizon }, () => 50 + Math.random() * 50);
  
  return JSON.stringify({
    load_forecast: load,
    wind_forecast: wind,
    solar_forecast: solar,
    price_forecast: price,
    battery_capacity_mwh: 100,
    battery_max_power_mw: 50,
    initial_soc_mwh: 50,
    optimization_mode: 'robust',
  });
}

function generateAnomalyRequest() {
  return JSON.stringify({
    values: Array.from({ length: 168 }, () => 45000 + Math.random() * 5000),
    target: 'load_mw',
    detector: 'isolation_forest',
  });
}

// =============================================================================
// TEST SCENARIOS
// =============================================================================

export default function () {
  group('Health Check', () => {
    const res = http.get(`${BASE_URL}/health`);
    check(res, {
      'health status is 200': (r) => r.status === 200,
      'health response is ok': (r) => r.json().status === 'ok',
    });
  });

  group('Forecast Endpoint', () => {
    const res = http.post(
      `${BASE_URL}/forecast/predict`,
      generateForecastRequest(),
      { headers, timeout: '10s' }
    );
    
    const success = check(res, {
      'forecast status is 200': (r) => r.status === 200,
      'forecast has predictions': (r) => {
        try {
          const body = r.json();
          return body.predictions && body.predictions.length > 0;
        } catch {
          return false;
        }
      },
    });

    forecastLatency.add(res.timings.duration);
    if (success) {
      successfulRequests.add(1);
    } else {
      errorRate.add(1);
    }
  });

  sleep(0.5);

  group('Optimization Endpoint', () => {
    const res = http.post(
      `${BASE_URL}/optimize/dispatch`,
      generateOptimizeRequest(),
      { headers, timeout: '30s' }
    );
    
    const success = check(res, {
      'optimize status is 200': (r) => r.status === 200,
      'optimize has schedule': (r) => {
        try {
          const body = r.json();
          return body.battery_charge_mw || body.charge_schedule;
        } catch {
          return false;
        }
      },
    });

    optimizeLatency.add(res.timings.duration);
    if (success) {
      successfulRequests.add(1);
    } else {
      errorRate.add(1);
    }
  });

  sleep(0.5);

  group('Anomaly Detection', () => {
    const res = http.post(
      `${BASE_URL}/anomaly/detect`,
      generateAnomalyRequest(),
      { headers, timeout: '10s' }
    );
    
    const success = check(res, {
      'anomaly status is 200': (r) => r.status === 200,
      'anomaly has results': (r) => {
        try {
          const body = r.json();
          return body.anomalies !== undefined || body.scores !== undefined;
        } catch {
          return false;
        }
      },
    });

    anomalyLatency.add(res.timings.duration);
    if (success) {
      successfulRequests.add(1);
    } else {
      errorRate.add(1);
    }
  });

  sleep(0.5);

  group('Monitoring Endpoints', () => {
    const driftRes = http.get(`${BASE_URL}/monitor/drift`, { headers });
    check(driftRes, {
      'drift status is 200': (r) => r.status === 200,
    });

    const metricsRes = http.get(`${BASE_URL}/metrics`);
    check(metricsRes, {
      'metrics status is 200': (r) => r.status === 200,
      'metrics contains gridpulse': (r) => r.body.includes('gridpulse_'),
    });
  });

  sleep(Math.random() * 2);
}

// =============================================================================
// SMOKE TEST
// =============================================================================

export function smokeTest() {
  const res = http.get(`${BASE_URL}/health`);
  check(res, {
    'smoke test passed': (r) => r.status === 200,
  });
}

// =============================================================================
// STRESS TEST
// =============================================================================

export const stressOptions = {
  stages: [
    { duration: '2m', target: 200 },
    { duration: '5m', target: 200 },
    { duration: '2m', target: 400 },
    { duration: '5m', target: 400 },
    { duration: '2m', target: 0 },
  ],
};

// =============================================================================
// SPIKE TEST
// =============================================================================

export const spikeOptions = {
  stages: [
    { duration: '10s', target: 10 },
    { duration: '1m', target: 10 },
    { duration: '10s', target: 500 },  // Spike!
    { duration: '3m', target: 500 },
    { duration: '10s', target: 10 },
    { duration: '1m', target: 10 },
    { duration: '10s', target: 0 },
  ],
};

// =============================================================================
// SOAK TEST (Long duration)
// =============================================================================

export const soakOptions = {
  stages: [
    { duration: '5m', target: 50 },
    { duration: '3h', target: 50 },  // 3 hour soak
    { duration: '5m', target: 0 },
  ],
};
