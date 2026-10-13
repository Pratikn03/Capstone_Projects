"""kga.server.service -- Autonomous Safety Gate microservice and HTTP serving adapter."""

from __future__ import annotations

import http.server
import json
from typing import Any, Dict, Optional

import numpy as np

from kga.gateway.safe_gateway import SafeInferenceGateway
from kga.observability.metrics import METRICS


class GatewayRequestHandler(http.server.BaseHTTPRequestHandler):
    """Standard-library HTTP request handler for the KGA Autonomous Safety Gateway."""

    gateway: Optional[SafeInferenceGateway] = None

    def do_GET(self) -> None:
        """Handle health probes and metrics scraping."""
        if self.path == "/healthz":
            self._handle_healthz()
        elif self.path == "/metrics":
            self._handle_metrics()
        else:
            self._send_json(404, {"error": "Not Found"})

    def do_POST(self) -> None:
        """Handle inference prediction and operator overrides."""
        if self.path == "/v1/predict":
            self._handle_predict()
        elif self.path == "/circuit-breaker/reset":
            self._handle_reset()
        else:
            self._send_json(404, {"error": "Not Found"})

    def _handle_healthz(self) -> None:
        if self.gateway is None:
            self._send_json(503, {"status": "uninitialized"})
            return

        cb_status = self.gateway.circuit_breaker.get_status()
        response = {
            "status": "healthy" if cb_status.state.value == "closed" else "degraded",
            "circuit_state": cb_status.state.value,
            "failure_count": cb_status.failure_count,
            "trip_count": cb_status.trip_count,
            "last_trip_reason": cb_status.last_trip_reason,
        }
        status_code = 200 if cb_status.state.value != "open" else 503
        self._send_json(status_code, response)

    def _handle_metrics(self) -> None:
        text = METRICS.generate_prometheus_text()
        encoded = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _handle_predict(self) -> None:
        if self.gateway is None:
            self._send_json(503, {"error": "Gateway not initialized"})
            return

        content_len = int(self.headers.get("Content-Length", 0))
        if content_len == 0:
            self._send_json(400, {"error": "Empty request body"})
            return

        body = self.rfile.read(content_len)
        try:
            payload = json.loads(body.decode("utf-8"))
            inputs = np.asarray(payload["inputs"], dtype=float)
            request_id = payload.get("request_id")
        except Exception as ex:
            self._send_json(400, {"error": f"Invalid JSON payload: {ex}"})
            return

        try:
            preds, decision = self.gateway.predict(inputs, request_id=request_id)
            self._send_json(
                200,
                {
                    "predictions": preds.tolist(),
                    "decision": decision.to_dict(),
                },
            )
        except Exception as ex:
            self._send_json(500, {"error": f"Inference pipeline failure: {ex}"})

    def _handle_reset(self) -> None:
        if self.gateway is None:
            self._send_json(503, {"error": "Gateway not initialized"})
            return

        self.gateway.circuit_breaker.reset()
        self._send_json(200, {"status": "reset", "circuit_state": self.gateway.circuit_breaker.state.value})

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        encoded = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def create_server(
    gateway: SafeInferenceGateway,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> http.server.HTTPServer:
    """Create a production HTTP server configured with the SafeInferenceGateway."""
    handler = GatewayRequestHandler
    handler.gateway = gateway
    return http.server.HTTPServer((host, port), handler)
