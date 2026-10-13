"""kga.observability.audit_logger -- Structured append-only JSONL audit log emitter."""

from __future__ import annotations

import datetime
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union

from kga.gateway.interface import GatewayDecision


class AuditLogger:
    """Thread-safe append-only structured audit logger for model routing decisions."""

    def __init__(self, log_file_path: Optional[Union[str, Path]] = None) -> None:
        self.log_file_path = Path(log_file_path) if log_file_path is not None else None
        self._lock = threading.Lock()
        if self.log_file_path is not None:
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

    def log_decision(
        self,
        decision: GatewayDecision,
        request_id: Optional[str] = None,
        client_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record an audit log entry for a gateway decision."""
        record = {
            "timestamp_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "request_id": request_id or "",
            "decision": decision.to_dict(),
            "client_metadata": client_metadata or {},
        }

        if self.log_file_path is not None:
            line = json.dumps(record, sort_keys=True) + "\n"
            with self._lock:
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    f.write(line)

        return record
