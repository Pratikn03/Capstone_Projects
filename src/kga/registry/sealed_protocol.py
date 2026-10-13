"""kga.registry.sealed_protocol -- Cryptographic protocol sealing and zero-trust verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ProtocolIntegrityError(RuntimeError):
    """Raised when an execution artifact does not match its sealed cryptographic digest."""
    pass


@dataclass(frozen=True)
class SealedProtocolManifest:
    """Immutable, cryptographically verifiable specification of a production KGA deployment."""

    protocol_name: str
    protocol_version: str
    model_name: str
    adapter_name: str
    weights_sha256: str
    calibration_data_sha256: str
    feature_schema: List[str]
    alpha: float
    delta: float
    m_min: int
    epsilon_calibrated: float
    decision_margin: float = 0.0
    created_at_iso: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def compute_manifest_digest(self) -> str:
        """Compute authoritative SHA-256 digest of the sealed configuration."""
        payload = {
            "protocol_name": self.protocol_name,
            "protocol_version": self.protocol_version,
            "model_name": self.model_name,
            "adapter_name": self.adapter_name,
            "weights_sha256": self.weights_sha256,
            "calibration_data_sha256": self.calibration_data_sha256,
            "feature_schema": self.feature_schema,
            "alpha": round(float(self.alpha), 6),
            "delta": round(float(self.delta), 6),
            "m_min": int(self.m_min),
            "epsilon_calibrated": round(float(self.epsilon_calibrated), 6),
            "decision_margin": round(float(self.decision_margin), 6),
        }
        canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical_json).hexdigest()

    def verify_runtime_artifacts(
        self,
        actual_weights_hash: Optional[str] = None,
        actual_calib_hash: Optional[str] = None,
        actual_feature_schema: Optional[List[str]] = None,
    ) -> bool:
        """Verify that runtime objects strictly match the sealed protocol."""
        if actual_weights_hash is not None and actual_weights_hash != self.weights_sha256:
            raise ProtocolIntegrityError(
                f"Model weights mismatch: sealed {self.weights_sha256[:16]}... "
                f"!= actual {actual_weights_hash[:16]}..."
            )

        if actual_calib_hash is not None and actual_calib_hash != self.calibration_data_sha256:
            raise ProtocolIntegrityError(
                f"Calibration data mismatch: sealed {self.calibration_data_sha256[:16]}... "
                f"!= actual {actual_calib_hash[:16]}..."
            )

        if actual_feature_schema is not None and actual_feature_schema != self.feature_schema:
            raise ProtocolIntegrityError(
                f"Feature schema mismatch: sealed {self.feature_schema} != actual {actual_feature_schema}"
            )

        return True

    def save(self, path: Union[str, Path]) -> None:
        """Save sealed protocol manifest to JSON file."""
        data = asdict(self)
        data["manifest_sha256"] = self.compute_manifest_digest()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Union[str, Path], verify_seal: bool = True) -> SealedProtocolManifest:
        """Load and verify a sealed protocol manifest."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        expected_seal = data.pop("manifest_sha256", None)
        manifest = cls(**data)

        if verify_seal and expected_seal is not None:
            actual_seal = manifest.compute_manifest_digest()
            if actual_seal != expected_seal:
                raise ProtocolIntegrityError(
                    f"Manifest seal verification failed: expected {expected_seal} != calculated {actual_seal}"
                )

        return manifest
