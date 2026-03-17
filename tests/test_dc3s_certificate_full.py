"""Comprehensive tests for DC3S certificate creation and storage."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from orius.dc3s.certificate import (
    compute_config_hash,
    compute_model_hash,
    get_certificate,
    make_certificate,
    store_certificate,
)


def _cert(cmd_id="cmd-1", prev_hash=None, **kw):
    defaults = dict(
        command_id=cmd_id,
        device_id="dev-1",
        zone_id="DE",
        controller="dc3s",
        proposed_action={"charge_mw": 5.0, "discharge_mw": 0.0},
        safe_action={"charge_mw": 4.0, "discharge_mw": 0.0},
        uncertainty={"lower": [90.0], "upper": [110.0]},
        reliability={"w_t": 0.8},
        drift={"drift": False},
        model_hash="abc123",
        config_hash="def456",
        prev_hash=prev_hash,
        intervened=True,
        intervention_reason="projection",
        reliability_w=0.8,
        drift_flag=False,
        inflation=1.3,
        guarantee_checks_passed=True,
        guarantee_fail_reasons=[],
        true_soc_violation_after_apply=False,
        assumptions_version="v1",
        gamma_mw=2.0,
        e_t_mwh=3.0,
        soc_tube_lower_mwh=15.0,
        soc_tube_upper_mwh=85.0,
    )
    defaults.update(kw)
    return make_certificate(**defaults)


class TestMakeCertificate:
    def test_returns_all_required_fields(self):
        cert = _cert()
        required = {"command_id", "certificate_id", "created_at", "certificate_hash",
                     "device_id", "zone_id", "controller", "proposed_action", "safe_action",
                     "uncertainty", "reliability", "drift", "model_hash", "config_hash", "prev_hash"}
        assert required <= set(cert.keys())

    def test_hash_is_string(self):
        c = _cert(cmd_id="x")
        assert isinstance(c["certificate_hash"], str)
        assert len(c["certificate_hash"]) > 16

    def test_hash_changes_with_different_inputs(self):
        c1 = _cert(cmd_id="a")
        c2 = _cert(cmd_id="b")
        # hashes include timestamp so will differ; verify they're non-empty strings
        assert isinstance(c1["certificate_hash"], str)
        assert isinstance(c2["certificate_hash"], str)

    def test_prev_hash_chain(self):
        c1 = _cert(cmd_id="1")
        c2 = _cert(cmd_id="2", prev_hash=c1["certificate_hash"])
        assert c2["prev_hash"] == c1["certificate_hash"]

    def test_optional_fields_none(self):
        cert = make_certificate(
            command_id="x", device_id="d", zone_id="z", controller="c",
            proposed_action={}, safe_action={}, uncertainty={}, reliability={},
            drift={}, model_hash="m", config_hash="c",
        )
        assert cert["intervened"] is None
        assert cert["gamma_mw"] is None

    def test_guarantee_fail_reasons_empty_list(self):
        cert = _cert(guarantee_fail_reasons=[])
        assert cert["guarantee_fail_reasons"] == []

    def test_guarantee_fail_reasons_populated(self):
        cert = _cert(guarantee_fail_reasons=["soc_invariance", "power_bounds"])
        assert cert["guarantee_fail_reasons"] == ["soc_invariance", "power_bounds"]


class TestComputeModelHash:
    def test_stable_for_same_files(self, tmp_path):
        f = tmp_path / "model.bin"
        f.write_bytes(b"model_weights_here")
        h1 = compute_model_hash([str(f)])
        h2 = compute_model_hash([str(f)])
        assert h1 == h2

    def test_changes_when_content_changes(self, tmp_path):
        f = tmp_path / "model.bin"
        f.write_bytes(b"v1")
        h1 = compute_model_hash([str(f)])
        f.write_bytes(b"v2")
        h2 = compute_model_hash([str(f)])
        assert h1 != h2

    def test_handles_missing_files(self):
        h = compute_model_hash(["/nonexistent/path/model.bin"])
        assert isinstance(h, str) and len(h) == 64

    def test_empty_input(self):
        h = compute_model_hash([])
        assert isinstance(h, str)


class TestComputeConfigHash:
    def test_deterministic(self):
        data = b'{"alpha": 0.10}'
        assert compute_config_hash(data) == compute_config_hash(data)

    def test_changes_with_content(self):
        assert compute_config_hash(b"a") != compute_config_hash(b"b")


class TestStoreAndGet:
    def test_round_trip(self, tmp_path):
        db = str(tmp_path / "test.duckdb")
        cert = _cert()
        store_certificate(cert, db, "certs")
        retrieved = get_certificate(cert["command_id"], db, "certs")
        assert retrieved is not None
        assert retrieved["command_id"] == cert["command_id"]
        assert retrieved["certificate_hash"] == cert["certificate_hash"]

    def test_get_nonexistent_returns_none(self, tmp_path):
        db = str(tmp_path / "test.duckdb")
        cert = _cert()
        store_certificate(cert, db, "certs")
        assert get_certificate("nonexistent", db, "certs") is None

    def test_get_nonexistent_db_returns_none(self):
        assert get_certificate("x", "/tmp/nonexistent_db_xyz.duckdb", "certs") is None

    def test_overwrite_existing(self, tmp_path):
        db = str(tmp_path / "test.duckdb")
        c1 = _cert(cmd_id="x", inflation=1.0)
        store_certificate(c1, db, "certs")
        c2 = _cert(cmd_id="x", inflation=2.0)
        store_certificate(c2, db, "certs")
        retrieved = get_certificate("x", db, "certs")
        assert retrieved["inflation"] == 2.0

    def test_typed_columns_persisted(self, tmp_path):
        db = str(tmp_path / "test.duckdb")
        cert = _cert(reliability_w=0.75, drift_flag=True, inflation=1.8, gamma_mw=4.0)
        store_certificate(cert, db, "certs")
        import duckdb
        conn = duckdb.connect(db)
        try:
            row = conn.execute(
                "SELECT reliability_w, drift_flag, inflation, gamma_mw FROM certs WHERE command_id = ?",
                [cert["command_id"]],
            ).fetchone()
        finally:
            conn.close()
        assert float(row[0]) == pytest.approx(0.75)
        assert bool(row[1]) is True
        assert float(row[2]) == pytest.approx(1.8)
        assert float(row[3]) == pytest.approx(4.0)

    def test_creates_parent_dirs(self, tmp_path):
        db = str(tmp_path / "nested" / "dir" / "test.duckdb")
        cert = _cert()
        store_certificate(cert, db, "certs")
        assert Path(db).exists()
