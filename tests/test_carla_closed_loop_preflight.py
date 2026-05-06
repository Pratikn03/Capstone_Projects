from __future__ import annotations

import tarfile
from pathlib import Path

from scripts.run_carla_closed_loop_validation import run_carla_closed_loop_validation


def _write_targz(path: Path, member_name: str) -> None:
    payload = path.parent / "payload.txt"
    payload.write_text("#!/bin/sh\n", encoding="utf-8")
    with tarfile.open(path, "w:gz") as archive:
        archive.add(payload, arcname=member_name)


def test_carla_preflight_blocks_linux_archive_on_local_mac_shape(tmp_path: Path) -> None:
    carla_root = tmp_path / "carla"
    carla_root.mkdir()
    _write_targz(carla_root / "CARLA_0.9.16.tar.gz", "CARLA_0.9.16/CarlaUE4.sh")

    manifest = run_carla_closed_loop_validation(
        carla_root=carla_root,
        out_dir=tmp_path / "reports",
        max_episodes=2,
        max_steps_per_episode=10,
    )

    assert manifest["carla_completed"] is False
    assert manifest["status"] in {
        "blocked_by_local_platform",
        "blocked_no_launcher",
        "runnable_preflight_passed",
    }
    assert "road deployment" in manifest["claim_boundary"]
    assert (tmp_path / "reports" / "carla_closed_loop_summary.csv").exists()
    assert (tmp_path / "reports" / "carla_closed_loop_traces.csv").exists()


def test_carla_completed_requires_imported_closed_loop_trace_rows(tmp_path: Path) -> None:
    carla_root = tmp_path / "carla"
    carla_root.mkdir()
    traces = tmp_path / "external_carla_traces.csv"
    traces.write_text(
        "\n".join(
            [
                "episode_id,step,stress_family,controller,certificate_valid,true_constraint_violated,postcondition_passed",
                "ep0,0,nominal,orius,1,0,1",
                "ep0,1,nominal,orius,1,0,1",
                "ep1,0,dropout,orius,1,0,1",
                "ep1,1,dropout,orius,1,0,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = run_carla_closed_loop_validation(
        carla_root=carla_root,
        out_dir=tmp_path / "reports",
        max_episodes=2,
        max_steps_per_episode=2,
        trace_input=traces,
        min_completed_episodes=2,
        min_completed_steps=4,
    )

    assert manifest["carla_completed"] is True
    assert manifest["status"] == "completed_closed_loop"
    assert manifest["episodes_completed"] == 2
    assert manifest["steps_completed"] == 4
    assert manifest["safety_violations"] == 0
