from __future__ import annotations

import scripts.build_data_integrity_audit as audit


def test_build_data_integrity_audit_reports_current_repo_truth(monkeypatch) -> None:
    monkeypatch.setattr(audit, "file_sha256", lambda path: f"sha256:{path.name}")

    report = audit.build_report()
    rows = {row["dataset"]: row for row in report["datasets"]}

    assert rows["OPSD Germany"]["defended_rows"] == 17377
    assert rows["OPSD Germany"]["split_gap_policy_holds"] is True

    assert rows["KITTI Odometry"]["raw_data_present"] == "manifest-only"
    assert rows["KITTI Odometry"]["actually_used_by_corresponding_adapter"] == "no"
    assert rows["KITTI Odometry"]["manifest_integrity"]["state"] == "ok"
    assert rows["KITTI Odometry"]["exact_blocker"] == "navigation_kitti_runtime_missing"

    assert rows["NASA C-MAPSS FD001"]["actually_used_by_corresponding_adapter"] == "partial"
    assert rows["NASA C-MAPSS FD001"]["canonical_runtime_present"] is False
    assert rows["NASA C-MAPSS FD001"]["support_runtime_present"] is True

    assert rows["UCI CCPP"]["raw_rows"] == 9568
    assert rows["BIDMC runtime / MIMIC-III boundary"]["mimic_iii_staged"] is False
    assert rows["BIDMC runtime / MIMIC-III boundary"]["actually_used_by_corresponding_adapter"] == "yes"
