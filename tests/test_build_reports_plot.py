from orius.publication.report_context import ReportContext
from scripts.build_reports import (
    build_formal_report,
    build_model_cards,
    plot_model_comparison,
    report_model_enabled,
)


def test_report_model_filter_accepts_gbm_aliases_and_blocks_sequence_models(tmp_path):
    ctx = ReportContext(
        repo_root=tmp_path,
        features_path=tmp_path / "features.parquet",
        splits_dir=tmp_path / "splits",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        publication_dir=tmp_path / "publication",
        model_filter=["gbm"],
    )

    assert report_model_enabled(ctx, "gbm") is True
    assert report_model_enabled(ctx, "lstm") is False
    assert report_model_enabled(ctx, "patchtst") is False


def test_plot_model_comparison_ignores_target_metadata(tmp_path):
    ctx = ReportContext(
        repo_root=tmp_path,
        features_path=tmp_path / "features.parquet",
        splits_dir=tmp_path / "splits",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        publication_dir=tmp_path / "publication",
    )
    metrics = {
        "targets": {
            "spo2_pct": {
                "n_features": 12,
                "gbm": {"rmse": 0.25, "mae": 0.18, "smape": 0.2},
                "retention_decision": "retained_incumbent",
                "retention_reason": "challenger_regressed_against_baseline",
            },
            "hr_bpm": {
                "n_features": 12,
                "gbm": {"rmse": 1.1, "mae": 0.7, "smape": 0.4},
            },
        }
    }

    out = plot_model_comparison(ctx, metrics)

    assert out == tmp_path / "reports" / "figures" / "model_comparison.png"
    assert out.exists()


def test_model_cards_and_formal_report_ignore_target_metadata(tmp_path):
    ctx = ReportContext(
        repo_root=tmp_path,
        features_path=tmp_path / "features.parquet",
        splits_dir=tmp_path / "splits",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        publication_dir=tmp_path / "publication",
    )
    metrics = {
        "targets": {
            "spo2_pct": {
                "n_features": 12,
                "gbm": {"rmse": 0.25, "mae": 0.18, "smape": 0.2, "mape": 0.01},
                "retention_decision": "retained_incumbent",
            }
        }
    }

    build_model_cards(ctx, metrics)
    build_formal_report(ctx, None, metrics=metrics, baselines={})

    assert (tmp_path / "reports" / "model_cards" / "spo2_pct.md").exists()
    assert (tmp_path / "reports" / "formal_evaluation_report.md").exists()
