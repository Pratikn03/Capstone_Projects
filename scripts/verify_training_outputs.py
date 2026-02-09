#!/usr/bin/env python3
"""
Verify training outputs: assert all expected artifacts exist.

This script validates that a training run produced all expected outputs:
- Model files (.pkl, .pt)
- Scalers and feature lists
- Metrics JSON files
- Conformal prediction artifacts
- Training manifests
- Reports and figures

Exit code 0 = all checks passed
Exit code 1 = missing or invalid artifacts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_ok(msg: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")


def print_fail(msg: str) -> None:
    """Print failure message."""
    print(f"{Colors.RED}✗{Colors.RESET} {msg}")


def print_warn(msg: str) -> None:
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {msg}")


def print_section(title: str) -> None:
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


class ArtifactChecker:
    """Validates training artifacts and reports."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.checks_passed = 0
        self.checks_failed = 0
    
    def check_file_exists(self, path: Path, description: str) -> bool:
        """Check if a file exists."""
        if path.exists():
            print_ok(f"{description}: {path}")
            self.checks_passed += 1
            return True
        else:
            print_fail(f"{description}: {path} (MISSING)")
            self.errors.append(f"Missing file: {path}")
            self.checks_failed += 1
            return False
    
    def check_directory_exists(self, path: Path, description: str) -> bool:
        """Check if a directory exists."""
        if path.exists() and path.is_dir():
            count = len(list(path.iterdir())) if path.is_dir() else 0
            print_ok(f"{description}: {path} ({count} items)")
            self.checks_passed += 1
            return True
        else:
            print_fail(f"{description}: {path} (MISSING)")
            self.errors.append(f"Missing directory: {path}")
            self.checks_failed += 1
            return False
    
    def check_json_valid(self, path: Path, description: str, required_keys: list[str] | None = None) -> bool:
        """Check if JSON file is valid and contains required keys."""
        if not self.check_file_exists(path, f"{description} (JSON)"):
            return False
        
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            
            if required_keys:
                missing = [k for k in required_keys if k not in data]
                if missing:
                    print_warn(f"  Missing keys in {path}: {missing}")
                    self.warnings.append(f"Incomplete JSON {path}: missing {missing}")
                else:
                    if self.verbose:
                        print(f"  All required keys present: {required_keys}")
            
            return True
        except json.JSONDecodeError as e:
            print_fail(f"  Invalid JSON in {path}: {e}")
            self.errors.append(f"Invalid JSON: {path}")
            self.checks_failed += 1
            return False
    
    def check_model_bundle(self, path: Path, model_type: str, target: str) -> bool:
        """Check model bundle completeness."""
        if model_type in ("gbm_lightgbm", "gbm_xgboost"):
            # Pickle file for GBM
            if not self.check_file_exists(path, f"GBM model {target}"):
                return False
            
            try:
                import pickle
                with open(path, "rb") as f:
                    bundle = pickle.load(f)
                
                required = ["model", "feature_cols", "target"]
                missing = [k for k in required if k not in bundle]
                if missing:
                    print_warn(f"  Model bundle missing keys: {missing}")
                    self.warnings.append(f"Incomplete bundle {path}: {missing}")
                else:
                    if self.verbose:
                        n_features = len(bundle.get("feature_cols", []))
                        print(f"  Features: {n_features}, Target: {bundle.get('target')}")
                
                return True
            except Exception as e:
                print_fail(f"  Failed to load pickle: {e}")
                self.errors.append(f"Invalid pickle: {path}")
                self.checks_failed += 1
                return False
        
        elif model_type in ("lstm", "tcn"):
            # PyTorch checkpoint for DL models
            if not self.check_file_exists(path, f"{model_type.upper()} model {target}"):
                return False
            
            try:
                import torch
                checkpoint = torch.load(path, map_location="cpu")
                
                required = ["model_state", "x_scaler", "y_scaler", "feature_cols", "target"]
                missing = [k for k in required if k not in checkpoint]
                if missing:
                    print_warn(f"  Checkpoint missing keys: {missing}")
                    self.warnings.append(f"Incomplete checkpoint {path}: {missing}")
                else:
                    if self.verbose:
                        n_features = len(checkpoint.get("feature_cols", []))
                        print(f"  Features: {n_features}, Target: {checkpoint.get('target')}")
                
                return True
            except Exception as e:
                print_fail(f"  Failed to load checkpoint: {e}")
                self.errors.append(f"Invalid checkpoint: {path}")
                self.checks_failed += 1
                return False
        
        return False
    
    def check_conformal_artifacts(self, artifacts_dir: Path, targets: list[str]) -> None:
        """Check conformal prediction artifacts."""
        print_section("Conformal Prediction Artifacts")
        
        for target in targets:
            # Conformal interval JSON
            conf_path = artifacts_dir / "uncertainty" / f"{target}_conformal.json"
            self.check_json_valid(conf_path, f"Conformal {target}", ["config", "meta"])
            
            # Calibration and test NPZ files
            cal_path = artifacts_dir / "backtests" / f"{target}_calibration.npz"
            test_path = artifacts_dir / "backtests" / f"{target}_test.npz"
            
            if cal_path.exists():
                try:
                    data = np.load(cal_path)
                    if "y_true" in data and "y_pred" in data:
                        print_ok(f"Calibration data {target}: {cal_path}")
                        self.checks_passed += 1
                    else:
                        print_warn(f"Calibration NPZ missing arrays: {cal_path}")
                        self.warnings.append(f"Incomplete NPZ: {cal_path}")
                except Exception as e:
                    print_fail(f"Invalid NPZ {cal_path}: {e}")
                    self.errors.append(f"Invalid NPZ: {cal_path}")
                    self.checks_failed += 1
            else:
                print_warn(f"Calibration data {target}: {cal_path} (MISSING)")
                self.warnings.append(f"Missing calibration: {cal_path}")
    
    def verify_training_run(
        self,
        models_dir: Path,
        reports_dir: Path,
        artifacts_dir: Path,
        targets: list[str],
        model_types: list[str],
        check_cv: bool = False,
    ) -> bool:
        """
        Comprehensive verification of a training run.
        
        Returns:
            True if all critical checks pass, False otherwise
        """
        # Check model artifacts
        print_section("Model Artifacts")
        for target in targets:
            for model_type in model_types:
                if model_type.startswith("gbm"):
                    path = models_dir / f"{model_type}_{target}.pkl"
                    self.check_model_bundle(path, model_type, target)
                elif model_type in ("lstm", "tcn"):
                    path = models_dir / f"{model_type}_{target}.pt"
                    self.check_model_bundle(path, model_type, target)
        
        # Check metrics
        print_section("Metrics and Reports")
        self.check_json_valid(
            reports_dir / "week2_metrics.json",
            "Week 2 metrics",
            required_keys=["targets"],
        )
        self.check_file_exists(
            reports_dir / "ml_vs_dl_comparison.md",
            "ML vs DL comparison",
        )
        
        # Check walk-forward backtest
        wf_path = reports_dir / "walk_forward_report.json"
        if wf_path.exists():
            self.check_json_valid(wf_path, "Walk-forward backtest", ["targets"])
        
        # Check CV results (if enabled)
        if check_cv:
            for target in targets:
                cv_path = reports_dir / f"{target}_cv_results.json"
                if cv_path.exists():
                    self.check_json_valid(cv_path, f"CV results {target}", ["n_splits"])
        
        # Check conformal artifacts
        self.check_conformal_artifacts(artifacts_dir, targets)
        
        # Check manifests
        print_section("Run Manifests")
        manifest_files = list(artifacts_dir.glob("manifest_*.json"))
        if manifest_files:
            for mf in manifest_files[:3]:  # Check first 3
                self.check_json_valid(mf, "Run manifest", ["run_id", "git", "system"])
        else:
            print_warn("No manifest files found")
            self.warnings.append("Missing run manifests")
        
        # Check figures directory
        print_section("Generated Figures")
        figures_dir = reports_dir / "figures"
        if figures_dir.exists():
            png_count = len(list(figures_dir.glob("*.png")))
            svg_count = len(list(figures_dir.glob("*.svg")))
            print_ok(f"Figures: {png_count} PNG, {svg_count} SVG in {figures_dir}")
            self.checks_passed += 1
        else:
            print_warn(f"Figures directory not found: {figures_dir}")
            self.warnings.append("Missing figures directory")
        
        # Summary
        print_section("Verification Summary")
        print(f"Checks passed: {Colors.GREEN}{self.checks_passed}{Colors.RESET}")
        print(f"Checks failed: {Colors.RED}{self.checks_failed}{Colors.RESET}")
        print(f"Warnings: {Colors.YELLOW}{len(self.warnings)}{Colors.RESET}")
        
        if self.errors:
            print(f"\n{Colors.RED}Critical Errors:{Colors.RESET}")
            for err in self.errors[:10]:  # Limit output
                print(f"  {Colors.RED}•{Colors.RESET} {err}")
        
        if self.warnings and self.verbose:
            print(f"\n{Colors.YELLOW}Warnings:{Colors.RESET}")
            for warn in self.warnings[:10]:
                print(f"  {Colors.YELLOW}•{Colors.RESET} {warn}")
        
        return self.checks_failed == 0


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Verify training outputs and artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("artifacts/models"),
        help="Models directory (default: artifacts/models)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Reports directory (default: reports)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Artifacts root directory (default: artifacts)",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["load_mw", "wind_mw", "solar_mw", "price_eur_mwh"],
        help="Target variables to check",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=["gbm_lightgbm", "lstm", "tcn"],
        help="Model types to verify",
    )
    parser.add_argument(
        "--check-cv",
        action="store_true",
        help="Verify cross-validation results",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    checker = ArtifactChecker(verbose=args.verbose)
    
    success = checker.verify_training_run(
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        artifacts_dir=args.artifacts_dir,
        targets=args.targets,
        model_types=args.model_types,
        check_cv=args.check_cv,
    )
    
    if success:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All checks PASSED{Colors.RESET}")
        sys.exit(0)
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Verification FAILED{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
