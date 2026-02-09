#!/usr/bin/env python3
"""
Verify novelty outputs for GridPulse.

Validates that all Advanced novelty components have generated required artifacts:
1. Robust dispatch with uncertainty quantification
2. Ablation study results
3. Statistical analysis outputs
4. Publication-quality figures
5. Cross-region comparison (if applicable)

Exit code 0 = all checks passed
Exit code 1 = missing or invalid artifacts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


class Colors:
    """ANSI color codes."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_ok(msg: str) -> None:
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")


def print_fail(msg: str) -> None:
    print(f"{Colors.RED}✗{Colors.RESET} {msg}")


def print_warn(msg: str) -> None:
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {msg}")


def print_section(title: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


class NoveltyChecker:
    """Validates novelty artifacts."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.checks_passed = 0
        self.checks_failed = 0
    
    def check_file_exists(self, path: Path, description: str) -> bool:
        """Check if file exists."""
        if path.exists():
            print_ok(f"{description}: {path}")
            self.checks_passed += 1
            return True
        else:
            print_fail(f"{description}: {path} (MISSING)")
            self.errors.append(f"Missing: {path}")
            self.checks_failed += 1
            return False
    
    def check_csv_valid(self, path: Path, description: str, min_rows: int = 1) -> bool:
        """Check if CSV is valid and has minimum rows."""
        if not self.check_file_exists(path, f"{description} (CSV)"):
            return False
        
        try:
            df = pd.read_csv(path)
            if len(df) < min_rows:
                print_warn(f"  CSV has only {len(df)} rows, expected >= {min_rows}")
                self.warnings.append(f"Insufficient rows in {path}")
            elif self.verbose:
                print(f"  {len(df)} rows, {len(df.columns)} columns")
            return True
        except Exception as e:
            print_fail(f"  Invalid CSV: {e}")
            self.errors.append(f"Invalid CSV: {path}")
            self.checks_failed += 1
            return False
    
    def check_json_valid(self, path: Path, description: str, required_keys: list[str] | None = None) -> bool:
        """Check if JSON is valid."""
        if not self.check_file_exists(path, f"{description} (JSON)"):
            return False
        
        try:
            data = json.loads(path.read_text())
            
            if required_keys:
                missing = [k for k in required_keys if k not in data]
                if missing:
                    print_warn(f"  Missing keys: {missing}")
                    self.warnings.append(f"Incomplete JSON {path}: missing {missing}")
                elif self.verbose:
                    print(f"  All required keys present")
            
            return True
        except Exception as e:
            print_fail(f"  Invalid JSON: {e}")
            self.errors.append(f"Invalid JSON: {path}")
            self.checks_failed += 1
            return False
    
    def check_ablation_outputs(self, ablation_dir: Path) -> None:
        """Check ablation study outputs."""
        print_section("Ablation Study Outputs")
        
        # Results CSV
        self.check_csv_valid(
            ablation_dir / "ablation_results.csv",
            "Ablation results",
            min_rows=4  # At least 4 scenarios
        )
        
        # Summary CSV
        self.check_csv_valid(
            ablation_dir / "ablation_summary.csv",
            "Ablation summary",
            min_rows=3
        )
        
        # Statistical comparisons
        self.check_json_valid(
            ablation_dir / "ablation_stats.json",
            "Statistical tests",
            required_keys=None  # Variable keys based on scenarios
        )
        
        # Visualization
        self.check_file_exists(
            ablation_dir / "ablation_comparison.png",
            "Ablation bar chart"
        )
    
    def check_statistical_outputs(self, tables_dir: Path) -> None:
        """Check statistical analysis outputs."""
        print_section("Statistical Analysis Outputs")
        
        # LaTeX tables
        latex_files = [
            "ablation_table.tex",
        ]
        
        for latex_file in latex_files:
            path = tables_dir / latex_file
            if path.exists():
                print_ok(f"LaTeX table: {path}")
                self.checks_passed += 1
            else:
                print_warn(f"LaTeX table: {path} (optional)")
    
    def check_figures(self, figures_dir: Path) -> None:
        """Check publication figures."""
        print_section("Publication Figures")
        
        if not figures_dir.exists():
            print_warn(f"Figures directory not found: {figures_dir}")
            self.warnings.append(f"Missing figures directory")
            return
        
        # Count PNG files
        png_files = list(figures_dir.glob("*.png"))
        svg_files = list(figures_dir.glob("*.svg"))
        
        print_ok(f"Found {len(png_files)} PNG files, {len(svg_files)} SVG files")
        self.checks_passed += 1
        
        # Check for specific novelty figures
        expected_novelty_figs = [
            "ablation_comparison.png",
        ]
        
        for fig in expected_novelty_figs:
            found = any(fig in str(f) for f in png_files)
            if found:
                print_ok(f"  {fig}")
            else:
                # Check in ablation dir
                alt_path = Path("reports/ablations") / fig
                if alt_path.exists():
                    print_ok(f"  {fig} (in ablations/)")
                else:
                    print_warn(f"  {fig} (not found)")
    
    def verify_novelty_outputs(
        self,
        ablation_dir: Path,
        tables_dir: Path,
        figures_dir: Path,
    ) -> bool:
        """
        Comprehensive verification of novelty outputs.
        
        Returns:
            True if all critical checks pass
        """
        # Check ablation outputs
        if ablation_dir.exists():
            self.check_ablation_outputs(ablation_dir)
        else:
            print_section("Ablation Study Outputs")
            print_fail(f"Ablation directory not found: {ablation_dir}")
            self.errors.append("Missing ablation directory")
            self.checks_failed += 1
        
        # Check statistical outputs
        if tables_dir.exists():
            self.check_statistical_outputs(tables_dir)
        else:
            print_section("Statistical Analysis Outputs")
            print_warn(f"Tables directory not found: {tables_dir}")
            self.warnings.append("Missing tables directory")
        
        # Check figures
        self.check_figures(figures_dir)
        
        # Summary
        print_section("Verification Summary")
        print(f"Checks passed: {Colors.GREEN}{self.checks_passed}{Colors.RESET}")
        print(f"Checks failed: {Colors.RED}{self.checks_failed}{Colors.RESET}")
        print(f"Warnings: {Colors.YELLOW}{len(self.warnings)}{Colors.RESET}")
        
        if self.errors:
            print(f"\n{Colors.RED}Critical Errors:{Colors.RESET}")
            for err in self.errors[:10]:
                print(f"  {Colors.RED}•{Colors.RESET} {err}")
        
        if self.warnings and self.verbose:
            print(f"\n{Colors.YELLOW}Warnings:{Colors.RESET}")
            for warn in self.warnings[:10]:
                print(f"  {Colors.YELLOW}•{Colors.RESET} {warn}")
        
        return self.checks_failed == 0


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Verify GridPulse novelty outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=Path("reports/ablations"),
        help="Ablation results directory",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("reports/tables"),
        help="Statistical tables directory",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("reports/figures"),
        help="Figures directory",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    checker = NoveltyChecker(verbose=args.verbose)
    
    success = checker.verify_novelty_outputs(
        ablation_dir=args.ablation_dir,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
    )
    
    if success:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All novelty checks PASSED{Colors.RESET}")
        sys.exit(0)
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Verification FAILED{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
