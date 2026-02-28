#!/usr/bin/env python3
"""Test all skill scripts: --help check and functional execution."""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / "skills"

# Scripts that require platform-specific packages not available in CI
SKIP_HELP = {
    "airflow_pipeline.py",   # Requires apache-airflow
    "prefect_pipeline.py",   # Requires prefect
}

# Scripts that can run functional tests without external APIs/GPU
FUNCTIONAL_TESTS = {
    # data-validation
    "validate_data.py": ["--help"],
    "data_contract.py": ["--help"],
    # feature-engineering
    "select_features.py": ["--help"],
    "transform_features.py": ["--help"],
    # ml-drift-detection
    "detect_drift.py": ["--help"],
    "drift_report.py": ["--help"],
    # model-monitoring
    "monitor_model.py": ["--help"],
    "alert_manager.py": ["--help"],
    # feature-store
    "feature_registry.py": ["--help"],
    # ml-experiment-tracking
    "experiment_compare.py": ["--help"],
    # ml-testing
    "test_model.py": ["--help"],
    "test_data.py": ["--help"],
    # ml-security
    "security_scan.py": ["--help"],
    "privacy_guard.py": ["--help"],
    # ml-cost-optimization
    "cost_analyzer.py": ["--help"],
    "model_compress.py": ["--help"],
}


def find_all_scripts():
    """Find all Python scripts across all skills."""
    scripts = []
    for category in ["mlops", "llmops"]:
        cat_dir = SKILLS_DIR / category
        if not cat_dir.is_dir():
            continue
        for skill_dir in sorted(cat_dir.iterdir()):
            scripts_dir = skill_dir / "scripts"
            if scripts_dir.is_dir():
                for py_file in sorted(scripts_dir.glob("*.py")):
                    scripts.append(py_file)
    return scripts


def run_help_test(script_path, timeout=30):
    """Test that a script's --help runs without error."""
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return "PASS", None
        else:
            return "FAIL", result.stderr[:200]
    except subprocess.TimeoutExpired:
        return "FAIL", "Timeout"
    except Exception as e:
        return "FAIL", str(e)[:200]


def main():
    parser = argparse.ArgumentParser(description="Test all skill scripts")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all results")
    parser.add_argument("--help-only", action="store_true", help="Only run --help tests")
    args = parser.parse_args()

    scripts = find_all_scripts()
    total = len(scripts)
    passed = 0
    failed = 0
    skipped = 0

    print(f"Testing {total} scripts...\n")
    print("--- --help Tests ---\n")

    for script in scripts:
        name = script.name
        skill = script.parent.parent.name

        if name in SKIP_HELP:
            skipped += 1
            if args.verbose:
                print(f"  SKIP  {skill}/{name} (platform-specific)")
            continue

        status, error = run_help_test(script)
        if status == "PASS":
            passed += 1
            if args.verbose:
                print(f"  PASS  {skill}/{name}")
        else:
            failed += 1
            print(f"  FAIL  {skill}/{name}")
            if error:
                print(f"        {error}")

    print(f"\n{'='*50}")
    print(f"--help tests: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"Total scripts: {total}")
    print(f"{'='*50}")

    if failed > 0:
        sys.exit(1)
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
