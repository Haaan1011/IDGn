"""Run the IDGn manuscript pipeline from two hospital exports and the frozen split."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from prepare import ROOT, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("clinical", "monitoring", "run"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--resume", action="store_true", help="Resume the same run with its original inputs.")
    args = parser.parse_args()
    run = args.run.resolve()
    inputs = {name: str(getattr(args, name).resolve(strict=True)) for name in ("clinical", "monitoring")}
    if args.split:
        inputs["split"] = str(args.split.resolve(strict=True))
    identity = run / "inputs.json"
    if args.resume:
        if json.loads(identity.read_text(encoding="utf-8")) != inputs:
            raise ValueError("Resume requires the same input paths.")
    else:
        prepare(**inputs, run=run)
        shutil.copytree(ROOT, run, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
        identity.write_text(json.dumps(inputs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logs = run / "logs"
    logs.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    gn = "outputs/monitor_level_shared_multitask_gru_addgate_27features_current_decision_tuned_20261015"
    out = "outputs/outcome27_model_family_comparison_20261016/predictions/test_predictions_all_models.csv"
    stages = [
        ("table1", "scripts/table1.py", "--clinical", inputs["clinical"]),
        ("gn27", "scripts/experiment/train_paper_gn.py", "--features", "27"),
        ("gn62", "scripts/experiment/train_paper_gn.py", "--features", "62"),
        ("gn_metrics", "scripts/experiment/evaluate_gn_metrics.py",
         "--prediction-dir", gn + "/predictions", "--output-dir", "outputs/gn_metrics"),
        ("sensitivity", "scripts/experiment/run_input_availability_sensitivity_27features.py"),
        ("shap_models", "scripts/experiment/monitor_level_27feature_next_monitor_20261015/run_monitor_level_gru_addgate_27features.py"),
        ("shap", "scripts/experiment/calculate_shap.py",
         "--output-dir", "outputs/supplementary_shap27"),
        ("outcome_validation", "scripts/experiment/run_outcome27_optimization_20261016.py", "--stage", "validation"),
        ("outcome_test", "scripts/experiment/run_outcome27_optimization_20261016.py", "--stage", "final"),
        ("outcome_comparison", "scripts/experiment/outcome27_model_family_comparison_20261016/run_comparison.py"),
        ("ohss_calibration_metrics", "scripts/experiment/evaluate_ohss_calibration.py",
         "--predictions", out, "--output-dir", "outputs/ohss_calibration"),
        ("candidate_regimens", "scripts/experiment/candidate_regimens.py", "--output-dir", "outputs/scenarios"),
        ("paper_metrics", "scripts/compare_paper_metrics.py"),
        ("full_shap", "scripts/experiment/calculate_shap.py",
         "--output-dir", "outputs/full_shap62", "--full"),
        ("architectures", "scripts/experiment/train_architectures.py"),
        ("demo_check", "scripts/check_scenario_demo.py"),
    ]
    for name, script, *options in stages:
        done = logs / f"{name}.done"
        if args.resume and done.is_file():
            continue
        print(f"Running {name}", flush=True)
        log = logs / f"{name}.log"
        with log.open("w", encoding="utf-8") as stream:
            status = subprocess.run([sys.executable, script, *options], cwd=run, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        if status.returncode:
            raise SystemExit(f"{name} failed. See {log}")
        done.touch()


if __name__ == "__main__":
    main()
