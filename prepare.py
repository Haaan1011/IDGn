"""Prepare the manuscript cohort in a private run directory."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "pipeline"


def prepare(clinical, monitoring, run, split=None):
    import pandas as pd

    paths = [Path(path).resolve(strict=True) for path in (clinical, monitoring)]
    run = Path(run).resolve()
    run.mkdir(mode=0o700, parents=True, exist_ok=False)
    data = run / "data"
    processed = data / "processed"
    splits = data / "splits"
    splits.mkdir(parents=True)
    config = run / "paths.yaml"
    config.write_text(
        f'raw_data:\n  clinical_table: "{paths[0].as_posix()}"\n'
        f'  monitoring_table: "{paths[1].as_posix()}"\n'
        f'outputs:\n  interim_data_dir: "{(data / "interim").as_posix()}"\n'
        f'  processed_data_dir: "{processed.as_posix()}"\n',
        encoding="utf-8",
    )
    env = {**os.environ, "PYTHONUTF8": "1"}
    log = run / "preprocessing.log"
    with log.open("w", encoding="utf-8") as stream:
        for script in (
            "scripts/ingest/run_standardize.py",
            "scripts/build_samples/run_build_samples.py",
            "scripts/ingest/run_male_factor_augment.py",
        ):
            subprocess.run(
                [sys.executable, str(ROOT / script), "--paths-config", str(config)],
                cwd=run, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True,
            )
    baseline = pd.read_csv(processed / "baseline_cycle_dataset.csv", dtype={"cycle_uid": str})
    if split:
        partition = pd.read_csv(Path(split).resolve(strict=True), dtype={"cycle_uid": str})[["cycle_uid", "split"]]
        split_source = "provided frozen manifest"
    else:
        sys.path.insert(0, str(ROOT))
        from models.common.splits import build_split_manifest
        from common.simple_yaml import load_yaml

        partition, _ = build_split_manifest(
            baseline, load_yaml(ROOT / "configs/splits/v1_split_manifest.yaml"),
            splits / "regenerated.csv",
        )
        partition = partition[["cycle_uid", "split"]]
        split_source = "regenerated using the original stratified procedure"
    expected = {"train": 2849, "valid": 611, "test": 611}
    if partition.cycle_uid.isna().any() or not partition.cycle_uid.is_unique:
        raise ValueError("The frozen split must contain exactly one row per cycle.")
    if partition.split.value_counts().to_dict() != expected or len(partition) != sum(expected.values()):
        raise ValueError("The manuscript requires the original 2849/611/611 split.")
    if not baseline.cycle_uid.is_unique or set(baseline.cycle_uid) != set(partition.cycle_uid):
        raise ValueError("The raw-data cohort differs from the frozen manuscript cohort.")
    partition.to_csv(splits / "split_manifest_v1.csv", index=False)
    clinical_table = pd.read_csv(
        data / "interim/clinical_standardized.csv",
        usecols=["cycle_uid", "complication_text"],
        dtype={"cycle_uid": str},
    )
    outcomes = baseline[["cycle_uid"]].merge(clinical_table, on="cycle_uid", validate="one_to_one")
    outcomes.to_csv(processed / "successful_cycle_labels_v1.csv", index=False)
    audit = {"cohort_cycles": len(baseline), "split": expected, "split_source": split_source}
    (run / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("clinical", "monitoring", "run"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    parser.add_argument("--split", type=Path)
    prepare(**vars(parser.parse_args()))
