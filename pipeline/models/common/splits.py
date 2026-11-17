from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def _combined_strata(frame: pd.DataFrame) -> pd.Series:
    live_birth = frame["target_live_birth"].fillna(0).astype(int).astype(str)
    ohss = frame["target_ohss_flag"].fillna(0).astype(int).astype(str)
    return live_birth + "_" + ohss


def _split_indices(frame: pd.DataFrame, seed: int, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    indices = np.arange(len(frame))
    try:
        train_idx, test_idx = next(splitter.split(indices, _combined_strata(frame)))
        return np.asarray(train_idx), np.asarray(test_idx)
    except ValueError:
        rng = np.random.default_rng(seed)
        permuted = rng.permutation(indices)
        cutoff = int(round(len(permuted) * (1.0 - test_size)))
        return permuted[:cutoff], permuted[cutoff:]


def build_split_manifest(
    source_df: pd.DataFrame,
    split_config: dict[str, Any],
    output_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"cycle_uid", "target_live_birth", "target_ohss_flag"}
    missing = required.difference(source_df.columns)
    if missing:
        raise ValueError(f"Source data missing required split columns: {sorted(missing)}")

    cycle_frame = (
        source_df.loc[:, ["cycle_uid", "target_live_birth", "target_ohss_flag"]]
        .drop_duplicates("cycle_uid")
        .reset_index(drop=True)
        .copy()
    )
    if len(cycle_frame) != source_df["cycle_uid"].nunique():
        raise ValueError("Split source must contain one row per cycle_uid.")

    seed = int(split_config["seed"])
    train_ratio = float(split_config["train_ratio"])
    valid_ratio = float(split_config["valid_ratio"])
    test_ratio = float(split_config["test_ratio"])
    if round(train_ratio + valid_ratio + test_ratio, 8) != 1.0:
        raise ValueError("Split ratios must sum to 1.0.")

    train_valid_idx, test_idx = _split_indices(cycle_frame, seed=seed, test_size=test_ratio)
    train_valid = cycle_frame.iloc[train_valid_idx].reset_index(drop=True)
    test_frame = cycle_frame.iloc[test_idx].reset_index(drop=True)

    rel_valid = valid_ratio / (train_ratio + valid_ratio)
    train_idx, valid_idx = _split_indices(train_valid, seed=seed + 1, test_size=rel_valid)
    train_frame = train_valid.iloc[train_idx].copy()
    valid_frame = train_valid.iloc[valid_idx].copy()

    split_manifest = pd.concat(
        [
            train_frame.assign(split="train"),
            valid_frame.assign(split="valid"),
            test_frame.assign(split="test"),
        ],
        ignore_index=True,
    )
    split_manifest["split_strata"] = _combined_strata(split_manifest)
    split_manifest = split_manifest.sort_values(["split", "cycle_uid"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_manifest.to_csv(output_path, index=False)

    summary = {
        "split_version": split_config.get("split_version", "v1"),
        "seed": seed,
        "train_cycles": int((split_manifest["split"] == "train").sum()),
        "valid_cycles": int((split_manifest["split"] == "valid").sum()),
        "test_cycles": int((split_manifest["split"] == "test").sum()),
        "total_cycles": int(len(split_manifest)),
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "test_ratio": test_ratio,
        "strata_counts": split_manifest["split_strata"].value_counts().to_dict(),
    }
    return split_manifest, summary


def load_split_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"cycle_uid", "split"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Split manifest missing required columns: {sorted(missing)}")
    return manifest
