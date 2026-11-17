#!/usr/bin/env python3
"""Frozen-split GRU(AddGate) models for next-real-monitor Gn dose prediction.

At real monitoring visit k, the model receives only information documented at
or before visit k and predicts the Gn dose recorded at visit k+1.  This is an
isolated experiment: it never writes to legacy daily-expanded result folders.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/monitor_level_gru_addgate_27features_next_monitor_20261015"
DATA_PATH = ROOT / "data/processed/snapshot_feature_dataset.csv"
MANIFEST_PATH = ROOT / "data/splits/split_manifest_v1.csv"
SEED = 20261015
BOOTSTRAP_N = 1000
MAX_HISTORY_VISITS = 9

STATIC_FEATURES = [
    "age", "bmi", "amh", "afc", "basal_fsh", "basal_lh", "basal_e2", "basal_p",
    "infertility_duration",
]
DYNAMIC_FEATURES = [
    "stim_day", "monitoring_order", "current_e2", "current_lh", "current_p", "current_fsh",
    "current_endometrium", "total_follicle_count", "max_follicle_diameter",
    "mean_follicle_diameter", "follicle_count_lt_10", "follicle_count_10_12",
    "follicle_count_13_15", "follicle_count_16_18", "follicle_count_gt_18",
    "previous_fsh_daily_dose", "previous_lh_daily_dose", "previous_hmg_daily_dose",
]
FEATURE_LABELS = {
    "age": "Age", "bmi": "Body mass index", "amh": "Anti-Müllerian hormone",
    "afc": "Antral follicle count", "basal_fsh": "Basal serum FSH",
    "basal_lh": "Basal serum LH", "basal_e2": "Basal serum estradiol",
    "basal_p": "Basal serum progesterone", "infertility_duration": "Infertility duration",
    "stim_day": "Stimulation day", "monitoring_order": "Monitoring visit",
    "current_e2": "Serum estradiol", "current_lh": "Serum LH",
    "current_p": "Serum progesterone", "current_fsh": "Serum FSH",
    "current_endometrium": "Endometrial thickness", "total_follicle_count": "Total follicle count",
    "max_follicle_diameter": "Maximum follicular diameter",
    "mean_follicle_diameter": "Mean follicular diameter",
    "follicle_count_lt_10": "Number of follicles measuring <10 mm",
    "follicle_count_10_12": "Number of follicles measuring 10–12 mm",
    "follicle_count_13_15": "Number of follicles measuring 13–15 mm",
    "follicle_count_16_18": "Number of follicles measuring 16–18 mm",
    "follicle_count_gt_18": "Number of follicles measuring >18 mm",
    "previous_fsh_daily_dose": "Previous FSH dose",
    "previous_lh_daily_dose": "Previous LH dose",
    "previous_hmg_daily_dose": "Previous HMG dose",
}
TARGETS = {
    "fsh": {"labels": ["low_or_none", "moderate", "high"], "dose_col": "next_fsh_daily_dose", "offset": 11},
    "lh": {"labels": ["dose_0", "dose_75", "dose_ge150"], "dose_col": "next_lh_daily_dose", "offset": 23},
    "hmg": {"labels": ["dose_0", "dose_75", "dose_ge150"], "dose_col": "next_hmg_daily_dose", "offset": 37},
}


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


phase866 = import_module(
    "monitor27_phase866",
    ROOT / "scripts/experiment/phase866_daily_expanded_model_comparison/run_phase866_daily_expanded_model_comparison.py",
)
phase864 = phase866.phase864


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dose_label(target: str, value: Any) -> str | None:
    dose = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(dose) or not 0 <= float(dose) <= 300:
        return None
    dose = float(dose)
    if target == "fsh":
        if dose < 80:
            return "low_or_none"
        if dose <= 160:
            return "moderate"
        return "high"
    if dose == 0:
        return "dose_0"
    if dose == 75:
        return "dose_75"
    if dose >= 150:
        return "dose_ge150"
    return None


def is_true(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin(["true", "1"])


def build_monitor_frame(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"cycle_uid", "visit_uid", "monitoring_order", "has_next_visit", "gn_day", *STATIC_FEATURES,
                *(f for f in DYNAMIC_FEATURES if f not in {"stim_day", "monitoring_order"}),
                "next_fsh_daily_dose", "next_lh_daily_dose", "next_hmg_daily_dose"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Required source columns absent: {missing}")
    frame = data.loc[is_true(data["has_next_visit"])].copy()
    frame["monitoring_order"] = pd.to_numeric(frame["monitoring_order"], errors="coerce")
    frame = frame[frame["monitoring_order"].notna() & frame["monitoring_order"].ge(1)].copy()
    frame["monitoring_order"] = frame["monitoring_order"].astype(int)
    frame["Day"] = frame["monitoring_order"]  # ordering key for the generic sequence builder only.
    raw_stim = pd.to_numeric(frame["gn_day"], errors="coerce")
    frame["stim_day"] = raw_stim.where(raw_stim.between(0, 30), np.nan)
    first_monitor = frame["monitoring_order"].eq(1)
    frame.loc[first_monitor & frame["stim_day"].isna(), "stim_day"] = 0.0
    prior_features = ["previous_fsh_daily_dose", "previous_lh_daily_dose", "previous_hmg_daily_dose"]
    for feature in prior_features:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
        frame.loc[first_monitor & frame[feature].isna(), feature] = 0.0
    for feature in STATIC_FEATURES + DYNAMIC_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    frame = frame.sort_values(["cycle_uid", "monitoring_order", "visit_uid"], kind="mergesort")
    measurements = ["current_e2", "current_lh", "current_p", "current_fsh", "current_endometrium"]
    frame[measurements] = frame.groupby("cycle_uid", sort=False)[measurements].ffill()
    frame["target_monitoring_order"] = frame["monitoring_order"] + 1
    frame = frame.sort_values(["cycle_uid", "monitoring_order", "visit_uid"], kind="mergesort").reset_index(drop=True)
    first_monitor = frame["monitoring_order"].eq(1)
    audit = {
        "rows_with_next_real_monitor": int(len(frame)),
        "cycles_with_next_real_monitor": int(frame["cycle_uid"].nunique()),
        "stimulation_day_source": "gn_day mapped to stim_day",
        "stimulation_day_cleaning": "values outside 0-30 set missing; missing/invalid at Monitor 1 set to 0",
        "stim_day_missing_after_cleaning": int(frame["stim_day"].isna().sum()),
        "monitor1_prior_dose_missing_after_encoding": {
            feature: int(frame.loc[first_monitor, feature].isna().sum()) for feature in prior_features
        },
        "static_features": STATIC_FEATURES,
        "dynamic_features": DYNAMIC_FEATURES,
        "feature_labels": FEATURE_LABELS,
    }
    return frame, audit


def build_raw(frame: pd.DataFrame, target: str):
    spec = TARGETS[target]
    working = frame.copy()
    working["ctfe_truth"] = working[spec["dose_col"]].map(lambda value: dose_label(target, value))
    working = working[working["ctfe_truth"].isin(spec["labels"])].copy()
    if working.empty:
        raise ValueError(f"{target}: no valid next-monitor labels")
    label_id = {label: index for index, label in enumerate(spec["labels"])}
    working["ctfe_truth_id"] = working["ctfe_truth"].map(label_id).astype(int)
    working["source_visit_uid"] = working["visit_uid"].astype(str)
    raw = phase864.build_daily_sequence_arrays(
        working,
        static_features=STATIC_FEATURES,
        dynamic_features=DYNAMIC_FEATURES,
        max_history_days=MAX_HISTORY_VISITS,
    )
    metadata = working.set_index("visit_uid")[["monitoring_order", "target_monitoring_order", spec["dose_col"], "stim_day"]]
    return raw, metadata


def make_loader(data: Any, mask: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(data.static[mask], dtype=torch.float32),
        torch.tensor(data.dynamic[mask], dtype=torch.float32),
        torch.tensor(data.mask[mask], dtype=torch.float32),
        torch.tensor(data.y[mask], dtype=torch.long),
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def predict(model: nn.Module, data: Any, mask: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, embeddings = [], []
    with torch.no_grad():
        for static_x, dynamic_x, obs_mask, _ in make_loader(data, mask, 2048, False, SEED):
            logits, embedding = model(static_x.to(device), dynamic_x.to(device), obs_mask.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            embeddings.append(embedding.cpu().numpy())
    return np.vstack(probabilities), np.vstack(embeddings)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float | int]:
    return {
        "n_records": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(range(n_classes)), average="weighted", zero_division=0)),
    }


def monitor_accuracy_bootstrap(test: pd.DataFrame, target: str, n_bootstrap: int = BOOTSTRAP_N) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED + TARGETS[target]["offset"] + 1000)
    for monitor, subset in test.groupby("monitoring_order", sort=True):
        cycle_to_accuracy = subset.groupby("cycle_uid").apply(lambda x: float((x["y_true"] == x["y_pred"]).mean()))
        per_cycle = cycle_to_accuracy.to_numpy(float)
        point = float(per_cycle.mean())
        sampled = rng.choice(per_cycle, size=(n_bootstrap, len(per_cycle)), replace=True).mean(axis=1)
        rows.append({
            "target": target.upper(), "monitoring_visit": int(monitor),
            "target_monitoring_visit": int(monitor) + 1, "n_records": int(len(subset)),
            "n_cycles": int(subset["cycle_uid"].nunique()), "accuracy": point,
            "ci_lower": float(np.quantile(sampled, 0.025)), "ci_upper": float(np.quantile(sampled, 0.975)),
            "bootstrap_replicates": int(n_bootstrap), "bootstrap_unit": "cycle_uid",
        })
    return pd.DataFrame(rows)


def train_target(target: str, frame: pd.DataFrame, manifest: pd.DataFrame, device: torch.device) -> tuple[dict[str, Any], pd.DataFrame]:
    spec = TARGETS[target]
    labels = list(spec["labels"])
    phase866.LABELS = labels
    phase866.NUM_CLASSES = len(labels)
    raw, metadata = build_raw(frame, target)
    split_map = manifest.set_index("cycle_uid")["split"]
    row_splits = raw.row_index["cycle_uid"].astype(str).map(split_map)
    if row_splits.isna().any():
        raise RuntimeError(f"{target}: rows without frozen split assignments")
    masks = {name: row_splits.eq(name).to_numpy(dtype=bool) for name in ["train", "valid", "test"]}
    data = phase864.prepare_arrays(raw, fit_mask=masks["train"])
    target_seed = SEED + int(spec["offset"])
    set_seed(target_seed)
    model = phase866.FusionGRUClassifier(
        data.static.shape[1], data.dynamic.shape[2], 96, len(labels), 0.15, mode="addgate"
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    criterion = nn.CrossEntropyLoss()
    best_state, best_epoch, best_score, wait = None, 0, -np.inf, 0
    curves = []
    for epoch in range(1, 71):
        model.train()
        losses = []
        for static_x, dynamic_x, obs_mask, y in make_loader(data, masks["train"], 1024, True, target_seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(static_x.to(device), dynamic_x.to(device), obs_mask.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        valid_prob, _ = predict(model, data, masks["valid"], device)
        valid_point = metrics(data.y[masks["valid"]], valid_prob.argmax(axis=1), len(labels))
        curves.append({"target": target.upper(), "epoch": epoch, "train_loss": float(np.mean(losses)), **valid_point})
        score = float(valid_point["weighted_f1"])
        print(f"[{target.upper()}] epoch={epoch:02d} loss={np.mean(losses):.5f} valid_wf1={score:.5f}", flush=True)
        if score > best_score + 1e-5:
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_epoch, best_score, wait = epoch, score, 0
        else:
            wait += 1
            if wait >= 12:
                break
    if best_state is None:
        raise RuntimeError(f"{target}: no validation-selected checkpoint")
    model.load_state_dict(best_state)
    checkpoint = {
        "state_dict": model.state_dict(), "target": target, "labels": labels,
        "static_features": STATIC_FEATURES, "dynamic_features": DYNAMIC_FEATURES,
        "feature_labels": FEATURE_LABELS, "best_epoch": best_epoch,
        "best_validation_weighted_f1": best_score,
        "protocol": "train fit; validation early stopping; independent test reported once",
        "target_definition": "information through true monitor k predicts dose recorded at true monitor k+1",
        "seed": target_seed,
    }
    torch.save(checkpoint, OUT / "checkpoints" / f"gru_addgate_{target}_monitor27.pt")
    pd.DataFrame(curves).to_csv(OUT / "training" / f"{target}_training_curve.csv", index=False)

    valid_prob, _ = predict(model, data, masks["valid"], device)
    test_prob, test_embedding = predict(model, data, masks["test"], device)
    test_pred = test_prob.argmax(axis=1)
    point = metrics(data.y[masks["test"]], test_pred, len(labels))
    test_frame = raw.row_index.loc[masks["test"]].reset_index(drop=True).copy()
    test_frame = test_frame.join(metadata, on="visit_uid", rsuffix="_source")
    test_frame["split"] = "test"
    test_frame["target"] = target.upper()
    test_frame["y_true"] = data.y[masks["test"]]
    test_frame["truth"] = [labels[int(value)] for value in data.y[masks["test"]]]
    test_frame["y_pred"] = test_pred
    test_frame["prediction"] = [labels[int(value)] for value in test_pred]
    for idx, label in enumerate(labels):
        test_frame[f"prob_{label}"] = test_prob[:, idx]
    test_frame.to_csv(OUT / "predictions" / f"{target}_test_predictions.csv", index=False)
    embedding_frame = test_frame[["cycle_uid", "visit_uid", "monitoring_order", "target_monitoring_order", "target", "y_true"]].copy()
    for idx in range(test_embedding.shape[1]):
        embedding_frame[f"embedding_{idx:03d}"] = test_embedding[:, idx]
    embedding_frame.to_csv(OUT / "embeddings" / f"{target}_test_embeddings.csv", index=False)
    monitor_table = monitor_accuracy_bootstrap(test_frame, target)
    monitor_table.to_csv(OUT / "tables" / f"{target}_test_accuracy_by_monitor_cycle_bootstrap.csv", index=False)
    coverage = {
        "target": target.upper(), "labels": labels, "best_epoch": best_epoch,
        "best_validation_weighted_f1": best_score,
        "records": {name: int(mask.sum()) for name, mask in masks.items()},
        "cycles": {name: int(raw.row_index.loc[mask, "cycle_uid"].nunique()) for name, mask in masks.items()},
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"target": target.upper(), "split": "test", "n_cycles": int(test_frame["cycle_uid"].nunique()), **point}, monitor_table, coverage


def main() -> None:
    for folder in ["checkpoints", "predictions", "embeddings", "training", "tables", "logs"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    expected = {"train": 2849, "valid": 611, "test": 611}
    if manifest["split"].value_counts().to_dict() != expected or manifest["cycle_uid"].nunique() != 4071:
        raise RuntimeError("Frozen split manifest failed the 2849/611/611 contract check")
    source = pd.read_csv(DATA_PATH, low_memory=False)
    frame, audit = build_monitor_frame(source)
    write_json(OUT / "logs" / "feature_contract.json", {
        "feature_count": 27, "static_features": STATIC_FEATURES, "dynamic_features": DYNAMIC_FEATURES,
        "display_names": FEATURE_LABELS, "target_definition": "true monitor k -> dose recorded at true monitor k+1",
        "monitor1_previous_dose_encoding": "0 = no prior Gn exposure",
    })
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_rows, monitor_rows, coverage = [], [], []
    for target in ["fsh", "lh", "hmg"]:
        point, monitor, target_coverage = train_target(target, frame, manifest, device)
        result_rows.append(point)
        monitor_rows.append(monitor)
        coverage.append(target_coverage)
        pd.DataFrame(result_rows).to_csv(OUT / "tables" / "final_test_metrics.csv", index=False)
        pd.concat(monitor_rows, ignore_index=True).to_csv(OUT / "tables" / "test_accuracy_by_monitor_cycle_bootstrap.csv", index=False)
        write_json(OUT / "logs" / "coverage.json", coverage)
    write_json(OUT / "logs" / "run_manifest.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "device": str(device), "torch_version": torch.__version__, "data_path": str(DATA_PATH.relative_to(ROOT)),
        "data_sha256": sha256(DATA_PATH), "split_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "split_manifest_sha256": sha256(MANIFEST_PATH), "data_audit": audit,
        "model": "GRU(AddGate)", "hyperparameters": {"hidden_dim": 96, "dropout": 0.15, "lr": 7e-4, "weight_decay": 2e-4, "max_epochs": 70, "patience": 12},
        "selection_metric": "validation weighted_f1", "test_accessed_during_selection": False,
        "monitor_accuracy_bootstrap": {"replicates": BOOTSTRAP_N, "unit": "cycle_uid"},
    })
    print(pd.DataFrame(result_rows).to_string(index=False))


if __name__ == "__main__":
    main()
