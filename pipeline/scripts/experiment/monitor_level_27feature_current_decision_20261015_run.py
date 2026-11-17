#!/usr/bin/env python3
"""Tune a 27-feature shared GRU(AddGate) for the current monitoring decision.

Clinical time point:
  information available through true Monitor k -> post-monitor Gn decision.

The post-monitor decision is stored as ``next_*_daily_dose`` in the snapshot
table because it becomes ``current_*_daily_dose`` at the following record.
Model selection and ensembling use the validation split only. The independent
test split is evaluated once after all choices are frozen.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import random
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/monitor_level_shared_multitask_gru_addgate_27features_current_decision_tuned_20261015"
HELPER_PATH = ROOT / "scripts/experiment/monitor_level_27feature_next_monitor_20261015/run_monitor_level_gru_addgate_27features.py"
DATA_PATH = ROOT / "data/processed/snapshot_feature_dataset.csv"
MANIFEST_PATH = ROOT / "data/splits/split_manifest_v1.csv"
SEED = 20261015
TARGET_ORDER = ("fsh", "lh", "hmg")
MAX_HISTORY_VISITS = 16
BOOTSTRAP_N = 1000

STATIC_FEATURES = [
    "age", "bmi", "amh", "afc", "basal_fsh", "basal_lh", "basal_e2", "basal_p",
    "infertility_duration",
]
DYNAMIC_FEATURES = [
    "stim_day", "monitoring_order", "current_e2", "current_lh", "current_p", "current_fsh",
    "current_endometrium", "total_follicle_count", "max_follicle_diameter",
    "mean_follicle_diameter", "follicle_count_lt_10", "follicle_count_10_12",
    "follicle_count_13_15", "follicle_count_16_18", "follicle_count_gt_18",
    "current_fsh_daily_dose", "current_lh_daily_dose", "current_hmg_daily_dose",
]
FEATURE_LABELS = {
    "age": "Age", "bmi": "Body mass index", "amh": "Anti-Mullerian hormone",
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
    "follicle_count_10_12": "Number of follicles measuring 10-12 mm",
    "follicle_count_13_15": "Number of follicles measuring 13-15 mm",
    "follicle_count_16_18": "Number of follicles measuring 16-18 mm",
    "follicle_count_gt_18": "Number of follicles measuring >18 mm",
    "current_fsh_daily_dose": "Previous FSH dose",
    "current_lh_daily_dose": "Previous LH dose",
    "current_hmg_daily_dose": "Previous HMG dose",
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


helper = import_module("current_decision_helper", HELPER_PATH)
phase864 = helper.phase864


@dataclass(frozen=True)
class Candidate:
    name: str
    hidden_dim: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    pooling: str
    latest_skip: bool
    adapter_dim: int
    label_smoothing: float = 0.0


CANDIDATES = [
    Candidate("base_h96", 96, 0.15, 7e-4, 2e-4, 1024, "last", False, 0),
    Candidate("latest_h96", 96, 0.15, 7e-4, 2e-4, 1024, "last", True, 0),
    Candidate("latest_adapter_h96", 96, 0.15, 7e-4, 2e-4, 1024, "last", True, 48),
    Candidate("latest_adapter_h128", 128, 0.15, 5e-4, 2e-4, 512, "last", True, 64),
    Candidate("attn_latest_adapter_h128", 128, 0.15, 5e-4, 2e-4, 512, "attention", True, 64),
    Candidate("attn_latest_adapter_h160", 160, 0.20, 3e-4, 1e-4, 512, "attention", True, 80),
    Candidate("attn_latest_adapter_smooth", 128, 0.10, 7e-4, 1e-4, 512, "attention", True, 64, 0.02),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def is_true(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin(["true", "1"])


def build_monitor_frame(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "cycle_uid", "visit_uid", "monitoring_order", "has_next_visit", "gn_day",
        *STATIC_FEATURES, *(feature for feature in DYNAMIC_FEATURES if feature not in {"stim_day", "monitoring_order"}),
        *(spec["dose_col"] for spec in TARGETS.values()),
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"Required source columns absent: {missing}")
    frame = source.loc[is_true(source["has_next_visit"])].copy()
    frame["monitoring_order"] = pd.to_numeric(frame["monitoring_order"], errors="coerce")
    frame = frame[frame["monitoring_order"].notna() & frame["monitoring_order"].ge(1)].copy()
    frame["monitoring_order"] = frame["monitoring_order"].astype(int)
    frame["Day"] = frame["monitoring_order"]
    raw_stim = pd.to_numeric(frame["gn_day"], errors="coerce")
    frame["stim_day"] = raw_stim.where(raw_stim.between(0, 30), np.nan)
    first_monitor = frame["monitoring_order"].eq(1)
    frame.loc[first_monitor & frame["stim_day"].isna(), "stim_day"] = 0.0
    for feature in STATIC_FEATURES + DYNAMIC_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    frame = frame.sort_values(["cycle_uid", "monitoring_order", "visit_uid"], kind="mergesort").reset_index(drop=True)
    measurements = ["current_e2", "current_lh", "current_p", "current_fsh", "current_endometrium"]
    frame[measurements] = frame.groupby("cycle_uid", sort=False)[measurements].ffill()
    for feature in ("current_fsh_daily_dose", "current_lh_daily_dose", "current_hmg_daily_dose"):
        frame.loc[first_monitor & frame[feature].isna(), feature] = 0.0
    frame["target_monitoring_order"] = frame["monitoring_order"] + 1
    audit = {
        "rows": int(len(frame)), "cycles": int(frame["cycle_uid"].nunique()),
        "clinical_target": "post-Monitor-k Gn decision",
        "stored_target_columns": {target: TARGETS[target]["dose_col"] for target in TARGET_ORDER},
        "pre_adjustment_dose_inputs": ["current_fsh_daily_dose", "current_lh_daily_dose", "current_hmg_daily_dose"],
        "feature_count": len(STATIC_FEATURES) + len(DYNAMIC_FEATURES),
    }
    return frame, audit


def prepare_master(frame: pd.DataFrame):
    working = frame.copy()
    working["ctfe_truth"] = "sequence_row"
    working["ctfe_truth_id"] = 0
    working["source_visit_uid"] = working["visit_uid"].astype(str)
    raw = phase864.build_daily_sequence_arrays(
        working, static_features=STATIC_FEATURES, dynamic_features=DYNAMIC_FEATURES,
        max_history_days=MAX_HISTORY_VISITS,
    )
    lookup = working.set_index("visit_uid")
    labels: dict[str, np.ndarray] = {}
    for target in TARGET_ORDER:
        label_to_id = {label: index for index, label in enumerate(TARGETS[target]["labels"])}
        labels[target] = (
            lookup.loc[raw.row_index["visit_uid"], TARGETS[target]["dose_col"]]
            .map(lambda value: helper.dose_label(target, value)).map(label_to_id)
            .fillna(-1).astype(int).to_numpy()
        )
    metadata = lookup.loc[raw.row_index["visit_uid"], ["monitoring_order", "target_monitoring_order", "stim_day"]].reset_index(drop=True)
    return raw, labels, metadata


def gather_last(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    index = mask.sum(dim=1).long().clamp_min(1) - 1
    return sequence[torch.arange(sequence.shape[0], device=sequence.device), index]


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, pooling: str):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.pooling = pooling
        self.attention = nn.Linear(hidden_dim, 1) if pooling == "attention" else None

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.rnn(sequence)
        if self.pooling == "last":
            return gather_last(outputs, mask)
        score = self.attention(torch.tanh(outputs)).squeeze(-1)
        score = score.masked_fill(~mask.bool(), -1e9)
        weights = torch.softmax(score, dim=1)
        return torch.sum(outputs * weights.unsqueeze(-1), dim=1)


class SharedDecisionGRU(nn.Module):
    def __init__(self, static_dim: int, dynamic_dim: int, config: Candidate):
        super().__init__()
        hidden = config.hidden_dim
        self.dynamic_encoder = TemporalEncoder(dynamic_dim, hidden, config.pooling)
        self.cross_encoder = TemporalEncoder(static_dim + dynamic_dim, hidden, config.pooling)
        self.cross_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.latest_skip = config.latest_skip
        if config.latest_skip:
            self.latest_projection = nn.Sequential(nn.Linear(dynamic_dim, hidden), nn.GELU())
            self.latest_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.fuse = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(config.dropout))
        heads: dict[str, nn.Module] = {}
        for target in TARGET_ORDER:
            if config.adapter_dim > 0:
                heads[target] = nn.Sequential(
                    nn.LayerNorm(hidden), nn.Linear(hidden, config.adapter_dim), nn.GELU(),
                    nn.Dropout(config.dropout), nn.Linear(config.adapter_dim, len(TARGETS[target]["labels"])),
                )
            else:
                heads[target] = nn.Linear(hidden, len(TARGETS[target]["labels"]))
        self.heads = nn.ModuleDict(heads)

    def encode(self, static_x: torch.Tensor, dynamic_x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        static_sequence = static_x.unsqueeze(1).expand(-1, dynamic_x.shape[1], -1)
        dynamic_h = self.dynamic_encoder(dynamic_x, mask)
        cross_h = self.cross_encoder(torch.cat([static_sequence, dynamic_x], dim=-1), mask)
        fused = dynamic_h + self.cross_gate(torch.cat([dynamic_h, cross_h], dim=1)) * cross_h
        if self.latest_skip:
            latest_h = self.latest_projection(gather_last(dynamic_x, mask))
            fused = fused + self.latest_gate(torch.cat([fused, latest_h], dim=1)) * latest_h
        return self.fuse(fused)

    def forward(self, static_x: torch.Tensor, dynamic_x: torch.Tensor, mask: torch.Tensor):
        embedding = self.encode(static_x, dynamic_x, mask)
        return {target: self.heads[target](embedding) for target in TARGET_ORDER}, embedding


def make_loader(data: Any, labels: dict[str, np.ndarray], row_mask: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    tensors = [
        torch.tensor(data.static[row_mask], dtype=torch.float32),
        torch.tensor(data.dynamic[row_mask], dtype=torch.float32),
        torch.tensor(data.mask[row_mask], dtype=torch.float32),
    ]
    tensors.extend(torch.tensor(labels[target][row_mask], dtype=torch.long) for target in TARGET_ORDER)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, generator=generator)


def predict(model: SharedDecisionGRU, data: Any, labels: dict[str, np.ndarray], row_mask: np.ndarray, device: torch.device):
    model.eval()
    probabilities = {target: [] for target in TARGET_ORDER}
    with torch.no_grad():
        for static_x, dynamic_x, obs_mask, *_ in make_loader(data, labels, row_mask, 2048, False, SEED):
            logits, _ = model(static_x.to(device), dynamic_x.to(device), obs_mask.to(device))
            for target in TARGET_ORDER:
                probabilities[target].append(torch.softmax(logits[target], dim=1).cpu().numpy())
    return {target: np.vstack(chunks) for target, chunks in probabilities.items()}


def metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    valid = y_true >= 0
    y = y_true[valid]
    pred = probability[valid].argmax(axis=1)
    labels = list(range(probability.shape[1]))
    return {
        "n_records": int(valid.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
    }


def train_candidate(config: Candidate, data: Any, labels: dict[str, np.ndarray], masks: dict[str, np.ndarray], device: torch.device):
    seed = SEED + sum(ord(char) for char in config.name)
    set_seed(seed)
    model = SharedDecisionGRU(data.static.shape[1], data.dynamic.shape[2], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    best_state, best_epoch, best_score, waiting = None, 0, -np.inf, 0
    curve: list[dict[str, float | int]] = []
    for epoch in range(1, 101):
        model.train()
        epoch_losses = []
        loader = make_loader(data, labels, masks["train"], config.batch_size, True, seed + epoch)
        for static_x, dynamic_x, obs_mask, y_fsh, y_lh, y_hmg in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(static_x.to(device), dynamic_x.to(device), obs_mask.to(device))
            batch_labels = {"fsh": y_fsh.to(device), "lh": y_lh.to(device), "hmg": y_hmg.to(device)}
            task_losses = []
            for target in TARGET_ORDER:
                valid = batch_labels[target] >= 0
                if valid.any():
                    task_losses.append(criterion(logits[target][valid], batch_labels[target][valid]))
            if not task_losses:
                continue
            loss = torch.stack(task_losses).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        valid_probability = predict(model, data, labels, masks["valid"], device)
        valid_metrics = {target: metrics(labels[target][masks["valid"]], valid_probability[target]) for target in TARGET_ORDER}
        score = float(np.mean([valid_metrics[target]["weighted_f1"] for target in TARGET_ORDER]))
        curve.append({
            "candidate": config.name, "epoch": epoch, "train_loss": float(np.mean(epoch_losses)),
            "mean_validation_weighted_f1": score,
            **{f"{target}_validation_weighted_f1": valid_metrics[target]["weighted_f1"] for target in TARGET_ORDER},
        })
        print(f"[{config.name}] epoch={epoch:03d} loss={np.mean(epoch_losses):.5f} val_mean_wf1={score:.5f}", flush=True)
        if score > best_score + 1e-5:
            best_state = deepcopy(model.state_dict())
            best_epoch, best_score, waiting = epoch, score, 0
        else:
            waiting += 1
            if waiting >= 15:
                break
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {config.name}")
    model.load_state_dict(best_state)
    valid_probability = predict(model, data, labels, masks["valid"], device)
    valid_metrics = {target: metrics(labels[target][masks["valid"]], valid_probability[target]) for target in TARGET_ORDER}
    checkpoint = {
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "config": asdict(config), "seed": seed, "best_epoch": best_epoch,
        "best_validation_mean_weighted_f1": best_score,
        "static_features": STATIC_FEATURES, "dynamic_features": DYNAMIC_FEATURES,
    }
    torch.save(checkpoint, OUT / "checkpoints" / f"{config.name}.pt")
    pd.DataFrame(curve).to_csv(OUT / "training" / f"{config.name}_curve.csv", index=False)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return valid_probability, valid_metrics, best_epoch, best_score


def apply_prior_adjustment(probability: np.ndarray, priors: np.ndarray, alpha: float) -> np.ndarray:
    adjusted = probability * np.power(np.clip(priors, 1e-8, None), alpha)[None, :]
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def select_ensemble(candidate_probabilities: dict[str, dict[str, np.ndarray]], labels: dict[str, np.ndarray], masks: dict[str, np.ndarray]):
    ranked = sorted(
        candidate_probabilities,
        key=lambda name: np.mean([
            metrics(labels[target][masks["valid"]], candidate_probabilities[name][target])["weighted_f1"]
            for target in TARGET_ORDER
        ]),
        reverse=True,
    )
    pool = ranked[:5]
    ensemble_rows = []
    best_combo, best_probs, best_score = None, None, -np.inf
    for size in (1, 2, 3):
        for combo in itertools.combinations(pool, size):
            probability = {
                target: np.mean([candidate_probabilities[name][target] for name in combo], axis=0)
                for target in TARGET_ORDER
            }
            task_metrics = {target: metrics(labels[target][masks["valid"]], probability[target]) for target in TARGET_ORDER}
            score = float(np.mean([task_metrics[target]["weighted_f1"] for target in TARGET_ORDER]))
            ensemble_rows.append({
                "members": "+".join(combo), "n_members": size, "mean_validation_weighted_f1": score,
                **{f"{target}_validation_weighted_f1": task_metrics[target]["weighted_f1"] for target in TARGET_ORDER},
            })
            if score > best_score + 1e-12:
                best_combo, best_probs, best_score = combo, probability, score
    if best_combo is None or best_probs is None:
        raise RuntimeError("No validation ensemble selected")
    pd.DataFrame(ensemble_rows).sort_values("mean_validation_weighted_f1", ascending=False).to_csv(
        OUT / "tables" / "validation_ensemble_selection.csv", index=False
    )
    return list(best_combo), best_probs, best_score


def tune_prior_adjustment(base_probability: dict[str, np.ndarray], labels: dict[str, np.ndarray], masks: dict[str, np.ndarray]):
    selected: dict[str, float] = {}
    rows = []
    train_labels = {target: labels[target][masks["train"]] for target in TARGET_ORDER}
    for target in TARGET_ORDER:
        valid_y = labels[target][masks["valid"]]
        counts = np.bincount(train_labels[target][train_labels[target] >= 0], minlength=len(TARGETS[target]["labels"])).astype(float)
        priors = counts / counts.sum()
        candidates = []
        for alpha in np.round(np.arange(-0.40, 0.401, 0.05), 2):
            probability = apply_prior_adjustment(base_probability[target], priors, float(alpha))
            payload = metrics(valid_y, probability)
            candidates.append((float(alpha), payload))
            rows.append({"target": target.upper(), "alpha": float(alpha), **payload})
        candidates.sort(key=lambda item: (item[1]["weighted_f1"], item[1]["accuracy"], -abs(item[0])), reverse=True)
        selected[target] = candidates[0][0]
    pd.DataFrame(rows).to_csv(OUT / "tables" / "validation_prior_adjustment.csv", index=False)
    return selected


def load_model(checkpoint_path: Path, static_dim: int, dynamic_dim: int, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = Candidate(**checkpoint["config"])
    model = SharedDecisionGRU(static_dim, dynamic_dim, config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def clustered_bootstrap(frame: pd.DataFrame, probability: np.ndarray, target: str) -> dict[str, float]:
    valid = frame["y_true"].to_numpy() >= 0
    work = frame.loc[valid].reset_index(drop=True)
    probs = probability[valid]
    cycle_groups = {cycle: indices.to_numpy() for cycle, indices in work.groupby("cycle_uid").groups.items()}
    cycles = np.array(list(cycle_groups), dtype=object)
    rng = np.random.default_rng(SEED + TARGETS[target]["offset"] + 5000)
    acc_values, wf1_values = [], []
    for _ in range(BOOTSTRAP_N):
        sampled = rng.choice(cycles, size=len(cycles), replace=True)
        indices = np.concatenate([cycle_groups[cycle] for cycle in sampled])
        y = work.loc[indices, "y_true"].to_numpy(int)
        pred = probs[indices].argmax(axis=1)
        acc_values.append(accuracy_score(y, pred))
        wf1_values.append(f1_score(y, pred, labels=list(range(probs.shape[1])), average="weighted", zero_division=0))
    return {
        "accuracy_ci_lower": float(np.quantile(acc_values, 0.025)),
        "accuracy_ci_upper": float(np.quantile(acc_values, 0.975)),
        "weighted_f1_ci_lower": float(np.quantile(wf1_values, 0.025)),
        "weighted_f1_ci_upper": float(np.quantile(wf1_values, 0.975)),
    }


def main() -> None:
    for folder in ("checkpoints", "predictions", "training", "tables", "logs"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    if manifest["split"].value_counts().to_dict() != {"train": 2849, "valid": 611, "test": 611}:
        raise RuntimeError("Frozen split manifest failed 2849/611/611 verification")
    source = pd.read_csv(DATA_PATH, low_memory=False)
    frame, data_audit = build_monitor_frame(source)
    raw, labels, metadata = prepare_master(frame)
    split_map = manifest.assign(cycle_uid=manifest["cycle_uid"].astype(str)).set_index("cycle_uid")["split"]
    row_splits = raw.row_index["cycle_uid"].astype(str).map(split_map)
    if row_splits.isna().any():
        raise RuntimeError("Rows absent from frozen split manifest")
    masks = {split: row_splits.eq(split).to_numpy(bool) for split in ("train", "valid", "test")}
    arrays = phase864.prepare_arrays(raw, fit_mask=masks["train"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidate_probabilities: dict[str, dict[str, np.ndarray]] = {}
    candidate_rows = []
    for config in CANDIDATES:
        probabilities, valid_metrics, best_epoch, best_score = train_candidate(config, arrays, labels, masks, device)
        candidate_probabilities[config.name] = probabilities
        candidate_rows.append({
            "candidate": config.name, "best_epoch": best_epoch,
            "mean_validation_weighted_f1": best_score,
            **{f"{target}_validation_accuracy": valid_metrics[target]["accuracy"] for target in TARGET_ORDER},
            **{f"{target}_validation_weighted_f1": valid_metrics[target]["weighted_f1"] for target in TARGET_ORDER},
        })
    pd.DataFrame(candidate_rows).sort_values("mean_validation_weighted_f1", ascending=False).to_csv(
        OUT / "tables" / "validation_candidate_selection.csv", index=False
    )

    selected_members, validation_ensemble, validation_score = select_ensemble(candidate_probabilities, labels, masks)
    selected_alpha = tune_prior_adjustment(validation_ensemble, labels, masks)
    train_priors = {}
    for target in TARGET_ORDER:
        train_y = labels[target][masks["train"]]
        counts = np.bincount(train_y[train_y >= 0], minlength=len(TARGETS[target]["labels"])).astype(float)
        train_priors[target] = counts / counts.sum()

    # The independent test set is accessed only after architecture, ensemble, and calibration are frozen.
    selected_test_probabilities = []
    for member in selected_members:
        model = load_model(OUT / "checkpoints" / f"{member}.pt", arrays.static.shape[1], arrays.dynamic.shape[2], device)
        selected_test_probabilities.append(predict(model, arrays, labels, masks["test"], device))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    test_probability = {
        target: np.mean([payload[target] for payload in selected_test_probabilities], axis=0)
        for target in TARGET_ORDER
    }
    test_probability = {
        target: apply_prior_adjustment(test_probability[target], train_priors[target], selected_alpha[target])
        for target in TARGET_ORDER
    }

    result_rows = []
    for target in TARGET_ORDER:
        y = labels[target][masks["test"]]
        valid = y >= 0
        probability = test_probability[target]
        test_rows = raw.row_index.loc[masks["test"]].reset_index(drop=True).copy()
        test_rows = pd.concat([test_rows, metadata.loc[masks["test"]].reset_index(drop=True)], axis=1)
        test_rows["target"] = target.upper()
        test_rows["y_true"] = y
        test_rows["y_pred"] = probability.argmax(axis=1)
        for index, label in enumerate(TARGETS[target]["labels"]):
            test_rows[f"prob_{label}"] = probability[:, index]
        test_rows.to_csv(OUT / "predictions" / f"{target}_test_predictions.csv", index=False)
        point = metrics(y, probability)
        ci = clustered_bootstrap(test_rows, probability, target)
        result_rows.append({
            "target": target.upper(), "split": "test", "n_cycles": int(test_rows.loc[valid, "cycle_uid"].nunique()),
            **point, **ci,
        })
    pd.DataFrame(result_rows).to_csv(OUT / "tables" / "final_test_metrics.csv", index=False)
    write_json(OUT / "logs" / "run_manifest.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED, "device": str(device),
        "data_audit": data_audit, "frozen_split": "2849 train / 611 validation / 611 test cycles",
        "target_definition": "information available through true Monitor k predicts the post-monitor-k Gn decision",
        "storage_note": "post-monitor decision uses next_* columns because it becomes current_* at the following record",
        "architecture": "dynamic GRU + static-dynamic GRU + AddGate + optional latest-monitor skip and task adapters",
        "selected_members": selected_members, "validation_ensemble_score_before_prior_adjustment": validation_score,
        "selected_prior_adjustment_alpha": selected_alpha,
        "selection_protocol": "candidate checkpoints, ensemble, and prior adjustment selected exclusively on validation; test evaluated once",
        "features": {"static": STATIC_FEATURES, "dynamic": DYNAMIC_FEATURES, "labels": FEATURE_LABELS},
        "candidates": [asdict(config) for config in CANDIDATES],
        "bootstrap": {"replicates": BOOTSTRAP_N, "unit": "cycle_uid", "interval": "percentile 95%"},
    })
    print("\nSELECTED MEMBERS", selected_members)
    print("SELECTED ALPHA", selected_alpha)
    print(pd.DataFrame(result_rows).to_string(index=False))


if __name__ == "__main__":
    main()
