#!/usr/bin/env python3
"""Optimize 27-feature true-Monitor GRU(AddGate) outcome heads.

Validation selects configurations. The final stage evaluates the frozen test
split once. Existing result directories are read-only and never overwritten.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "scripts/experiment/monitor_level_27feature_current_decision_20261015_run.py"
GN_OUT = ROOT / "outputs/monitor_level_shared_multitask_gru_addgate_27features_current_decision_tuned_20261015"
GN_CHECKPOINT = GN_OUT / "checkpoints/attn_latest_adapter_h128.pt"
OUT = ROOT / "outputs/gru_addgate_outcome27_optimization_20261016"
BASELINE_PATH = ROOT / "data/processed/baseline_cycle_dataset.csv"
LABEL_PATH = ROOT / "data/processed/successful_cycle_labels_v1.csv"
SEEDS = [20261016, 20261037, 20261059, 20261081, 20261103]
MAX_EPOCHS = 80
PATIENCE = 15
BOOTSTRAP_N = 1000


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = import_module("outcome27_base", BASE_SCRIPT)


@dataclass(frozen=True)
class Config:
    name: str
    task: str
    count_loss: str
    ohss_loss: str
    pretrained: bool
    freeze_encoder: bool = False
    balanced_batches: bool = False
    hidden_dim: int = 128
    dropout: float = 0.15
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    ohss_weight: float = 0.50


CONFIGS = [
    Config("random_joint_huber_bce", "both", "huber", "bce", False),
    Config("pretrained_joint_huber_bce", "both", "huber", "bce", True),
    Config("pretrained_joint_logmse_focal", "both", "logmse", "focal", True),
    Config("pretrained_joint_poisson_focal", "both", "poisson", "focal", True),
    Config("pretrained_joint_logmse_rank", "both", "logmse", "rank", True, balanced_batches=True),
    Config("pretrained_oocyte_logmse", "oocyte", "logmse", "none", True),
    Config("pretrained_oocyte_huber", "oocyte", "huber", "none", True),
    Config("pretrained_oocyte_poisson", "oocyte", "poisson", "none", True),
    Config("frozen_ohss_bce", "ohss", "none", "bce", True, freeze_encoder=True, balanced_batches=True),
    Config("pretrained_ohss_bce", "ohss", "none", "bce", True, balanced_batches=True),
    Config("pretrained_ohss_focal", "ohss", "none", "focal", True, balanced_batches=True),
    Config("pretrained_ohss_rank", "ohss", "none", "rank", True, balanced_batches=True),
]


@dataclass
class CycleArrays:
    static: np.ndarray
    dynamic: np.ndarray
    mask: np.ndarray
    oocytes: np.ndarray
    ohss: np.ndarray
    meta: pd.DataFrame


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def strict_ohss(value: Any) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip().lower()
    return int(any(term in text for term in ("moderate", "severe", "中度", "重度", "中重度")))


def prepare_cycle_data() -> tuple[CycleArrays, dict[str, Any]]:
    manifest = pd.read_csv(base.MANIFEST_PATH)[["cycle_uid", "split"]]
    if manifest["split"].value_counts().to_dict() != {"train": 2849, "valid": 611, "test": 611}:
        raise RuntimeError("Frozen cycle split verification failed")
    source = pd.read_csv(base.DATA_PATH, low_memory=False)
    source["monitoring_order"] = pd.to_numeric(source["monitoring_order"], errors="coerce")
    frame = source.loc[source["monitoring_order"].notna() & source["monitoring_order"].ge(1)].copy()
    frame["monitoring_order"] = frame["monitoring_order"].astype(int)
    frame["Day"] = frame["monitoring_order"]
    raw_stim = pd.to_numeric(frame["gn_day"], errors="coerce")
    frame["stim_day"] = raw_stim.where(raw_stim.between(0, 30), np.nan)
    first = frame["monitoring_order"].eq(1)
    frame.loc[first & frame["stim_day"].isna(), "stim_day"] = 0.0
    for feature in base.STATIC_FEATURES + base.DYNAMIC_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    for feature in ("current_fsh_daily_dose", "current_lh_daily_dose", "current_hmg_daily_dose"):
        frame.loc[first & frame[feature].isna(), feature] = 0.0
    frame = frame.sort_values(["cycle_uid", "monitoring_order", "visit_uid"], kind="mergesort").reset_index(drop=True)
    measurements = ["current_e2", "current_lh", "current_p", "current_fsh", "current_endometrium"]
    frame[measurements] = frame.groupby("cycle_uid", sort=False)[measurements].ffill()
    frame["ctfe_truth"] = "cycle_outcome"
    frame["ctfe_truth_id"] = 0
    frame["source_visit_uid"] = frame["visit_uid"].astype(str)
    raw = base.phase864.build_daily_sequence_arrays(
        frame, static_features=base.STATIC_FEATURES, dynamic_features=base.DYNAMIC_FEATURES,
        max_history_days=base.MAX_HISTORY_VISITS,
    )
    split_map = manifest.assign(cycle_uid=manifest["cycle_uid"].astype(str)).set_index("cycle_uid")["split"]
    row_split = raw.row_index["cycle_uid"].astype(str).map(split_map)
    if row_split.isna().any():
        raise RuntimeError("Sequence rows absent from frozen split")
    train_rows = row_split.eq("train").to_numpy(bool)
    prepared = base.phase864.prepare_arrays(raw, fit_mask=train_rows)

    last_visits = frame.groupby("cycle_uid", sort=False).tail(1)[["cycle_uid", "visit_uid", "monitoring_order"]].copy()
    last_lookup = {str(uid): str(visit) for uid, visit in zip(last_visits["cycle_uid"], last_visits["visit_uid"])}
    selected = np.array([
        str(visit) == last_lookup.get(str(cycle), "")
        for visit, cycle in zip(raw.row_index["visit_uid"], raw.row_index["cycle_uid"])
    ], dtype=bool)
    selected_index = np.flatnonzero(selected)
    if len(selected_index) != 4071:
        raise RuntimeError(f"Expected 4,071 final-monitor cycles, found {len(selected_index)}")

    baseline = pd.read_csv(BASELINE_PATH, low_memory=False)[["cycle_uid", "target_oocytes"]]
    labels = pd.read_csv(LABEL_PATH, low_memory=False)[["cycle_uid", "complication_text"]]
    outcomes = baseline.merge(labels, on="cycle_uid", how="left", validate="one_to_one")
    outcomes["cycle_uid"] = outcomes["cycle_uid"].astype(str)
    outcomes["oocytes"] = pd.to_numeric(outcomes["target_oocytes"], errors="coerce")
    outcomes["ohss"] = outcomes["complication_text"].map(strict_ohss).astype(int)
    outcome_lookup = outcomes.set_index("cycle_uid")

    meta = raw.row_index.loc[selected].reset_index(drop=True).copy()
    meta["cycle_uid"] = meta["cycle_uid"].astype(str)
    meta["split"] = meta["cycle_uid"].map(split_map)
    meta["monitoring_order"] = meta["cycle_uid"].map(
        last_visits.assign(cycle_uid=last_visits["cycle_uid"].astype(str)).set_index("cycle_uid")["monitoring_order"]
    ).astype(int)
    oocytes = outcome_lookup.loc[meta["cycle_uid"], "oocytes"].to_numpy(float)
    ohss = outcome_lookup.loc[meta["cycle_uid"], "ohss"].to_numpy(int)
    if not np.isfinite(oocytes).all() or (oocytes < 0).any():
        raise RuntimeError("Invalid oocyte endpoint")

    data = CycleArrays(
        static=prepared.static[selected_index].astype(np.float32),
        dynamic=prepared.dynamic[selected_index].astype(np.float32),
        mask=prepared.mask[selected_index].astype(np.float32),
        oocytes=oocytes.astype(np.float32), ohss=ohss.astype(np.int64), meta=meta,
    )
    audit = {
        "sample_unit": "one final pre-retrieval Monitor sequence per cycle_uid",
        "feature_count": len(base.STATIC_FEATURES) + len(base.DYNAMIC_FEATURES),
        "static_features": base.STATIC_FEATURES, "dynamic_features": base.DYNAMIC_FEATURES,
        "split_counts": meta["split"].value_counts().to_dict(),
        "ohss_positive_counts": meta.assign(ohss=ohss).groupby("split")["ohss"].sum().astype(int).to_dict(),
        "monitoring_order_quantiles": meta["monitoring_order"].quantile([0.5, 0.9, 0.95, 0.99, 1.0]).to_dict(),
        "prediction_time": "last recorded pre-retrieval monitoring state",
        "ohss_endpoint": "documented moderate-to-severe OHSS",
        "oocyte_mean": float(oocytes.mean()), "oocyte_sd": float(oocytes.std()),
        "preprocessing": "median imputation and standardization fitted on training sequence rows only",
    }
    return data, audit


class OutcomeGRU(nn.Module):
    def __init__(self, static_dim: int, dynamic_dim: int, config: Config):
        super().__init__()
        hidden = config.hidden_dim
        self.config = config
        self.dynamic_encoder = base.TemporalEncoder(dynamic_dim, hidden, "attention")
        self.cross_encoder = base.TemporalEncoder(static_dim + dynamic_dim, hidden, "attention")
        self.cross_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.latest_projection = nn.Sequential(nn.Linear(dynamic_dim, hidden), nn.GELU())
        self.latest_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.fuse = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(config.dropout))
        self.oocyte_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(64, 1)
        )
        self.ohss_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(64, 1)
        )

    def encoder_parameters(self):
        modules = [self.dynamic_encoder, self.cross_encoder, self.cross_gate, self.latest_projection, self.latest_gate, self.fuse]
        return [parameter for module in modules for parameter in module.parameters()]

    def encode(self, static_x, dynamic_x, mask):
        static_sequence = static_x.unsqueeze(1).expand(-1, dynamic_x.shape[1], -1)
        dynamic_h = self.dynamic_encoder(dynamic_x, mask)
        cross_h = self.cross_encoder(torch.cat([static_sequence, dynamic_x], dim=-1), mask)
        fused = dynamic_h + self.cross_gate(torch.cat([dynamic_h, cross_h], dim=1)) * cross_h
        latest = self.latest_projection(base.gather_last(dynamic_x, mask))
        fused = fused + self.latest_gate(torch.cat([fused, latest], dim=1)) * latest
        return self.fuse(fused)

    def forward(self, static_x, dynamic_x, mask):
        h = self.encode(static_x, dynamic_x, mask)
        raw_count = self.oocyte_head(h).squeeze(1)
        logit = self.ohss_head(h).squeeze(1)
        if self.config.count_loss in ("huber", "poisson"):
            count = F.softplus(raw_count)
        else:
            count = torch.expm1(raw_count.clamp(min=0.0, max=math.log1p(60.0)))
        return count, raw_count, logit


def initialize_pretrained(model: OutcomeGRU) -> dict[str, Any]:
    if not GN_CHECKPOINT.exists():
        raise FileNotFoundError(GN_CHECKPOINT)
    checkpoint = torch.load(GN_CHECKPOINT, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    own = model.state_dict()
    prefixes = ("dynamic_encoder.", "cross_encoder.", "cross_gate.", "latest_projection.", "latest_gate.", "fuse.")
    compatible = {name: value for name, value in state.items() if name.startswith(prefixes) and name in own and own[name].shape == value.shape}
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "source": str(GN_CHECKPOINT), "loaded_tensor_count": len(compatible),
        "missing_after_partial_load": list(missing), "unexpected": list(unexpected),
    }


def masks(data: CycleArrays) -> dict[str, np.ndarray]:
    return {split: data.meta["split"].eq(split).to_numpy(bool) for split in ("train", "valid", "test")}


def make_loader(data, select, batch_size, shuffle, seed, balanced=False):
    index = np.flatnonzero(select)
    dataset = TensorDataset(
        torch.tensor(data.static[index], dtype=torch.float32),
        torch.tensor(data.dynamic[index], dtype=torch.float32),
        torch.tensor(data.mask[index], dtype=torch.float32),
        torch.tensor(data.oocytes[index], dtype=torch.float32),
        torch.tensor(data.ohss[index], dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    if balanced:
        y = data.ohss[index]
        counts = np.bincount(y, minlength=2).astype(float)
        weights = torch.tensor(1.0 / np.maximum(counts[y], 1.0), dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(index), replacement=True, generator=generator)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def predict(model, data, select, device):
    model.eval()
    counts, probabilities = [], []
    with torch.no_grad():
        for static_x, dynamic_x, obs_mask, *_ in make_loader(data, select, 1024, False, 1):
            count, _, logit = model(static_x.to(device), dynamic_x.to(device), obs_mask.to(device))
            counts.append(count.cpu().numpy())
            probabilities.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(counts), np.concatenate(probabilities)


def regression_metrics(y, pred):
    mae = float(mean_absolute_error(y, pred))
    mean_abs = float(np.mean(np.abs(y)))
    return {
        "r2": float(r2_score(y, pred)), "mae": mae,
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
        "relative_mae": float(mae / mean_abs) if mean_abs > 0 else float("nan"),
        "relative_mae_percent": float(100.0 * mae / mean_abs) if mean_abs > 0 else float("nan"),
        "bias": float(np.mean(pred - y)),
    }


def focal_loss(logit, target, alpha=0.90, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    probability = torch.sigmoid(logit)
    p_t = target * probability + (1.0 - target) * (1.0 - probability)
    alpha_t = target * alpha + (1.0 - target) * (1.0 - alpha)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


def ohss_loss(logit, target, kind, pos_weight):
    bce = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight)
    if kind == "bce":
        return bce
    if kind == "focal":
        return focal_loss(logit, target)
    if kind == "rank":
        pos = logit[target > 0.5]
        neg = logit[target <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            return bce
        ranking = F.softplus(-(pos[:, None] - neg[None, :])).mean()
        return 0.25 * bce + 0.75 * ranking
    raise ValueError(kind)


def count_loss(count, raw_count, target, kind, target_sd):
    if kind == "huber":
        return F.smooth_l1_loss(count, target, beta=2.0) / max(target_sd, 1.0)
    if kind == "poisson":
        return F.poisson_nll_loss(count.clamp_min(1e-5), target, log_input=False, full=False) / max(float(target.mean()), 1.0)
    if kind == "logmse":
        return F.mse_loss(raw_count, torch.log1p(target))
    raise ValueError(kind)


def train_one(config, seed, data, data_masks, device):
    set_seed(seed)
    model = OutcomeGRU(data.static.shape[1], data.dynamic.shape[2], config).to(device)
    pretraining = initialize_pretrained(model) if config.pretrained else {"source": None, "loaded_tensor_count": 0}
    if config.freeze_encoder:
        for parameter in model.encoder_parameters():
            parameter.requires_grad = False
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    train_y = data.ohss[data_masks["train"]]
    n_pos = int(train_y.sum())
    n_neg = int(len(train_y) - n_pos)
    pos_weight = torch.tensor(math.sqrt(n_neg / max(n_pos, 1)), dtype=torch.float32, device=device)
    target_sd = float(data.oocytes[data_masks["train"]].std())
    best = {"oocyte": {"value": np.inf, "epoch": 0, "state": None, "wait": 0}, "ohss": {"value": -np.inf, "epoch": 0, "state": None, "wait": 0}}
    curve = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        loader = make_loader(
            data, data_masks["train"], config.batch_size, True, seed + epoch,
            balanced=config.balanced_batches,
        )
        for static_x, dynamic_x, obs_mask, y_count, y_ohss in loader:
            static_x, dynamic_x, obs_mask = static_x.to(device), dynamic_x.to(device), obs_mask.to(device)
            y_count, y_ohss = y_count.to(device), y_ohss.to(device)
            optimizer.zero_grad(set_to_none=True)
            count, raw_count, logit = model(static_x, dynamic_x, obs_mask)
            terms = []
            if config.task in ("both", "oocyte"):
                terms.append(count_loss(count, raw_count, y_count, config.count_loss, target_sd))
            if config.task in ("both", "ohss"):
                weight = config.ohss_weight if config.task == "both" else 1.0
                terms.append(weight * ohss_loss(logit, y_ohss, config.ohss_loss, pos_weight))
            loss = torch.stack(terms).sum()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 3.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        valid_count, valid_prob = predict(model, data, data_masks["valid"], device)
        y_count = data.oocytes[data_masks["valid"]]
        y_ohss = data.ohss[data_masks["valid"]]
        reg = regression_metrics(y_count, valid_count)
        auc = float(roc_auc_score(y_ohss, valid_prob))
        ap = float(average_precision_score(y_ohss, valid_prob))
        if config.task in ("both", "oocyte"):
            if reg["rmse"] < best["oocyte"]["value"] - 1e-5:
                best["oocyte"] = {"value": reg["rmse"], "epoch": epoch, "state": deepcopy(model.state_dict()), "wait": 0}
            else:
                best["oocyte"]["wait"] += 1
        if config.task in ("both", "ohss"):
            if auc > best["ohss"]["value"] + 1e-5:
                best["ohss"] = {"value": auc, "epoch": epoch, "state": deepcopy(model.state_dict()), "wait": 0}
            else:
                best["ohss"]["wait"] += 1
        curve.append({
            "candidate": config.name, "seed": seed, "epoch": epoch, "train_loss": float(np.mean(losses)),
            "validation_r2": reg["r2"], "validation_mae": reg["mae"], "validation_rmse": reg["rmse"],
            "validation_relative_mae": reg["relative_mae"], "validation_roc_auc": auc,
            "validation_average_precision": ap,
        })
        active_waits = []
        if config.task in ("both", "oocyte"):
            active_waits.append(best["oocyte"]["wait"])
        if config.task in ("both", "ohss"):
            active_waits.append(best["ohss"]["wait"])
        if epoch == 1 or epoch % 10 == 0:
            print(f"[{config.name} seed={seed}] epoch={epoch:03d} rmse={reg['rmse']:.4f} r2={reg['r2']:.4f} auc={auc:.4f} ap={ap:.4f}", flush=True)
        if active_waits and all(wait >= PATIENCE for wait in active_waits):
            break

    (OUT / "training").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curve).to_csv(OUT / "training" / f"{config.name}_seed{seed}_curve.csv", index=False)
    result = {"config": config, "seed": seed, "pretraining": pretraining}
    for task in ("oocyte", "ohss"):
        if best[task]["state"] is None:
            continue
        model.load_state_dict(best[task]["state"])
        valid_count, valid_prob = predict(model, data, data_masks["valid"], device)
        reg = regression_metrics(data.oocytes[data_masks["valid"]], valid_count)
        clf = {
            "roc_auc": float(roc_auc_score(data.ohss[data_masks["valid"]], valid_prob)),
            "average_precision": float(average_precision_score(data.ohss[data_masks["valid"]], valid_prob)),
        }
        checkpoint_name = f"{config.name}_seed{seed}_best_{task}.pt"
        checkpoint_path = OUT / "checkpoints" / checkpoint_name
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": {name: value.detach().cpu() for name, value in best[task]["state"].items()},
            "config": asdict(config), "seed": seed, "selected_task": task,
            "best_epoch": best[task]["epoch"], "pretraining": pretraining,
            "static_features": base.STATIC_FEATURES, "dynamic_features": base.DYNAMIC_FEATURES,
        }, checkpoint_path)
        result[task] = {
            "checkpoint": str(checkpoint_path), "best_epoch": best[task]["epoch"],
            "valid_count": valid_count, "valid_prob": valid_prob, "regression": reg, "classification": clf,
        }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def load_model(path, data, device):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    config = Config(**bundle["config"])
    model = OutcomeGRU(data.static.shape[1], data.dynamic.shape[2], config).to(device)
    model.load_state_dict(bundle["state_dict"])
    return model


def validation_stage():
    for folder in ("audit", "checkpoints", "training", "predictions", "tables", "logs"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    data, audit = prepare_cycle_data()
    data_masks = masks(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_json(OUT / "audit" / "data_contract.json", audit)
    rows, first_runs = [], {}
    for config in CONFIGS:
        run = train_one(config, SEEDS[0], data, data_masks, device)
        first_runs[config.name] = run
        row = {"candidate": config.name, "task": config.task, "seed": SEEDS[0]}
        if "oocyte" in run:
            row.update({f"oocyte_{key}": value for key, value in run["oocyte"]["regression"].items()})
            row["oocyte_best_epoch"] = run["oocyte"]["best_epoch"]
        if "ohss" in run:
            row.update({f"ohss_{key}": value for key, value in run["ohss"]["classification"].items()})
            row["ohss_best_epoch"] = run["ohss"]["best_epoch"]
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "tables" / "validation_candidate_screen.csv", index=False)

    oocyte_candidates = table.loc[table["oocyte_rmse"].notna()].sort_values(
        ["oocyte_rmse", "oocyte_mae", "oocyte_r2"], ascending=[True, True, False]
    )
    ohss_candidates = table.loc[table["ohss_roc_auc"].notna()].sort_values(
        ["ohss_roc_auc", "ohss_average_precision"], ascending=[False, False]
    )
    selected_names = {"oocyte": str(oocyte_candidates.iloc[0]["candidate"]), "ohss": str(ohss_candidates.iloc[0]["candidate"])}
    multiseed_runs = {task: [first_runs[name]] for task, name in selected_names.items()}
    trained_cache = {}
    for task, name in selected_names.items():
        config = next(item for item in CONFIGS if item.name == name)
        for seed in SEEDS[1:]:
            key = (name, seed)
            if key not in trained_cache:
                trained_cache[key] = train_one(config, seed, data, data_masks, device)
            multiseed_runs[task].append(trained_cache[key])

    final_validation = []
    selection = {}
    for task in ("oocyte", "ohss"):
        task_runs = multiseed_runs[task]
        if task == "oocyte":
            predictions = np.mean([run["oocyte"]["valid_count"] for run in task_runs], axis=0)
            payload = regression_metrics(data.oocytes[data_masks["valid"]], predictions)
            final_validation.append({"target": "Oocytes retrieved", "configuration": selected_names[task], **payload})
            selection[task] = {
                "configuration": selected_names[task],
                "checkpoints": [run["oocyte"]["checkpoint"] for run in task_runs],
                "validation_metrics": payload,
            }
        else:
            probabilities = np.mean([run["ohss"]["valid_prob"] for run in task_runs], axis=0)
            payload = {
                "roc_auc": float(roc_auc_score(data.ohss[data_masks["valid"]], probabilities)),
                "average_precision": float(average_precision_score(data.ohss[data_masks["valid"]], probabilities)),
            }
            final_validation.append({"target": "Moderate-to-severe OHSS", "configuration": selected_names[task], **payload})
            selection[task] = {
                "configuration": selected_names[task],
                "checkpoints": [run["ohss"]["checkpoint"] for run in task_runs],
                "validation_metrics": payload,
            }
    pd.DataFrame(final_validation).to_csv(OUT / "tables" / "validation_selected_ensembles.csv", index=False)
    write_json(OUT / "logs" / "frozen_validation_selection.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_accessed_in_this_stage": False,
        "selection_rules": {"oocyte": "minimum validation RMSE", "ohss": "maximum validation ROC-AUC"},
        "seeds": SEEDS, "selection": selection, "candidate_configs": [asdict(config) for config in CONFIGS],
        "stop_rule": "finite prespecified candidate screen followed by five-seed ensemble; no test-driven continuation",
    })
    print("\nVALIDATION CANDIDATES")
    print(table.to_string(index=False))
    print("\nFROZEN SELECTION")
    print(pd.DataFrame(final_validation).to_string(index=False))


def bootstrap(y_count, pred_count, y_ohss, probability):
    rng = np.random.default_rng(SEEDS[0])
    rows = []
    n = len(y_count)
    for iteration in range(BOOTSTRAP_N):
        index = rng.integers(0, n, size=n)
        reg = regression_metrics(y_count[index], pred_count[index])
        if len(np.unique(y_ohss[index])) < 2:
            continue
        rows.append({
            "iteration": iteration + 1, **reg,
            "roc_auc": float(roc_auc_score(y_ohss[index], probability[index])),
        })
    return pd.DataFrame(rows)


def final_stage():
    selection = json.loads((OUT / "logs" / "frozen_validation_selection.json").read_text(encoding="utf-8"))["selection"]
    data, audit = prepare_cycle_data()
    data_masks = masks(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count_predictions = []
    for path in selection["oocyte"]["checkpoints"]:
        model = load_model(path, data, device)
        count, _ = predict(model, data, data_masks["test"], device)
        count_predictions.append(count)
        del model
    ohss_predictions = []
    for path in selection["ohss"]["checkpoints"]:
        model = load_model(path, data, device)
        _, probability = predict(model, data, data_masks["test"], device)
        ohss_predictions.append(probability)
        del model
    count_prediction = np.mean(count_predictions, axis=0)
    ohss_probability = np.mean(ohss_predictions, axis=0)
    y_count = data.oocytes[data_masks["test"]]
    y_ohss = data.ohss[data_masks["test"]]
    reg = regression_metrics(y_count, count_prediction)
    clf = {
        "roc_auc": float(roc_auc_score(y_ohss, ohss_probability)),
        "average_precision": float(average_precision_score(y_ohss, ohss_probability)),
    }
    test_rows = data.meta.loc[data_masks["test"]].reset_index(drop=True).copy()
    test_rows["oocytes"] = y_count
    test_rows["predicted_oocytes"] = count_prediction
    test_rows["ohss_mod_sev"] = y_ohss
    test_rows["predicted_ohss_probability"] = ohss_probability
    test_rows.to_csv(OUT / "predictions" / "test_predictions.csv", index=False)
    boot = bootstrap(y_count, count_prediction, y_ohss, ohss_probability)
    boot.to_csv(OUT / "tables" / "test_cycle_bootstrap_1000.csv", index=False)
    ci_rows = []
    for metric in ("r2", "mae", "rmse", "relative_mae", "relative_mae_percent", "roc_auc"):
        values = boot[metric].dropna()
        ci_rows.append({
            "metric": metric, "lower_95ci": float(values.quantile(0.025)),
            "upper_95ci": float(values.quantile(0.975)), "bootstrap_replicates": int(len(values)),
            "bootstrap_unit": "cycle_uid",
        })
    pd.DataFrame(ci_rows).to_csv(OUT / "tables" / "test_metrics_95ci.csv", index=False)
    result = {
        "sample": {"n_cycles": int(data_masks["test"].sum()), "ohss_positive": int(y_ohss.sum())},
        "oocyte_configuration": selection["oocyte"]["configuration"],
        "ohss_configuration": selection["ohss"]["configuration"],
        "oocyte": reg, "ohss": clf,
        "metric_note": "RMSE is the conventional root mean squared error; relative_mae is MAE divided by mean observed oocyte count.",
    }
    write_json(OUT / "tables" / "final_test_metrics.json", result)
    pd.DataFrame([
        {
            "Outcome": "Oocytes retrieved", "Configuration": selection["oocyte"]["configuration"],
            "R2": reg["r2"], "MAE": reg["mae"], "RMSE": reg["rmse"],
            "Relative MAE": reg["relative_mae"], "Relative MAE (%)": reg["relative_mae_percent"], "ROC-AUC": np.nan,
        },
        {
            "Outcome": "Moderate-to-severe OHSS", "Configuration": selection["ohss"]["configuration"],
            "R2": np.nan, "MAE": np.nan, "RMSE": np.nan, "Relative MAE": np.nan,
            "Relative MAE (%)": np.nan, "ROC-AUC": clf["roc_auc"],
        },
    ]).to_csv(OUT / "tables" / "final_test_summary.csv", index=False)
    write_json(OUT / "logs" / "final_evaluation_manifest.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_source": "frozen_validation_selection.json", "test_used_for_selection_in_this_experiment": False,
        "known_caveat": "The same fixed test cycles were inspected in earlier project experiments.",
        "bootstrap": {"unit": "cycle_uid", "replicates": BOOTSTRAP_N, "interval": "percentile 95%"},
    })
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("validation", "final"))
    args = parser.parse_args()
    if args.stage == "validation":
        validation_stage()
    else:
        final_stage()


if __name__ == "__main__":
    main()
