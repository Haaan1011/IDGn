#!/usr/bin/env python3
"""Compute numerical SHAP GradientExplainer values for frozen 27- or 62-feature GRU models.

No data split, checkpoint, model parameter, or prediction is changed.  SHAP is
computed on the frozen validation records, stratified by observed history
length, so the fixed sequence mask supplied to the GRU is valid for every
background/evaluated pair in an explainer call.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
MODEL_OUT = ROOT / "outputs/monitor_level_gru_addgate_27features_next_monitor_20261015"
RUNNER_PATH = ROOT / "scripts/experiment/monitor_level_27feature_next_monitor_20261015/run_monitor_level_gru_addgate_27features.py"
TARGETS = ("fsh", "lh", "hmg")
SEED = 20261016
BACKGROUND_MAX = 64
NSAMPLES = 64
EXPLAIN_BATCH_SIZE = 128

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full", action="store_true", help="Explain the 62-feature neural components.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stratified_background(indices: np.ndarray, y: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministically sample a class-proportional SHAP background."""
    if len(indices) <= maximum:
        return np.sort(indices)
    groups = {int(label): indices[y[indices] == label] for label in np.unique(y[indices])}
    sizes = {label: len(values) for label, values in groups.items()}
    quotas = {label: max(1, int(round(maximum * size / len(indices)))) for label, size in sizes.items()}
    while sum(quotas.values()) > maximum:
        label = max(quotas, key=lambda item: (quotas[item], sizes[item]))
        if quotas[label] > 1:
            quotas[label] -= 1
        else:
            break
    while sum(quotas.values()) < maximum:
        label = max(quotas, key=lambda item: (sizes[item] - quotas[item], sizes[item]))
        if quotas[label] < sizes[label]:
            quotas[label] += 1
        else:
            break
    selected = [rng.choice(groups[label], size=min(quotas[label], sizes[label]), replace=False) for label in sorted(groups)]
    return np.sort(np.concatenate(selected))


class FixedLengthLogitModel(nn.Module):
    """Expose logits with a fixed, valid GRU mask for one observed history length."""
    def __init__(self, model: nn.Module, history_length: int, max_history: int, task: str):
        super().__init__()
        self.model = model
        self.task = task
        self.register_buffer(
            "fixed_mask",
            torch.cat([torch.ones(history_length), torch.zeros(max_history - history_length)]).float(),
        )

    def forward(self, static_x: torch.Tensor, dynamic_x: torch.Tensor) -> torch.Tensor:
        mask = self.fixed_mask.unsqueeze(0).expand(static_x.shape[0], -1)
        logits, _ = self.model(static_x, dynamic_x, mask)
        return logits[self.task] if isinstance(logits, dict) else logits


def make_detail(
    target: str,
    index: np.ndarray,
    raw_index: pd.DataFrame,
    static_values: np.ndarray,
    dynamic_values: np.ndarray,
    predicted_class: np.ndarray,
    feature_labels: dict[str, str],
    static_features: list[str],
    dynamic_features: list[str],
) -> pd.DataFrame:
    sample = raw_index.iloc[index].reset_index(drop=True)
    identity = pd.DataFrame({
        "target": target.upper(),
        "sample_id": [f"{target.upper()}_{int(i)}" for i in index],
        "cycle_uid": sample["cycle_uid"].astype(str).to_numpy(),
        "visit_uid": sample["visit_uid"].astype(str).to_numpy(),
        "predicted_class": predicted_class.astype(int),
    })
    blocks = []
    for source, features, values in (("static", static_features, static_values), ("dynamic", dynamic_features, dynamic_values)):
        n, f = values.shape
        block = identity.loc[identity.index.repeat(f)].reset_index(drop=True)
        block["feature"] = np.tile(features, n)
        block["feature_label"] = block["feature"].map(feature_labels)
        block["source"] = source
        block["shap_value"] = values.reshape(-1)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def mean_abs_summary(detail: pd.DataFrame) -> pd.DataFrame:
    summary = detail.groupby(["target", "feature", "feature_label", "source"], as_index=False).agg(
        mean_abs_shap_value=("shap_value", lambda x: float(np.mean(np.abs(x)))),
        mean_shap_value=("shap_value", "mean"),
        n_validation_records=("sample_id", "nunique"),
    )
    return summary.sort_values(["target", "mean_abs_shap_value"], ascending=[True, False], kind="mergesort")


def main() -> None:
    args = parse_args()
    for directory in (args.output_dir / "tables", args.output_dir / "qa"):
        directory.mkdir(parents=True, exist_ok=True)
    import shap
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    runner = import_module("monitor27_runner_actual_shap", RUNNER_PATH)
    source = pd.read_csv(runner.DATA_PATH, low_memory=False)
    manifest = pd.read_csv(runner.MANIFEST_PATH)
    split_map = manifest.set_index("cycle_uid")["split"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_details, all_backgrounds, target_manifest = [], [], []
    if args.full:
        sys.path.insert(0, str(ROOT / "scripts/experiment"))
        import train_paper_gn as full

        full.base.STATIC_FEATURES = full.STATIC_FEATURES
        full.base.DYNAMIC_FEATURES = full.DYNAMIC_FEATURES
        frame, _ = full.base.build_monitor_frame(source)
        full_raw, full_labels, _ = full.base.prepare_master(frame)
        full_split = full_raw.row_index["cycle_uid"].astype(str).map(split_map)
        full_data = full.base.phase864.prepare_arrays(full_raw, fit_mask=full_split.eq("train").to_numpy())
        runner.STATIC_FEATURES = full.STATIC_FEATURES
        runner.DYNAMIC_FEATURES = full.DYNAMIC_FEATURES
        for feature in runner.STATIC_FEATURES + runner.DYNAMIC_FEATURES:
            runner.FEATURE_LABELS.setdefault(feature, feature.replace("_", " ").capitalize())

    for target_number, target in enumerate(TARGETS):
        spec = runner.TARGETS[target]
        labels = list(spec["labels"])
        runner.phase866.LABELS = labels
        runner.phase866.NUM_CLASSES = len(labels)
        if args.full:
            raw, data, row_split = full_raw, full_data, full_split
            data.y = full_labels[target]
            member = full.FROZEN_SELECTION[target]["neural"]
            checkpoint_path = ROOT / "outputs/gn_full62/checkpoints" / f"{member}.pt"
            model = full.base.load_model(checkpoint_path, data.static.shape[1], data.dynamic.shape[2], device)
        else:
            frame, _ = runner.build_monitor_frame(source)
            raw, _ = runner.build_raw(frame, target)
            row_split = raw.row_index["cycle_uid"].astype(str).map(split_map)
            data = runner.phase864.prepare_arrays(raw, fit_mask=row_split.eq("train").to_numpy(bool))
            checkpoint_path = MODEL_OUT / "checkpoints" / f"gru_addgate_{target}_monitor27.pt"
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model = runner.phase866.FusionGRUClassifier(
                data.static.shape[1], data.dynamic.shape[2], 96, len(labels), 0.15, mode="addgate"
            ).to(device)
            model.load_state_dict(checkpoint["state_dict"])
        train_mask = row_split.eq("train").to_numpy(bool) & (data.y >= 0)
        valid_mask = row_split.eq("valid").to_numpy(bool) & (data.y >= 0)
        if row_split.isna().any() or not valid_mask.any():
            raise RuntimeError(f"{target}: frozen split assignment failed")
        model.eval()

        history_length = data.mask.sum(axis=1).astype(int)
        valid_indices = np.flatnonzero(valid_mask)
        static_parts, dynamic_parts, index_parts, predicted_parts = [], [], [], []
        target_background_rows = []
        for length in sorted(np.unique(history_length[valid_mask])):
            train_indices = np.flatnonzero(train_mask & (history_length == length))
            eval_indices = np.flatnonzero(valid_mask & (history_length == length))
            if len(train_indices) == 0 or len(eval_indices) == 0:
                raise RuntimeError(f"{target}: no background/evaluation rows for history length {length}")
            rng = np.random.default_rng(SEED + 1000 * target_number + 10 * int(length))
            background_indices = stratified_background(train_indices, data.y, BACKGROUND_MAX, rng)
            background = [
                torch.tensor(data.static[background_indices], dtype=torch.float32, device=device),
                torch.tensor(data.dynamic[background_indices], dtype=torch.float32, device=device),
            ]
            wrapper = FixedLengthLogitModel(model, int(length), data.dynamic.shape[1], target).to(device)
            wrapper.eval()
            with torch.backends.cudnn.flags(enabled=False):
                explainer = shap.GradientExplainer(wrapper, background, batch_size=2048)
                for start in range(0, len(eval_indices), EXPLAIN_BATCH_SIZE):
                    batch_indices = eval_indices[start:start + EXPLAIN_BATCH_SIZE]
                    values_input = [
                        torch.tensor(data.static[batch_indices], dtype=torch.float32, device=device),
                        torch.tensor(data.dynamic[batch_indices], dtype=torch.float32, device=device),
                    ]
                    with torch.no_grad():
                        prediction = wrapper(*values_input).argmax(dim=1).detach().cpu().numpy()
                    shap_values = explainer.shap_values(
                        values_input, nsamples=NSAMPLES,
                        rseed=SEED + 100000 * target_number + 1000 * int(length) + start,
                    )
                    static_class_index = np.broadcast_to(
                        prediction[:, None, None],
                        (len(prediction), shap_values[0].shape[1], 1),
                    )
                    dynamic_class_index = np.broadcast_to(
                        prediction[:, None, None, None],
                        (*shap_values[1].shape[:3], 1),
                    )
                    static_selected = np.take_along_axis(shap_values[0], static_class_index, axis=2).squeeze(2)
                    dynamic_selected = np.take_along_axis(shap_values[1], dynamic_class_index, axis=3).squeeze(3).sum(axis=1)
                    static_parts.append(static_selected)
                    dynamic_parts.append(dynamic_selected)
                    index_parts.append(batch_indices)
                    predicted_parts.append(prediction)
            records = raw.row_index.iloc[background_indices].copy()
            records["target"] = target.upper(); records["history_length"] = int(length)
            records["background_rank"] = np.arange(1, len(records) + 1)
            target_background_rows.append(records)
            print(f"[{target.upper()}] history={length} background={len(background_indices)} validation={len(eval_indices)}", flush=True)

        index_array = np.concatenate(index_parts)
        order = np.argsort(index_array)
        detail = make_detail(
            target, index_array[order], raw.row_index,
            np.vstack(static_parts)[order], np.vstack(dynamic_parts)[order], np.concatenate(predicted_parts)[order],
            runner.FEATURE_LABELS, runner.STATIC_FEATURES, runner.DYNAMIC_FEATURES,
        )
        if detail["sample_id"].nunique() != int(valid_mask.sum()) or detail["feature"].nunique() != (62 if args.full else 27):
            raise AssertionError(f"{target}: incomplete SHAP detail output")
        detail.to_csv(args.output_dir / "tables" / f"{target}_validation_shap_detail.csv", index=False)
        all_details.append(detail)
        all_backgrounds.append(pd.concat(target_background_rows, ignore_index=True))
        target_manifest.append({
            "target": target.upper(), "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
            "validation_records": int(valid_mask.sum()), "training_records": int(train_mask.sum()),
            "history_lengths": sorted(int(x) for x in np.unique(history_length[valid_mask])),
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    detail = pd.concat(all_details, ignore_index=True)
    background = pd.concat(all_backgrounds, ignore_index=True)
    summary = mean_abs_summary(detail)
    detail.to_csv(args.output_dir / "tables" / "validation_shap_detail.csv", index=False)
    summary.to_csv(args.output_dir / "tables" / "validation_mean_abs_shap_summary.csv", index=False)
    background.to_csv(args.output_dir / "tables" / "shap_background_records.csv", index=False)
    qa = {
        "method": "shap.GradientExplainer (Expected Gradients)", "shap_version": shap.__version__,
        "model_output_explained": "logit for each record's predicted dose class",
        "dynamic_aggregation": "Signed SHAP values were summed over observed history visits before absolute aggregation by feature.",
        "background": "Class-proportional frozen-training records sampled separately within each observed history length.",
        "background_max_per_target_history_length": BACKGROUND_MAX, "nsamples": NSAMPLES,
        "validation_records_per_target": detail.groupby("target")["sample_id"].nunique().astype(int).to_dict(),
        "features_per_target": summary.groupby("target")["feature"].nunique().astype(int).to_dict(),
        "non_finite_shap_values": int((~np.isfinite(detail["shap_value"])).sum()),
        "seed": SEED,
    }
    manifest_payload = {
        "model_contract": f"Frozen {62 if args.full else 27}-feature neural GRU(AddGate) checkpoints",
        "source_dataset": str(runner.DATA_PATH), "source_dataset_sha256": sha256(runner.DATA_PATH),
        "split_manifest": str(runner.MANIFEST_PATH), "split_manifest_sha256": sha256(runner.MANIFEST_PATH),
        "split_used_for_shap": "validation", "targets": target_manifest, "seed": SEED,
        "device": str(device), "torch_version": torch.__version__, "shap_version": shap.__version__,
    }
    (args.output_dir / "qa" / "shap_method.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "qa" / "run_manifest.json").write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    main()
