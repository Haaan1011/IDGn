#!/usr/bin/env python3
"""Train holdout-only UI-reduced GRU(AddGate) models and audit final test metrics."""
from __future__ import annotations

import argparse
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/paper_audit_20260715"
MANIFEST = ROOT / "data/splits/split_manifest_v1.csv"
CONTRACT = (
    ROOT
    / "results/result1_deep_temporal_gn_dose_phase870_gru_addgate_ui_reduced/logs/ui_reduced_feature_contract.json"
)
MONITORING = ROOT / "data/processed/layer1_strategy_dataset.csv"
BASELINE = ROOT / "data/processed/baseline_cycle_dataset.csv"
SEED = 20260715
BOOTSTRAP_N = 1000

FSH_LABELS = ["stop", "low_dose", "medium_low", "medium_high", "high_dose"]
LH_LABELS = ["dose_0", "dose_75", "dose_ge150"]
HMG_LABELS = ["dose_0", "dose_75", "dose_150", "dose_ge225"]
TARGETS = {
    "fsh": {
        "labels": FSH_LABELS,
        "label_col": "daily_fsh_dose_bin",
        "previous_feature": "previous_fsh_daily_dose",
    },
    "lh": {
        "labels": LH_LABELS,
        "label_col": "daily_lh_dose_bin_grouped",
        "previous_feature": "previous_lh_daily_dose",
    },
    "hmg": {
        "labels": HMG_LABELS,
        "label_col": "daily_hmg_dose_bin_grouped",
        "previous_feature": "previous_hmg_daily_dose",
    },
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
    "phase866_holdout_base",
    ROOT
    / "scripts/experiment/phase866_daily_expanded_model_comparison/run_phase866_daily_expanded_model_comparison.py",
)
phase864 = phase866.phase864
phase866_final = import_module(
    "phase866_final_holdout_base",
    ROOT
    / "scripts/experiment/phase866_daily_expanded_model_comparison/run_gru_addgate_fsh_lh3_hmg4.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(data: Any, mask: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(data.static[mask], dtype=torch.float32),
        torch.tensor(data.dynamic[mask], dtype=torch.float32),
        torch.tensor(data.mask[mask], dtype=torch.float32),
        torch.tensor(data.y[mask], dtype=torch.long),
    )
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def predict(model: nn.Module, data: Any, mask: np.ndarray, device: torch.device):
    model.eval()
    probabilities, embeddings = [], []
    with torch.no_grad():
        for static_x, dynamic_x, obs_mask, _ in loader(data, mask, 2048, False):
            logits, embedding = model(static_x.to(device), dynamic_x.to(device), obs_mask.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            embeddings.append(embedding.cpu().numpy())
    return np.vstack(probabilities), np.vstack(embeddings)


def multiclass_metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray, n_classes: int):
    result = {
        "n_records": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=list(range(n_classes)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=list(range(n_classes)), average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y, pred, labels=list(range(n_classes)), average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, labels=list(range(n_classes)), average="macro", zero_division=0)),
    }
    y_onehot = label_binarize(y, classes=np.arange(n_classes))
    try:
        result["roc_auc_ovr_macro"] = float(
            roc_auc_score(y, prob, labels=np.arange(n_classes), multi_class="ovr", average="macro")
        )
    except ValueError:
        result["roc_auc_ovr_macro"] = float("nan")
    try:
        result["macro_ap"] = float(average_precision_score(y_onehot, prob, average="macro"))
    except ValueError:
        result["macro_ap"] = float("nan")
    return result


def previous_label(target: str, dose: float) -> int | None:
    if not np.isfinite(dose):
        return None
    if target == "fsh":
        label = phase866.fsh_label_from_dose(dose)
    elif target == "lh":
        label = phase866_final.lh3_label(dose)
    else:
        label = phase866_final.hmg4_label(dose)
    labels = TARGETS[target]["labels"]
    return labels.index(label) if label in labels else None


def last_dynamic(raw: Any) -> np.ndarray:
    output = np.full((len(raw.y), raw.dynamic.shape[2]), np.nan)
    lengths = raw.mask.sum(axis=1).astype(int)
    for index, length in enumerate(lengths):
        if length:
            output[index] = raw.dynamic[index, length - 1]
    return output


def save_predictions(
    target: str,
    split: str,
    raw: Any,
    mask: np.ndarray,
    prob: np.ndarray,
    embedding: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    pred = prob.argmax(axis=1)
    frame = raw.row_index.loc[mask].reset_index(drop=True).copy()
    frame["split"] = split
    frame["target"] = target.upper()
    frame["y_true"] = raw.y[mask]
    frame["truth"] = [labels[int(value)] for value in raw.y[mask]]
    frame["y_pred"] = pred
    frame["prediction"] = [labels[int(value)] for value in pred]
    for index, label in enumerate(labels):
        frame[f"prob_{label}"] = prob[:, index]
    frame.to_csv(OUT / "predictions" / f"{target}_{split}_predictions.csv", index=False)
    emb = frame[["cycle_uid", "visit_uid", "Day", "split", "target", "y_true"]].copy()
    for index in range(embedding.shape[1]):
        emb[f"embedding_{index:03d}"] = embedding[:, index]
    emb.to_csv(OUT / "embeddings" / f"{target}_{split}_embeddings.csv", index=False)
    return frame


def validation_baselines(target: str, raw: Any, train_mask: np.ndarray, valid_mask: np.ndarray):
    labels = TARGETS[target]["labels"]
    n_classes = len(labels)
    rows = []
    counts = np.bincount(raw.y[train_mask], minlength=n_classes)
    majority = int(np.argmax(counts))
    majority_pred = np.full(valid_mask.sum(), majority)
    majority_prob = np.eye(n_classes)[majority_pred]
    rows.append({"target": target.upper(), "model": "Majority baseline", **multiclass_metrics(raw.y[valid_mask], majority_pred, majority_prob, n_classes)})

    feature = TARGETS[target]["previous_feature"]
    matrix = last_dynamic(raw)
    feature_index = raw.dynamic_features.index(feature)
    previous_pred = []
    for dose in matrix[valid_mask, feature_index]:
        value = previous_label(target, float(dose))
        previous_pred.append(majority if value is None else value)
    previous_pred = np.asarray(previous_pred, dtype=int)
    previous_prob = np.eye(n_classes)[previous_pred]
    rows.append({"target": target.upper(), "model": "Previous-dose baseline", **multiclass_metrics(raw.y[valid_mask], previous_pred, previous_prob, n_classes)})

    X = np.hstack([raw.static, matrix])
    lasso = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            penalty="l1",
            solver="saga",
            C=1.0,
            max_iter=1500,
            tol=1e-3,
            multi_class="multinomial",
            random_state=SEED,
            n_jobs=-1,
        ),
    )
    lasso.fit(X[train_mask], raw.y[train_mask])
    lasso_pred = lasso.predict(X[valid_mask]).astype(int)
    lasso_raw_prob = lasso.predict_proba(X[valid_mask])
    lasso_prob = np.zeros((valid_mask.sum(), n_classes))
    for source_col, class_id in enumerate(lasso[-1].classes_):
        lasso_prob[:, int(class_id)] = lasso_raw_prob[:, source_col]
    rows.append({"target": target.upper(), "model": "LASSO logistic regression", **multiclass_metrics(raw.y[valid_mask], lasso_pred, lasso_prob, n_classes)})
    return rows


def bootstrap_metrics(predictions: pd.DataFrame, labels: list[str], n_boot: int = BOOTSTRAP_N):
    probability_columns = [f"prob_{label}" for label in labels]
    cycles = predictions["cycle_uid"].astype(str).unique()
    grouped = {key: value.index.to_numpy() for key, value in predictions.groupby(predictions["cycle_uid"].astype(str))}
    rng = np.random.default_rng(SEED)
    samples = []
    for iteration in range(n_boot):
        sampled_cycles = rng.choice(cycles, size=len(cycles), replace=True)
        indexes = np.concatenate([grouped[cycle] for cycle in sampled_cycles])
        sample = predictions.loc[indexes]
        metrics = multiclass_metrics(
            sample["y_true"].to_numpy(int),
            sample["y_pred"].to_numpy(int),
            sample[probability_columns].to_numpy(float),
            len(labels),
        )
        metrics["bootstrap_iteration"] = iteration + 1
        samples.append(metrics)
    return pd.DataFrame(samples)


def summarize_ci(point: dict[str, Any], bootstrap: pd.DataFrame, target: str):
    rows = []
    for metric in ["accuracy", "macro_f1", "roc_auc_ovr_macro", "macro_ap"]:
        values = pd.to_numeric(bootstrap[metric], errors="coerce").dropna()
        rows.append(
            {
                "task": target.upper(),
                "metric": metric,
                "estimate": point[metric],
                "ci_lower": float(values.quantile(0.025)),
                "ci_upper": float(values.quantile(0.975)),
                "bootstrap_valid_replicates": int(len(values)),
                "bootstrap_total_replicates": BOOTSTRAP_N,
                "bootstrap_unit": "cycle_uid cluster",
            }
        )
    return rows


def train_target(target: str, panel: pd.DataFrame, contract: dict[str, Any], manifest: pd.DataFrame, device: torch.device):
    labels = TARGETS[target]["labels"]
    phase866.LABELS = labels
    phase866.NUM_CLASSES = len(labels)
    feature_groups = {
        "static_features": contract["static_features"],
        "dynamic_features": contract["dynamic_features"],
    }
    raw = phase866_final.prepare_target_raw(panel, feature_groups, target, SimpleNamespace(max_history_days=16))
    split_map = manifest.set_index("cycle_uid")["split"]
    split_values = raw.row_index["cycle_uid"].map(split_map)
    if split_values.isna().any():
        raise RuntimeError(f"{target}: samples absent from split manifest")
    masks = {name: split_values.eq(name).to_numpy() for name in ["train", "valid", "test"]}
    data = phase864.prepare_arrays(raw, fit_mask=masks["train"])

    seed = SEED + {"fsh": 11, "lh": 23, "hmg": 37}[target]
    set_seed(seed)
    model = phase866.FusionGRUClassifier(
        data.static.shape[1], data.dynamic.shape[2], 96, len(labels), 0.15, mode="addgate"
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    criterion = nn.CrossEntropyLoss()
    best_state, best_epoch, best_score, wait = None, 0, -np.inf, 0
    curve = []
    for epoch in range(1, 71):
        model.train()
        losses = []
        for static_x, dynamic_x, obs_mask, y in loader(data, masks["train"], 1024, True):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(static_x.to(device), dynamic_x.to(device), obs_mask.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        valid_prob, _ = predict(model, data, masks["valid"], device)
        valid_pred = valid_prob.argmax(axis=1)
        valid_metrics = multiclass_metrics(data.y[masks["valid"]], valid_pred, valid_prob, len(labels))
        score = valid_metrics["weighted_f1"]
        curve.append({"target": target.upper(), "epoch": epoch, "train_loss": np.mean(losses), **valid_metrics})
        print(f"[{target}] epoch={epoch:02d} loss={np.mean(losses):.5f} valid_wf1={score:.5f}", flush=True)
        if score > best_score + 1e-5:
            best_score, best_epoch, wait = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 12:
                break
    if best_state is None:
        raise RuntimeError(f"{target}: no checkpoint selected")
    model.load_state_dict(best_state)
    checkpoint_path = OUT / "checkpoints" / f"gru_addgate_{target}_holdout.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "target": target,
            "labels": labels,
            "static_features": raw.static_features,
            "dynamic_features": raw.dynamic_features,
            "best_epoch": best_epoch,
            "best_validation_weighted_f1": best_score,
            "hyperparameters": {
                "hidden_dim": 96,
                "dropout": 0.15,
                "epochs_max": 70,
                "patience": 12,
                "batch_size": 1024,
                "lr": 7e-4,
                "weight_decay": 2e-4,
                "seed": seed,
                "loss": "standard cross entropy",
                "selection_metric": "validation weighted_f1",
            },
            "manifest_sha256": sha256(MANIFEST),
            "contract_sha256": sha256(CONTRACT),
        },
        checkpoint_path,
    )
    pd.DataFrame(curve).to_csv(OUT / "training" / f"{target}_training_curve.csv", index=False)

    validation_prob, validation_embedding = predict(model, data, masks["valid"], device)
    validation_frame = save_predictions(target, "valid", raw, masks["valid"], validation_prob, validation_embedding, labels)
    validation_point = multiclass_metrics(data.y[masks["valid"]], validation_prob.argmax(axis=1), validation_prob, len(labels))

    # Test inference is deliberately performed only after validation-selected checkpoint is frozen above.
    test_prob, test_embedding = predict(model, data, masks["test"], device)
    test_frame = save_predictions(target, "test", raw, masks["test"], test_prob, test_embedding, labels)
    test_point = multiclass_metrics(data.y[masks["test"]], test_prob.argmax(axis=1), test_prob, len(labels))

    all_prob, all_embedding = predict(model, data, np.ones(len(data.y), dtype=bool), device)
    save_predictions(target, "all_manifest", raw, np.ones(len(data.y), dtype=bool), all_prob, all_embedding, labels)

    class_report = classification_report(
        test_frame["y_true"], test_frame["y_pred"], labels=list(range(len(labels))), target_names=labels,
        output_dict=True, zero_division=0,
    )
    class_rows = []
    for label in labels:
        stats = class_report[label]
        class_rows.append(
            {
                "target": target.upper(),
                "class": label,
                "testing_samples": int(stats["support"]),
                "precision": stats["precision"],
                "recall": stats["recall"],
                "f1": stats["f1-score"],
            }
        )

    daily_rows = []
    for day, subset in test_frame.groupby("Day"):
        day_metrics = multiclass_metrics(
            subset["y_true"].to_numpy(int), subset["y_pred"].to_numpy(int),
            subset[[f"prob_{label}" for label in labels]].to_numpy(float), len(labels),
        )
        daily_rows.append(
            {
                "target": target.upper(),
                "stimulation_day": int(day),
                "prediction_records_n": int(len(subset)),
                "unique_cycles_n": int(subset["cycle_uid"].nunique()),
                "small_sample_flag": int(len(subset) < 30),
                **day_metrics,
            }
        )
    bootstrap = bootstrap_metrics(test_frame, labels)
    bootstrap.to_csv(OUT / "bootstrap" / f"{target}_test_cluster_bootstrap_1000.csv", index=False)
    ci_rows = summarize_ci(test_point, bootstrap, target)
    baseline_rows = validation_baselines(target, raw, masks["train"], masks["valid"])
    baseline_rows.append({"target": target.upper(), "model": "GRU(AddGate)", **validation_point})
    coverage = {
        "target": target.upper(),
        "manifest_cycles": manifest["split"].value_counts().to_dict(),
        "model_cycles": {
            split: int(raw.row_index.loc[mask, "cycle_uid"].nunique()) for split, mask in masks.items()
        },
        "model_records": {split: int(mask.sum()) for split, mask in masks.items()},
        "best_epoch": best_epoch,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return test_point, class_rows, daily_rows, ci_rows, baseline_rows, coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    for directory in ["checkpoints", "predictions", "embeddings", "training", "tables", "bootstrap", "logs"]:
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    manifest = pd.read_csv(MANIFEST)
    if manifest["cycle_uid"].nunique() != 4071 or manifest["split"].value_counts().to_dict() != {"train": 2849, "valid": 611, "test": 611}:
        raise RuntimeError("Frozen manifest does not match 4071 / 2849 / 611 / 611 contract")
    monitoring = pd.read_csv(MONITORING, low_memory=False)
    baseline = pd.read_csv(BASELINE, low_memory=False)
    panel = phase864.build_daily_no_sw_panel(monitoring, baseline, min_day=1, max_day=20)
    panel = phase866_final.add_grouped_labels(panel)
    for feature in contract["static_features"] + contract["dynamic_features"]:
        if feature not in panel.columns:
            raise RuntimeError(f"UI-reduced feature absent: {feature}")
        panel[feature] = pd.to_numeric(panel[feature], errors="coerce")

    point_rows, class_rows, daily_rows, ci_rows, baseline_rows, coverage_rows = [], [], [], [], [], []
    for target in ["fsh", "lh", "hmg"]:
        point, classes, daily, cis, baselines, coverage = train_target(target, panel, contract, manifest, device)
        point_rows.append({"task": target.upper(), "split": "test", **point})
        class_rows.extend(classes)
        daily_rows.extend(daily)
        ci_rows.extend(cis)
        baseline_rows.extend(baselines)
        coverage_rows.append(coverage)
        pd.DataFrame(point_rows).to_csv(OUT / "tables/final_dose_test_metrics.csv", index=False)
        pd.DataFrame(class_rows).to_csv(OUT / "tables/final_dose_test_classwise_metrics.csv", index=False)
        pd.DataFrame(daily_rows).to_csv(OUT / "tables/supplementary_table_s4_daily_test_performance.csv", index=False)
        pd.DataFrame(ci_rows).to_csv(OUT / "tables/final_dose_metrics_with_95ci.csv", index=False)
        pd.DataFrame(baseline_rows).to_csv(OUT / "tables/validation_baseline_comparison_all_targets.csv", index=False)
        (OUT / "logs/dose_model_coverage.json").write_text(json.dumps(coverage_rows, indent=2), encoding="utf-8")

    s1 = pd.DataFrame(baseline_rows)
    s1 = s1[s1["target"].isin(["LH", "HMG"])]
    s1.to_csv(OUT / "tables/supplementary_table_s1_lh_hmg_validation_baselines.csv", index=False)
    run_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": "7bc2bfc96b8e52e51aab20f84646b66e16f38ebc",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": sha256(MANIFEST),
        "feature_contract": str(CONTRACT.relative_to(ROOT)),
        "feature_contract_sha256": sha256(CONTRACT),
        "monitoring_sha256": sha256(MONITORING),
        "baseline_sha256": sha256(BASELINE),
        "protocol": "train fit; validation early stopping/selection; test one-time report; no OOF/CV",
        "bootstrap_replicates": BOOTSTRAP_N,
    }
    (OUT / "logs/holdout_dose_run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(point_rows).to_string(index=False))


if __name__ == "__main__":
    main()
