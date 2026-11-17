#!/usr/bin/env python3
"""Run full-original-variable architecture, ablation, and frozen-test audits."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/paper_audit_20260715"
COMPARISON_SCRIPT = OUT / "scripts/run_lh_hmg_validation_architecture_comparison.py"
FULL_CONTRACT = OUT / "logs/full_original_feature_contract.json"
SEED_OFFSETS = [0, 101, 202, 303, 404]
TARGET_BASE_SEED = {"fsh": 20260726, "lh": 20260738, "hmg": 20260752}
TARGETS = ["fsh", "lh", "hmg"]
METRICS = ["accuracy", "macro_f1", "weighted_f1", "roc_auc_ovr_macro", "macro_ap"]

ABLATION_GROUPS = {
    "baseline variables": None,
    "visit timing": {
        "Day", "treatment_count", "fresh_treatment_count", "monitoring_order", "cycle_day",
        "gn_day", "days_since_previous_visit", "visits_seen", "evaluation_day",
        "baseline_only_feature_row",
    },
    "current hormones/endometrium": {
        "current_e2", "current_lh", "current_p", "current_fsh", "current_endometrium",
        "delta_e2", "delta_lh", "delta_p", "delta_fsh", "delta_endometrium",
        "p_lh_ratio", "delta_p_per_day", "delta_lh_per_day",
    },
    "follicle profile": {
        "total_follicle_count", "left_follicle_count", "right_follicle_count",
        "max_follicle_diameter", "mean_follicle_diameter", "mature_follicle_count",
        "mature_follicle_share", "large_follicle_share", "dominant_follicle_count",
        "follicle_count_lt_10",
        "follicle_count_10_12", "follicle_count_13_15", "follicle_count_16_18",
        "follicle_count_gt_18", "growing_follicle_count", "medium_plus_follicle_count",
        "ohss_follicle_load_score", "follicle_size_weighted_count", "follicle_maturity_index",
        "mid_follicle_share", "large_to_mature_follicle_ratio", "e2_per_mature_follicle",
        "e2_per_weighted_follicle",
    },
    "previous dose history": {
        "previous_fsh_daily_dose", "previous_lh_daily_dose", "previous_hmg_daily_dose",
        "previous_lh_like_hmg_daily_dose", "previous_gn_dose",
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


comparison = import_module("validation_comparison_multiseed_base", COMPARISON_SCRIPT)
base = comparison.base


def seeded_loader(data: Any, mask: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(data.static[mask], dtype=torch.float32),
        torch.tensor(data.dynamic[mask], dtype=torch.float32),
        torch.tensor(data.mask[mask], dtype=torch.float32),
        torch.tensor(data.y[mask], dtype=torch.long),
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def fit_model(
    target: str,
    model_key: str,
    data: Any,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    device: torch.device,
    init_seed: int,
    loader_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    labels = base.TARGETS[target]["labels"]
    base.phase866.LABELS = labels
    base.phase866.NUM_CLASSES = len(labels)
    base.set_seed(init_seed)
    model = base.phase866.make_neural_model(
        model_key, data, SimpleNamespace(hidden_dim=96, dropout=0.15)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_epoch = 0
    best_score = -np.inf
    wait = 0
    curve = []
    for epoch in range(1, 71):
        model.train()
        losses = []
        for static_x, dynamic_x, obs_mask, y in seeded_loader(
            data, train_mask, 1024, True, loader_seed
        ):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(static_x.to(device), dynamic_x.to(device), obs_mask.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        prob, _ = base.predict(model, data, valid_mask, device)
        pred = prob.argmax(axis=1)
        metric = base.multiclass_metrics(data.y[valid_mask], pred, prob, len(labels))
        score = float(metric["weighted_f1"])
        curve.append({
            "target": target.upper(), "model_key": model_key, "init_seed": init_seed,
            "loader_seed": loader_seed, "epoch": epoch,
            "train_loss": float(np.mean(losses)), **metric,
        })
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            wait = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 12:
                break
    if best_state is None:
        raise RuntimeError((target, model_key, init_seed))
    model.load_state_dict(best_state)
    prob, _ = base.predict(model, data, valid_mask, device)
    pred = prob.argmax(axis=1)
    metric = base.multiclass_metrics(data.y[valid_mask], pred, prob, len(labels))
    metric.update({
        "target": target.upper(), "model_key": model_key, "init_seed": init_seed,
        "loader_seed": loader_seed, "best_epoch": best_epoch,
        "selection_split": "valid", "test_accessed": False,
    })
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metric, pd.DataFrame(curve)


def build_common_data():
    manifest = pd.read_csv(base.MANIFEST)
    expected = {"train": 2849, "valid": 611, "test": 611}
    if manifest["cycle_uid"].nunique() != 4071 or manifest["split"].value_counts().to_dict() != expected:
        raise RuntimeError("Frozen split contract changed")
    monitoring = pd.read_csv(base.MONITORING, low_memory=False)
    baseline_data = pd.read_csv(base.BASELINE, low_memory=False)
    panel, feature_groups, audit = base.phase864.prepare_daily_ctfe_frame(
        monitoring, baseline_data, min_day=1, max_day=20
    )
    panel = base.phase866_final.add_grouped_labels(panel)
    contract = {
        "contract_name": "full_original_variables_no_same_day_dose_leakage",
        "static_features": list(feature_groups["static_features"]),
        "dynamic_features": list(feature_groups["dynamic_features"]),
        "static_feature_count": len(feature_groups["static_features"]),
        "dynamic_feature_count": len(feature_groups["dynamic_features"]),
        "source_audit": audit,
    }
    dynamic_partition = set().union(*(set(v) for k, v in ABLATION_GROUPS.items() if k != "baseline variables"))
    expected_dynamic = set(contract["dynamic_features"])
    if dynamic_partition != expected_dynamic:
        raise RuntimeError({
            "missing_from_ablation_partition": sorted(expected_dynamic - dynamic_partition),
            "unexpected_in_ablation_partition": sorted(dynamic_partition - expected_dynamic),
        })
    FULL_CONTRACT.parent.mkdir(parents=True, exist_ok=True)
    FULL_CONTRACT.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    panel["__baseline_zero__"] = 0.0
    for feature in contract["static_features"] + contract["dynamic_features"]:
        panel[feature] = pd.to_numeric(panel[feature], errors="coerce")
    return panel, contract, manifest


def init_architecture_seed_rows(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def run_architecture(panel, contract, manifest, device: torch.device, selected_targets=None) -> pd.DataFrame:
    selected_targets = list(selected_targets or TARGETS)
    tag = "" if selected_targets == TARGETS else "_" + "_".join(selected_targets)
    path = OUT / f"tables/full_validation_architecture_comparison_5seed_runs{tag}.csv"
    curve_path = OUT / f"training/full_validation_architecture_5seed_curves{tag}.csv"
    rows = init_architecture_seed_rows(path).to_dict("records")
    finished = {(str(row["target"]).lower(), str(row["model_key"]), int(row["init_seed"])) for row in rows}
    curves = pd.read_csv(curve_path).to_dict("records") if curve_path.exists() else []
    for target in selected_targets:
        _raw, data, train_mask, valid_mask = comparison.prepare_data(target, panel, contract, manifest)
        for group, display_name, model_key in comparison.ARCH_ROWS:
            for seed_index, offset in enumerate(SEED_OFFSETS):
                init_seed = TARGET_BASE_SEED[target] + offset
                loader_seed = base.SEED + offset
                key = (target, model_key, init_seed)
                if key in finished:
                    continue
                metric, curve = fit_model(
                    target, model_key, data, train_mask, valid_mask, device, init_seed, loader_seed
                )
                rows.append({"Model group": group, "Model": display_name, **metric})
                curves.extend(curve.to_dict("records"))
                pd.DataFrame(rows).to_csv(path, index=False)
                pd.DataFrame(curves).to_csv(curve_path, index=False)
                finished.add(key)
                print(f"ARCH {target.upper()} {model_key} seed={seed_index + 1}/5 complete", flush=True)
    return pd.DataFrame(rows)


def init_ablation_seed_rows(path: Path, architecture_runs: pd.DataFrame, selected_targets=None) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for target in list(selected_targets or TARGETS):
        full_runs = architecture_runs.loc[
            architecture_runs["target"].eq(target.upper())
            & architecture_runs["model_key"].eq("gru_addgate")
        ].copy()
        for _, row in full_runs.iterrows():
            payload = row.to_dict()
            payload["ablation"] = "All full variables"
            rows.append(payload)
    result = pd.DataFrame(rows)
    result.to_csv(path, index=False)
    return result


def prepare_ablation_data(target, panel, contract, manifest, removed):
    labels = base.TARGETS[target]["labels"]
    base.phase866.LABELS = labels
    base.phase866.NUM_CLASSES = len(labels)
    static_features = [feature for feature in contract["static_features"] if feature not in removed]
    if not static_features:
        static_features = ["__baseline_zero__"]
    dynamic_features = [feature for feature in contract["dynamic_features"] if feature not in removed]
    raw = base.phase866_final.prepare_target_raw(
        panel,
        {"static_features": static_features, "dynamic_features": dynamic_features},
        target,
        SimpleNamespace(max_history_days=16),
    )
    split_map = manifest.set_index("cycle_uid")["split"]
    split_values = raw.row_index["cycle_uid"].map(split_map)
    train_mask = split_values.eq("train").to_numpy()
    valid_mask = split_values.eq("valid").to_numpy()
    data = base.phase864.prepare_arrays(raw, fit_mask=train_mask)
    return data, train_mask, valid_mask


def run_ablation(panel, contract, manifest, device, architecture_runs, selected_targets=None):
    selected_targets = list(selected_targets or TARGETS)
    tag = "" if selected_targets == TARGETS else "_" + "_".join(selected_targets)
    path = OUT / f"tables/full_validation_ablation_5seed_runs{tag}.csv"
    curve_path = OUT / f"training/full_validation_ablation_5seed_curves{tag}.csv"
    rows = init_ablation_seed_rows(path, architecture_runs, selected_targets).to_dict("records")
    finished = {
        (str(row["target"]).lower(), str(row["ablation"]), int(row["init_seed"])) for row in rows
    }
    curves = pd.read_csv(curve_path).to_dict("records") if curve_path.exists() else []
    groups = dict(ABLATION_GROUPS)
    groups["baseline variables"] = set(contract["static_features"])
    for target in selected_targets:
        for group_name, removed in groups.items():
            data, train_mask, valid_mask = prepare_ablation_data(target, panel, contract, manifest, removed)
            ablation_name = f"Without {group_name}"
            for seed_index, offset in enumerate(SEED_OFFSETS):
                init_seed = TARGET_BASE_SEED[target] + offset
                loader_seed = base.SEED + offset
                key = (target, ablation_name, init_seed)
                if key in finished:
                    continue
                metric, curve = fit_model(
                    target, "gru_addgate", data, train_mask, valid_mask, device, init_seed, loader_seed
                )
                rows.append({"ablation": ablation_name, **metric})
                curve["ablation"] = ablation_name
                curves.extend(curve.to_dict("records"))
                pd.DataFrame(rows).to_csv(path, index=False)
                pd.DataFrame(curves).to_csv(curve_path, index=False)
                finished.add(key)
                print(f"ABL {target.upper()} {group_name} seed={seed_index + 1}/5 complete", flush=True)
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def mean_sd(value: pd.Series) -> str:
    return f"{value.mean():.3f} ± {value.std(ddof=1):.3f}"


def aggregate_architecture(runs: pd.DataFrame) -> None:
    order = {key: index for index, (_group, _model, key) in enumerate(comparison.ARCH_ROWS)}
    for target in TARGETS:
        subset = runs.loc[runs["target"].eq(target.upper())].copy()
        counts = subset.groupby("model_key")["init_seed"].nunique()
        if len(counts) != 16 or not counts.eq(5).all():
            raise RuntimeError(f"{target}: architecture seed coverage incomplete: {counts.to_dict()}")
        rows = []
        raw_rows = []
        for group, display, key in comparison.ARCH_ROWS:
            model = subset.loc[subset["model_key"].eq(key)]
            row = {"Model group": group, "Model": display}
            raw = {"task": target.upper(), "Model group": group, "Model": display, "model_key": key, "n_seeds": 5}
            for metric in METRICS:
                values = pd.to_numeric(model[metric])
                row[metric] = mean_sd(values)
                raw[f"{metric}_mean"] = values.mean()
                raw[f"{metric}_sd"] = values.std(ddof=1)
            rows.append(row)
            raw_rows.append(raw)
        formatted = pd.DataFrame(rows).rename(columns={
            "accuracy": "Accuracy", "macro_f1": "Macro-F1", "weighted_f1": "Weighted-F1",
            "roc_auc_ovr_macro": "Macro AUC", "macro_ap": "Macro AP",
        })
        formatted.to_csv(OUT / f"tables/table_full_validation_architecture_comparison_{target}_mean_sd.csv", index=False)
        (OUT / f"tables/table_full_validation_architecture_comparison_{target}_mean_sd.md").write_text(
            markdown_table(formatted), encoding="utf-8"
        )
        pd.DataFrame(raw_rows).to_csv(
            OUT / f"tables/table_full_validation_architecture_comparison_{target}_mean_sd_raw.csv", index=False
        )


def aggregate_ablation(runs: pd.DataFrame) -> None:
    order = ["All full variables"] + [f"Without {name}" for name in ABLATION_GROUPS]
    rows = []
    raw_rows = []
    for target in TARGETS:
        subset = runs.loc[runs["target"].eq(target.upper())].copy()
        counts = subset.groupby("ablation")["init_seed"].nunique()
        if len(counts) != 6 or not counts.eq(5).all():
            raise RuntimeError(f"{target}: ablation seed coverage incomplete: {counts.to_dict()}")
        for ablation in order:
            model = subset.loc[subset["ablation"].eq(ablation)]
            row = {"Task": target.upper(), "Ablation setting": ablation, "Records": int(model["n_records"].iloc[0])}
            raw = {"task": target.upper(), "ablation": ablation, "n_seeds": 5, "n_records": int(model["n_records"].iloc[0])}
            for metric in METRICS:
                values = pd.to_numeric(model[metric])
                row[metric] = mean_sd(values)
                raw[f"{metric}_mean"] = values.mean()
                raw[f"{metric}_sd"] = values.std(ddof=1)
            rows.append(row)
            raw_rows.append(raw)
    formatted = pd.DataFrame(rows).rename(columns={
        "accuracy": "Accuracy", "macro_f1": "Macro-F1", "weighted_f1": "Weighted-F1",
        "roc_auc_ovr_macro": "Macro AUC", "macro_ap": "Macro AP",
    })
    formatted.to_csv(OUT / "tables/supplementary_full_validation_ablation_mean_sd.csv", index=False)
    (OUT / "tables/supplementary_full_validation_ablation_mean_sd.md").write_text(
        markdown_table(formatted), encoding="utf-8"
    )
    pd.DataFrame(raw_rows).to_csv(OUT / "tables/supplementary_full_validation_ablation_mean_sd_raw.csv", index=False)


def cluster_bootstrap(predictions: pd.DataFrame, labels: list[str], target: str, n_boot: int = 1000) -> pd.DataFrame:
    probability_columns = [f"prob_{label}" for label in labels]
    predictions = predictions.reset_index(drop=True)
    cycles = predictions["cycle_uid"].astype(str).unique()
    grouped = {
        key: value.index.to_numpy()
        for key, value in predictions.groupby(predictions["cycle_uid"].astype(str), sort=False)
    }
    rng = np.random.default_rng(base.SEED + {"fsh": 11, "lh": 23, "hmg": 37}[target])
    rows = []
    for iteration in range(n_boot):
        sampled_cycles = rng.choice(cycles, size=len(cycles), replace=True)
        indexes = np.concatenate([grouped[cycle] for cycle in sampled_cycles])
        sample = predictions.loc[indexes]
        metric = base.multiclass_metrics(
            sample["y_true"].to_numpy(int),
            sample["y_pred"].to_numpy(int),
            sample[probability_columns].to_numpy(float),
            len(labels),
        )
        rows.append({"bootstrap_iteration": iteration + 1, **metric})
    return pd.DataFrame(rows)


def train_full_gru_addgate_test(panel, contract, manifest, device: torch.device) -> pd.DataFrame:
    ci_rows = []
    point_rows = []
    for target in TARGETS:
        labels = base.TARGETS[target]["labels"]
        base.phase866.LABELS = labels
        base.phase866.NUM_CLASSES = len(labels)
        raw, data, train_mask, valid_mask = comparison.prepare_data(target, panel, contract, manifest)
        split_map = manifest.set_index("cycle_uid")["split"]
        split_values = raw.row_index["cycle_uid"].map(split_map)
        test_mask = split_values.eq("test").to_numpy()
        init_seed = TARGET_BASE_SEED[target]
        loader_seed = base.SEED
        base.set_seed(init_seed)
        model = base.phase866.make_neural_model(
            "gru_addgate", data, SimpleNamespace(hidden_dim=96, dropout=0.15)
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
        criterion = nn.CrossEntropyLoss()
        best_state = None
        best_epoch = 0
        best_score = -np.inf
        wait = 0
        curve = []
        for epoch in range(1, 71):
            model.train()
            losses = []
            for static_x, dynamic_x, obs_mask, y in seeded_loader(
                data, train_mask, 1024, True, loader_seed
            ):
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(static_x.to(device), dynamic_x.to(device), obs_mask.to(device))
                loss = criterion(logits, y.to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            valid_prob, _ = base.predict(model, data, valid_mask, device)
            valid_pred = valid_prob.argmax(axis=1)
            valid_metric = base.multiclass_metrics(
                data.y[valid_mask], valid_pred, valid_prob, len(labels)
            )
            score = float(valid_metric["weighted_f1"])
            curve.append({
                "target": target.upper(), "epoch": epoch,
                "train_loss": float(np.mean(losses)), **valid_metric,
            })
            if score > best_score + 1e-5:
                best_score = score
                best_epoch = epoch
                wait = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= 12:
                    break
        if best_state is None:
            raise RuntimeError(f"{target}: no full-variable final checkpoint")
        model.load_state_dict(best_state)
        checkpoint = OUT / "checkpoints" / f"full_gru_addgate_{target}_holdout.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(), "target": target, "labels": labels,
            "static_features": raw.static_features, "dynamic_features": raw.dynamic_features,
            "best_epoch": best_epoch, "best_validation_weighted_f1": best_score,
            "init_seed": init_seed, "loader_seed": loader_seed,
            "selection_split": "valid", "test_accessed_during_selection": False,
        }, checkpoint)
        pd.DataFrame(curve).to_csv(
            OUT / "training" / f"full_gru_addgate_{target}_holdout_curve.csv", index=False
        )

        # The test split is materialized only after the validation-selected checkpoint is frozen.
        test_prob, _ = base.predict(model, data, test_mask, device)
        test_pred = test_prob.argmax(axis=1)
        point = base.multiclass_metrics(data.y[test_mask], test_pred, test_prob, len(labels))
        prediction = raw.row_index.loc[test_mask, ["cycle_uid", "visit_uid", "Day"]].reset_index(drop=True)
        prediction["target"] = target.upper()
        prediction["y_true"] = data.y[test_mask]
        prediction["y_pred"] = test_pred
        for index, label in enumerate(labels):
            prediction[f"prob_{label}"] = test_prob[:, index]
        prediction.to_csv(
            OUT / "predictions" / f"full_gru_addgate_{target}_test_predictions.csv", index=False
        )
        bootstrap = cluster_bootstrap(prediction, labels, target, n_boot=1000)
        bootstrap.to_csv(
            OUT / "bootstrap" / f"full_gru_addgate_{target}_test_cluster_bootstrap_1000.csv",
            index=False,
        )
        point_rows.append({
            "task": target.upper(), "split": "test", "n_cycles": int(prediction["cycle_uid"].nunique()),
            "best_epoch": best_epoch, "init_seed": init_seed, **point,
        })
        for metric in METRICS:
            values = pd.to_numeric(bootstrap[metric], errors="coerce").dropna()
            ci_rows.append({
                "task": target.upper(), "metric": metric, "estimate": float(point[metric]),
                "ci_lower": float(values.quantile(0.025)),
                "ci_upper": float(values.quantile(0.975)),
                "bootstrap_valid_replicates": int(len(values)),
                "bootstrap_total_replicates": 1000,
                "bootstrap_unit": "cycle_uid cluster",
                "n_records": int(len(prediction)), "n_cycles": int(prediction["cycle_uid"].nunique()),
            })
        pd.DataFrame(point_rows).to_csv(
            OUT / "tables/full_gru_addgate_final_test_metrics.csv", index=False
        )
        pd.DataFrame(ci_rows).to_csv(
            OUT / "tables/full_gru_addgate_final_test_metrics_with_95ci.csv", index=False
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"FINAL {target.upper()} full GRU(AddGate) complete", flush=True)
    return pd.DataFrame(ci_rows)


def ci_cell(row: pd.Series) -> str:
    return (
        f"{float(row['estimate']):.3f} "
        f"(95% CI {float(row['ci_lower']):.3f}–{float(row['ci_upper']):.3f})"
    )


def aggregate_full_vs_parsimonious_test_ci() -> None:
    full = pd.read_csv(OUT / "tables/full_gru_addgate_final_test_metrics_with_95ci.csv")
    parsimonious = pd.read_csv(OUT / "tables/final_dose_metrics_with_95ci.csv")
    metric_display = [
        ("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"),
        ("weighted_f1", "Weighted-F1"), ("roc_auc_ovr_macro", "Macro AUC"),
        ("macro_ap", "Macro AP"),
    ]
    rows = []
    for target in ["FSH", "LH", "HMG"]:
        full_target = full.loc[full["task"].eq(target)].set_index("metric")
        pars_target = parsimonious.loc[parsimonious["task"].eq(target)].set_index("metric")
        row = {"Target": target}
        for metric, display in metric_display:
            row[f"Full {display}"] = ci_cell(full_target.loc[metric])
            row[f"Parsimonious {display}"] = ci_cell(pars_target.loc[metric])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "tables/full_vs_parsimonious_gru_addgate_test_with_95ci.csv", index=False)
    (OUT / "tables/full_vs_parsimonious_gru_addgate_test_with_95ci.md").write_text(
        markdown_table(table), encoding="utf-8"
    )

    full_rows = []
    for target in ["FSH", "LH", "HMG"]:
        subset = full.loc[full["task"].eq(target)].set_index("metric")
        row = {
            "Task": target,
            "Test records": int(subset["n_records"].iloc[0]),
            "Test cycles": int(subset["n_cycles"].iloc[0]),
        }
        for metric, display in metric_display:
            row[display] = ci_cell(subset.loc[metric])
        full_rows.append(row)
    full_table = pd.DataFrame(full_rows)
    full_table.to_csv(OUT / "tables/full_gru_addgate_final_test_with_95ci.csv", index=False)
    (OUT / "tables/full_gru_addgate_final_test_with_95ci.md").write_text(
        markdown_table(full_table), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["architecture", "ablation", "final", "aggregate", "all"], default="all"
    )
    parser.add_argument("--target", choices=["all", "fsh", "lh", "hmg"], default="all")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    for directory in ["tables", "training", "predictions", "bootstrap", "checkpoints", "logs"]:
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    panel, contract, manifest = build_common_data()
    selected_targets = TARGETS if args.target == "all" else [args.target]
    architecture_path = OUT / "tables/full_validation_architecture_comparison_5seed_runs.csv"
    if args.mode in {"architecture", "all"}:
        architecture_runs = run_architecture(panel, contract, manifest, device, selected_targets)
    elif args.mode in {"ablation", "aggregate"}:
        architecture_runs = pd.read_csv(architecture_path)
    else:
        architecture_runs = pd.DataFrame()
    if args.mode in {"ablation", "all"}:
        ablation_runs = run_ablation(
            panel, contract, manifest, device, architecture_runs, selected_targets
        )
    elif args.mode == "aggregate":
        ablation_runs = pd.read_csv(OUT / "tables/full_validation_ablation_5seed_runs.csv")
    else:
        ablation_runs = pd.DataFrame()
    if args.mode in {"final", "all"}:
        train_full_gru_addgate_test(panel, contract, manifest, device)
        aggregate_full_vs_parsimonious_test_ci()
    if args.mode in {"aggregate", "all"}:
        aggregate_architecture(architecture_runs)
        aggregate_ablation(ablation_runs)
        if (OUT / "tables/full_gru_addgate_final_test_metrics_with_95ci.csv").exists():
            aggregate_full_vs_parsimonious_test_ci()
        manifest_out = {
            "protocol": "Full original variables; five prespecified seeds on fixed training/validation split; test not accessed for model comparison or ablation",
            "seed_offsets": SEED_OFFSETS,
            "target_base_seeds": TARGET_BASE_SEED,
            "n_seeds": 5,
            "summary": "mean and sample SD across seeds",
            "full_feature_counts": {
                "static": len(contract["static_features"]),
                "dynamic": len(contract["dynamic_features"]),
            },
            "final_test": "Prespecified GRU(AddGate), base seed, validation-frozen checkpoint, 1000-cycle-cluster-bootstrap 95% CI",
            "manifest_sha256": base.sha256(base.MANIFEST),
            "contract_sha256": base.sha256(FULL_CONTRACT),
        }
        (OUT / "logs/full_multiseed_validation_and_test_ci_manifest.json").write_text(
            json.dumps(manifest_out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
