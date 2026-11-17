"""Repeat the 16 architecture comparisons across five seeds on validation."""
import pandas as pd
import torch

from train_paper_gn import ROOT, STATIC_FEATURES, DYNAMIC_FEATURES, base, import_module


def main():
    historical = import_module(
        "architecture_training",
        ROOT / "outputs/paper_audit_20260715/scripts/run_full_variable_multiseed_tables.py",
    )
    out = ROOT / "outputs/architecture"
    for folder in ("training", "tables"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    historical.OUT = out
    base.STATIC_FEATURES = STATIC_FEATURES
    base.DYNAMIC_FEATURES = DYNAMIC_FEATURES
    source = pd.read_csv(base.DATA_PATH, low_memory=False)
    manifest = pd.read_csv(base.MANIFEST_PATH, dtype={"cycle_uid": str})
    frame, _ = base.build_monitor_frame(source)
    raw, labels, _ = base.prepare_master(frame)
    split = raw.row_index.cycle_uid.astype(str).map(manifest.set_index("cycle_uid").split)
    if split.isna().any():
        raise ValueError("Records missing from the manuscript split.")
    train = split.eq("train").to_numpy()
    valid = split.eq("valid").to_numpy()
    data = base.phase864.prepare_arrays(raw, fit_mask=train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_path = out / "tables/validation_runs.csv"
    rows = pd.read_csv(metrics_path).to_dict("records") if metrics_path.is_file() else []
    completed = {(r["target"], r["model_key"], r["init_seed"]) for r in rows}
    for target in base.TARGET_ORDER:
        historical.base.TARGETS[target]["labels"] = base.TARGETS[target]["labels"]
        data.y = labels[target]
        evaluable = data.y >= 0
        for group, name, key in historical.comparison.ARCH_ROWS:
            for offset in historical.SEED_OFFSETS:
                seed = historical.TARGET_BASE_SEED[target] + offset
                if (target.upper(), key, seed) in completed:
                    continue
                result, curve = historical.fit_model(
                    target, key, data, train & evaluable, valid & evaluable, device,
                    seed, historical.base.SEED + offset,
                )
                rows.append({"Model group": group, "Model": name, **result})
                curve.to_csv(out / "training" / f"{target}_{key}_{seed}.csv", index=False)
                pd.DataFrame(rows).to_csv(metrics_path, index=False)
    summary = pd.DataFrame(rows).groupby(["target", "Model group", "Model"], sort=False)
    summary[historical.METRICS].agg(["mean", "std"]).to_csv(out / "tables/validation_mean_sd.csv")


if __name__ == "__main__":
    main()
