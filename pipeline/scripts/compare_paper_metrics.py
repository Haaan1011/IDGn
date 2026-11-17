"""Compare recalculated point estimates with the 11.12 manuscript."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


OUT = Path(__file__).resolve().parents[1] / "outputs"
GN27 = OUT / "monitor_level_shared_multitask_gru_addgate_27features_current_decision_tuned_20261015"


def main():
    rows = []
    table3, paired = [], []

    def compare(item, target, metric, actual, paper):
        rows.append({"item": item, "target": target, "metric": metric, "recalculated": actual,
                     "manuscript": paper, "difference": actual - paper,
                     "matches_3_decimals": f"{actual:.3f}" == f"{paper:.3f}"})

    for root, label, expected in (
        (GN27, "Table 3 (27)", [(0.732, 0.730), (0.841, 0.841), (0.806, 0.809)]),
        (OUT / "gn_full62", "Table 3 (62)", [(0.732, 0.730), (0.838, 0.839), (0.810, 0.813)]),
    ):
        observed = pd.read_csv(root / "tables/test_metrics.csv").set_index("target")
        for target, pair in zip(("FSH", "LH", "HMG"), expected):
            for metric, reference in zip(("accuracy", "weighted_f1"), pair):
                compare(label, target, metric, observed.loc[target, metric], reference)
    for target, offset in (("fsh", 11), ("lh", 23), ("hmg", 37)):
        cols = ["cycle_uid", "monitoring_order", "y_true", "y_pred"]
        full = pd.read_csv(OUT / "gn_full62/predictions" / f"{target}_full62_test_predictions.csv",
                           usecols=cols)
        small = pd.read_csv(GN27 / "predictions" / f"{target}_all_expert_hybrid_test_predictions.csv",
                            usecols=cols)
        keys = cols[:-1]
        frame = full.merge(small, on=keys, suffixes=("_62", "_27"), validate="one_to_one")
        if len(frame) != len(full) or len(frame) != len(small):
            raise ValueError(f"{target}: full and parsimonious test rows differ.")
        y = frame.y_true.to_numpy(int)
        predictions = [frame[f"y_pred_{n}"].to_numpy(int) for n in (62, 27)]
        def score(index):
            return np.asarray([[accuracy_score(y[index], p[index]),
                                f1_score(y[index], p[index], average="weighted", zero_division=0)]
                               for p in predictions])
        points = score(np.arange(len(frame)))
        groups = list(frame.groupby("cycle_uid").indices.values())
        rng = np.random.default_rng(20261015 + offset + 5000)
        samples = np.asarray([score(np.concatenate([groups[i] for i in indices]))
                              for indices in rng.integers(len(groups), size=(1000, len(groups)))])
        for i, n in enumerate((62, 27)):
            for j, metric in enumerate(("accuracy", "weighted_f1")):
                lower, upper = np.quantile(samples[:, i, j], [0.025, 0.975])
                table3.append({"target": target.upper(), "features": n, "metric": metric,
                               "estimate": points[i, j], "lower95": lower, "upper95": upper})
        for i, diff in enumerate(samples[:, 1, 1] - samples[:, 0, 1], start=1):
            paired.append({"target": target.upper(), "replicate": i, "delta_weighted_f1": diff})
    pd.DataFrame(table3).to_csv(OUT / "table3_paired_cycle_bootstrap.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "table3_paired_bootstrap_replicates.csv", index=False)
    gn = pd.read_csv(OUT / "gn_metrics/source_data/gn_test_metrics.csv").set_index("Target")
    for target, auc, ap in zip(("FSH", "LH", "HMG"), (0.884, 0.946, 0.936), (0.774, 0.826, 0.849)):
        compare("Fig. 4", target, "macro ROC-AUC", gn.loc[target, "Macro-AUC"], auc)
        compare("Fig. 4", target, "macro AP", gn.loc[target, "Average precision"], ap)
    outcomes = pd.read_csv(OUT / "outcome27_model_family_comparison_20261016/tables/all_model_metrics_and_deltas_vs_gru.csv")
    oocytes = outcomes.loc[outcomes.target.eq("Oocytes") & outcomes.model.eq("GRU(AddGate)")].iloc[0]
    ohss = outcomes.loc[outcomes.target.eq("OHSS") & outcomes.model.eq("GradientBoosting")
                         & outcomes.representation.eq("GRU embedding + tabular summary")].iloc[0]
    for metric, reference in (("r2", 0.639), ("mae", 3.073), ("rmse", 4.080)):
        compare("Fig. 5", "Oocytes", metric, oocytes[metric], reference)
    compare("Fig. 5", "OHSS", "ROC-AUC", ohss["roc_auc"], 0.731)
    pd.DataFrame(rows).to_csv(OUT / "manuscript_metric_comparison.csv", index=False)


if __name__ == "__main__":
    main()
