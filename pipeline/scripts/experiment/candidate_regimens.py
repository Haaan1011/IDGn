"""Fit validation-only probability calibration and score candidate Gn regimens."""
import argparse
from itertools import product
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import f1_score, log_loss
import torch

from train_paper_gn import PARSIMONIOUS_OUT, FROZEN_SELECTION, base
from run_outcome27_optimization_20261016 import strict_ohss, BASELINE_PATH, LABEL_PATH
from outcome27_model_family_comparison_20261016.run_comparison import classifiers


def scale(probability, temperature):
    return softmax(np.log(np.clip(probability, 1e-12, 1.0)) / temperature, axis=1)


def fit_filter(probability, truth):
    result = minimize_scalar(
        lambda log_t: log_loss(truth, scale(probability, np.exp(log_t)), labels=[0, 1, 2]),
        bounds=(-4, 4), method="bounded",
    )
    if not result.success:
        raise RuntimeError("Validation temperature optimization failed.")
    temperature = float(np.exp(result.x))
    calibrated = scale(probability, temperature)
    positives = np.eye(3, dtype=int)[truth]
    thresholds = np.linspace(0.01, 0.99, 99)
    threshold = max(thresholds, key=lambda t: f1_score(positives.ravel(), (calibrated >= t).ravel()))
    return {"temperature": temperature, "threshold": float(threshold)}


def retain(probability, threshold):
    return sorted(set(np.flatnonzero(probability >= threshold)) | {int(probability.argmax())})


def score_regimens(bundle, probabilities, embedding):
    targets = base.TARGET_ORDER
    scaled = {
        t: scale(np.asarray(probabilities[t], dtype=float).reshape(1, 3), bundle["filters"][t]["temperature"])[0]
        for t in targets
    }
    classes = [retain(scaled[t], bundle["filters"][t]["threshold"]) for t in targets]
    doses = np.asarray(list(product(*[
        [bundle["representative_dose"][t][k] for k in group] for t, group in zip(targets, classes)
    ])), dtype=float)
    features = np.column_stack([np.repeat(np.asarray(embedding).reshape(1, -1), len(doses), axis=0), doses])
    candidates = pd.DataFrame(doses, columns=targets)
    candidates["predicted_oocytes"] = bundle["oocyte"].predict(features)
    candidates["predicted_ohss_risk"] = bundle["ohss"].predict_proba(features)[:, 1]
    candidates["total_dose"] = doses.sum(axis=1)
    candidates = candidates.sort_values(["predicted_ohss_risk", "predicted_oocytes", "total_dose"],
                                       ascending=[True, False, True]).reset_index(drop=True)
    candidates["selected"] = candidates.index == 0
    return candidates


@torch.inference_mode()
def encode(model, arrays, rows, device):
    pieces = []
    for ix in np.array_split(rows, max(1, (len(rows) + 511) // 512)):
        inputs = (arrays.static[ix], arrays.dynamic[ix], arrays.mask[ix])
        _, embedding = model(*(torch.as_tensor(x, device=device) for x in inputs))
        pieces.append(embedding.cpu().numpy())
    return np.concatenate(pieces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filters = {}
    for target in base.TARGET_ORDER:
        predictions = pd.read_csv(PARSIMONIOUS_OUT / "predictions" / f"{target}_validation_probabilities.csv")
        valid = predictions.loc[predictions.y_true.ge(0)]
        filters[target] = fit_filter(valid.filter(regex=r"^prob_").to_numpy(), valid.y_true.to_numpy(int))

    source = pd.read_csv(base.DATA_PATH, low_memory=False)
    manifest = pd.read_csv(base.MANIFEST_PATH, dtype={"cycle_uid": str})
    frame, _ = base.build_monitor_frame(source)
    raw, labels, _ = base.prepare_master(frame)
    split = raw.row_index.cycle_uid.astype(str).map(manifest.set_index("cycle_uid").split)
    train = split.eq("train").to_numpy()
    eligible = train & np.logical_and.reduce([labels[t] >= 0 for t in base.TARGET_ORDER])
    rows = np.flatnonzero(eligible)
    arrays = base.phase864.prepare_arrays(raw, fit_mask=train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neural = FROZEN_SELECTION["fsh"]["neural"]
    model = base.load_model(PARSIMONIOUS_OUT / "checkpoints" / f"{neural}.pt",
                            arrays.static.shape[1], arrays.dynamic.shape[2], device)
    features = encode(model, arrays, rows, device)
    doses = frame.set_index("visit_uid").loc[
        raw.row_index.iloc[rows].visit_uid, [base.TARGETS[t]["dose_col"] for t in base.TARGET_ORDER]
    ].to_numpy(float)
    clinical = pd.read_csv(BASELINE_PATH, dtype={"cycle_uid": str})[["cycle_uid", "target_oocytes"]]
    outcome = pd.read_csv(LABEL_PATH, dtype={"cycle_uid": str})[["cycle_uid", "complication_text"]]
    y = clinical.merge(outcome, on="cycle_uid", validate="one_to_one").set_index("cycle_uid")
    y = y.loc[raw.row_index.iloc[rows].cycle_uid.astype(str)]
    ohss = y.complication_text.map(strict_ohss).to_numpy()
    x = np.column_stack([features, doses])
    oocyte = GradientBoostingRegressor(n_estimators=400, learning_rate=0.025, max_depth=2,
                                       min_samples_leaf=10, random_state=20261016).fit(x, y.target_oocytes)
    risk = classifiers(1)["GradientBoosting"].fit(x, ohss)
    representatives = {
        t: [float(pd.Series(doses[labels[t][rows] == c, j]).mode().iloc[0]) for c in range(3)]
        for j, t in enumerate(base.TARGET_ORDER)
    }
    bundle = {"filters": filters, "representative_dose": representatives, "oocyte": oocyte, "ohss": risk}
    bundle["feature_contract"] = {"static": base.STATIC_FEATURES, "dynamic": base.DYNAMIC_FEATURES,
                                  "max_visits": base.MAX_HISTORY_VISITS}
    bundle["gn_run"] = str(PARSIMONIOUS_OUT.relative_to(base.ROOT))
    bundle["preprocessing"] = base.phase864.fit_sequence_preprocessing(raw, train)
    joblib.dump(bundle, args.output_dir / "scenario_models.joblib")
    (args.output_dir / "filters.json").write_text(json.dumps({
        "calibration_fit": "validation multiclass log loss",
        "threshold_fit": "validation pooled one-vs-rest F1 over 0.01 to 0.99, retaining argmax",
        "filters": filters, "representative_dose": representatives,
        "n_training_records": len(rows),
    }, indent=2) + "\n", encoding="utf-8")
    reference = score_regimens(bundle, {t: [0.25, 0.50, 0.25] for t in base.TARGET_ORDER},
                               np.zeros(features.shape[1]))
    reference.to_csv(args.output_dir / "synthetic_scenario.csv", index=False)


if __name__ == "__main__":
    main()
