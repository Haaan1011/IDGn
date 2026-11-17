"""Local tabular research demonstration using synthetic defaults."""
from pathlib import Path
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "experiment"))
from candidate_regimens import score_regimens
from train_paper_gn import ROOT, FROZEN_SELECTION, base, masked_temporal_features


@st.cache_resource
def load_models(path):
    bundle = joblib.load(path)
    cpu = torch.device("cpu")
    checkpoint = ROOT / bundle["gn_run"] / "checkpoints"
    networks, experts = {}, {}
    for task, item in FROZEN_SELECTION.items():
        member = item["neural"]
        if member not in networks:
            networks[member] = base.load_model(checkpoint / f"{member}.pt", 9, 18, cpu)
        experts[task] = joblib.load(checkpoint / f"{task}_{item['expert']}.joblib")
    experts["lh"].set_params(device="cpu", n_jobs=2)
    return bundle, networks, experts


def neural_shap(network, task, inputs, reference, category):
    import shap

    class Logit(torch.nn.Module):
        def forward(self, static, dynamic):
            mask = inputs[2].expand(static.shape[0], -1)
            logits, _ = network(static, dynamic, mask)
            return logits[task][:, category:category+1]

    explainer = shap.GradientExplainer(Logit(), list(reference))
    static, dynamic = explainer.shap_values(list(inputs[:2]), nsamples=64, rseed=20261016)
    values = np.r_[static.reshape(-1), dynamic.reshape(inputs[1].shape[1], -1).sum(axis=0)]
    return pd.DataFrame({
        "Feature": [base.FEATURE_LABELS[f] for f in base.STATIC_FEATURES + base.DYNAMIC_FEATURES],
        "SHAP value (neural logit)": values,
    }).sort_values("SHAP value (neural logit)", key=abs, ascending=False).head(8).set_index("Feature")


def main():
    st.set_page_config(page_title="IDGn reproduction", layout="wide")
    st.title("Individualized Gn regimen scenarios")
    st.caption("Research demonstration with synthetic defaults. Doses are category representatives.")
    path = ROOT / "outputs/scenarios/scenario_models.joblib"
    if not path.is_file():
        st.error("Run the candidate_regimens stage first.")
        st.stop()
    bundle, networks, experts = load_models(str(path))
    panel_in, panel_out = st.columns((1, 2))
    with panel_in:
        st.subheader("Baseline information")
        static_defaults = [30, 22, 3.0, 12, 6, 4, 42, 0.5, 2]
        baseline = st.data_editor(pd.DataFrame([static_defaults], columns=base.STATIC_FEATURES),
                                  hide_index=True, key="baseline")
        st.subheader("Monitoring visits")
        visit_defaults = [
            [0, 1, 30, 2, 0.4, 6, 4, 11, 6, 4.5, 11, 0, 0, 0, 0, 0, 0, 0],
            [5, 2, 350, 2, 0.6, 8, 7, 11, 12, 9.5, 7, 4, 0, 0, 0, 150, 0, 0],
        ]
        editable = [f for f in base.DYNAMIC_FEATURES if f != "monitoring_order"]
        defaults = pd.DataFrame(visit_defaults, columns=base.DYNAMIC_FEATURES)[editable]
        visits = st.data_editor(defaults, num_rows="dynamic", hide_index=True, key="visits")
    with panel_out:
        if visits.empty:
            st.info("Enter at least one monitoring visit.")
            return
        visits = visits.copy().reset_index(drop=True)
        visits["monitoring_order"] = np.arange(1, len(visits) + 1)
        dynamic = visits[base.DYNAMIC_FEATURES].apply(pd.to_numeric, errors="coerce")
        measurements = ["current_e2", "current_lh", "current_p", "current_fsh", "current_endometrium"]
        dynamic[measurements] = dynamic[measurements].ffill()
        first_dose = ["current_fsh_daily_dose", "current_lh_daily_dose", "current_hmg_daily_dose"]
        dynamic.loc[0, first_dose] = dynamic.loc[0, first_dose].fillna(0.0)
        if dynamic["stim_day"].dropna().diff().lt(0).any() or dynamic["stim_day"].dropna().lt(0).any():
            st.error("Enter nonnegative stimulation days in chronological order.")
            return
        dynamic["stim_day"] = dynamic["stim_day"].where(dynamic["stim_day"].between(0, 30))
        if pd.isna(dynamic.loc[0, "stim_day"]):
            dynamic.loc[0, "stim_day"] = 0.0
        n = min(len(dynamic), base.MAX_HISTORY_VISITS)
        shape = (1, base.MAX_HISTORY_VISITS, len(base.DYNAMIC_FEATURES))
        history = np.full(shape, np.nan)
        history[0, :n] = dynamic.tail(n).to_numpy(float)
        mask = np.zeros(shape[:2], dtype=np.float32)
        mask[0, :n] = 1
        prep = bundle["preprocessing"]
        static, dynamic = base.phase864.transform_sequence(baseline.to_numpy(float), history, mask, prep)
        tabular = masked_temporal_features(SimpleNamespace(static=static, dynamic=dynamic, mask=mask))
        tensors = tuple(torch.from_numpy(x) for x in (static, dynamic, mask))
        with torch.inference_mode():
            outputs = {name: model(*tensors) for name, model in networks.items()}
        probabilities = {}
        for task, choice in FROZEN_SELECTION.items():
            neural = outputs[choice["neural"]][0][task].softmax(dim=1).numpy()[0]
            expert = experts[task].predict_proba(tabular)[0]
            probabilities[task] = (1.0 - choice["weight"]) * neural + choice["weight"] * expert
        embedding = outputs[FROZEN_SELECTION["fsh"]["neural"]][1].numpy()[0]
        rows = score_regimens(bundle, probabilities, embedding)
        st.subheader("Candidate doses and predicted outcomes")
        st.dataframe(rows, hide_index=True)
        selected = rows.iloc[0]
        for target, col in zip(base.TARGET_ORDER, st.columns(3)):
            others = [key for key in base.TARGET_ORDER if key != target]
            curve = rows.loc[(rows[others] == selected[others]).all(axis=1)].sort_values(target)
            col.subheader(target.upper())
            col.dataframe(curve[[target, "predicted_oocytes", "predicted_ohss_risk"]],
                          hide_index=True)
        if st.button("Calculate individual SHAP explanations"):
            ref_static, ref_dynamic = base.phase864.transform_sequence(
                prep[0].statistics_[None, :], np.broadcast_to(prep[2].statistics_, shape), mask, prep,
            )
            reference = tuple(torch.from_numpy(x) for x in (ref_static, ref_dynamic))
            for task, col in zip(base.TARGET_ORDER, st.columns(3)):
                col.subheader(task.upper())
                model = networks[FROZEN_SELECTION[task]["neural"]]
                representative = np.asarray(bundle["representative_dose"][task])
                category = int(np.flatnonzero(representative == selected[task])[0])
                col.dataframe(neural_shap(model, task, tensors, reference, category))


if __name__ == "__main__":
    main()
