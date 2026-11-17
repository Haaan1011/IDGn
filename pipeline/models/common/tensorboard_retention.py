from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable


DEFAULT_RETAINED_DISPLAY_RUNS = {
    "phase8_candidate_scoring_diag_20260513_183216",
    "phase8_layer3_paper_metric_lift_20260514_115412",
    "phase8_ohss_imbalance_search_20260514_110104",
    "phase8_layer1_splitdose_action_opt_v3_20260512_192404",
    "phase8_layer1_ctfe_auxiliary_20260520_190706",
    "phase8_ctfe_neural_optimization_20260520_195130",
    "phase8_ctfe_neural_ensemble_20260521_155259",
    "phase8_ctfe_neural_stratified_ensemble_20260521_174449",
    "phase8_ctfe_label_stability_audit_20260525_131839",
    "phase8_ctfe_repeated_split_stability_20260525_133017",
    "phase8_ctfe_stratified_sampling_20260525_142629",
    "phase8_ctfe_mild_cost_sensitive_20260525_202437",
    "phase8_ctfe_cost_sensitive_fusion_20260525_211246",
    "phase8_ctfe_stage_gated_calibration_20260526_120654",
    "phase8_ctfe_stage_auxiliary_20260526_124453",
    "phase8_ctfe_kong_aligned_v2_20260526_135002",
    "phase8_ctfe_label_definition_source_audit_20260526_194916",
    "phase8_ctfe_hierarchical_high_dose_20260526_210416",
    "phase8_34_ctfe_ordered_transition_best",
    "phase8_35_ctfe_early_high_selective_best",
    "phase8_37_layer3_et_endpoint_best",
    "phase837_layer3_et_task_best_20260528_192206",
    "phase8_38_layer3_calibration_shap_review",
    "phase8_39_layer3_stability_bootstrap",
    "phase8_40_layer3_temporal_aggregation",
    "phase8_41_layer3_ensemble_stability",
    "phase8_42_ctfe_high_dose_multitask_sparse",
}


def _safe_child(root: Path, child: Path) -> Path:
    resolved_root = root.resolve()
    resolved_child = child.resolve()
    if resolved_child.parent != resolved_root:
        raise ValueError(f"TensorBoard retention refuses path outside run root: {resolved_child}")
    return resolved_child


def reset_display_slot(run_root: str | Path, slot_name: str) -> Path:
    root = Path(run_root)
    if not slot_name or "/" in slot_name or "\\" in slot_name or slot_name in {".", ".."}:
        raise ValueError("TensorBoard display slot must be a direct directory name.")
    root.mkdir(parents=True, exist_ok=True)
    slot = _safe_child(root, root / slot_name)
    if slot.exists():
        shutil.rmtree(slot)
    overview = slot / "_overview"
    overview.mkdir(parents=True, exist_ok=True)
    return overview


def prune_display_runs(
    run_root: str | Path,
    *,
    keep_run_names: Iterable[str] = DEFAULT_RETAINED_DISPLAY_RUNS,
    apply: bool = False,
) -> dict[str, list[str]]:
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    keep = {str(name) for name in keep_run_names}
    retained: list[str] = []
    removable: list[str] = []
    for entry in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name):
        _safe_child(root, entry)
        if entry.name in keep:
            retained.append(entry.name)
        else:
            removable.append(entry.name)
    removed: list[str] = []
    if apply:
        for name in removable:
            shutil.rmtree(_safe_child(root, root / name))
            removed.append(name)
    return {
        "retained": retained,
        "would_remove": removable,
        "removed": removed,
    }
