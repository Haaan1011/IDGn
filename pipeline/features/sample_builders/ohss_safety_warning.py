from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from models.common.config import read_yaml
from models.common.paths import REPO_ROOT, resolve_repo_path


ID_COLUMNS = [
    "visit_uid",
    "canonical_visit_key",
    "cycle_uid",
    "cycle_id",
    "art_id",
    "monitoring_order",
    "monitoring_date",
]


@dataclass(frozen=True)
class SafetyWarningBuildResult:
    cycle_labels_path: Path
    snapshot_dataset_path: Path
    feature_manifest_path: Path
    incidence_table_path: Path
    source_audit_path: Path
    feature_audit_path: Path
    report_path: Path
    cycle_count: int
    snapshot_count: int
    positive_cycles: int


def _boolish(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y", "是"})


def _documented_ohss(series: pd.Series, keywords: list[str]) -> pd.Series:
    text = series.fillna("").astype(str)
    mask = pd.Series(False, index=series.index)
    for keyword in keywords:
        mask |= text.str.contains(str(keyword), case=False, regex=False)
    return mask.astype(int)


def _risk_control_flag(clinical: pd.DataFrame, candidate_fields: list[str]) -> tuple[pd.Series, list[str]]:
    present = [field for field in candidate_fields if field in clinical.columns]
    flag = pd.Series(False, index=clinical.index)
    for field in present:
        values = clinical[field]
        if pd.api.types.is_numeric_dtype(values):
            flag |= pd.to_numeric(values, errors="coerce").fillna(0).gt(0)
        else:
            normalized = values.fillna("").astype(str).str.strip().str.lower()
            flag |= ~normalized.isin({"", "0", "false", "否", "none", "nan", "unknown"})
    return flag.astype(int), present


def build_cycle_labels(clinical: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = set(config.get("required_fields", []))
    missing = sorted(required.difference(clinical.columns))
    if missing:
        raise KeyError(f"Cannot build OHSS safety-warning label; missing required field(s): {missing}")

    cohort_flag = str(config.get("cohort_flag", "cohort_strict_eligible"))
    if cohort_flag not in clinical.columns:
        raise KeyError(f"Configured cohort flag is missing: {cohort_flag}")
    eligible = clinical.loc[_boolish(clinical[cohort_flag])].copy()
    if eligible.empty:
        raise ValueError("Formal cohort is empty while building OHSS safety-warning labels")
    if eligible["cycle_uid"].duplicated().any():
        raise ValueError("Clinical standardized input must contain one row per cycle_uid")

    label_cfg = config["label_definition"]
    documented_cfg = label_cfg["documented_ohss"]
    proxy_cfg = label_cfg["high_response_proxy"]
    intervention_cfg = label_cfg.get("clinician_risk_control", {})

    documented = _documented_ohss(
        eligible[documented_cfg["source_field"]],
        list(documented_cfg.get("keywords", ["OHSS"])),
    )
    oocytes = pd.to_numeric(eligible[proxy_cfg["source_field"]], errors="coerce")
    high_response = oocytes.gt(float(proxy_cfg.get("threshold", 20))).astype(int)
    intervention, included_intervention_fields = _risk_control_flag(
        eligible,
        list(intervention_cfg.get("candidate_fields", [])),
    )

    labels = eligible[["cycle_uid", "cycle_id", "art_id"]].copy()
    labels["documented_ohss_flag"] = documented.to_numpy(dtype=int)
    labels["retrieved_oocytes_gt20_flag"] = high_response.to_numpy(dtype=int)
    labels["clinician_risk_control_flag"] = intervention.to_numpy(dtype=int)
    labels["ohss_related_safety_warning"] = (
        labels[
            [
                "documented_ohss_flag",
                "retrieved_oocytes_gt20_flag",
                "clinician_risk_control_flag",
            ]
        ]
        .max(axis=1)
        .astype(int)
    )
    labels["strict_ohss_safety_warning"] = (
        labels[["documented_ohss_flag", "retrieved_oocytes_gt20_flag"]].max(axis=1).astype(int)
    )
    labels["label_source"] = labels.apply(
        lambda row: "|".join(
            source
            for source, active in [
                ("documented_ohss", row["documented_ohss_flag"]),
                ("retrieved_oocytes_gt20", row["retrieved_oocytes_gt20_flag"]),
                ("clinician_risk_control", row["clinician_risk_control_flag"]),
            ]
            if int(active) == 1
        )
        or "negative",
        axis=1,
    )

    audit = {
        "formal_cohort_cycles": int(len(labels)),
        "documented_ohss_positive": int(labels["documented_ohss_flag"].sum()),
        "retrieved_oocytes_gt20_positive": int(labels["retrieved_oocytes_gt20_flag"].sum()),
        "clinician_risk_control_positive": int(labels["clinician_risk_control_flag"].sum()),
        "primary_positive": int(labels["ohss_related_safety_warning"].sum()),
        "strict_positive": int(labels["strict_ohss_safety_warning"].sum()),
        "primary_strict_mismatch": int(
            (
                labels["ohss_related_safety_warning"]
                != labels["strict_ohss_safety_warning"]
            ).sum()
        ),
        "included_intervention_fields": included_intervention_fields,
        "risk_control_field_status": (
            "included_explicit_fields" if included_intervention_fields else "not_available_not_forced"
        ),
    }
    return labels, audit


def _forbidden_reason(column: str, feature_policy: dict[str, Any]) -> str | None:
    lowered = column.lower()
    if column in set(feature_policy.get("forbidden_exact", [])):
        return "forbidden_exact"
    if any(lowered.startswith(str(prefix).lower()) for prefix in feature_policy.get("forbidden_prefixes", [])):
        return "forbidden_prefix"
    if any(str(token).lower() in lowered for token in feature_policy.get("forbidden_substrings", [])):
        return "forbidden_substring"
    return None


def build_snapshot_modeling_dataset(
    snapshot: pd.DataFrame,
    labels: pd.DataFrame,
    split_manifest: pd.DataFrame,
    model_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    required_snapshot = {"cycle_uid", "visit_uid"}
    missing_snapshot = sorted(required_snapshot.difference(snapshot.columns))
    if missing_snapshot:
        raise KeyError(f"Snapshot dataset missing required identifier(s): {missing_snapshot}")
    if snapshot["visit_uid"].duplicated().any():
        raise ValueError("snapshot_feature_dataset contains duplicated visit_uid values")

    feature_policy = model_config["feature_policy"]
    configured = list(dict.fromkeys(feature_policy.get("allowed_features", [])))
    available = [column for column in configured if column in snapshot.columns]
    unavailable = [column for column in configured if column not in snapshot.columns]

    audit_rows: list[dict[str, Any]] = []
    selected: list[str] = []
    for column in configured:
        reason = _forbidden_reason(column, feature_policy)
        present = column in snapshot.columns
        include = bool(present and reason is None)
        if include:
            selected.append(column)
        audit_rows.append(
            {
                "column": column,
                "present_in_snapshot": present,
                "included_as_feature": include,
                "exclusion_reason": reason or ("" if present else "missing_optional_feature"),
            }
        )

    forbidden_hits = [column for column in selected if _forbidden_reason(column, feature_policy)]
    if forbidden_hits:
        raise ValueError(f"Leakage guard rejected selected feature(s): {forbidden_hits}")
    if not selected:
        raise ValueError("No allowed OHSS safety-warning feature is available")

    model_frame = snapshot[[column for column in ID_COLUMNS if column in snapshot.columns] + selected].copy()
    model_frame = model_frame.merge(
        labels[
            [
                "cycle_uid",
                "ohss_related_safety_warning",
                "strict_ohss_safety_warning",
                "documented_ohss_flag",
                "retrieved_oocytes_gt20_flag",
                "clinician_risk_control_flag",
            ]
        ],
        on="cycle_uid",
        how="inner",
        validate="many_to_one",
    )
    model_frame = model_frame.merge(
        split_manifest[["cycle_uid", "split"]].drop_duplicates("cycle_uid"),
        on="cycle_uid",
        how="left",
        validate="many_to_one",
    )
    if model_frame["split"].isna().any():
        missing_cycles = model_frame.loc[model_frame["split"].isna(), "cycle_uid"].nunique()
        raise ValueError(f"{missing_cycles} cycle(s) are missing from split_manifest_v1")
    if set(model_frame["split"].unique()) != {"train", "valid", "test"}:
        raise ValueError("Modeling dataset must contain train, valid, and test splits")

    manifest = {
        "task_name": model_config["task_name"],
        "dataset_unit": "one row per current monitoring snapshot",
        "group_key": "cycle_uid",
        "split_source": model_config["paths"]["split_manifest"],
        "feature_columns": selected,
        "missing_optional_features": unavailable,
        "identifier_columns": [column for column in ID_COLUMNS if column in model_frame.columns],
        "label_columns": [
            "ohss_related_safety_warning",
            "strict_ohss_safety_warning",
            "documented_ohss_flag",
            "retrieved_oocytes_gt20_flag",
            "clinician_risk_control_flag",
        ],
        "leakage_guard": {
            "forbidden_exact": feature_policy.get("forbidden_exact", []),
            "forbidden_prefixes": feature_policy.get("forbidden_prefixes", []),
            "forbidden_substrings": feature_policy.get("forbidden_substrings", []),
            "selected_forbidden_hits": forbidden_hits,
        },
    }
    return model_frame, manifest, pd.DataFrame(audit_rows)


def _incidence_table(labels: pd.DataFrame, snapshot_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_by_cycle = snapshot_frame[["cycle_uid", "split"]].drop_duplicates("cycle_uid")
    cycle_frame = labels.merge(split_by_cycle, on="cycle_uid", how="left", validate="one_to_one")
    for split_name in ["all", "train", "valid", "test"]:
        cycle_subset = cycle_frame if split_name == "all" else cycle_frame.loc[cycle_frame["split"] == split_name]
        snapshot_subset = snapshot_frame if split_name == "all" else snapshot_frame.loc[snapshot_frame["split"] == split_name]
        rows.append(
            {
                "split": split_name,
                "cycles": int(len(cycle_subset)),
                "positive_cycles": int(cycle_subset["ohss_related_safety_warning"].sum()),
                "negative_cycles": int((cycle_subset["ohss_related_safety_warning"] == 0).sum()),
                "positive_cycle_rate": float(cycle_subset["ohss_related_safety_warning"].mean()),
                "snapshots": int(len(snapshot_subset)),
                "positive_snapshots": int(snapshot_subset["ohss_related_safety_warning"].sum()),
                "positive_snapshot_rate": float(snapshot_subset["ohss_related_safety_warning"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_from_config(
    target_config_path: str | Path = "configs/targets/ohss_safety_warning_v1.yaml",
    model_config_path: str | Path = "configs/models/ohss_safety_warning_v1.yaml",
) -> SafetyWarningBuildResult:
    target_config = read_yaml(resolve_repo_path(target_config_path))
    model_config = read_yaml(resolve_repo_path(model_config_path))

    clinical = pd.read_csv(resolve_repo_path(target_config["sources"]["clinical_dataset"]))
    snapshot = pd.read_csv(resolve_repo_path(target_config["sources"]["snapshot_dataset"]))
    split_manifest = pd.read_csv(resolve_repo_path(target_config["sources"]["split_manifest"]))

    labels, label_audit = build_cycle_labels(clinical, target_config)
    model_frame, feature_manifest, feature_audit = build_snapshot_modeling_dataset(
        snapshot,
        labels,
        split_manifest,
        model_config,
    )
    incidence = _incidence_table(labels, model_frame)

    outputs = target_config["outputs"]
    cycle_labels_path = resolve_repo_path(outputs["cycle_labels"])
    snapshot_dataset_path = resolve_repo_path(outputs["snapshot_dataset"])
    feature_manifest_path = resolve_repo_path(outputs["feature_manifest"])
    incidence_path = resolve_repo_path(outputs["incidence_table"])
    source_audit_path = resolve_repo_path(outputs["source_audit"])
    feature_audit_path = resolve_repo_path(outputs["feature_audit"])
    report_path = resolve_repo_path(outputs["build_report"])
    for path in [
        cycle_labels_path,
        snapshot_dataset_path,
        feature_manifest_path,
        incidence_path,
        source_audit_path,
        feature_audit_path,
        report_path,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    labels.to_csv(cycle_labels_path, index=False)
    model_frame.to_csv(snapshot_dataset_path, index=False)
    feature_manifest_path.write_text(
        json.dumps(feature_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    incidence.to_csv(incidence_path, index=False)
    source_rows = [
        {
            "label_component": "documented_ohss",
            "source_field": target_config["label_definition"]["documented_ohss"]["source_field"],
            "field_available": True,
            "positive_cycles": label_audit["documented_ohss_positive"],
            "status": "included",
        },
        {
            "label_component": "retrieved_oocytes_gt20",
            "source_field": target_config["label_definition"]["high_response_proxy"]["source_field"],
            "field_available": True,
            "positive_cycles": label_audit["retrieved_oocytes_gt20_positive"],
            "status": "label_only_excluded_from_features",
        },
        {
            "label_component": "clinician_risk_control",
            "source_field": "|".join(label_audit["included_intervention_fields"]),
            "field_available": bool(label_audit["included_intervention_fields"]),
            "positive_cycles": label_audit["clinician_risk_control_positive"],
            "status": label_audit["risk_control_field_status"],
        },
    ]
    pd.DataFrame(source_rows).to_csv(source_audit_path, index=False)
    feature_audit.to_csv(feature_audit_path, index=False)

    report = [
        "# OHSS safety-warning label build report",
        "",
        "## Definition",
        "",
        "- Primary: documented OHSS OR retrieved oocytes > 20 OR an explicit clinician risk-control field.",
        "- Strict: documented OHSS OR retrieved oocytes > 20.",
        "- Retrieved oocytes are used only to construct the cycle label and are excluded from model features.",
        "",
        "## Incidence",
        "",
        _dataframe_to_markdown(incidence),
        "",
        "## Source audit",
        "",
        f"- Documented OHSS cycles: {label_audit['documented_ohss_positive']}.",
        f"- Retrieved oocytes > 20 cycles: {label_audit['retrieved_oocytes_gt20_positive']}.",
        f"- Explicit risk-control cycles: {label_audit['clinician_risk_control_positive']}.",
        f"- Primary positive cycles: {label_audit['primary_positive']}.",
        f"- Strict positive cycles: {label_audit['strict_positive']}.",
        f"- Primary/strict mismatch cycles: {label_audit['primary_strict_mismatch']}.",
        f"- Risk-control field status: {label_audit['risk_control_field_status']}.",
        "",
        "## Leakage guard",
        "",
        f"- Selected feature count: {len(feature_manifest['feature_columns'])}.",
        f"- Missing optional features: {feature_manifest['missing_optional_features']}.",
        f"- Selected forbidden hits: {feature_manifest['leakage_guard']['selected_forbidden_hits']}.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return SafetyWarningBuildResult(
        cycle_labels_path=cycle_labels_path,
        snapshot_dataset_path=snapshot_dataset_path,
        feature_manifest_path=feature_manifest_path,
        incidence_table_path=incidence_path,
        source_audit_path=source_audit_path,
        feature_audit_path=feature_audit_path,
        report_path=report_path,
        cycle_count=int(labels["cycle_uid"].nunique()),
        snapshot_count=int(len(model_frame)),
        positive_cycles=int(labels["ohss_related_safety_warning"].sum()),
    )
