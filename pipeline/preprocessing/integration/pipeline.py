from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from common.simple_yaml import load_yaml
from preprocessing.cleaning.standardization import (
    build_monitoring_base_keys,
    extract_monitoring_follicles,
    extract_monitoring_medications,
    standardize_clinical_dataframe,
    standardize_monitoring_visit_dataframe,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _ensure_dir(path_value: str | Path) -> Path:
    path = _resolve_repo_path(path_value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _render_standardization_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 3 Standardization Report",
        "",
        "## Clinical Table",
        f"- Row count: {summary['clinical']['row_count']}",
        f"- Distinct cycle_uid count: {summary['clinical']['distinct_cycle_uid_count']}",
        f"- Duplicate cycle_uid count: {summary['clinical']['duplicate_cycle_uid_count']}",
        f"- Long-protocol candidate count: {summary['clinical']['protocol_candidate_count']}",
        f"- Strict cohort-eligible count: {summary['clinical']['strict_cohort_count']}",
        "",
        "## Official Cohort",
        "- Filters: 治疗方案 ∈ [IVF, ICSI, PESA-ICSI, TESA-ICSI, 供精-IVF] + GnRH-a超长方案 + 治疗次数 = 1",
    ]
    for step in summary["clinical"].get("official_cohort_flow", []):
        lines.append(f"- {step['step']}: {step['count']}")
    lines.extend([
        "",
        "## Target Notes",
        "- `mii` is derived as `oocytes - gv - mi`.",
        "",
        "## Monitoring Tables",
        f"- Standardized visit count: {summary['monitoring']['visit_count']}",
        f"- Medication row count: {summary['monitoring']['medication_count']}",
        f"- Follicle row count: {summary['monitoring']['follicle_count']}",
        f"- Duplicate canonical visit key count: {summary['monitoring']['duplicate_canonical_visit_count']}",
        "",
        "## Carry-Forward Strategy",
        "- `cycle_id` and `art_id` are inherited from `第1次` by row position for later sheets where the source workbook omits those keys.",
        "- Empty later-sheet rows are discarded by requiring both `monitoring_date` and `cycle_uid`.",
        "",
        "## Sheet Summary",
    ])
    for row in summary["monitoring"]["sheet_rows"]:
        lines.append(
            "- {sheet_name}: raw_rows={raw_rows} active_visits={active_visit_rows} medications={medication_rows} follicles={follicle_rows} inherited_cycle_id_rows={inherited_cycle_rows}".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def run_standardization_pipeline(
    paths_config_path: str | Path,
    clinical_schema_path: str | Path,
    monitoring_schema_path: str | Path,
    cohort_config_path: str | Path,
) -> dict[str, Any]:
    paths_config = load_yaml(paths_config_path)
    clinical_schema = load_yaml(clinical_schema_path)
    monitoring_schema = load_yaml(monitoring_schema_path)
    cohort_config = load_yaml(cohort_config_path)

    interim_dir = _ensure_dir(paths_config["outputs"]["interim_data_dir"])
    clinical_path = _resolve_repo_path(paths_config["raw_data"]["clinical_table"])
    monitoring_path = _resolve_repo_path(paths_config["raw_data"]["monitoring_table"])

    clinical_df = pd.read_excel(clinical_path, sheet_name=clinical_schema.get("sheet_name", "Sheet1"))
    clinical_standardized, clinical_resolved_map = standardize_clinical_dataframe(clinical_df, clinical_schema, cohort_config)
    official_flow = [
        {"step": "raw_clinical_rows", "count": int(len(clinical_standardized))},
        {"step": "with_core_identifier", "count": int(clinical_standardized["has_core_identifier"].fillna(False).sum())},
        {
            "step": "ivf_or_icsi_treatment_plan",
            "count": int(
                (
                    clinical_standardized["has_core_identifier"].fillna(False)
                    & clinical_standardized["treatment_plan_is_ivf_icsi_scope"].fillna(False)
                ).sum()
            ),
        },
        {
            "step": "gnrha_superlong_only",
            "count": int(
                (
                    clinical_standardized["has_core_identifier"].fillna(False)
                    & clinical_standardized["treatment_plan_is_ivf_icsi_scope"].fillna(False)
                    & clinical_standardized["protocol_matches_official_protocol"].fillna(False)
                ).sum()
            ),
        },
        {"step": "treatment_count_eq_1", "count": int(clinical_standardized["cohort_strict_eligible"].fillna(False).sum())},
    ]

    workbook = pd.ExcelFile(monitoring_path)
    expected_start, expected_end = monitoring_schema.get("sheet_range", [1, 15])
    expected_sheets = [
        str(monitoring_schema["sheet_pattern"]).format(n=index)
        for index in range(int(expected_start), int(expected_end) + 1)
    ]

    first_sheet_name = expected_sheets[0]
    first_sheet_df = pd.read_excel(monitoring_path, sheet_name=first_sheet_name)
    base_keys = build_monitoring_base_keys(first_sheet_df, monitoring_schema)

    visit_frames: list[pd.DataFrame] = []
    medication_frames: list[pd.DataFrame] = []
    follicle_frames: list[pd.DataFrame] = []
    sheet_rows: list[dict[str, Any]] = []

    for sheet_index, sheet_name in enumerate(expected_sheets, start=1):
        if sheet_name not in workbook.sheet_names:
            sheet_rows.append(
                {
                    "sheet_name": sheet_name,
                    "sheet_index": sheet_index,
                    "raw_rows": 0,
                    "active_visit_rows": 0,
                    "medication_rows": 0,
                    "follicle_rows": 0,
                    "inherited_cycle_rows": 0,
                }
            )
            continue

        raw_sheet_df = pd.read_excel(monitoring_path, sheet_name=sheet_name)
        visit_full = standardize_monitoring_visit_dataframe(raw_sheet_df, base_keys, monitoring_schema, sheet_name, sheet_index)
        visit_active = visit_full[visit_full["is_active_visit"]].copy()
        medication_long = extract_monitoring_medications(raw_sheet_df, visit_full, monitoring_schema)
        follicle_long = extract_monitoring_follicles(raw_sheet_df, visit_full, monitoring_schema)

        visit_frames.append(visit_active)
        medication_frames.append(medication_long)
        follicle_frames.append(follicle_long)
        sheet_rows.append(
            {
                "sheet_name": sheet_name,
                "sheet_index": sheet_index,
                "raw_rows": int(len(raw_sheet_df)),
                "active_visit_rows": int(len(visit_active)),
                "medication_rows": int(len(medication_long)),
                "follicle_rows": int(len(follicle_long)),
                "inherited_cycle_rows": int(visit_full["cycle_id_inherited_from_sheet1"].fillna(False).sum()),
            }
        )

    monitoring_visits = pd.concat(visit_frames, ignore_index=True) if visit_frames else pd.DataFrame()
    monitoring_medications = pd.concat(medication_frames, ignore_index=True) if medication_frames else pd.DataFrame()
    monitoring_follicles = pd.concat(follicle_frames, ignore_index=True) if follicle_frames else pd.DataFrame()

    visit_agg = pd.DataFrame()
    if not monitoring_visits.empty:
        visit_agg = monitoring_visits.groupby("cycle_uid", dropna=False).agg(
            monitoring_visit_count=("visit_uid", "size"),
            first_monitoring_date=("monitoring_date", "min"),
            last_monitoring_date=("monitoring_date", "max"),
            max_monitoring_order=("monitoring_order", "max"),
        )

    medication_agg = pd.DataFrame()
    if not monitoring_medications.empty:
        medication_agg = monitoring_medications.groupby("cycle_uid", dropna=False).agg(
            medication_record_count=("visit_uid", "size"),
            gn_related_medication_count=("is_gn_related", "sum"),
        )

    follicle_agg = pd.DataFrame()
    if not monitoring_follicles.empty:
        follicle_agg = monitoring_follicles.groupby("cycle_uid", dropna=False).agg(
            follicle_record_count=("visit_uid", "size"),
        )

    cycle_master = clinical_standardized.copy()
    for agg_df in [visit_agg, medication_agg, follicle_agg]:
        if not agg_df.empty:
            cycle_master = cycle_master.merge(agg_df.reset_index(), on="cycle_uid", how="left")

    for column in [
        "monitoring_visit_count",
        "medication_record_count",
        "gn_related_medication_count",
        "follicle_record_count",
    ]:
        if column in cycle_master.columns:
            cycle_master[column] = cycle_master[column].fillna(0).astype("Int64")
    cycle_master["has_monitoring_data"] = cycle_master.get("monitoring_visit_count", pd.Series(0, index=cycle_master.index)).gt(0).astype("boolean")

    clinical_standardized.to_csv(interim_dir / "clinical_standardized.csv", index=False, encoding="utf-8-sig")
    monitoring_visits.to_csv(interim_dir / "monitoring_visits_standardized.csv", index=False, encoding="utf-8-sig")
    monitoring_medications.to_csv(interim_dir / "monitoring_medications_long.csv", index=False, encoding="utf-8-sig")
    monitoring_follicles.to_csv(interim_dir / "monitoring_follicles_long.csv", index=False, encoding="utf-8-sig")
    cycle_master.to_csv(interim_dir / "cycle_master_index.csv", index=False, encoding="utf-8-sig")

    summary = {
        "phase": "phase_3_standardization_and_integration",
        "clinical": {
            "row_count": int(len(clinical_standardized)),
            "distinct_cycle_uid_count": int(clinical_standardized["cycle_uid"].nunique(dropna=True)),
            "duplicate_cycle_uid_count": int(clinical_standardized["cycle_uid"].duplicated().sum()),
            "protocol_candidate_count": int(clinical_standardized["protocol_matches_official_protocol"].fillna(False).sum()),
            "strict_cohort_count": int(clinical_standardized["cohort_strict_eligible"].fillna(False).sum()),
            "resolved_fields": clinical_resolved_map,
            "official_cohort_flow": official_flow,
        },
        "monitoring": {
            "visit_count": int(len(monitoring_visits)),
            "medication_count": int(len(monitoring_medications)),
            "follicle_count": int(len(monitoring_follicles)),
            "duplicate_canonical_visit_count": int(
                monitoring_visits["canonical_visit_key"].duplicated().sum()
            )
            if not monitoring_visits.empty
            else 0,
            "sheet_rows": sheet_rows,
        },
        "outputs": [
            str(interim_dir / "clinical_standardized.csv"),
            str(interim_dir / "monitoring_visits_standardized.csv"),
            str(interim_dir / "monitoring_medications_long.csv"),
            str(interim_dir / "monitoring_follicles_long.csv"),
            str(interim_dir / "cycle_master_index.csv"),
            str(interim_dir / "standardization_summary.json"),
            str(interim_dir / "standardization_report.md"),
        ],
    }
    _write_json(interim_dir / "standardization_summary.json", summary)
    _write_markdown(interim_dir / "standardization_report.md", _render_standardization_report(summary))
    return summary
