from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from common.simple_yaml import load_yaml
from preprocessing.intake.profiling import build_column_profile, summarize_dataframe
from preprocessing.schema.mapping import (
    build_clinical_field_mapping,
    build_monitoring_field_mapping,
    resolve_key_columns,
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


def _top_unmapped(mapping_df: pd.DataFrame, limit: int = 20) -> list[str]:
    if mapping_df.empty:
        return []
    subset = mapping_df[~mapping_df["is_configured"]]
    return subset["raw_field_name"].astype(str).head(limit).tolist()


def _configured_missing(mapping_df: pd.DataFrame) -> list[str]:
    if mapping_df.empty:
        return []
    subset = mapping_df[mapping_df["is_configured"] & ~mapping_df["present_in_source"]]
    return subset["raw_field_name"].astype(str).tolist()


def _render_summary_markdown(
    title: str,
    source_path: Path,
    summary: dict[str, Any],
    unmapped_columns: list[str],
    configured_missing: list[str],
) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Source: `{source_path}`",
        f"- Rows: {summary['row_count']}",
        f"- Columns: {summary['column_count']}",
        f"- Duplicate rows: {summary['duplicate_row_count']}",
        f"- Data kinds: {summary['data_kind_counts']}",
        "",
        "## Key Summary",
    ]
    for key_name, info in summary["key_summary"]["resolved_keys"].items():
        if info.get("presence_count") is not None:
            lines.append(f"- {key_name}: raw=`{info['raw_column']}` sheet_presence={info['presence_count']}")
        else:
            lines.append(
                f"- {key_name}: raw=`{info['raw_column']}` missing={info['missing_count']} duplicates={info['duplicate_count']}"
            )
    lines.append(f"- composite_duplicate_count: {summary['key_summary']['composite_duplicate_count']}")
    lines.extend(["", "## Top Missing Columns"])
    for item in summary["top_missing_columns"]:
        lines.append(f"- {item['column_name']}: {item['missing_rate']:.2%}")
    lines.extend(["", "## Top IQR Outlier Columns"])
    if summary["top_outlier_columns"]:
        for item in summary["top_outlier_columns"]:
            lines.append(
                f"- {item['column_name']}: count={item['outlier_count_iqr']} rate={item['outlier_rate_iqr']:.2%}"
            )
    else:
        lines.append("- No numeric outliers captured by the IQR rule.")
    lines.extend(["", "## Mapping Gaps"])
    lines.append(f"- Unmapped column count: {summary.get('unmapped_column_count', len(unmapped_columns))}")
    if unmapped_columns:
        lines.append(f"- Unmapped examples: {', '.join(unmapped_columns[:20])}")
    lines.append(
        f"- Configured but absent column count: {summary.get('configured_missing_count', len(configured_missing))}"
    )
    if configured_missing:
        lines.append(f"- Missing configured columns: {', '.join(configured_missing[:20])}")
    return "\n".join(lines) + "\n"


def run_ingest_pipeline(
    paths_config_path: str | Path,
    clinical_schema_path: str | Path,
    monitoring_schema_path: str | Path,
) -> dict[str, Any]:
    paths_config = load_yaml(paths_config_path)
    clinical_schema = load_yaml(clinical_schema_path)
    monitoring_schema = load_yaml(monitoring_schema_path)

    dictionary_dir = _ensure_dir("data/dictionary")
    profiles_dir = _ensure_dir(paths_config["outputs"]["profiles_dir"])

    clinical_path = _resolve_repo_path(paths_config["raw_data"]["clinical_table"])
    monitoring_path = _resolve_repo_path(paths_config["raw_data"]["monitoring_table"])

    clinical_sheet = clinical_schema.get("sheet_name", "Sheet1")
    clinical_df = pd.read_excel(clinical_path, sheet_name=clinical_sheet)
    ignored_clinical_fields = {str(item) for item in clinical_schema.get("ignored_raw_fields", [])}
    if ignored_clinical_fields:
        clinical_df = clinical_df.drop(columns=[column for column in ignored_clinical_fields if column in clinical_df.columns])
    clinical_keys = resolve_key_columns(list(clinical_df.columns), clinical_schema.get("keys", {}))
    clinical_mapping = build_clinical_field_mapping(list(clinical_df.columns), clinical_schema)
    clinical_profile = build_column_profile(
        clinical_df,
        table_name=clinical_schema.get("table_name", "clinical_table"),
        sheet_name=clinical_sheet,
    )
    clinical_summary = summarize_dataframe(
        clinical_df,
        table_name=clinical_schema.get("table_name", "clinical_table"),
        column_profile=clinical_profile,
        key_columns=clinical_keys,
    )
    clinical_unmapped = _top_unmapped(clinical_mapping)
    clinical_missing = _configured_missing(clinical_mapping)
    clinical_summary["source_path"] = str(clinical_path)
    clinical_summary["sheet_name"] = clinical_sheet
    clinical_summary["unmapped_column_count"] = int((~clinical_mapping["is_configured"]).sum())
    clinical_summary["configured_missing_count"] = len(clinical_missing)

    clinical_mapping.to_csv(dictionary_dir / "clinical_table_field_mapping.csv", index=False, encoding="utf-8-sig")
    clinical_profile.to_csv(profiles_dir / "clinical_table_columns.csv", index=False, encoding="utf-8-sig")
    _write_json(profiles_dir / "clinical_table_summary.json", clinical_summary)
    _write_markdown(
        profiles_dir / "clinical_table_report.md",
        _render_summary_markdown("Clinical Table Profile", clinical_path, clinical_summary, clinical_unmapped, clinical_missing),
    )

    monitoring_workbook = pd.ExcelFile(monitoring_path)
    start_sheet, end_sheet = monitoring_schema.get("sheet_range", [1, 15])
    expected_sheets = [str(monitoring_schema["sheet_pattern"]).format(n=index) for index in range(int(start_sheet), int(end_sheet) + 1)]
    actual_columns_by_sheet: dict[str, list[str]] = {}
    monitoring_sheet_rows: list[dict[str, Any]] = []
    monitoring_column_frames: list[pd.DataFrame] = []

    for sheet_name in expected_sheets:
        if sheet_name not in monitoring_workbook.sheet_names:
            monitoring_sheet_rows.append(
                {
                    "sheet_name": sheet_name,
                    "exists": False,
                    "row_count": 0,
                    "column_count": 0,
                    "duplicate_row_count": None,
                    "composite_duplicate_count": None,
                }
            )
            actual_columns_by_sheet[sheet_name] = []
            continue

        sheet_df = pd.read_excel(monitoring_path, sheet_name=sheet_name)
        actual_columns_by_sheet[sheet_name] = [str(column) for column in sheet_df.columns]
        sheet_keys = resolve_key_columns(list(sheet_df.columns), monitoring_schema.get("keys", {}))
        sheet_profile = build_column_profile(
            sheet_df,
            table_name=monitoring_schema.get("table_name", "monitoring_table"),
            sheet_name=sheet_name,
        )
        sheet_summary = summarize_dataframe(
            sheet_df,
            table_name=monitoring_schema.get("table_name", "monitoring_table"),
            column_profile=sheet_profile,
            key_columns=sheet_keys,
        )
        monitoring_sheet_rows.append(
            {
                "sheet_name": sheet_name,
                "exists": True,
                "row_count": sheet_summary["row_count"],
                "column_count": sheet_summary["column_count"],
                "duplicate_row_count": sheet_summary["duplicate_row_count"],
                "composite_duplicate_count": sheet_summary["key_summary"]["composite_duplicate_count"],
                "top_missing_column": sheet_summary["top_missing_columns"][0]["column_name"] if sheet_summary["top_missing_columns"] else "",
                "top_missing_rate": sheet_summary["top_missing_columns"][0]["missing_rate"] if sheet_summary["top_missing_columns"] else 0.0,
            }
        )
        monitoring_column_frames.append(sheet_profile)

    monitoring_mapping = build_monitoring_field_mapping(actual_columns_by_sheet, monitoring_schema)
    monitoring_columns = pd.concat(monitoring_column_frames, ignore_index=True) if monitoring_column_frames else pd.DataFrame()
    monitoring_sheet_summary = pd.DataFrame(monitoring_sheet_rows)
    monitoring_unmapped = _top_unmapped(monitoring_mapping)
    monitoring_missing = _configured_missing(monitoring_mapping)
    monitoring_union_coverage = monitoring_mapping[
        ["raw_field_name", "field_group", "is_configured", "present_in_source", "presence_count", "sheet_list"]
    ].copy()

    monitoring_key_rows = monitoring_mapping[monitoring_mapping["field_group"] == "keys"]
    monitoring_key_summary = {
        "resolved_keys": {
            row["standard_field_name"]: {
                "raw_column": row["raw_field_name"],
                "presence_count": int(row["presence_count"]),
            }
            for _, row in monitoring_key_rows.iterrows()
        },
        "composite_duplicate_count": None,
    }

    monitoring_summary = {
        "table_name": monitoring_schema.get("table_name", "monitoring_table"),
        "source_path": str(monitoring_path),
        "expected_sheet_count": len(expected_sheets),
        "existing_sheet_count": int(monitoring_sheet_summary["exists"].sum()) if not monitoring_sheet_summary.empty else 0,
        "union_column_count": int(monitoring_mapping["raw_field_name"].nunique()) if not monitoring_mapping.empty else 0,
        "configured_missing_count": len(monitoring_missing),
        "unmapped_column_count": int((~monitoring_mapping["is_configured"]).sum()) if not monitoring_mapping.empty else 0,
        "sheet_rows": monitoring_sheet_rows,
        "key_summary": monitoring_key_summary,
    }

    monitoring_mapping.to_csv(dictionary_dir / "monitoring_table_field_mapping.csv", index=False, encoding="utf-8-sig")
    monitoring_sheet_summary.to_csv(profiles_dir / "monitoring_sheet_summary.csv", index=False, encoding="utf-8-sig")
    monitoring_columns.to_csv(profiles_dir / "monitoring_table_columns.csv", index=False, encoding="utf-8-sig")
    monitoring_union_coverage.to_csv(profiles_dir / "monitoring_union_column_coverage.csv", index=False, encoding="utf-8-sig")
    _write_json(profiles_dir / "monitoring_table_summary.json", monitoring_summary)
    _write_markdown(
        profiles_dir / "monitoring_table_report.md",
        _render_summary_markdown(
            "Monitoring Table Profile",
            monitoring_path,
            {
                "row_count": int(monitoring_sheet_summary["row_count"].sum()) if not monitoring_sheet_summary.empty else 0,
                "column_count": monitoring_summary["union_column_count"],
                "duplicate_row_count": int(monitoring_sheet_summary["duplicate_row_count"].fillna(0).sum()) if not monitoring_sheet_summary.empty else 0,
                "data_kind_counts": monitoring_columns["data_kind"].value_counts().to_dict() if not monitoring_columns.empty else {},
                "top_missing_columns": (
                    monitoring_columns.sort_values(["missing_rate", "column_name"], ascending=[False, True])
                    .head(15)[["column_name", "missing_rate"]]
                    .drop_duplicates()
                    .to_dict(orient="records")
                    if not monitoring_columns.empty
                    else []
                ),
                "top_outlier_columns": (
                    monitoring_columns[monitoring_columns["outlier_count_iqr"] > 0]
                    .sort_values(["outlier_count_iqr", "column_name"], ascending=[False, True])
                    .head(15)[["column_name", "outlier_count_iqr", "outlier_rate_iqr"]]
                    .drop_duplicates()
                    .to_dict(orient="records")
                    if not monitoring_columns.empty
                    else []
                ),
                "key_summary": monitoring_key_summary,
                "unmapped_column_count": monitoring_summary["unmapped_column_count"],
                "configured_missing_count": monitoring_summary["configured_missing_count"],
            },
            monitoring_unmapped,
            monitoring_missing,
        ),
    )

    manifest = {
        "phase": "phase_2_intake_and_profile",
        "clinical_path": str(clinical_path),
        "monitoring_path": str(monitoring_path),
        "generated_files": [
            str(dictionary_dir / "clinical_table_field_mapping.csv"),
            str(dictionary_dir / "monitoring_table_field_mapping.csv"),
            str(profiles_dir / "clinical_table_columns.csv"),
            str(profiles_dir / "clinical_table_summary.json"),
            str(profiles_dir / "clinical_table_report.md"),
            str(profiles_dir / "monitoring_sheet_summary.csv"),
            str(profiles_dir / "monitoring_table_columns.csv"),
            str(profiles_dir / "monitoring_union_column_coverage.csv"),
            str(profiles_dir / "monitoring_table_summary.json"),
            str(profiles_dir / "monitoring_table_report.md"),
        ],
    }
    _write_json(profiles_dir / "ingest_manifest.json", manifest)
    return manifest
