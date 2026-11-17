from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.simple_yaml import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _safe_text(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    return normalized.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "None": pd.NA,
            "none": pd.NA,
            "<NA>": pd.NA,
        }
    )


def _normalize_binary(series: pd.Series) -> pd.Series:
    normalized = _safe_text(series).fillna("").str.lower()
    result = pd.Series(pd.NA, index=series.index, dtype="Int64")
    result.loc[normalized.isin({"1", "true", "yes", "y", "是", "有", "阳性"})] = 1
    result.loc[normalized.isin({"0", "false", "no", "n", "否", "无", "阴性"})] = 0
    return result


def _derive_male_factor_flag(frame: pd.DataFrame) -> pd.Series:
    diagnosis_columns = [column for column in ["诊断2", "诊断3", "诊断4", "诊断5", "诊断6"] if column in frame.columns]
    if not diagnosis_columns:
        return pd.Series(pd.NA, index=frame.index, dtype="Int64")

    tokens = [
        r"（?夫",
        r"男方",
        r"精子",
        r"精液",
        r"精索",
        r"睾丸",
        r"附睾",
        r"少精",
        r"弱精",
        r"畸精",
        r"无精",
        r"少弱",
        r"供精",
        r"PESA",
        r"TESA",
        r"夫）",
    ]
    pattern = re.compile("|".join(tokens), flags=re.IGNORECASE)
    flag = pd.Series(False, index=frame.index, dtype="boolean")
    for column in diagnosis_columns:
        text = _safe_text(frame[column]).fillna("")
        flag = flag | text.str.contains(pattern, na=False)
    return flag.astype("Int64")


def _derive_sperm_source_group(treatment_plan: pd.Series) -> pd.Series:
    normalized = _safe_text(treatment_plan).fillna("")
    result = pd.Series(pd.NA, index=treatment_plan.index, dtype="string")
    result.loc[normalized.str.contains("PESA", case=False, na=False, regex=False)] = "PESA"
    result.loc[normalized.str.contains("TESA", case=False, na=False, regex=False)] = "TESA"
    result.loc[normalized.str.contains("供精", case=False, na=False, regex=False)] = "供精"
    self_mask = result.isna() & normalized.notna()
    result.loc[self_mask] = "自体"
    return result


def _prepare_male_factor_frame(raw_clinical_path: Path) -> pd.DataFrame:
    raw_frame = pd.read_excel(raw_clinical_path, sheet_name="Sheet1")
    male_frame = pd.DataFrame(index=raw_frame.index)
    male_frame["cycle_uid"] = (
        raw_frame["ART号"].astype("string").str.strip().fillna("")
        + "__"
        + raw_frame["总编号"].astype("string").str.strip().fillna("")
    )
    male_frame["male_age"] = pd.to_numeric(raw_frame.get("男年龄(岁)"), errors="coerce")
    male_frame["male_factor_infertility_flag"] = _derive_male_factor_flag(raw_frame)
    male_frame["sperm_source_group"] = _derive_sperm_source_group(raw_frame.get("治疗方案"))
    male_frame["fertilization_method"] = _safe_text(raw_frame.get("授精方式"))
    diagnosis_text = (
        _safe_text(raw_frame.get("诊断2")).fillna("")
        + " "
        + _safe_text(raw_frame.get("诊断3")).fillna("")
        + " "
        + _safe_text(raw_frame.get("诊断4")).fillna("")
        + " "
        + _safe_text(raw_frame.get("诊断5")).fillna("")
        + " "
        + _safe_text(raw_frame.get("诊断6")).fillna("")
    ).str.strip()
    male_frame["male_factor_source_text"] = diagnosis_text
    male_frame = male_frame.dropna(subset=["cycle_uid"])
    male_frame = male_frame.drop_duplicates(subset=["cycle_uid"], keep="first")
    return male_frame


def _augment_frame(frame: pd.DataFrame, male_frame: pd.DataFrame) -> pd.DataFrame:
    if "cycle_uid" not in frame.columns:
        return frame.copy()

    merged = frame.copy()
    male_index = male_frame.set_index("cycle_uid")
    male_columns = [
        "male_age",
        "male_factor_infertility_flag",
        "sperm_source_group",
        "fertilization_method",
        "male_factor_source_text",
    ]
    for column in male_columns:
        if column not in male_index.columns:
            continue
        mapped = merged["cycle_uid"].map(male_index[column])
        if column not in merged.columns:
            merged[column] = mapped
        else:
            merged[column] = merged[column].combine_first(mapped)
    return merged


def _append_unique_columns(existing: list[str], additions: list[str]) -> list[str]:
    result = list(existing)
    for column in additions:
        if column not in result:
            result.append(column)
    return result


def main() -> None:
    parser = ArgumentParser(description="Append male-factor features from raw clinical workbook into processed datasets.")
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=REPO_ROOT / "configs/paths/local.example.yaml",
        help="Path config that points to the raw workbook locations.",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write a markdown summary under docs/review.",
    )
    args = parser.parse_args()

    paths_config = load_yaml(_resolve_repo_path(args.paths_config))
    raw_clinical_path = _resolve_repo_path(paths_config["raw_data"]["clinical_table"])
    interim_dir = _resolve_repo_path(paths_config["outputs"]["interim_data_dir"])
    processed_dir = _resolve_repo_path(paths_config["outputs"]["processed_data_dir"])
    report_dir = _resolve_repo_path("docs/review")
    report_dir.mkdir(parents=True, exist_ok=True)

    if not raw_clinical_path.exists():
        raise FileNotFoundError(f"Raw clinical workbook not found: {raw_clinical_path}")

    male_frame = _prepare_male_factor_frame(raw_clinical_path)
    male_summary = {
        "rows": int(len(male_frame)),
        "male_age_non_null": int(male_frame["male_age"].notna().sum()),
        "male_factor_positive": int(male_frame["male_factor_infertility_flag"].fillna(0).sum()),
        "sperm_source_counts": male_frame["sperm_source_group"].fillna("NA").value_counts(dropna=False).to_dict(),
        "fertilization_method_counts": male_frame["fertilization_method"].fillna("NA").value_counts(dropna=False).to_dict(),
    }

    interim_clinical_path = interim_dir / "clinical_standardized.csv"
    if not interim_clinical_path.exists():
        raise FileNotFoundError(f"Missing interim clinical table: {interim_clinical_path}")
    clinical_standardized = pd.read_csv(interim_clinical_path)
    clinical_standardized = _augment_frame(clinical_standardized, male_frame)
    clinical_standardized.to_csv(interim_clinical_path, index=False, encoding="utf-8-sig")

    processed_files = [
        processed_dir / "baseline_cycle_dataset.csv",
        processed_dir / "snapshot_feature_dataset.csv",
        processed_dir / "layer1_strategy_dataset.csv",
        processed_dir / "layer2_snapshot_dataset.csv",
        processed_dir / "layer3_snapshot_dataset.csv",
    ]
    for path in processed_files:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = _augment_frame(frame, male_frame)
        frame.to_csv(path, index=False, encoding="utf-8-sig")

    feature_manifest_path = processed_dir / "feature_manifest.json"
    if feature_manifest_path.exists():
        feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
        male_columns = [
            "male_age",
            "male_factor_infertility_flag",
            "sperm_source_group",
            "fertilization_method",
        ]
        feature_manifest["baseline_feature_columns"] = _append_unique_columns(
            feature_manifest.get("baseline_feature_columns", []),
            male_columns,
        )
        feature_manifest["snapshot_feature_columns"] = _append_unique_columns(
            feature_manifest.get("snapshot_feature_columns", []),
            male_columns,
        )
        feature_manifest_path.write_text(json.dumps(feature_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.write_report:
        report_path = report_dir / "male_factor_feature_augmentation.md"
        lines = [
            "# Male-factor feature augmentation",
            "",
            f"- raw workbook: {raw_clinical_path}",
            f"- augmented rows: {male_summary['rows']}",
            f"- male age non-null rows: {male_summary['male_age_non_null']}",
            f"- male factor positive rows: {male_summary['male_factor_positive']}",
            f"- sperm source counts: {male_summary['sperm_source_counts']}",
            f"- fertilization method counts: {male_summary['fertilization_method_counts']}",
            "",
            "## Added feature columns",
            "- male_age",
            "- male_factor_infertility_flag",
            "- sperm_source_group",
            "- fertilization_method",
        ]
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"raw_clinical_path": str(raw_clinical_path), "male_summary": male_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
