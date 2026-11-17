from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.simple_yaml import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

BOOL_TRUE_TOKENS = {"true", "1", "yes", "y"}
BOOL_FALSE_TOKENS = {"false", "0", "no", "n"}

BASELINE_FEATURE_SOURCE_MAP = {
    "age": "age",
    "male_age": "male_age",
    "bmi": "bmi",
    "infertility_duration": "infertility_duration",
    "diagnosis_primary_secondary": "diagnosis_primary_secondary",
    "afc": "afc",
    "amh": "amh",
    "basal_fsh": "fsh",
    "basal_lh": "lh",
    "basal_e2": "e2",
    "basal_p": "p",
    "protocol": "protocol",
    "initial_gn_dose": "initial_gn_dose",
    "male_factor_infertility_flag": "male_factor_infertility_flag",
    "sperm_source_group": "sperm_source_group",
    "fertilization_method": "fertilization_method",
}
CONTEXT_FEATURE_SOURCE_MAP = {
    "procedure_year": "procedure_year",
    "treatment_count": "treatment_count",
    "fresh_treatment_count": "fresh_treatment_count",
}
MONITORING_FEATURE_SOURCE_MAP = {
    "monitoring_order": "monitoring_order",
    "cycle_day": "cycle_day",
    "gn_day": "gn_day",
    "current_e2": "e2",
    "current_lh": "lh",
    "current_p": "p",
    "current_fsh": "fsh",
    "current_endometrium": "endometrium",
    "current_fsh_daily_dose": "current_fsh_daily_dose",
    "current_lh_daily_dose": "current_lh_daily_dose",
    "current_hmg_daily_dose": "current_hmg_daily_dose",
    "current_lh_like_hmg_daily_dose": "current_lh_like_hmg_daily_dose",
    "cumulative_fsh_dose": "cumulative_fsh_dose",
    "cumulative_lh_dose": "cumulative_lh_dose",
    "cumulative_hmg_dose": "cumulative_hmg_dose",
    "cumulative_lh_like_hmg_dose": "cumulative_lh_like_hmg_dose",
}


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


def _load_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=parse_dates or [], low_memory=False)


def _unique_preserve_order(items: list[str]) -> list[str]:
    unique_items: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique_items.append(item)
    return unique_items


def _coerce_boolish(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
            "y": True,
            "n": False,
        }
    )
    return mapped.astype("boolean")


def _coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _safe_divide_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two numeric series while keeping zero denominators as missing."""
    numerator_numeric = pd.to_numeric(numerator, errors="coerce")
    denominator_numeric = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    result = numerator_numeric / denominator_numeric
    return result.replace([np.inf, -np.inf], np.nan)


def _normalize_text(series: pd.Series) -> pd.Series:
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


def _keyword_mask(series: pd.Series, keywords: list[str]) -> pd.Series:
    normalized = _normalize_text(series).fillna("")
    if not keywords:
        return pd.Series(False, index=series.index, dtype="boolean")
    mask = pd.Series(False, index=series.index, dtype="boolean")
    for keyword in keywords:
        mask = mask | normalized.str.contains(re.escape(str(keyword)), case=False, na=False, regex=True)
    return mask.astype("boolean")


def _build_strategy_action(delta: pd.Series, threshold: float) -> pd.Series:
    result = pd.Series(pd.NA, index=delta.index, dtype="string")
    result.loc[delta > threshold] = "increase"
    result.loc[delta < -threshold] = "decrease"
    between_mask = delta.notna() & delta.ge(-threshold) & delta.le(threshold)
    result.loc[between_mask] = "maintain"
    return result


def _normalize_laterality(series: pd.Series) -> pd.Series:
    normalized = _normalize_text(series)
    upper = normalized.fillna("").str.upper()
    result = pd.Series(pd.NA, index=series.index, dtype="string")
    result.loc[upper.isin(["1", "1.0", "L", "LEFT"])] = "L"
    result.loc[upper.isin(["2", "2.0", "R", "RIGHT"])] = "R"
    result.loc[upper.str.contains("左", na=False)] = "L"
    result.loc[upper.str.contains("右", na=False)] = "R"
    return result


def _keyword_any_mask(series: pd.Series, keywords: list[str]) -> pd.Series:
    normalized = _normalize_text(series).fillna("").str.lower()
    mask = pd.Series(False, index=series.index, dtype="boolean")
    for keyword in keywords:
        if not str(keyword).strip():
            continue
        mask = mask | normalized.str.contains(re.escape(str(keyword).lower()), case=False, na=False, regex=True)
    return mask.astype("boolean")


def _classify_medication_groups(
    meds_df: pd.DataFrame,
    feature_config: dict[str, Any],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Classify stimulation medications into mutually exclusive FSH, LH, and HMG groups."""
    med_config = feature_config.get("medication", {})
    text = (
        _normalize_text(meds_df.get("medication_class", pd.Series(index=meds_df.index))).fillna("")
        + " "
        + _normalize_text(meds_df.get("medication_name", pd.Series(index=meds_df.index))).fillna("")
    )
    pure_fsh_keywords = [str(item).lower() for item in med_config.get("pure_fsh_class_keywords", ["fsh", "果纳芬", "普丽康", "丽申宝"])]
    lh_keywords = [str(item).lower() for item in med_config.get("lh_class_keywords", ["lh", "乐芮"])]
    hmg_keywords = [str(item).lower() for item in med_config.get("hmg_class_keywords", ["hmg", "高纯度hmg", "gn(fsh+lh)", "贺美奇"])]

    hmg_mask = _keyword_any_mask(text, hmg_keywords)
    lh_mask = _keyword_any_mask(text, lh_keywords) & ~hmg_mask
    pure_fsh_mask = _keyword_any_mask(text, pure_fsh_keywords) & ~lh_mask & ~hmg_mask
    return pure_fsh_mask.astype("boolean"), lh_mask.astype("boolean"), hmg_mask.astype("boolean")


def _prepare_clinical_dataframe(
    clinical_df: pd.DataFrame,
    cohort_flag: str,
) -> pd.DataFrame:
    clinical_df = clinical_df.copy()
    for column in [
        "has_core_identifier",
        "within_source_year_range",
        "protocol_matches_official_protocol",
        "protocol_matches_long_protocol",
        "treatment_plan_is_ivf_icsi_scope",
        "fertilization_is_ivf_or_icsi",
        "treatment_count_eq_1_flag",
        "first_stimulation_cycle_flag",
        "cohort_protocol_only_eligible",
        "cohort_strict_eligible",
        "has_monitoring_data",
    ]:
        if column in clinical_df.columns:
            clinical_df[column] = _coerce_boolish(clinical_df[column])

    clinical_df = _coerce_numeric_columns(
        clinical_df,
        [
            "procedure_year",
            "treatment_count",
            "fresh_treatment_count",
            "age",
            "male_age",
            "bmi",
            "infertility_duration",
            "afc",
            "amh",
            "fsh",
            "lh",
            "e2",
            "p",
            "initial_gn_dose",
            "gn_start_cycle_day",
            "total_gn_dose",
            "total_gn_days",
            "oocytes",
            "mii",
            "monitoring_visit_count",
            "medication_record_count",
            "gn_related_medication_count",
            "follicle_record_count",
        ],
    )

    for column in [
        "procedure_date",
        "transfer_date",
        "gn_start_date",
        "first_monitoring_date",
        "last_monitoring_date",
    ]:
        if column in clinical_df.columns:
            clinical_df[column] = pd.to_datetime(clinical_df[column], errors="coerce")

    for column in [
        "protocol",
        "treatment_plan",
        "diagnosis_1",
        "diagnosis_primary_secondary",
        "fertilization_method",
        "sperm_source_group",
        "male_factor_source_text",
        "complication_text",
    ]:
        if column in clinical_df.columns:
            clinical_df[column] = _normalize_text(clinical_df[column])

    clinical_df["target_ohss_flag"] = _keyword_mask(clinical_df["complication_text"], ["OHSS"]).astype("Int64")
    clinical_df["target_clinical_pregnancy"] = pd.to_numeric(clinical_df["clinical_pregnancy"], errors="coerce").astype("Int64")
    clinical_df["target_live_birth"] = pd.to_numeric(clinical_df["live_birth"], errors="coerce").astype("Int64")

    cohort_mask = _coerce_boolish(clinical_df[cohort_flag]).fillna(False)
    clinical_df = clinical_df.loc[cohort_mask].copy()
    clinical_df["cohort_flag_used"] = cohort_flag
    return clinical_df


def _prepare_visits_dataframe(visits_df: pd.DataFrame) -> pd.DataFrame:
    visits_df = visits_df.copy()
    visits_df = _coerce_numeric_columns(
        visits_df,
        [
            "source_sheet_index",
            "source_row_index",
            "monitoring_order",
            "days_from_lmp",
            "interval_days",
            "cycle_day",
            "gn_day",
            "e2",
            "lh",
            "p",
            "fsh",
            "endometrium",
        ],
    )
    for column in [
        "cycle_id_inherited_from_sheet1",
        "art_id_inherited_from_sheet1",
        "is_active_visit",
    ]:
        if column in visits_df.columns:
            visits_df[column] = _coerce_boolish(visits_df[column])
    for column in ["monitoring_date", "lmp_date"]:
        if column in visits_df.columns:
            visits_df[column] = pd.to_datetime(visits_df[column], errors="coerce")
    return visits_df


def _prepare_medication_features(
    meds_df: pd.DataFrame,
    feature_config: dict[str, Any],
) -> pd.DataFrame:
    meds_df = meds_df.copy()
    meds_df = _coerce_numeric_columns(
        meds_df,
        [
            "source_sheet_index",
            "source_row_index",
            "monitoring_order",
            "medication_slot",
            "medication_value",
            "medication_days",
            "medication_total_amount",
        ],
    )
    if "monitoring_date" in meds_df.columns:
        meds_df["monitoring_date"] = pd.to_datetime(meds_df["monitoring_date"], errors="coerce")
    for column in ["medication_name", "medication_class", "medication_unit", "medication_frequency"]:
        if column in meds_df.columns:
            meds_df[column] = _normalize_text(meds_df[column])

    pure_fsh_mask, lh_mask, hmg_mask = _classify_medication_groups(meds_df, feature_config)
    meds_df["is_pure_fsh"] = pure_fsh_mask
    meds_df["is_lh"] = lh_mask
    meds_df["is_hmg"] = hmg_mask
    meds_df["is_lh_like_hmg"] = (lh_mask | hmg_mask).astype("boolean")
    meds_df["is_stimulation_gn"] = (pure_fsh_mask | lh_mask | hmg_mask).astype("boolean")

    visit_base = meds_df.groupby("visit_uid", dropna=False).agg(
        total_medication_count=("medication_slot", "size"),
        pure_fsh_medication_count=("is_pure_fsh", "sum"),
        lh_medication_count=("is_lh", "sum"),
        hmg_medication_count=("is_hmg", "sum"),
        lh_like_hmg_medication_count=("is_lh_like_hmg", "sum"),
        stimulation_gn_medication_count=("is_stimulation_gn", "sum"),
    )
    pure_fsh = meds_df[meds_df["is_pure_fsh"].fillna(False)].copy()
    pure_fsh_agg = pure_fsh.groupby("visit_uid", dropna=False).agg(
        visit_pure_fsh_daily_dose=("medication_value", "sum"),
        visit_pure_fsh_total_amount=("medication_total_amount", "sum"),
    )
    lh = meds_df[meds_df["is_lh"].fillna(False)].copy()
    lh_agg = lh.groupby("visit_uid", dropna=False).agg(
        visit_lh_daily_dose=("medication_value", "sum"),
        visit_lh_total_amount=("medication_total_amount", "sum"),
    )
    hmg = meds_df[meds_df["is_hmg"].fillna(False)].copy()
    hmg_agg = hmg.groupby("visit_uid", dropna=False).agg(
        visit_hmg_daily_dose=("medication_value", "sum"),
        visit_hmg_total_amount=("medication_total_amount", "sum"),
    )
    lh_like_hmg = meds_df[meds_df["is_lh_like_hmg"].fillna(False)].copy()
    lh_like_hmg_agg = lh_like_hmg.groupby("visit_uid", dropna=False).agg(
        visit_lh_like_hmg_daily_dose=("medication_value", "sum"),
        visit_lh_like_hmg_total_amount=("medication_total_amount", "sum"),
    )
    stimulation = meds_df[meds_df["is_stimulation_gn"].fillna(False)].copy()
    stimulation_agg = stimulation.groupby("visit_uid", dropna=False).agg(
        visit_stimulation_gn_daily_dose=("medication_value", "sum"),
        visit_stimulation_gn_total_amount=("medication_total_amount", "sum"),
    )

    result = (
        visit_base
        .join(pure_fsh_agg, how="left")
        .join(lh_agg, how="left")
        .join(hmg_agg, how="left")
        .join(lh_like_hmg_agg, how="left")
        .join(stimulation_agg, how="left")
        .reset_index()
    )
    for column in [
        "total_medication_count",
        "pure_fsh_medication_count",
        "lh_medication_count",
        "hmg_medication_count",
        "lh_like_hmg_medication_count",
        "stimulation_gn_medication_count",
        "visit_pure_fsh_daily_dose",
        "visit_pure_fsh_total_amount",
        "visit_lh_daily_dose",
        "visit_lh_total_amount",
        "visit_hmg_daily_dose",
        "visit_hmg_total_amount",
        "visit_lh_like_hmg_daily_dose",
        "visit_lh_like_hmg_total_amount",
        "visit_stimulation_gn_daily_dose",
        "visit_stimulation_gn_total_amount",
    ]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _apply_same_cycle_forward_fill(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Forward-fill only already-visible dynamic values within the same cycle."""
    fill_columns = [column for column in columns if column in frame.columns]
    if not fill_columns:
        return frame
    result = frame.sort_values(["cycle_uid", "monitoring_order", "monitoring_date", "visit_uid"]).copy()
    result[fill_columns] = result.groupby("cycle_uid", dropna=False)[fill_columns].ffill()
    return result.sort_index()


def _add_ohss_high_response_features(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Add OHSS-oriented high-response features visible at the current snapshot."""
    result = snapshot.copy()

    def numeric(column: str) -> pd.Series:
        if column not in result.columns:
            return pd.Series(np.nan, index=result.index, dtype=float)
        return pd.to_numeric(result[column], errors="coerce")

    follicle_10_12 = numeric("follicle_count_10_12").fillna(0.0)
    follicle_13_15 = numeric("follicle_count_13_15").fillna(0.0)
    follicle_16_18 = numeric("follicle_count_16_18").fillna(0.0)
    follicle_gt_18 = numeric("follicle_count_gt_18").fillna(0.0)

    result["growing_follicle_count"] = follicle_10_12 + follicle_13_15 + follicle_16_18 + follicle_gt_18
    result["medium_plus_follicle_count"] = follicle_13_15 + follicle_16_18 + follicle_gt_18
    result["ohss_follicle_load_score"] = (
        0.5 * follicle_10_12
        + 1.0 * follicle_13_15
        + 1.5 * follicle_16_18
        + 2.0 * follicle_gt_18
    )

    current_e2 = numeric("current_e2")
    total_follicles = numeric("total_follicle_count")
    mature_follicles = numeric("mature_follicle_count")
    growing_follicles = result["growing_follicle_count"]

    result["e2_per_total_follicle"] = _safe_divide_series(current_e2, total_follicles)
    result["e2_per_growing_follicle"] = _safe_divide_series(current_e2, growing_follicles)
    result["e2_per_mature_follicle"] = _safe_divide_series(current_e2, mature_follicles)
    result["e2_follicle_load_interaction"] = current_e2 * growing_follicles

    delta_e2 = numeric("delta_e2")
    previous_e2 = current_e2 - delta_e2
    result["delta_e2_per_day"] = _safe_divide_series(delta_e2, numeric("days_since_previous_visit"))
    result["e2_growth_ratio"] = _safe_divide_series(delta_e2, previous_e2)

    amh = numeric("amh")
    afc = numeric("afc")
    age = numeric("age")
    result["amh_afc_product"] = amh * afc
    result["high_response_reserve_index"] = _safe_divide_series(amh * afc, age)

    result["gn_per_growing_follicle"] = _safe_divide_series(numeric("cumulative_gn_dose"), growing_follicles)
    result["lh_like_share_current_gn"] = _safe_divide_series(
        numeric("current_lh_like_hmg_daily_dose"),
        numeric("current_gn_dose"),
    )

    return result


def _add_layer2_regression_features(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Add current-snapshot regression features for oocyte yield and MII maturity."""
    result = snapshot.copy()

    def numeric(column: str) -> pd.Series:
        if column not in result.columns:
            return pd.Series(np.nan, index=result.index, dtype=float)
        return pd.to_numeric(result[column], errors="coerce")

    follicle_lt_10 = numeric("follicle_count_lt_10").fillna(0.0)
    follicle_10_12 = numeric("follicle_count_10_12").fillna(0.0)
    follicle_13_15 = numeric("follicle_count_13_15").fillna(0.0)
    follicle_16_18 = numeric("follicle_count_16_18").fillna(0.0)
    follicle_gt_18 = numeric("follicle_count_gt_18").fillna(0.0)

    total_follicles = numeric("total_follicle_count")
    mature_follicles = numeric("mature_follicle_count")
    current_e2 = numeric("current_e2")
    days_since_previous = numeric("days_since_previous_visit")

    weighted_count = (
        0.25 * follicle_lt_10
        + 0.50 * follicle_10_12
        + 1.00 * follicle_13_15
        + 1.50 * follicle_16_18
        + 1.75 * follicle_gt_18
    )
    result["follicle_size_weighted_count"] = weighted_count
    result["follicle_maturity_index"] = _safe_divide_series(weighted_count, total_follicles)
    result["mid_follicle_share"] = _safe_divide_series(follicle_13_15 + follicle_16_18, total_follicles)
    result["large_to_mature_follicle_ratio"] = _safe_divide_series(
        follicle_16_18 + follicle_gt_18,
        mature_follicles,
    )
    result["e2_per_mature_follicle"] = _safe_divide_series(current_e2, mature_follicles)
    result["e2_per_weighted_follicle"] = _safe_divide_series(current_e2, weighted_count)
    result["p_lh_ratio"] = _safe_divide_series(numeric("current_p"), numeric("current_lh"))
    result["delta_p_per_day"] = _safe_divide_series(numeric("delta_p"), days_since_previous)
    result["delta_lh_per_day"] = _safe_divide_series(numeric("delta_lh"), days_since_previous)

    current_gn = numeric("current_gn_dose")
    cumulative_gn = numeric("cumulative_gn_dose")
    result["gn_per_total_follicle"] = _safe_divide_series(current_gn, total_follicles)
    result["gn_per_mature_follicle"] = _safe_divide_series(current_gn, mature_follicles)
    result["follicle_per_1000iu_gn"] = _safe_divide_series(total_follicles * 1000.0, cumulative_gn)
    result["mature_follicle_per_1000iu_gn"] = _safe_divide_series(mature_follicles * 1000.0, cumulative_gn)
    result["current_fsh_share_gn"] = _safe_divide_series(numeric("current_fsh_daily_dose"), current_gn)
    result["lh_like_share_current_gn"] = _safe_divide_series(numeric("current_lh_like_hmg_daily_dose"), current_gn)
    result["current_e2_per_gn"] = _safe_divide_series(current_e2, current_gn)
    result["amh_afc_product"] = numeric("amh") * numeric("afc")
    result["reserve_age_index"] = _safe_divide_series(numeric("amh") * numeric("afc"), numeric("age"))
    result["follicle_to_afc_ratio"] = _safe_divide_series(total_follicles, numeric("afc"))
    result["follicle_reserve_gap"] = total_follicles - numeric("afc")
    result["amh_follicle_interaction"] = numeric("amh") * total_follicles
    return result


def _prepare_follicle_features(follicles_df: pd.DataFrame) -> pd.DataFrame:
    follicles_df = follicles_df.copy()
    follicles_df = _coerce_numeric_columns(
        follicles_df,
        [
            "source_sheet_index",
            "source_row_index",
            "monitoring_order",
            "follicle_slot",
            "follicle_order",
            "follicle_long_axis",
            "follicle_short_axis",
            "follicle_count",
            "follicle_mean_diameter",
        ],
    )
    if "monitoring_date" in follicles_df.columns:
        follicles_df["monitoring_date"] = pd.to_datetime(follicles_df["monitoring_date"], errors="coerce")

    follicles_df["follicle_laterality"] = _normalize_laterality(follicles_df["follicle_laterality"])
    follicles_df["effective_diameter"] = follicles_df["follicle_mean_diameter"]
    diameter_fallback = follicles_df[["follicle_long_axis", "follicle_short_axis"]].mean(axis=1)
    follicles_df["effective_diameter"] = follicles_df["effective_diameter"].fillna(diameter_fallback)
    follicles_df["follicle_count"] = follicles_df["follicle_count"].fillna(0).clip(lower=0)

    visit_index = pd.Index(follicles_df["visit_uid"].dropna().unique(), name="visit_uid")
    result = pd.DataFrame(index=visit_index)
    result["total_follicle_count"] = follicles_df.groupby("visit_uid", dropna=False)["follicle_count"].sum()

    side_counts = (
        follicles_df.dropna(subset=["follicle_laterality"])
        .pivot_table(
            index="visit_uid",
            columns="follicle_laterality",
            values="follicle_count",
            aggfunc="sum",
            fill_value=0,
        )
        .rename(columns={"L": "left_follicle_count", "R": "right_follicle_count"})
    )
    result = result.join(side_counts, how="left")

    result["max_follicle_diameter"] = follicles_df.groupby("visit_uid", dropna=False)["effective_diameter"].max()
    diameter_weight = follicles_df["follicle_count"].fillna(0)
    weighted_sum = (follicles_df["effective_diameter"].fillna(0) * diameter_weight).groupby(follicles_df["visit_uid"]).sum()
    weight_sum = diameter_weight.groupby(follicles_df["visit_uid"]).sum()
    result["mean_follicle_diameter"] = weighted_sum / weight_sum.replace(0, np.nan)

    bucket_rules = {
        "follicle_count_lt_10": follicles_df["effective_diameter"] < 10,
        "follicle_count_10_12": follicles_df["effective_diameter"].ge(10) & follicles_df["effective_diameter"].le(12),
        "follicle_count_13_15": follicles_df["effective_diameter"].ge(13) & follicles_df["effective_diameter"].le(15),
        "follicle_count_16_18": follicles_df["effective_diameter"].ge(16) & follicles_df["effective_diameter"].le(18),
        "follicle_count_gt_18": follicles_df["effective_diameter"] > 18,
    }
    for column_name, mask in bucket_rules.items():
        result[column_name] = (
            follicles_df.loc[mask].groupby("visit_uid", dropna=False)["follicle_count"].sum()
        )

    result["mature_follicle_count"] = (
        result["follicle_count_13_15"].fillna(0)
        + result["follicle_count_16_18"].fillna(0)
        + result["follicle_count_gt_18"].fillna(0)
    )
    result["mature_follicle_share"] = result["mature_follicle_count"] / result["total_follicle_count"].replace(0, np.nan)
    result["large_follicle_share"] = (
        result["follicle_count_16_18"].fillna(0) + result["follicle_count_gt_18"].fillna(0)
    ) / result["total_follicle_count"].replace(0, np.nan)
    result["dominant_follicle_count"] = result["follicle_count_gt_18"].fillna(0)

    result = result.reset_index()
    count_columns = [
        "total_follicle_count",
        "left_follicle_count",
        "right_follicle_count",
        "follicle_count_lt_10",
        "follicle_count_10_12",
        "follicle_count_13_15",
        "follicle_count_16_18",
        "follicle_count_gt_18",
        "mature_follicle_count",
        "dominant_follicle_count",
    ]
    for column in count_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    for column in ["mature_follicle_share", "large_follicle_share"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    return result



def _filter_layer3_et_eligible_snapshots(snapshot_dataset: pd.DataFrame, clinical_df: pd.DataFrame) -> pd.DataFrame:
    """Return Layer3 snapshots limited to strict-cohort cycles with ET embryos > 0.

    `et_embryo_count` is a downstream transfer variable. It is used only as a
    denominator/sample filter for clinical outcome modeling and is deliberately
    not copied into the returned modeling frame.
    """
    if "et_embryo_count" not in clinical_df.columns:
        raise KeyError("clinical_df must contain et_embryo_count mapped from raw ET胚胎")
    et_count = pd.to_numeric(clinical_df["et_embryo_count"], errors="coerce").fillna(0)
    eligible_cycles = set(clinical_df.loc[et_count.gt(0), "cycle_uid"].dropna().astype(str))
    filtered = snapshot_dataset[snapshot_dataset["cycle_uid"].astype(str).isin(eligible_cycles)].copy()
    forbidden = [
        column
        for column in filtered.columns
        if column.lower().startswith("et_") or "embry" in column.lower() or "transfer" in column.lower()
    ]
    if forbidden:
        filtered = filtered.drop(columns=forbidden)
    return filtered


def _build_baseline_dataset(
    clinical_df: pd.DataFrame,
    feature_config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    baseline_columns = [
        feature_name
        for feature_name in feature_config.get("baseline", {}).get("enabled", [])
        if feature_name in BASELINE_FEATURE_SOURCE_MAP and feature_name != "initial_gn_dose"
    ]
    context_columns = [
        feature_name
        for feature_name in feature_config.get("context", {}).get("enabled", [])
        if feature_name in CONTEXT_FEATURE_SOURCE_MAP
    ]

    dataset = clinical_df[["cycle_uid", "cycle_id", "art_id"]].copy()
    for feature_name in baseline_columns:
        dataset[feature_name] = clinical_df[BASELINE_FEATURE_SOURCE_MAP[feature_name]]
    for feature_name in context_columns:
        dataset[feature_name] = clinical_df[CONTEXT_FEATURE_SOURCE_MAP[feature_name]]

    dataset["observed_initial_gn_dose"] = clinical_df["initial_gn_dose"]
    dataset["target_oocytes"] = clinical_df["oocytes"]
    dataset["target_mii"] = clinical_df["mii"]
    dataset["target_ohss_flag"] = clinical_df["target_ohss_flag"]
    dataset["target_clinical_pregnancy"] = clinical_df["target_clinical_pregnancy"]
    dataset["target_live_birth"] = clinical_df["target_live_birth"]
    dataset["cohort_flag_used"] = clinical_df["cohort_flag_used"]
    return dataset, baseline_columns + context_columns, ["observed_initial_gn_dose"]


def _build_snapshot_dataset(
    clinical_df: pd.DataFrame,
    visits_df: pd.DataFrame,
    medication_features: pd.DataFrame,
    follicle_features: pd.DataFrame,
    feature_config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    cohort_cycles = clinical_df["cycle_uid"].dropna().unique().tolist()
    visits_df = visits_df[visits_df["cycle_uid"].isin(cohort_cycles)].copy()
    visits_df = visits_df.sort_values(["cycle_uid", "monitoring_order", "monitoring_date", "visit_uid"]).reset_index(drop=True)

    snapshot = visits_df[
        [
            "visit_uid",
            "canonical_visit_key",
            "cycle_uid",
            "cycle_id",
            "art_id",
            "monitoring_order",
            "monitoring_date",
            "lmp_date",
            "days_from_lmp",
            "interval_days",
            "cycle_day",
            "gn_day",
            "e2",
            "lh",
            "p",
            "fsh",
            "endometrium",
        ]
    ].copy()
    snapshot = snapshot.merge(
        clinical_df[
            [
                "cycle_uid",
                "procedure_year",
                "age",
                "male_age",
                "bmi",
                "infertility_duration",
                "diagnosis_1",
                "diagnosis_primary_secondary",
                "afc",
                "amh",
                "fsh",
                "fertilization_method",
                "male_factor_infertility_flag",
                "male_factor_source_text",
                "sperm_source_group",
                "lh",
                "e2",
                "p",
                "protocol",
                "initial_gn_dose",
                "treatment_count",
                "fresh_treatment_count",
                "oocytes",
                "mii",
                "target_ohss_flag",
                "target_clinical_pregnancy",
                "target_live_birth",
                "cohort_flag_used",
            ]
        ],
        on="cycle_uid",
        how="left",
        suffixes=("", "_baseline"),
    )

    snapshot = snapshot.merge(medication_features, on="visit_uid", how="left")
    snapshot = snapshot.merge(follicle_features, on="visit_uid", how="left")

    snapshot = snapshot.rename(
        columns={
            "age": "age",
            "bmi": "bmi",
            "infertility_duration": "infertility_duration",
            "afc": "afc",
            "amh": "amh",
            "fsh_baseline": "basal_fsh",
            "lh_baseline": "basal_lh",
            "e2_baseline": "basal_e2",
            "p_baseline": "basal_p",
            "e2": "current_e2",
            "lh": "current_lh",
            "p": "current_p",
            "fsh": "current_fsh",
            "endometrium": "current_endometrium",
            "visit_pure_fsh_daily_dose": "current_fsh_daily_dose",
            "visit_lh_daily_dose": "current_lh_daily_dose",
            "visit_hmg_daily_dose": "current_hmg_daily_dose",
            "visit_lh_like_hmg_daily_dose": "current_lh_like_hmg_daily_dose",
            "visit_pure_fsh_total_amount": "visit_fsh_total_amount",
            "visit_lh_total_amount": "visit_lh_total_amount",
            "visit_hmg_total_amount": "visit_hmg_total_amount",
            "visit_lh_like_hmg_total_amount": "visit_lh_like_hmg_total_amount",
            "visit_stimulation_gn_daily_dose": "current_gn_dose",
            "visit_stimulation_gn_total_amount": "visit_gn_total_amount",
        }
    )

    snapshot["current_fsh_daily_dose"] = pd.to_numeric(snapshot["current_fsh_daily_dose"], errors="coerce").fillna(0.0)
    snapshot["current_lh_daily_dose"] = pd.to_numeric(snapshot["current_lh_daily_dose"], errors="coerce").fillna(0.0)
    snapshot["current_hmg_daily_dose"] = pd.to_numeric(snapshot["current_hmg_daily_dose"], errors="coerce").fillna(0.0)
    snapshot["current_lh_like_hmg_daily_dose"] = pd.to_numeric(
        snapshot["current_lh_like_hmg_daily_dose"], errors="coerce"
    ).fillna(snapshot["current_lh_daily_dose"] + snapshot["current_hmg_daily_dose"]).fillna(0.0)
    snapshot["visit_fsh_total_amount"] = pd.to_numeric(snapshot["visit_fsh_total_amount"], errors="coerce").fillna(0.0)
    snapshot["visit_lh_total_amount"] = pd.to_numeric(snapshot["visit_lh_total_amount"], errors="coerce").fillna(0.0)
    snapshot["visit_hmg_total_amount"] = pd.to_numeric(snapshot["visit_hmg_total_amount"], errors="coerce").fillna(0.0)
    snapshot["visit_lh_like_hmg_total_amount"] = pd.to_numeric(
        snapshot["visit_lh_like_hmg_total_amount"], errors="coerce"
    ).fillna(snapshot["visit_lh_total_amount"] + snapshot["visit_hmg_total_amount"]).fillna(0.0)
    snapshot["current_gn_dose"] = pd.to_numeric(snapshot["current_gn_dose"], errors="coerce").fillna(0.0)
    snapshot["visit_gn_total_amount"] = pd.to_numeric(snapshot["visit_gn_total_amount"], errors="coerce").fillna(0.0)
    snapshot["total_medication_count"] = pd.to_numeric(snapshot["total_medication_count"], errors="coerce").fillna(0.0)
    snapshot["pure_fsh_medication_count"] = pd.to_numeric(snapshot["pure_fsh_medication_count"], errors="coerce").fillna(0.0)
    snapshot["lh_medication_count"] = pd.to_numeric(snapshot["lh_medication_count"], errors="coerce").fillna(0.0)
    snapshot["hmg_medication_count"] = pd.to_numeric(snapshot["hmg_medication_count"], errors="coerce").fillna(0.0)
    snapshot["lh_like_hmg_medication_count"] = pd.to_numeric(
        snapshot["lh_like_hmg_medication_count"], errors="coerce"
    ).fillna(snapshot["lh_medication_count"] + snapshot["hmg_medication_count"]).fillna(0.0)
    snapshot["stimulation_gn_medication_count"] = pd.to_numeric(
        snapshot["stimulation_gn_medication_count"], errors="coerce"
    ).fillna(0.0)

    snapshot = _apply_same_cycle_forward_fill(
        snapshot,
        [
            "cycle_day",
            "gn_day",
            "current_e2",
            "current_lh",
            "current_p",
            "current_fsh",
            "current_endometrium",
        ],
    )

    for column in [
        "total_follicle_count",
        "left_follicle_count",
        "right_follicle_count",
        "follicle_count_lt_10",
        "follicle_count_10_12",
        "follicle_count_13_15",
        "follicle_count_16_18",
        "follicle_count_gt_18",
    ]:
        if column in snapshot.columns:
            snapshot[column] = pd.to_numeric(snapshot[column], errors="coerce").fillna(0.0)

    snapshot["visits_seen"] = snapshot.groupby("cycle_uid").cumcount() + 1
    snapshot["days_since_previous_visit"] = snapshot.groupby("cycle_uid")["monitoring_date"].diff().dt.days
    snapshot["cumulative_fsh_dose"] = snapshot.groupby("cycle_uid")["visit_fsh_total_amount"].cumsum()
    snapshot["cumulative_lh_dose"] = snapshot.groupby("cycle_uid")["visit_lh_total_amount"].cumsum()
    snapshot["cumulative_hmg_dose"] = snapshot.groupby("cycle_uid")["visit_hmg_total_amount"].cumsum()
    snapshot["cumulative_lh_like_hmg_dose"] = snapshot.groupby("cycle_uid")["visit_lh_like_hmg_total_amount"].cumsum()
    snapshot["cumulative_gn_dose"] = snapshot.groupby("cycle_uid")["visit_gn_total_amount"].cumsum()

    delta_columns = {
        "current_e2": "delta_e2",
        "current_lh": "delta_lh",
        "current_p": "delta_p",
        "current_fsh": "delta_fsh",
        "current_endometrium": "delta_endometrium",
    }
    for source_column, delta_column in delta_columns.items():
        snapshot[delta_column] = snapshot[source_column] - snapshot.groupby("cycle_uid")[source_column].shift(1)

    snapshot = _add_ohss_high_response_features(snapshot)
    snapshot = _add_layer2_regression_features(snapshot)

    snapshot["previous_fsh_daily_dose"] = snapshot.groupby("cycle_uid")["current_fsh_daily_dose"].shift(1)
    snapshot["previous_lh_daily_dose"] = snapshot.groupby("cycle_uid")["current_lh_daily_dose"].shift(1)
    snapshot["previous_hmg_daily_dose"] = snapshot.groupby("cycle_uid")["current_hmg_daily_dose"].shift(1)
    snapshot["previous_lh_like_hmg_daily_dose"] = snapshot.groupby("cycle_uid")["current_lh_like_hmg_daily_dose"].shift(1)
    snapshot["previous_gn_dose"] = snapshot.groupby("cycle_uid")["current_gn_dose"].shift(1)
    snapshot["next_fsh_daily_dose"] = snapshot.groupby("cycle_uid")["current_fsh_daily_dose"].shift(-1)
    snapshot["next_lh_daily_dose"] = snapshot.groupby("cycle_uid")["current_lh_daily_dose"].shift(-1)
    snapshot["next_hmg_daily_dose"] = snapshot.groupby("cycle_uid")["current_hmg_daily_dose"].shift(-1)
    snapshot["next_lh_like_hmg_daily_dose"] = snapshot.groupby("cycle_uid")["current_lh_like_hmg_daily_dose"].shift(-1)
    snapshot["next_gn_dose"] = snapshot.groupby("cycle_uid")["current_gn_dose"].shift(-1)
    snapshot["next_monitoring_order"] = snapshot.groupby("cycle_uid")["monitoring_order"].shift(-1)
    snapshot["next_monitoring_date"] = snapshot.groupby("cycle_uid")["monitoring_date"].shift(-1)
    snapshot["next_visit_interval_days"] = (snapshot["next_monitoring_date"] - snapshot["monitoring_date"]).dt.days
    snapshot["observed_gn_dose_delta"] = snapshot["next_gn_dose"] - snapshot["current_gn_dose"]
    action_threshold = float(feature_config.get("settings", {}).get("action_delta_threshold", 12.5))
    snapshot["observed_action_label"] = _build_strategy_action(snapshot["observed_gn_dose_delta"], action_threshold)
    snapshot["has_next_visit"] = snapshot["next_monitoring_order"].notna().astype("boolean")
    min_gn_day = float(feature_config.get("settings", {}).get("strategy_min_gn_day", 1))
    snapshot["strategy_eligible_flag"] = (
        snapshot["has_next_visit"].fillna(False)
        & (
            snapshot["gn_day"].fillna(0).ge(min_gn_day)
            | snapshot["current_gn_dose"].gt(0)
        )
    ).astype("boolean")

    monitoring_feature_columns = [
        feature_name
        for feature_name in feature_config.get("monitoring", {}).get("enabled", [])
        if feature_name in snapshot.columns
    ]
    follicle_feature_columns = [
        feature_name
        for feature_name in feature_config.get("follicles", {}).get("enabled", [])
        if feature_name in snapshot.columns
    ]
    ohss_high_response_feature_columns = [
        feature_name
        for feature_name in feature_config.get("ohss_high_response", {}).get("enabled", [])
        if feature_name in snapshot.columns
    ]
    layer2_regression_feature_columns = [
        feature_name
        for feature_name in feature_config.get("layer2_regression", {}).get("enabled", [])
        if feature_name in snapshot.columns
    ]
    baseline_feature_columns = [
        feature_name
        for feature_name in feature_config.get("baseline", {}).get("enabled", [])
        if feature_name in BASELINE_FEATURE_SOURCE_MAP
    ]
    context_feature_columns = [
        feature_name
        for feature_name in feature_config.get("context", {}).get("enabled", [])
        if feature_name in CONTEXT_FEATURE_SOURCE_MAP
    ]

    column_order = [
        "visit_uid",
        "canonical_visit_key",
        "cycle_uid",
        "cycle_id",
        "art_id",
        "monitoring_order",
        "monitoring_date",
    ]
    column_order.extend(baseline_feature_columns)
    column_order.extend(context_feature_columns)
    column_order.extend(monitoring_feature_columns)
    column_order.extend(follicle_feature_columns)
    column_order.extend(ohss_high_response_feature_columns)
    column_order.extend(layer2_regression_feature_columns)
    column_order.extend(
        [
            "visit_gn_total_amount",
            "total_medication_count",
            "pure_fsh_medication_count",
            "lh_medication_count",
            "hmg_medication_count",
            "lh_like_hmg_medication_count",
            "stimulation_gn_medication_count",
            "previous_fsh_daily_dose",
            "previous_lh_daily_dose",
            "previous_hmg_daily_dose",
            "previous_lh_like_hmg_daily_dose",
            "previous_gn_dose",
            "next_fsh_daily_dose",
            "next_lh_daily_dose",
            "next_hmg_daily_dose",
            "next_lh_like_hmg_daily_dose",
            "next_gn_dose",
            "next_monitoring_order",
            "next_monitoring_date",
            "next_visit_interval_days",
            "observed_gn_dose_delta",
            "observed_action_label",
            "has_next_visit",
            "strategy_eligible_flag",
            "target_oocytes",
            "target_mii",
            "target_ohss_flag",
            "target_clinical_pregnancy",
            "target_live_birth",
            "cohort_flag_used",
        ]
    )

    snapshot["target_oocytes"] = snapshot["oocytes"]
    snapshot["target_mii"] = snapshot["mii"]
    snapshot = snapshot.loc[:, ~snapshot.columns.duplicated()].copy()
    final_columns = [column for column in _unique_preserve_order(column_order) if column in snapshot.columns]
    snapshot = snapshot[final_columns].copy()
    return snapshot, (
        baseline_feature_columns
        + context_feature_columns
        + monitoring_feature_columns
        + follicle_feature_columns
        + ohss_high_response_feature_columns
        + layer2_regression_feature_columns
    ), [
        "visit_gn_total_amount",
        "next_gn_dose",
        "next_monitoring_order",
        "next_monitoring_date",
        "next_visit_interval_days",
        "observed_gn_dose_delta",
        "observed_action_label",
        "has_next_visit",
        "strategy_eligible_flag",
    ]


def _build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 4 Sample Build Report",
        "",
        "## Cohort",
        f"- Cohort flag: `{summary['cohort_flag']}`",
        f"- Eligible cycle count: {summary['cohort_cycle_count']}",
        "",
        "## Datasets",
        f"- Baseline cycle dataset rows: {summary['baseline_cycle_rows']}",
        f"- Snapshot dataset rows: {summary['snapshot_rows']}",
        f"- Layer 1 strategy rows: {summary['layer1_rows']}",
        f"- Layer 2 snapshot rows: {summary['layer2_rows']}",
        f"- Layer 3 snapshot rows: {summary['layer3_rows']}",
        "",
        "## Targets",
        f"- OHSS positive cycles: {summary['target_counts']['ohss_positive_cycles']}",
        f"- Clinical pregnancy positive cycles: {summary['target_counts']['clinical_pregnancy_positive_cycles']}",
        f"- Live birth positive cycles: {summary['target_counts']['live_birth_positive_cycles']}",
        "",
        "## Leakage Guardrails",
        "- Snapshot features are built only from baseline fields plus visit-level data available at or before the current monitoring visit.",
        "- Future-only fields such as `procedure_date`, `procedure_year`, `transfer_date`, `total_gn_dose`, and `total_gn_days` are excluded from processed modeling datasets.",
        "- Strategy samples use the next-visit Gn change only as an observational reference column, not as an input feature.",
    ]
    return "\n".join(lines) + "\n"


def run_sample_build_pipeline(
    paths_config_path: str | Path,
    feature_config_path: str | Path,
) -> dict[str, Any]:
    paths_config = load_yaml(paths_config_path)
    feature_config = load_yaml(feature_config_path)

    interim_dir = _resolve_repo_path(paths_config["outputs"]["interim_data_dir"])
    processed_dir = _ensure_dir(paths_config["outputs"]["processed_data_dir"])

    clinical_df = _load_csv(
        interim_dir / "clinical_standardized.csv",
        parse_dates=["procedure_date", "transfer_date", "gn_start_date"],
    )
    visits_df = _load_csv(
        interim_dir / "monitoring_visits_standardized.csv",
        parse_dates=["monitoring_date", "lmp_date"],
    )
    meds_df = _load_csv(
        interim_dir / "monitoring_medications_long.csv",
        parse_dates=["monitoring_date"],
    )
    follicles_df = _load_csv(
        interim_dir / "monitoring_follicles_long.csv",
        parse_dates=["monitoring_date"],
    )

    cohort_flag = str(feature_config.get("settings", {}).get("cohort_flag", "cohort_strict_eligible"))
    clinical_prepared = _prepare_clinical_dataframe(clinical_df, cohort_flag=cohort_flag)
    visits_prepared = _prepare_visits_dataframe(visits_df)
    medication_features = _prepare_medication_features(meds_df, feature_config)
    follicle_features = _prepare_follicle_features(follicles_df)

    baseline_dataset, baseline_features, baseline_reference_columns = _build_baseline_dataset(
        clinical_prepared,
        feature_config,
    )
    snapshot_dataset, snapshot_features, snapshot_reference_columns = _build_snapshot_dataset(
        clinical_prepared,
        visits_prepared,
        medication_features,
        follicle_features,
        feature_config,
    )

    layer1_dataset = snapshot_dataset[snapshot_dataset["strategy_eligible_flag"].fillna(False)].copy()
    layer2_dataset = snapshot_dataset.copy()
    layer3_dataset = _filter_layer3_et_eligible_snapshots(snapshot_dataset, clinical_df)

    baseline_dataset.to_csv(processed_dir / "baseline_cycle_dataset.csv", index=False, encoding="utf-8-sig")
    snapshot_dataset.to_csv(processed_dir / "snapshot_feature_dataset.csv", index=False, encoding="utf-8-sig")
    layer1_dataset.to_csv(processed_dir / "layer1_strategy_dataset.csv", index=False, encoding="utf-8-sig")
    layer2_dataset.to_csv(processed_dir / "layer2_snapshot_dataset.csv", index=False, encoding="utf-8-sig")
    layer3_dataset.to_csv(processed_dir / "layer3_snapshot_dataset.csv", index=False, encoding="utf-8-sig")

    feature_manifest = {
        "cohort_flag": cohort_flag,
        "baseline_feature_columns": baseline_features,
        "snapshot_feature_columns": snapshot_features,
        "baseline_reference_columns": baseline_reference_columns,
        "snapshot_reference_columns": snapshot_reference_columns,
        "target_columns": {
            "layer2": ["target_oocytes", "target_mii", "target_ohss_flag"],
            "layer3": ["target_clinical_pregnancy", "target_live_birth"],
        },
        "layer3_sample_filter": {
            "definition": "strict cohort + ET胚胎 > 0",
            "source_standardized_column": "et_embryo_count",
            "source_raw_column": "ET胚胎",
            "used_as_feature": False
        },
        "excluded_for_leakage": [
            "procedure_date",
            "procedure_year",
            "transfer_date",
            "et_embryo_count",
            "total_gn_dose",
            "total_gn_days",
            "clinical_pregnancy",
            "live_birth",
        ],
    }
    _write_json(processed_dir / "feature_manifest.json", feature_manifest)

    summary = {
        "phase": "phase_4_sample_building",
        "cohort_flag": cohort_flag,
        "cohort_cycle_count": int(len(baseline_dataset)),
        "baseline_cycle_rows": int(len(baseline_dataset)),
        "snapshot_rows": int(len(snapshot_dataset)),
        "layer1_rows": int(len(layer1_dataset)),
        "layer2_rows": int(len(layer2_dataset)),
        "layer3_rows": int(len(layer3_dataset)),
        "target_counts": {
            "layer3_et_positive_cycles": int(layer3_dataset["cycle_uid"].nunique()),
            "layer3_et_snapshot_rows": int(len(layer3_dataset)),
            "layer3_clinical_pregnancy_positive_cycles": int(layer3_dataset.drop_duplicates("cycle_uid")["target_clinical_pregnancy"].fillna(0).sum()),
            "layer3_live_birth_positive_cycles": int(layer3_dataset.drop_duplicates("cycle_uid")["target_live_birth"].fillna(0).sum()),
            "ohss_positive_cycles": int(baseline_dataset["target_ohss_flag"].fillna(0).sum()),
            "clinical_pregnancy_positive_cycles": int(baseline_dataset["target_clinical_pregnancy"].fillna(0).sum()),
            "live_birth_positive_cycles": int(baseline_dataset["target_live_birth"].fillna(0).sum()),
        },
        "outputs": [
            str(processed_dir / "baseline_cycle_dataset.csv"),
            str(processed_dir / "snapshot_feature_dataset.csv"),
            str(processed_dir / "layer1_strategy_dataset.csv"),
            str(processed_dir / "layer2_snapshot_dataset.csv"),
            str(processed_dir / "layer3_snapshot_dataset.csv"),
            str(processed_dir / "feature_manifest.json"),
            str(processed_dir / "sample_build_summary.json"),
            str(processed_dir / "sample_build_report.md"),
        ],
    }
    _write_json(processed_dir / "sample_build_summary.json", summary)
    _write_markdown(processed_dir / "sample_build_report.md", _build_report(summary))
    return summary
