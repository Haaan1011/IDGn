from __future__ import annotations

import re
from typing import Any

import pandas as pd


POSITIVE_BINARY_TOKENS = {
    "1",
    "y",
    "yes",
    "true",
    "是",
    "有",
    "阳性",
}
NEGATIVE_BINARY_TOKENS = {
    "0",
    "n",
    "no",
    "false",
    "否",
    "无",
    "阴性",
}
GN_RELATED_KEYWORDS = [
    "gn",
    "hmg",
    "fsh",
    "尿促",
    "果纳芬",
    "普丽康",
    "丽申宝",
    "乐芮",
]


def normalize_text_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    return normalized.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "None": pd.NA,
            "none": pd.NA,
            "NaT": pd.NA,
            "<NA>": pd.NA,
        }
    )


def normalize_identifier_series(series: pd.Series) -> pd.Series:
    return normalize_text_series(series)


def coerce_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def coerce_datetime_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    parsed = pd.to_datetime(series, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.notna()
    if numeric_mask.any():
        parsed.loc[numeric_mask] = pd.to_datetime(
            numeric.loc[numeric_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )
    return parsed


def coerce_binary_flag(series: pd.Series) -> pd.Series:
    normalized = normalize_text_series(series)
    numeric = pd.to_numeric(normalized, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="Int64")

    result.loc[numeric.eq(0)] = 0
    result.loc[numeric.eq(1)] = 1

    text_mask = normalized.notna() & result.isna()
    lowered = normalized[text_mask].str.lower()
    positive_mask = lowered.isin(POSITIVE_BINARY_TOKENS)
    negative_mask = lowered.isin(NEGATIVE_BINARY_TOKENS)
    result.loc[lowered.index[positive_mask]] = 1
    result.loc[lowered.index[negative_mask]] = 0
    return result


def contains_keywords(series: pd.Series, keywords: list[str]) -> pd.Series:
    normalized = normalize_text_series(series).fillna("")
    if not keywords:
        return pd.Series(False, index=series.index, dtype="boolean")
    pattern = "|".join(re.escape(keyword) for keyword in keywords)
    return normalized.str.contains(pattern, case=False, na=False, regex=True).astype("boolean")


def build_cycle_uid(cycle_id: pd.Series, art_id: pd.Series) -> pd.Series:
    cycle_part = normalize_identifier_series(cycle_id)
    art_part = normalize_identifier_series(art_id)
    result = pd.Series(pd.NA, index=cycle_part.index, dtype="string")

    both_mask = cycle_part.notna() & art_part.notna()
    art_only_mask = cycle_part.isna() & art_part.notna()
    cycle_only_mask = cycle_part.notna() & art_part.isna()

    result.loc[both_mask] = art_part.loc[both_mask] + "__" + cycle_part.loc[both_mask]
    result.loc[art_only_mask] = "art__" + art_part.loc[art_only_mask]
    result.loc[cycle_only_mask] = "cycle__" + cycle_part.loc[cycle_only_mask]
    return result


def normalize_follicle_laterality(series: pd.Series) -> pd.Series:
    normalized = normalize_text_series(series)
    result = normalized.copy()
    if normalized.empty:
        return result
    upper = normalized.fillna("").str.upper()
    result.loc[upper.isin(["1", "1.0"])] = "L"
    result.loc[upper.isin(["2", "2.0"])] = "R"
    result.loc[upper.str.contains("L|左", regex=True)] = "L"
    result.loc[upper.str.contains("R|右", regex=True)] = "R"
    return result


def normalize_primary_secondary_infertility(series: pd.Series) -> pd.Series:
    normalized = normalize_text_series(series)
    result = pd.Series(pd.NA, index=series.index, dtype="string")
    filled = normalized.fillna("")
    result.loc[filled.str.contains("原发", regex=False)] = "原发不孕"
    result.loc[filled.str.contains("继发", regex=False)] = "继发不孕"
    return result


def _diagnosis_has_male_marker(series: pd.Series) -> pd.Series:
    normalized = normalize_text_series(series).fillna("")
    return (
        normalized.str.contains("\(\u592b\)", na=False, regex=True)
        | normalized.str.contains("\uff08\u592b\uff09", na=False, regex=True)
    ).astype("boolean")


def _derive_male_factor_flag(frame: pd.DataFrame) -> pd.Series:
    diagnosis_columns = [column for column in ["diagnosis_2", "diagnosis_3", "diagnosis_4", "diagnosis_5", "diagnosis_6"] if column in frame.columns]
    if not diagnosis_columns:
        return pd.Series(pd.NA, index=frame.index, dtype="Int64")

    flag = pd.Series(False, index=frame.index, dtype="boolean")
    for column in diagnosis_columns:
        flag = flag | _diagnosis_has_male_marker(frame[column])
    return flag.astype("Int64")


def _build_male_factor_source_text(frame: pd.DataFrame) -> pd.Series:
    diagnosis_columns = [column for column in ["diagnosis_2", "diagnosis_3", "diagnosis_4", "diagnosis_5", "diagnosis_6"] if column in frame.columns]
    if not diagnosis_columns:
        return pd.Series(pd.NA, index=frame.index, dtype="string")

    def _join_row(row: pd.Series) -> str | pd._libs.missing.NAType:
        parts = []
        for value in row:
            if pd.isna(value):
                continue
            text = str(value).strip()
            if not text:
                continue
            if "(夫)" in text or "（夫）" in text:
                parts.append(text)
        return "；".join(parts) if parts else pd.NA

    source_text = frame[diagnosis_columns].apply(_join_row, axis=1).astype("string")
    return source_text.replace({"": pd.NA})


def _derive_sperm_source_group(series: pd.Series) -> pd.Series:
    normalized = normalize_text_series(series)
    result = pd.Series(pd.NA, index=series.index, dtype="string")
    if normalized.empty:
        return result

    upper = normalized.fillna("").str.upper()
    result.loc[upper.str.contains("PESA", na=False, regex=False)] = "PESA"
    result.loc[upper.str.contains("TESA", na=False, regex=False)] = "TESA"
    result.loc[upper.str.contains("供精", na=False, regex=False)] = "供精"
    self_mask = result.isna() & normalized.notna()
    result.loc[self_mask] = "自体"
    return result




def is_gn_related_medication(name: pd.Series, medication_class: pd.Series) -> pd.Series:
    name_match = contains_keywords(name, GN_RELATED_KEYWORDS)
    class_match = contains_keywords(medication_class, GN_RELATED_KEYWORDS)
    return (name_match | class_match).astype("boolean")


def _resolve_configured_series(
    df: pd.DataFrame,
    configured_value: str | list[str],
) -> tuple[pd.Series, str | None]:
    candidates = configured_value if isinstance(configured_value, list) else [configured_value]
    for candidate in candidates:
        if candidate in df.columns:
            return df[candidate].copy(), candidate
    return pd.Series(pd.NA, index=df.index), None


def _add_configured_columns(
    output_df: pd.DataFrame,
    source_df: pd.DataFrame,
    config_block: dict[str, Any],
) -> dict[str, str | None]:
    resolved_map: dict[str, str | None] = {}
    for standard_name, raw_value in config_block.items():
        series, raw_column = _resolve_configured_series(source_df, raw_value)
        output_df[standard_name] = series
        resolved_map[standard_name] = raw_column
    return resolved_map


def standardize_clinical_dataframe(
    clinical_df: pd.DataFrame,
    schema_config: dict[str, Any],
    cohort_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    standardized = pd.DataFrame(index=clinical_df.index)
    standardized["source_row_index"] = clinical_df.index.astype("int64")

    resolved_map: dict[str, str | None] = {}
    for section in ["keys", "baseline_fields", "context_fields", "targets"]:
        resolved_map.update(_add_configured_columns(standardized, clinical_df, schema_config.get(section, {})))

    standardized["cycle_id"] = normalize_identifier_series(standardized["cycle_id"])
    standardized["art_id"] = normalize_identifier_series(standardized["art_id"])
    standardized["cycle_uid"] = build_cycle_uid(standardized["cycle_id"], standardized["art_id"])

    numeric_fields = [
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
        "total_gn_dose",
        "total_gn_days",
        "et_embryo_count",
        "treatment_count",
        "fresh_treatment_count",
        "gn_start_cycle_day",
        "oocytes",
        "gv",
        "mi",
    ]
    for column in numeric_fields:
        if column in standardized.columns:
            standardized[column] = coerce_numeric_series(standardized[column])

    oocytes = standardized.get("oocytes", pd.Series(pd.NA, index=standardized.index, dtype="Float64"))
    gv = standardized.get("gv", pd.Series(pd.NA, index=standardized.index, dtype="Float64"))
    mi = standardized.get("mi", pd.Series(pd.NA, index=standardized.index, dtype="Float64"))
    standardized["mii"] = oocytes - gv - mi
    resolved_map["mii"] = "derived: 获卵数 - GV - MI"

    for column in ["procedure_date", "transfer_date", "gn_start_date"]:
        if column in standardized.columns:
            standardized[column] = coerce_datetime_series(standardized[column])

    for column in ["protocol", "treatment_plan", "diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4", "diagnosis_5", "diagnosis_6", "fertilization_method", "complication_text"]:
        if column in standardized.columns:
            standardized[column] = normalize_text_series(standardized[column])

    for column in ["clinical_pregnancy", "preterm_birth", "term_birth"]:
        if column in standardized.columns:
            standardized[column] = coerce_binary_flag(standardized[column])
    preterm_birth = standardized.get("preterm_birth", pd.Series(pd.NA, index=standardized.index, dtype="Int64"))
    term_birth = standardized.get("term_birth", pd.Series(pd.NA, index=standardized.index, dtype="Int64"))
    standardized["live_birth"] = ((preterm_birth.fillna(0) + term_birth.fillna(0)) > 0).astype("Int64")
    standardized["diagnosis_primary_secondary"] = normalize_primary_secondary_infertility(
        standardized.get("diagnosis_1", pd.Series(pd.NA, index=standardized.index, dtype="string"))
    )
    standardized["male_factor_infertility_flag"] = _derive_male_factor_flag(standardized)
    standardized["male_factor_source_text"] = _build_male_factor_source_text(standardized)
    standardized["sperm_source_group"] = _derive_sperm_source_group(
        standardized.get("treatment_plan", pd.Series(pd.NA, index=standardized.index, dtype="string"))
    )
    resolved_map["male_factor_infertility_flag"] = "derived: 诊断2-6"
    resolved_map["male_factor_source_text"] = "derived: 诊断2-6"
    resolved_map["sperm_source_group"] = "derived: 治疗方案"

    inclusion = cohort_config.get("inclusion", {})
    source_metadata = cohort_config.get("source_metadata", {})
    source_year_range = source_metadata.get("source_year_range", [])
    source_year_min = int(source_year_range[0]) if len(source_year_range) > 0 else None
    source_year_max = int(source_year_range[1]) if len(source_year_range) > 1 else None

    procedure_year = standardized["procedure_date"].dt.year.astype("Int64")
    standardized["procedure_year"] = procedure_year
    if source_year_min is not None and source_year_max is not None:
        standardized["within_source_year_range"] = (
            procedure_year.notna() & procedure_year.ge(source_year_min) & procedure_year.le(source_year_max)
        ).astype("boolean")
    else:
        standardized["within_source_year_range"] = pd.Series(pd.NA, index=standardized.index, dtype="boolean")

    protocol_exact_values = [str(item) for item in inclusion.get("protocol_exact_values", [])]
    if protocol_exact_values:
        standardized["protocol_matches_official_protocol"] = standardized["protocol"].isin(protocol_exact_values).astype(
            "boolean"
        )
    else:
        standardized["protocol_matches_official_protocol"] = contains_keywords(
            standardized["protocol"],
            list(inclusion.get("protocol_keywords", [])),
        )
    standardized["protocol_matches_long_protocol"] = standardized["protocol_matches_official_protocol"]

    treatment_plan_allowed_values = [str(item) for item in inclusion.get("treatment_plan_allowed_values", [])]
    if treatment_plan_allowed_values:
        standardized["treatment_plan_is_ivf_icsi_scope"] = standardized["treatment_plan"].isin(
            treatment_plan_allowed_values
        ).astype("boolean")
    else:
        standardized["treatment_plan_is_ivf_icsi_scope"] = contains_keywords(
            standardized["treatment_plan"],
            ["ivf", "icsi"],
        )
    standardized["fertilization_is_ivf_or_icsi"] = contains_keywords(
        standardized["fertilization_method"],
        ["ivf", "icsi"],
    )
    standardized["has_core_identifier"] = (
        standardized["cycle_uid"].notna() & standardized["cycle_id"].notna() & standardized["art_id"].notna()
    ).astype("boolean")

    first_treatment_field = str(inclusion.get("first_treatment_field", "treatment_count"))
    first_treatment_value = inclusion.get("first_treatment_value", 1)
    if first_treatment_field in standardized.columns:
        standardized["treatment_count_eq_1_flag"] = standardized[first_treatment_field].eq(first_treatment_value).astype(
            "boolean"
        )
    else:
        standardized["treatment_count_eq_1_flag"] = pd.Series(pd.NA, index=standardized.index, dtype="boolean")
    standardized["first_stimulation_cycle_flag"] = standardized["treatment_count_eq_1_flag"]
    standardized["cohort_protocol_only_eligible"] = (
        standardized["has_core_identifier"]
        & standardized["treatment_plan_is_ivf_icsi_scope"]
        & standardized["protocol_matches_official_protocol"]
    ).astype("boolean")
    standardized["cohort_strict_eligible"] = (
        standardized["cohort_protocol_only_eligible"]
        & standardized["treatment_count_eq_1_flag"]
    ).astype("boolean")

    column_order = [
        "cycle_uid",
        "cycle_id",
        "art_id",
        "source_row_index",
        "procedure_date",
        "procedure_year",
        "transfer_date",
        "et_embryo_count",
        "protocol",
        "treatment_plan",
        "treatment_count",
        "fresh_treatment_count",
        "diagnosis_1",
        "diagnosis_2",
        "diagnosis_3",
        "diagnosis_4",
        "diagnosis_5",
        "diagnosis_6",
        "diagnosis_primary_secondary",
        "male_factor_infertility_flag",
        "male_factor_source_text",
        "sperm_source_group",
        "male_age",
        "fertilization_method",
        "complication_text",
        "age",
        "bmi",
        "infertility_duration",
        "afc",
        "amh",
        "fsh",
        "lh",
        "e2",
        "p",
        "initial_gn_dose",
        "gn_start_date",
        "gn_start_cycle_day",
        "total_gn_dose",
        "total_gn_days",
        "oocytes",
        "gv",
        "mi",
        "mii",
        "clinical_pregnancy",
        "preterm_birth",
        "term_birth",
        "live_birth",
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
    ]
    standardized = standardized[[column for column in column_order if column in standardized.columns]]
    return standardized, resolved_map


def build_monitoring_base_keys(
    first_sheet_df: pd.DataFrame,
    schema_config: dict[str, Any],
) -> pd.DataFrame:
    base = pd.DataFrame(index=first_sheet_df.index)
    base["source_row_index"] = first_sheet_df.index.astype("int64")
    cycle_id, _ = _resolve_configured_series(first_sheet_df, schema_config.get("keys", {}).get("cycle_id", []))
    art_id, _ = _resolve_configured_series(first_sheet_df, schema_config.get("keys", {}).get("art_id", []))
    base["cycle_id"] = normalize_identifier_series(cycle_id)
    base["art_id"] = normalize_identifier_series(art_id)
    base["cycle_uid"] = build_cycle_uid(base["cycle_id"], base["art_id"])
    return base


def standardize_monitoring_visit_dataframe(
    monitoring_df: pd.DataFrame,
    base_keys: pd.DataFrame,
    schema_config: dict[str, Any],
    sheet_name: str,
    sheet_index: int,
) -> pd.DataFrame:
    standardized = pd.DataFrame(index=monitoring_df.index)
    standardized["source_sheet_name"] = sheet_name
    standardized["source_sheet_index"] = sheet_index
    standardized["source_row_index"] = monitoring_df.index.astype("int64")

    aligned_base = base_keys.reindex(monitoring_df.index)
    key_config = schema_config.get("keys", {})

    for key_name in ["cycle_id", "art_id"]:
        raw_series, raw_column = _resolve_configured_series(monitoring_df, key_config.get(key_name, []))
        normalized = normalize_identifier_series(raw_series)
        inherited_mask = normalized.isna()
        if raw_column is None:
            standardized[key_name] = aligned_base[key_name]
            standardized[f"{key_name}_inherited_from_sheet1"] = pd.Series(True, index=monitoring_df.index, dtype="boolean")
        else:
            standardized[key_name] = normalized.fillna(aligned_base[key_name])
            standardized[f"{key_name}_inherited_from_sheet1"] = inherited_mask.astype("boolean")

    for key_name in ["monitoring_date", "monitoring_order"]:
        raw_series, _ = _resolve_configured_series(monitoring_df, key_config.get(key_name, []))
        standardized[key_name] = raw_series

    _add_configured_columns(standardized, monitoring_df, schema_config.get("anchor_fields", {}))
    _add_configured_columns(standardized, monitoring_df, schema_config.get("core_fields", {}))

    standardized["cycle_uid"] = build_cycle_uid(standardized["cycle_id"], standardized["art_id"])
    standardized["monitoring_date"] = coerce_datetime_series(standardized["monitoring_date"])
    standardized["monitoring_order"] = coerce_numeric_series(standardized["monitoring_order"]).round().astype("Int64")
    standardized.loc[standardized["monitoring_order"].isna(), "monitoring_order"] = sheet_index

    if "lmp_date" in standardized.columns:
        standardized["lmp_date"] = coerce_datetime_series(standardized["lmp_date"])

    for column in ["days_from_lmp", "interval_days", "cycle_day", "gn_day", "e2", "lh", "p", "fsh", "endometrium"]:
        if column in standardized.columns:
            standardized[column] = coerce_numeric_series(standardized[column])

    standardized["canonical_visit_key"] = (
        standardized["cycle_uid"].fillna("missing_cycle")
        + "__m"
        + standardized["monitoring_order"].astype("string").fillna("missing_order")
    )
    standardized["visit_uid"] = (
        standardized["canonical_visit_key"] + "__s" + str(sheet_index) + "__r" + standardized["source_row_index"].astype("string")
    )
    standardized["is_active_visit"] = (
        standardized["cycle_uid"].notna() & standardized["monitoring_date"].notna()
    ).astype("boolean")

    column_order = [
        "visit_uid",
        "canonical_visit_key",
        "cycle_uid",
        "cycle_id",
        "art_id",
        "source_sheet_name",
        "source_sheet_index",
        "source_row_index",
        "cycle_id_inherited_from_sheet1",
        "art_id_inherited_from_sheet1",
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
        "is_active_visit",
    ]
    return standardized[[column for column in column_order if column in standardized.columns]]


def extract_monitoring_medications(
    monitoring_df: pd.DataFrame,
    visit_df: pd.DataFrame,
    schema_config: dict[str, Any],
) -> pd.DataFrame:
    active_visits = visit_df[visit_df["is_active_visit"]].copy()
    if active_visits.empty:
        return pd.DataFrame(
            columns=[
                "visit_uid",
                "canonical_visit_key",
                "cycle_uid",
                "cycle_id",
                "art_id",
                "source_sheet_name",
                "source_sheet_index",
                "source_row_index",
                "monitoring_order",
                "monitoring_date",
                "medication_slot",
                "medication_name",
                "medication_class",
                "medication_unit",
                "medication_value",
                "medication_frequency",
                "medication_days",
                "medication_total_amount",
                "is_gn_related",
            ]
        )

    pattern = schema_config.get("medication_pattern", {})
    max_index = int(pattern.get("max_index", 0))
    rows: list[pd.DataFrame] = []

    for slot in range(1, max_index + 1):
        frame = active_visits[
            [
                "visit_uid",
                "canonical_visit_key",
                "cycle_uid",
                "cycle_id",
                "art_id",
                "source_sheet_name",
                "source_sheet_index",
                "source_row_index",
                "monitoring_order",
                "monitoring_date",
            ]
        ].copy()
        frame["medication_slot"] = slot
        field_templates = {
            "medication_name": pattern.get("name_field"),
            "medication_class": pattern.get("class_field"),
            "medication_unit": pattern.get("unit_field"),
            "medication_value": pattern.get("value_field"),
            "medication_frequency": pattern.get("frequency_field"),
            "medication_days": pattern.get("days_field"),
            "medication_total_amount": pattern.get("total_amount_field"),
        }
        for standard_name, template in field_templates.items():
            if template and template.format(i=slot) in monitoring_df.columns:
                frame[standard_name] = monitoring_df.loc[active_visits.index, template.format(i=slot)].values
            else:
                frame[standard_name] = pd.NA

        for column in ["medication_name", "medication_class", "medication_unit", "medication_frequency"]:
            frame[column] = normalize_text_series(frame[column])
        for column in ["medication_value", "medication_days", "medication_total_amount"]:
            frame[column] = coerce_numeric_series(frame[column])
        frame["is_gn_related"] = is_gn_related_medication(frame["medication_name"], frame["medication_class"])

        has_content = frame[
            [
                "medication_name",
                "medication_class",
                "medication_value",
                "medication_days",
                "medication_total_amount",
            ]
        ].notna().any(axis=1)
        rows.append(frame.loc[has_content])

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def extract_monitoring_follicles(
    monitoring_df: pd.DataFrame,
    visit_df: pd.DataFrame,
    schema_config: dict[str, Any],
) -> pd.DataFrame:
    active_visits = visit_df[visit_df["is_active_visit"]].copy()
    if active_visits.empty:
        return pd.DataFrame(
            columns=[
                "visit_uid",
                "canonical_visit_key",
                "cycle_uid",
                "cycle_id",
                "art_id",
                "source_sheet_name",
                "source_sheet_index",
                "source_row_index",
                "monitoring_order",
                "monitoring_date",
                "follicle_slot",
                "follicle_laterality",
                "follicle_order",
                "follicle_long_axis",
                "follicle_short_axis",
                "follicle_count",
                "follicle_mean_diameter",
            ]
        )

    pattern = schema_config.get("follicle_detail_pattern", {})
    max_index = int(pattern.get("max_index", 0))
    rows: list[pd.DataFrame] = []

    for slot in range(1, max_index + 1):
        frame = active_visits[
            [
                "visit_uid",
                "canonical_visit_key",
                "cycle_uid",
                "cycle_id",
                "art_id",
                "source_sheet_name",
                "source_sheet_index",
                "source_row_index",
                "monitoring_order",
                "monitoring_date",
            ]
        ].copy()
        frame["follicle_slot"] = slot
        field_templates = {
            "follicle_laterality": pattern.get("lateral_field"),
            "follicle_order": pattern.get("order_field"),
            "follicle_long_axis": pattern.get("long_axis_field"),
            "follicle_short_axis": pattern.get("short_axis_field"),
            "follicle_count": pattern.get("count_field"),
            "follicle_mean_diameter": pattern.get("mean_diameter_field"),
        }
        for standard_name, template in field_templates.items():
            if template and template.format(i=slot) in monitoring_df.columns:
                frame[standard_name] = monitoring_df.loc[active_visits.index, template.format(i=slot)].values
            else:
                frame[standard_name] = pd.NA

        frame["follicle_laterality"] = normalize_follicle_laterality(frame["follicle_laterality"])
        for column in [
            "follicle_order",
            "follicle_long_axis",
            "follicle_short_axis",
            "follicle_count",
            "follicle_mean_diameter",
        ]:
            frame[column] = coerce_numeric_series(frame[column])

        has_content = frame[
            [
                "follicle_laterality",
                "follicle_order",
                "follicle_long_axis",
                "follicle_short_axis",
                "follicle_count",
                "follicle_mean_diameter",
            ]
        ].notna().any(axis=1)
        rows.append(frame.loc[has_content])

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
