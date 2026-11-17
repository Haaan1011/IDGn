from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.simple_yaml import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SuccessfulCycleBuildResult:
    cycle_labels_path: Path
    incidence_path: Path
    split_distribution_path: Path
    trigger_audit_path: Path
    missingness_audit_path: Path
    semantics_audit_path: Path
    sensitivity_audit_path: Path
    build_summary_path: Path
    formal_cohort_cycles: int
    trigger_cycles: int
    exact_evaluable_cycles: int
    exact_positive_cycles: int


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _boolish(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().isin({"1", "true", "yes", "y", "是"})


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _nullable_binary(value: pd.Series, evaluable: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=value.index, dtype="Int64")
    result.loc[evaluable] = value.loc[evaluable].astype(int)
    return result


def _require_columns(frame: pd.DataFrame, columns: set[str], frame_name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise KeyError(f"{frame_name} missing required columns: {missing}")


def identify_trigger_medications(
    medications: pd.DataFrame,
    trigger_config: dict[str, Any],
) -> pd.Series:
    mask = pd.Series(False, index=medications.index)
    accepted_classes = {
        str(value).strip().lower() for value in trigger_config.get("accepted_classes", [])
    }
    for field in trigger_config.get("class_fields", []):
        if field in medications.columns:
            normalized = medications[field].fillna("").astype(str).str.strip().str.lower()
            mask |= normalized.isin(accepted_classes)

    name_field = str(trigger_config.get("name_field", "medication_name"))
    if name_field in medications.columns:
        accepted_names = {
            str(value).strip().lower() for value in trigger_config.get("accepted_names", [])
        }
        normalized_names = medications[name_field].fillna("").astype(str).str.strip().str.lower()
        mask |= normalized_names.isin(accepted_names)
    return mask


def select_trigger_visits(
    medications: pd.DataFrame,
    clinical: pd.DataFrame,
    trigger_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(
        medications,
        {
            "cycle_uid",
            "visit_uid",
            "canonical_visit_key",
            "monitoring_order",
            "monitoring_date",
            "medication_name",
        },
        "monitoring medications",
    )
    _require_columns(clinical, {"cycle_uid", "procedure_date"}, "clinical")

    trigger_rows = medications.loc[identify_trigger_medications(medications, trigger_config)].copy()
    trigger_rows["monitoring_date"] = pd.to_datetime(
        trigger_rows["monitoring_date"], errors="coerce"
    )

    visit_keys = [
        "cycle_uid",
        "visit_uid",
        "canonical_visit_key",
        "monitoring_order",
        "monitoring_date",
    ]
    trigger_visits = (
        trigger_rows.groupby(visit_keys, dropna=False)
        .agg(
            trigger_medication_row_count=("medication_name", "size"),
            trigger_medication_names=(
                "medication_name",
                lambda values: "|".join(sorted(set(values.dropna().astype(str)))),
            ),
        )
        .reset_index()
    )
    class_fields = [
        field for field in trigger_config.get("class_fields", []) if field in trigger_rows.columns
    ]
    if class_fields:
        class_text = trigger_rows[class_fields].fillna("").astype(str).agg("|".join, axis=1)
        trigger_rows = trigger_rows.assign(_trigger_class_text=class_text)
        classes = (
            trigger_rows.groupby(visit_keys, dropna=False)["_trigger_class_text"]
            .agg(lambda values: "|".join(sorted({item for item in values if item})))
            .rename("trigger_medication_classes")
            .reset_index()
        )
        trigger_visits = trigger_visits.merge(classes, on=visit_keys, how="left")
    else:
        trigger_visits["trigger_medication_classes"] = ""

    procedure = clinical[["cycle_uid", "procedure_date"]].copy()
    procedure["procedure_date"] = pd.to_datetime(procedure["procedure_date"], errors="coerce")
    trigger_visits = trigger_visits.merge(
        procedure, on="cycle_uid", how="left", validate="many_to_one"
    )
    trigger_visits["days_trigger_to_procedure"] = (
        trigger_visits["procedure_date"] - trigger_visits["monitoring_date"]
    ).dt.days
    visit_counts = trigger_visits.groupby("cycle_uid")["visit_uid"].nunique()
    trigger_visits["trigger_visit_count_in_cycle"] = (
        trigger_visits["cycle_uid"].map(visit_counts).astype("Int64")
    )

    selected = (
        trigger_visits.sort_values(
            ["cycle_uid", "monitoring_order", "monitoring_date", "visit_uid"],
            na_position="first",
        )
        .groupby("cycle_uid", as_index=False)
        .tail(1)
        .copy()
    )
    selected["trigger_selection_rule"] = np.where(
        selected["trigger_visit_count_in_cycle"].gt(1),
        "maximum_monitoring_order_among_multiple_hcg_visits",
        "single_hcg_visit",
    )
    selected = selected.rename(
        columns={
            "visit_uid": "trigger_visit_uid",
            "canonical_visit_key": "trigger_canonical_visit_key",
            "monitoring_order": "trigger_monitoring_order",
            "monitoring_date": "trigger_monitoring_date",
        }
    )
    return selected.reset_index(drop=True), trigger_visits.reset_index(drop=True)


def aggregate_follicles(
    follicles: pd.DataFrame,
    follicle_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(
        follicles,
        {
            "cycle_uid",
            "visit_uid",
            "canonical_visit_key",
            "monitoring_order",
            "monitoring_date",
            str(follicle_config["diameter_field"]),
            str(follicle_config["count_field"]),
        },
        "monitoring follicles",
    )
    frame = follicles.copy()
    diameter_field = str(follicle_config["diameter_field"])
    fallback_fields = [
        field for field in follicle_config.get("fallback_diameter_fields", []) if field in frame
    ]
    count_field = str(follicle_config["count_field"])
    frame["effective_diameter"] = _numeric(frame[diameter_field])
    if fallback_fields:
        fallback = frame[fallback_fields].apply(_numeric).mean(axis=1)
        frame["effective_diameter"] = frame["effective_diameter"].fillna(fallback)
    frame["effective_follicle_count"] = _numeric(frame[count_field])
    frame["monitoring_date"] = pd.to_datetime(frame["monitoring_date"], errors="coerce")

    valid_count = frame["effective_follicle_count"].notna() & frame[
        "effective_follicle_count"
    ].ge(0)
    frame["count_for_aggregation"] = frame["effective_follicle_count"].where(valid_count)
    non_integer = valid_count & (
        frame["effective_follicle_count"] - frame["effective_follicle_count"].round()
    ).abs().gt(1e-9)

    keys = ["cycle_uid", "visit_uid", "canonical_visit_key"]
    visit = (
        frame.groupby(keys, dropna=False)
        .agg(
            monitoring_order=("monitoring_order", "max"),
            monitoring_date=("monitoring_date", "max"),
            follicle_row_count=("effective_follicle_count", "size"),
            valid_diameter_row_count=("effective_diameter", lambda values: int(values.notna().sum())),
            missing_diameter_row_count=("effective_diameter", lambda values: int(values.isna().sum())),
            invalid_count_row_count=(
                "effective_follicle_count",
                lambda values: int(values.isna().sum() + values.lt(0).sum()),
            ),
            total_follicle_count=("count_for_aggregation", "sum"),
            max_effective_diameter=("effective_diameter", "max"),
            max_grouped_follicle_count=("count_for_aggregation", "max"),
        )
        .reset_index()
    )
    non_integer_counts = (
        frame.loc[non_integer].groupby(keys).size().rename("non_integer_count_row_count")
    )
    visit = visit.merge(non_integer_counts.reset_index(), on=keys, how="left")
    visit["non_integer_count_row_count"] = (
        visit["non_integer_count_row_count"].fillna(0).astype(int)
    )

    threshold = float(follicle_config["threshold_mm"])
    plausible_max = float(follicle_config["plausible_max_diameter_mm_sensitivity"])
    masks = {
        "follicles_ge14": frame["effective_diameter"].ge(threshold),
        "follicles_gt14": frame["effective_diameter"].gt(threshold),
        "follicles_ge13p5": frame["effective_diameter"].ge(
            float(follicle_config["threshold_sensitivity_mm"][0])
        ),
        "follicles_ge14_le40": frame["effective_diameter"].ge(threshold)
        & frame["effective_diameter"].le(plausible_max),
    }
    for column, mask in masks.items():
        counts = (
            frame.loc[mask]
            .groupby(keys)["count_for_aggregation"]
            .sum()
            .rename(column)
            .reset_index()
        )
        visit = visit.merge(counts, on=keys, how="left")
        visit[column] = visit[column].fillna(0.0)
    return visit, frame


def _latest_visit(
    cycle_visits: pd.DataFrame,
    before_date: pd.Timestamp | None = None,
    before_order: float | int | None = None,
    allow_same_visit: bool = False,
) -> pd.Series | None:
    candidates = cycle_visits
    if before_date is not None and pd.notna(before_date):
        if allow_same_visit:
            date_mask = candidates["monitoring_date"].le(before_date)
        else:
            date_mask = candidates["monitoring_date"].lt(before_date)
        if before_order is not None and pd.notna(before_order):
            same_day = candidates["monitoring_date"].eq(before_date)
            order_mask = candidates["monitoring_order"].le(before_order)
            if not allow_same_visit:
                order_mask = candidates["monitoring_order"].lt(before_order)
            date_mask |= same_day & order_mask
        candidates = candidates.loc[date_mask]
    if candidates.empty:
        return None
    return candidates.sort_values(
        ["monitoring_date", "monitoring_order", "visit_uid"], na_position="first"
    ).iloc[-1]


def build_cycle_labels(
    clinical: pd.DataFrame,
    visits: pd.DataFrame,
    medications: pd.DataFrame,
    follicles: pd.DataFrame,
    split_manifest: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    cohort_flag = str(config["cohort_flag"])
    _require_columns(
        clinical,
        {
            "cycle_uid",
            "cycle_id",
            "art_id",
            "procedure_date",
            "complication_text",
            "oocytes",
            "mii",
            "clinical_pregnancy",
            "live_birth",
            cohort_flag,
        },
        "clinical",
    )
    _require_columns(split_manifest, {"cycle_uid", "split"}, "split manifest")
    _require_columns(visits, {"cycle_uid", "visit_uid", "canonical_visit_key"}, "monitoring visits")

    formal = clinical.loc[_boolish(clinical[cohort_flag])].copy()
    if formal.empty:
        raise ValueError("Formal cohort is empty")
    if formal["cycle_uid"].duplicated().any():
        raise ValueError("Formal clinical cohort must contain one row per cycle_uid")

    formal_cycles = set(formal["cycle_uid"])
    medications = medications.loc[medications["cycle_uid"].isin(formal_cycles)].copy()
    follicles = follicles.loc[follicles["cycle_uid"].isin(formal_cycles)].copy()
    visits = visits.loc[visits["cycle_uid"].isin(formal_cycles)].copy()

    selected_trigger, all_trigger_visits = select_trigger_visits(
        medications, formal, config["trigger_detection"]
    )
    follicle_visits, follicle_rows = aggregate_follicles(
        follicles, config["follicle_counting"]
    )
    exact = selected_trigger.merge(
        follicle_visits,
        left_on=["cycle_uid", "trigger_visit_uid", "trigger_canonical_visit_key"],
        right_on=["cycle_uid", "visit_uid", "canonical_visit_key"],
        how="left",
        validate="one_to_one",
    )
    exact["same_visit_follicle_match"] = exact["follicle_row_count"].notna()
    exact["exact_evaluable"] = (
        exact["same_visit_follicle_match"]
        & exact["valid_diameter_row_count"].fillna(0).gt(0)
        & exact["invalid_count_row_count"].fillna(0).eq(0)
        & exact["non_integer_count_row_count"].fillna(0).eq(0)
    )

    follicle_groups = {
        cycle_uid: group.copy()
        for cycle_uid, group in follicle_visits.groupby("cycle_uid", sort=False)
    }
    selected_by_cycle = exact.set_index("cycle_uid", drop=False)
    proxy_rows: list[dict[str, Any]] = []
    prior_rows: list[dict[str, Any]] = []
    for row in formal.itertuples(index=False):
        cycle_uid = row.cycle_uid
        cycle_visits = follicle_groups.get(cycle_uid)
        if cycle_visits is None or cycle_visits.empty:
            continue

        selected = selected_by_cycle.loc[cycle_uid] if cycle_uid in selected_by_cycle.index else None
        if selected is not None and bool(selected["exact_evaluable"]):
            proxy_visit = cycle_visits.loc[
                cycle_visits["visit_uid"].eq(selected["trigger_visit_uid"])
            ].iloc[0]
            proxy_source = "exact_trigger_visit"
        elif selected is not None:
            proxy_visit = _latest_visit(
                cycle_visits,
                before_date=selected["trigger_monitoring_date"],
                before_order=selected["trigger_monitoring_order"],
                allow_same_visit=False,
            )
            proxy_source = "last_pre_trigger_proxy"
        else:
            procedure_date = pd.to_datetime(getattr(row, "procedure_date"), errors="coerce")
            proxy_visit = _latest_visit(
                cycle_visits,
                before_date=procedure_date,
                allow_same_visit=False,
            )
            proxy_source = "last_observed_before_procedure_proxy"
        if proxy_visit is not None:
            proxy_rows.append(
                {
                    "cycle_uid": cycle_uid,
                    "proxy_visit_uid": proxy_visit["visit_uid"],
                    "proxy_canonical_visit_key": proxy_visit["canonical_visit_key"],
                    "proxy_monitoring_order": proxy_visit["monitoring_order"],
                    "proxy_monitoring_date": proxy_visit["monitoring_date"],
                    "proxy_follicles_ge14": proxy_visit["follicles_ge14"],
                    "proxy_source": proxy_source,
                }
            )

        if selected is not None:
            prior_visit = _latest_visit(
                cycle_visits,
                before_date=selected["trigger_monitoring_date"],
                before_order=selected["trigger_monitoring_order"],
                allow_same_visit=False,
            )
            if prior_visit is not None:
                prior_rows.append(
                    {
                        "cycle_uid": cycle_uid,
                        "pre_trigger_visit_uid": prior_visit["visit_uid"],
                        "pre_trigger_follicles_ge14": prior_visit["follicles_ge14"],
                        "pre_trigger_gap_days": (
                            selected["trigger_monitoring_date"]
                            - prior_visit["monitoring_date"]
                        ).days,
                    }
                )

    proxy = pd.DataFrame(
        proxy_rows,
        columns=[
            "cycle_uid",
            "proxy_visit_uid",
            "proxy_canonical_visit_key",
            "proxy_monitoring_order",
            "proxy_monitoring_date",
            "proxy_follicles_ge14",
            "proxy_source",
        ],
    )
    prior = pd.DataFrame(
        prior_rows,
        columns=[
            "cycle_uid",
            "pre_trigger_visit_uid",
            "pre_trigger_follicles_ge14",
            "pre_trigger_gap_days",
        ],
    )
    labels = formal[
        [
            "cycle_uid",
            "cycle_id",
            "art_id",
            "procedure_date",
            "complication_text",
            "oocytes",
            "mii",
            "clinical_pregnancy",
            "live_birth",
        ]
    ].copy()
    labels = labels.merge(
        split_manifest[["cycle_uid", "split"]].drop_duplicates("cycle_uid"),
        on="cycle_uid",
        how="left",
        validate="one_to_one",
    )
    if labels["split"].isna().any():
        raise ValueError("Every formal cohort cycle must exist in the split manifest")

    exact_columns = [
        "cycle_uid",
        "trigger_visit_uid",
        "trigger_canonical_visit_key",
        "trigger_monitoring_order",
        "trigger_monitoring_date",
        "trigger_medication_row_count",
        "trigger_medication_names",
        "trigger_medication_classes",
        "trigger_visit_count_in_cycle",
        "trigger_selection_rule",
        "days_trigger_to_procedure",
        "same_visit_follicle_match",
        "exact_evaluable",
        "follicle_row_count",
        "valid_diameter_row_count",
        "missing_diameter_row_count",
        "invalid_count_row_count",
        "non_integer_count_row_count",
        "total_follicle_count",
        "max_effective_diameter",
        "max_grouped_follicle_count",
        "follicles_ge14",
        "follicles_gt14",
        "follicles_ge13p5",
        "follicles_ge14_le40",
    ]
    labels = labels.merge(exact[exact_columns], on="cycle_uid", how="left", validate="one_to_one")
    labels = labels.merge(proxy, on="cycle_uid", how="left", validate="one_to_one")
    labels = labels.merge(prior, on="cycle_uid", how="left", validate="one_to_one")

    lower = float(config["follicle_counting"]["lower_success_count"])
    upper = float(config["follicle_counting"]["upper_success_count"])
    exact_evaluable = labels["exact_evaluable"].eq(True)
    proxy_evaluable = labels["proxy_follicles_ge14"].notna()
    prior_evaluable = labels["pre_trigger_follicles_ge14"].notna()
    labels["kong_aligned_exact_success"] = _nullable_binary(
        labels["follicles_ge14"].between(lower, upper), exact_evaluable
    )
    labels["kong_aligned_last_observed_proxy_success"] = _nullable_binary(
        labels["proxy_follicles_ge14"].between(lower, upper), proxy_evaluable
    )
    labels["kong_strict_pre_trigger_proxy_success"] = _nullable_binary(
        labels["pre_trigger_follicles_ge14"].between(lower, upper), prior_evaluable
    )
    labels["kong_exact_gt14_sensitivity"] = _nullable_binary(
        labels["follicles_gt14"].between(lower, upper), exact_evaluable
    )
    labels["kong_exact_ge13p5_sensitivity"] = _nullable_binary(
        labels["follicles_ge13p5"].between(lower, upper), exact_evaluable
    )
    labels["kong_exact_plausible_diameter_sensitivity"] = _nullable_binary(
        labels["follicles_ge14_le40"].between(lower, upper), exact_evaluable
    )
    offset_evaluable = exact_evaluable & labels["days_trigger_to_procedure"].eq(
        int(config["trigger_detection"]["expected_days_before_procedure"])
    )
    labels["kong_exact_expected_offset_sensitivity"] = _nullable_binary(
        labels["follicles_ge14"].between(lower, upper), offset_evaluable
    )

    oocytes = _numeric(labels["oocytes"])
    mii = _numeric(labels["mii"])
    documented_ohss = labels["complication_text"].fillna("").astype(str).str.contains(
        str(config["project_composite"]["safety_primary"]["documented_ohss_keyword"]),
        case=False,
        regex=False,
    )
    labels["documented_ohss_flag"] = documented_ohss.astype(int)
    labels["retrieved_oocytes_gt20_flag"] = oocytes.gt(
        float(config["project_composite"]["safety_primary"]["high_response_threshold"])
    ).astype(int)
    labels["effectiveness_mii_positive_flag"] = mii.gt(
        float(config["project_composite"]["effectiveness"]["threshold"])
    ).astype(int)
    labels["safety_no_ohss_warning_flag"] = (
        ~documented_ohss
        & oocytes.le(
            float(config["project_composite"]["safety_primary"]["high_response_threshold"])
        )
    ).astype(int)
    labels["safety_no_documented_ohss_flag"] = (~documented_ohss).astype(int)
    labels["benefit_recorded_live_birth_flag"] = _numeric(labels["live_birth"]).eq(1).astype(int)
    labels["benefit_recorded_clinical_pregnancy_flag"] = (
        _numeric(labels["clinical_pregnancy"]).eq(1).astype(int)
    )

    labels["project_composite_success_secondary"] = (
        labels["effectiveness_mii_positive_flag"].eq(1)
        & labels["safety_no_ohss_warning_flag"].eq(1)
        & labels["benefit_recorded_live_birth_flag"].eq(1)
    ).astype(int)
    labels["project_composite_success_recorded_clinical_pregnancy"] = (
        labels["effectiveness_mii_positive_flag"].eq(1)
        & labels["safety_no_ohss_warning_flag"].eq(1)
        & labels["benefit_recorded_clinical_pregnancy_flag"].eq(1)
    ).astype(int)
    labels["project_composite_success_documented_ohss_only"] = (
        labels["effectiveness_mii_positive_flag"].eq(1)
        & labels["safety_no_documented_ohss_flag"].eq(1)
        & labels["benefit_recorded_live_birth_flag"].eq(1)
    ).astype(int)
    labels["project_composite_success_exact_process_anchored"] = _nullable_binary(
        labels["project_composite_success_secondary"].eq(1)
        & labels["kong_aligned_exact_success"].eq(1),
        labels["kong_aligned_exact_success"].notna(),
    )
    labels["project_composite_success_proxy_process_anchored"] = (
        labels["project_composite_success_secondary"].eq(1)
        & labels["kong_aligned_last_observed_proxy_success"].eq(1)
    ).astype(int)

    label_roles = {
        "kong_aligned_exact_success": "primary_kong_process",
        "kong_aligned_last_observed_proxy_success": "kong_proxy_sensitivity",
        "kong_strict_pre_trigger_proxy_success": "timing_sensitivity",
        "effectiveness_mii_positive_flag": "project_component",
        "safety_no_ohss_warning_flag": "project_component",
        "benefit_recorded_live_birth_flag": "project_component",
        "benefit_recorded_clinical_pregnancy_flag": "project_component_sensitivity",
        "project_composite_success_secondary": "secondary_project_composite",
        "project_composite_success_recorded_clinical_pregnancy": "project_sensitivity",
        "project_composite_success_documented_ohss_only": "project_sensitivity",
        "project_composite_success_exact_process_anchored": "project_sensitivity",
        "project_composite_success_proxy_process_anchored": "project_sensitivity",
    }
    incidence_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for label, role in label_roles.items():
        evaluable = labels[label].notna()
        positives = int(labels.loc[evaluable, label].sum())
        incidence_rows.append(
            {
                "label": label,
                "role": role,
                "denominator_cycles": int(evaluable.sum()),
                "positive_cycles": positives,
                "negative_cycles": int(evaluable.sum()) - positives,
                "positive_rate": float(labels.loc[evaluable, label].mean()),
                "missing_or_not_evaluable_cycles": int((~evaluable).sum()),
            }
        )
        for split_name in ["train", "valid", "test"]:
            split_mask = labels["split"].eq(split_name)
            split_evaluable = evaluable & split_mask
            split_positive = int(labels.loc[split_evaluable, label].sum())
            split_denominator = int(split_evaluable.sum())
            split_rows.append(
                {
                    "label": label,
                    "role": role,
                    "split": split_name,
                    "denominator_cycles": split_denominator,
                    "positive_cycles": split_positive,
                    "negative_cycles": split_denominator - split_positive,
                    "positive_rate": (
                        float(labels.loc[split_evaluable, label].mean())
                        if split_denominator
                        else np.nan
                    ),
                    "missing_or_not_evaluable_cycles": int((split_mask & ~evaluable).sum()),
                }
            )

    formal_n = len(labels)
    trigger_n = int(labels["trigger_visit_uid"].notna().sum())
    exact_n = int(labels["kong_aligned_exact_success"].notna().sum())
    trigger_visit_n = len(all_trigger_visits)
    dual_trigger_visits = int(
        all_trigger_visits["trigger_medication_row_count"].gt(1).sum()
    )
    multi_trigger_cycles = int(
        all_trigger_visits.groupby("cycle_uid")["visit_uid"].nunique().gt(1).sum()
    )
    visit_lookup = visits[
        ["visit_uid", "cycle_uid", "canonical_visit_key"]
    ].drop_duplicates("visit_uid")
    trigger_visit_link = all_trigger_visits.merge(
        visit_lookup.rename(
            columns={
                "cycle_uid": "visit_cycle_uid",
                "canonical_visit_key": "visit_canonical_visit_key",
            }
        ),
        on="visit_uid",
        how="left",
        validate="many_to_one",
    )
    trigger_visit_found = trigger_visit_link["visit_cycle_uid"].notna()
    trigger_cycle_match = (
        trigger_visit_found
        & trigger_visit_link["cycle_uid"].eq(trigger_visit_link["visit_cycle_uid"])
    )
    trigger_canonical_match = (
        trigger_visit_found
        & trigger_visit_link["canonical_visit_key"].eq(
            trigger_visit_link["visit_canonical_visit_key"]
        )
    )
    same_visit_follicle_matches = int(
        labels["same_visit_follicle_match"].eq(True).sum()
    )
    all_trigger_follicle_link = all_trigger_visits.merge(
        follicle_visits[
            ["cycle_uid", "visit_uid", "canonical_visit_key", "follicle_row_count"]
        ],
        on=["cycle_uid", "visit_uid", "canonical_visit_key"],
        how="left",
        validate="one_to_one",
    )
    all_trigger_follicle_matches = int(
        all_trigger_follicle_link["follicle_row_count"].notna().sum()
    )
    trigger_candidate_rows = medications.loc[
        identify_trigger_medications(medications, config["trigger_detection"])
    ].copy()
    duplicate_columns = [
        column
        for column in [
            "visit_uid",
            "medication_slot",
            "medication_name",
            "medication_class",
            "medication_category",
            "medication_value",
            "medication_unit",
            "medication_frequency",
            "medication_days",
        ]
        if column in trigger_candidate_rows.columns
    ]
    exact_duplicate_trigger_rows = int(
        trigger_candidate_rows.duplicated(duplicate_columns, keep=False).sum()
    )
    trigger_audit_rows = [
        ("formal_cohort_cycles", formal_n, formal_n, 1.0),
        (
            "structured_hcg_medication_rows",
            len(trigger_candidate_rows),
            formal_n,
            len(trigger_candidate_rows) / formal_n,
        ),
        ("cycles_with_structured_hcg_trigger", trigger_n, formal_n, trigger_n / formal_n),
        ("structured_hcg_trigger_visits", trigger_visit_n, formal_n, trigger_visit_n / formal_n),
        (
            "all_trigger_visits_with_same_visit_follicles",
            all_trigger_follicle_matches,
            trigger_visit_n,
            all_trigger_follicle_matches / trigger_visit_n,
        ),
        (
            "selected_trigger_cycles_with_same_visit_follicles",
            same_visit_follicle_matches,
            trigger_n,
            float(same_visit_follicle_matches / trigger_n),
        ),
        (
            "trigger_visits_found_in_standardized_visit_table",
            int(trigger_visit_found.sum()),
            trigger_visit_n,
            float(trigger_visit_found.mean()),
        ),
        (
            "trigger_visit_cycle_uid_matches_standardized_visit",
            int(trigger_cycle_match.sum()),
            trigger_visit_n,
            float(trigger_cycle_match.mean()),
        ),
        (
            "trigger_visit_canonical_key_matches_standardized_visit",
            int(trigger_canonical_match.sum()),
            trigger_visit_n,
            float(trigger_canonical_match.mean()),
        ),
        ("exact_evaluable_cycles", exact_n, formal_n, exact_n / formal_n),
        ("cycles_without_structured_hcg_trigger", formal_n - trigger_n, formal_n, (formal_n - trigger_n) / formal_n),
        ("selected_trigger_cycles_without_follicles", trigger_n - exact_n, trigger_n, (trigger_n - exact_n) / trigger_n),
        ("same_visit_dual_or_multi_hcg_medication_visits", dual_trigger_visits, trigger_visit_n, dual_trigger_visits / trigger_visit_n),
        (
            "exact_duplicate_hcg_medication_rows",
            exact_duplicate_trigger_rows,
            len(trigger_candidate_rows),
            exact_duplicate_trigger_rows / len(trigger_candidate_rows),
        ),
        ("cycles_with_multiple_hcg_visits", multi_trigger_cycles, formal_n, multi_trigger_cycles / formal_n),
        (
            "selected_trigger_exactly_2_days_before_procedure",
            int(labels["days_trigger_to_procedure"].eq(2).sum()),
            trigger_n,
            float(labels["days_trigger_to_procedure"].eq(2).sum() / trigger_n),
        ),
        (
            "selected_trigger_non_2_day_offset",
            int(labels["days_trigger_to_procedure"].notna().sum() - labels["days_trigger_to_procedure"].eq(2).sum()),
            trigger_n,
            float(
                (
                    labels["days_trigger_to_procedure"].notna().sum()
                    - labels["days_trigger_to_procedure"].eq(2).sum()
                )
                / trigger_n
            ),
        ),
        (
            "trigger_visits_with_diameter_above_40mm",
            int(labels["max_effective_diameter"].gt(40).sum()),
            exact_n,
            float(labels["max_effective_diameter"].gt(40).sum() / exact_n),
        ),
        (
            "trigger_visits_with_grouped_count_above_20",
            int(labels["max_grouped_follicle_count"].gt(20).sum()),
            exact_n,
            float(labels["max_grouped_follicle_count"].gt(20).sum() / exact_n),
        ),
    ]
    for offset, count in (
        labels.loc[labels["trigger_visit_uid"].notna(), "days_trigger_to_procedure"]
        .value_counts(dropna=False)
        .sort_index()
        .items()
    ):
        offset_name = "missing" if pd.isna(offset) else str(int(offset))
        trigger_audit_rows.append(
            (
                f"selected_trigger_offset_days_{offset_name}",
                int(count),
                trigger_n,
                int(count) / trigger_n,
            )
        )
    trigger_audit = pd.DataFrame(
        trigger_audit_rows, columns=["metric", "numerator", "denominator", "rate"]
    )

    missingness_rows = []
    for field, source, interpretation in [
        ("trigger_visit_uid", "monitoring_medications_long", "missing means no structured HCG trigger was found"),
        ("follicle_row_count", "monitoring_follicles_long", "missing means no same-trigger-visit follicle rows"),
        ("follicles_ge14", "monitoring_follicles_long", "missing means Kong exact label is not evaluable"),
        ("proxy_follicles_ge14", "monitoring_follicles_long", "fallback proxy count; not exact trigger-day evidence"),
        ("oocytes", "clinical_standardized", "retrieved-oocyte count used only in project safety label"),
        ("mii", "clinical_standardized", "MII count used only in project effectiveness label"),
        ("complication_text", "clinical_standardized", "missing is treated as no documented OHSS, not proof that OHSS did not occur"),
        ("clinical_pregnancy", "clinical_standardized", "1 is a recorded positive; missing is no recorded positive, not an explicit negative"),
        ("live_birth", "clinical_standardized", "derived binary from preterm_birth + term_birth > 0"),
        ("split", "split_manifest_v1", "patient/cycle-level shared split"),
    ]:
        missingness_rows.append(
            {
                "field": field,
                "source": source,
                "total_cycles": formal_n,
                "observed_cycles": int(labels[field].notna().sum()),
                "missing_cycles": int(labels[field].isna().sum()),
                "missing_rate": float(labels[field].isna().mean()),
                "interpretation": interpretation,
            }
        )

    semantics = pd.DataFrame(
        [
            {
                "concept": "Kong process success",
                "source_table": "monitoring_medications_long + monitoring_follicles_long",
                "source_field": "medication_class/category; medication_name; follicle_mean_diameter; follicle_count",
                "standardized_rule": "last structured HCG visit by maximum monitoring_order; same visit effective diameter >=14 mm; sum follicle_count",
                "output_field": "kong_aligned_exact_success",
                "status": "primary_exact_subset",
                "caveat": "not evaluable when structured trigger or same-visit follicles are absent",
            },
            {
                "concept": "Kong fallback proxy",
                "source_table": "monitoring_follicles_long",
                "source_field": "follicle_mean_diameter; follicle_count",
                "standardized_rule": "exact visit when available; otherwise last pre-trigger or last observed before procedure",
                "output_field": "kong_aligned_last_observed_proxy_success",
                "status": "sensitivity_only",
                "caveat": "proxy rows must never be described as exact trigger-day observations",
            },
            {
                "concept": "Project effectiveness",
                "source_table": "clinical_standardized",
                "source_field": "mii",
                "standardized_rule": "mii > 0",
                "output_field": "effectiveness_mii_positive_flag",
                "status": "secondary_component",
                "caveat": "objective minimum mature-oocyte response; no weighted score",
            },
            {
                "concept": "Project safety",
                "source_table": "clinical_standardized",
                "source_field": "complication_text; oocytes",
                "standardized_rule": "no documented OHSS and oocytes <=20",
                "output_field": "safety_no_ohss_warning_flag",
                "status": "secondary_component",
                "caveat": "inverse of the existing documented-OHSS or high-response warning; not an OHSS diagnosis",
            },
            {
                "concept": "Project benefit",
                "source_table": "clinical_standardized",
                "source_field": "live_birth",
                "standardized_rule": "live_birth == 1",
                "output_field": "benefit_recorded_live_birth_flag",
                "status": "secondary_component",
                "caveat": "recorded live birth; absence is not independently adjudicated",
            },
            {
                "concept": "Project composite success",
                "source_table": "derived from audited components",
                "source_field": "mii; complication_text; oocytes; live_birth",
                "standardized_rule": "effectiveness AND safety AND recorded live birth",
                "output_field": "project_composite_success_secondary",
                "status": "secondary_sensitivity_only",
                "caveat": "logical conjunction with no subjective weights; not a replacement for individual endpoints",
            },
        ]
    )

    exact_comparable = (
        labels["kong_aligned_exact_success"].notna()
        & labels["kong_strict_pre_trigger_proxy_success"].notna()
    )
    sensitivity = pd.DataFrame(
        [
            {
                "analysis": "Kong exact primary >=14 mm",
                "denominator_cycles": exact_n,
                "positive_cycles": int(labels["kong_aligned_exact_success"].sum()),
                "positive_rate": float(labels["kong_aligned_exact_success"].mean()),
                "discordant_vs_exact_cycles": 0,
                "note": "Primary exact label.",
            },
            {
                "analysis": "Kong exact >14 mm",
                "denominator_cycles": exact_n,
                "positive_cycles": int(labels["kong_exact_gt14_sensitivity"].sum()),
                "positive_rate": float(labels["kong_exact_gt14_sensitivity"].mean()),
                "discordant_vs_exact_cycles": int(
                    (
                        labels["kong_exact_gt14_sensitivity"]
                        != labels["kong_aligned_exact_success"]
                    ).fillna(False).sum()
                ),
                "note": "Boundary sensitivity.",
            },
            {
                "analysis": "Kong exact >=13.5 mm",
                "denominator_cycles": exact_n,
                "positive_cycles": int(labels["kong_exact_ge13p5_sensitivity"].sum()),
                "positive_rate": float(labels["kong_exact_ge13p5_sensitivity"].mean()),
                "discordant_vs_exact_cycles": int(
                    (
                        labels["kong_exact_ge13p5_sensitivity"]
                        != labels["kong_aligned_exact_success"]
                    ).fillna(False).sum()
                ),
                "note": "Half-millimeter threshold sensitivity.",
            },
            {
                "analysis": "Kong exact >=14 mm excluding diameter >40 mm rows",
                "denominator_cycles": exact_n,
                "positive_cycles": int(labels["kong_exact_plausible_diameter_sensitivity"].sum()),
                "positive_rate": float(labels["kong_exact_plausible_diameter_sensitivity"].mean()),
                "discordant_vs_exact_cycles": int(
                    (
                        labels["kong_exact_plausible_diameter_sensitivity"]
                        != labels["kong_aligned_exact_success"]
                    ).fillna(False).sum()
                ),
                "note": "Data-quality sensitivity only; primary preserves recorded values.",
            },
            {
                "analysis": "Kong exact with expected trigger-to-procedure offset of 2 days",
                "denominator_cycles": int(labels["kong_exact_expected_offset_sensitivity"].notna().sum()),
                "positive_cycles": int(labels["kong_exact_expected_offset_sensitivity"].sum()),
                "positive_rate": float(labels["kong_exact_expected_offset_sensitivity"].mean()),
                "discordant_vs_exact_cycles": 0,
                "note": "Restricts exact denominator to the dominant 2-day workflow.",
            },
            {
                "analysis": "Kong exact-or-fallback proxy all formal cycles",
                "denominator_cycles": int(labels["kong_aligned_last_observed_proxy_success"].notna().sum()),
                "positive_cycles": int(labels["kong_aligned_last_observed_proxy_success"].sum()),
                "positive_rate": float(labels["kong_aligned_last_observed_proxy_success"].mean()),
                "discordant_vs_exact_cycles": 0,
                "note": "Fills only exact-missing cycles; proxy source is retained per cycle.",
            },
            {
                "analysis": "Kong strictly pre-trigger visit proxy",
                "denominator_cycles": int(labels["kong_strict_pre_trigger_proxy_success"].notna().sum()),
                "positive_cycles": int(labels["kong_strict_pre_trigger_proxy_success"].sum()),
                "positive_rate": float(labels["kong_strict_pre_trigger_proxy_success"].mean()),
                "discordant_vs_exact_cycles": int(
                    (
                        labels.loc[exact_comparable, "kong_strict_pre_trigger_proxy_success"]
                        != labels.loc[exact_comparable, "kong_aligned_exact_success"]
                    ).sum()
                ),
                "note": "Timing sensitivity; expected to differ because follicles grow before trigger.",
            },
        ]
    )

    summary = {
        "task_version": config["task_version"],
        "formal_cohort_cycles": formal_n,
        "structured_trigger_cycles": trigger_n,
        "structured_trigger_visits": trigger_visit_n,
        "exact_evaluable_cycles": exact_n,
        "exact_positive_cycles": int(labels["kong_aligned_exact_success"].sum()),
        "exact_positive_rate": float(labels["kong_aligned_exact_success"].mean()),
        "proxy_evaluable_cycles": int(labels["kong_aligned_last_observed_proxy_success"].notna().sum()),
        "proxy_positive_cycles": int(labels["kong_aligned_last_observed_proxy_success"].sum()),
        "project_composite_secondary_positive_cycles": int(
            labels["project_composite_success_secondary"].sum()
        ),
        "project_composite_secondary_positive_rate": float(
            labels["project_composite_success_secondary"].mean()
        ),
        "multiple_hcg_visit_cycles": multi_trigger_cycles,
        "same_visit_multi_hcg_medication_visits": dual_trigger_visits,
        "selected_trigger_without_follicles": trigger_n - exact_n,
        "cycles_without_structured_trigger": formal_n - trigger_n,
    }
    audit_frames = {
        "incidence": pd.DataFrame(incidence_rows),
        "split_distribution": pd.DataFrame(split_rows),
        "trigger_audit": trigger_audit,
        "missingness_audit": pd.DataFrame(missingness_rows),
        "semantics_audit": semantics,
        "sensitivity_audit": sensitivity,
    }
    return labels, audit_frames, summary


def build_from_config(
    config_path: str | Path = "configs/targets/successful_cycle_v1.yaml",
) -> SuccessfulCycleBuildResult:
    config = load_yaml(_resolve(config_path))
    sources = config["sources"]
    clinical = pd.read_csv(_resolve(sources["clinical_dataset"]), low_memory=False)
    visits = pd.read_csv(_resolve(sources["monitoring_visits_dataset"]), low_memory=False)
    medications = pd.read_csv(
        _resolve(sources["monitoring_medications_dataset"]), low_memory=False
    )
    follicles = pd.read_csv(
        _resolve(sources["monitoring_follicles_dataset"]), low_memory=False
    )
    split_manifest = pd.read_csv(_resolve(sources["split_manifest"]), low_memory=False)

    labels, audits, summary = build_cycle_labels(
        clinical,
        visits,
        medications,
        follicles,
        split_manifest,
        config,
    )
    outputs = config["outputs"]
    output_paths = {name: _resolve(path) for name, path in outputs.items()}
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    labels.to_csv(output_paths["cycle_labels"], index=False)
    for name in [
        "incidence",
        "split_distribution",
        "trigger_audit",
        "missingness_audit",
        "semantics_audit",
        "sensitivity_audit",
    ]:
        audits[name].to_csv(output_paths[name], index=False)
    output_paths["build_summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return SuccessfulCycleBuildResult(
        cycle_labels_path=output_paths["cycle_labels"],
        incidence_path=output_paths["incidence"],
        split_distribution_path=output_paths["split_distribution"],
        trigger_audit_path=output_paths["trigger_audit"],
        missingness_audit_path=output_paths["missingness_audit"],
        semantics_audit_path=output_paths["semantics_audit"],
        sensitivity_audit_path=output_paths["sensitivity_audit"],
        build_summary_path=output_paths["build_summary"],
        formal_cohort_cycles=int(summary["formal_cohort_cycles"]),
        trigger_cycles=int(summary["structured_trigger_cycles"]),
        exact_evaluable_cycles=int(summary["exact_evaluable_cycles"]),
        exact_positive_cycles=int(summary["exact_positive_cycles"]),
    )
