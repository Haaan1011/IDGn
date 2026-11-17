from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd


def resolve_key_columns(actual_columns: list[str], key_config: dict[str, Any]) -> dict[str, str | None]:
    resolved: dict[str, str | None] = {}
    column_set = set(actual_columns)
    for standard_name, candidates in key_config.items():
        raw_name = None
        for candidate in candidates:
            if candidate in column_set:
                raw_name = candidate
                break
        resolved[standard_name] = raw_name
    return resolved


def build_clinical_field_mapping(actual_columns: list[str], schema_config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    configured_raw_names: set[str] = set()
    sections = ["keys", "baseline_fields", "context_fields", "targets"]

    for section in sections:
        config_block = schema_config.get(section, {})
        for standard_name, raw_value in config_block.items():
            if isinstance(raw_value, list):
                for raw_name in raw_value:
                    configured_raw_names.add(raw_name)
                    rows.append(
                        {
                            "source_table": schema_config.get("table_name", "clinical_table"),
                            "sheet_scope": schema_config.get("sheet_name", ""),
                            "raw_field_name": raw_name,
                            "standard_field_name": standard_name,
                            "field_group": section,
                            "is_configured": True,
                            "present_in_source": raw_name in actual_columns,
                            "notes": "",
                        }
                    )
            else:
                raw_name = str(raw_value)
                configured_raw_names.add(raw_name)
                rows.append(
                    {
                        "source_table": schema_config.get("table_name", "clinical_table"),
                        "sheet_scope": schema_config.get("sheet_name", ""),
                        "raw_field_name": raw_name,
                        "standard_field_name": standard_name,
                        "field_group": section,
                        "is_configured": True,
                        "present_in_source": raw_name in actual_columns,
                        "notes": "",
                    }
                )

    for raw_name in actual_columns:
        if raw_name in configured_raw_names:
            continue
        rows.append(
            {
                "source_table": schema_config.get("table_name", "clinical_table"),
                "sheet_scope": schema_config.get("sheet_name", ""),
                "raw_field_name": raw_name,
                "standard_field_name": "",
                "field_group": "unmapped",
                "is_configured": False,
                "present_in_source": True,
                "notes": "",
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["is_configured", "field_group", "standard_field_name", "raw_field_name"],
        ascending=[False, True, True, True],
    )


def build_monitoring_field_mapping(
    actual_columns_by_sheet: dict[str, list[str]],
    schema_config: dict[str, Any],
) -> pd.DataFrame:
    presence_map: dict[str, list[str]] = defaultdict(list)
    for sheet_name, columns in actual_columns_by_sheet.items():
        for column in columns:
            presence_map[column].append(sheet_name)

    union_columns = sorted(presence_map)
    rows: list[dict[str, Any]] = []
    configured_raw_names: set[str] = set()

    for section in ["keys", "anchor_fields", "core_fields"]:
        config_block = schema_config.get(section, {})
        for standard_name, raw_name in config_block.items():
            raw_names = raw_name if isinstance(raw_name, list) else [raw_name]
            for candidate_raw_name in raw_names:
                configured_raw_names.add(candidate_raw_name)
                rows.append(
                    {
                        "source_table": schema_config.get("table_name", "monitoring_table"),
                        "sheet_scope": "all_expected_sheets",
                        "raw_field_name": candidate_raw_name,
                        "standard_field_name": standard_name,
                        "field_group": section,
                        "is_configured": True,
                        "present_in_source": candidate_raw_name in presence_map,
                        "presence_count": len(presence_map.get(candidate_raw_name, [])),
                        "sheet_list": ",".join(sorted(presence_map.get(candidate_raw_name, []))),
                        "notes": "",
                    }
                )

    medication_pattern = schema_config.get("medication_pattern", {})
    medication_max_index = int(medication_pattern.get("max_index", 0))
    medication_fields = {
        "name_field": "medication_name",
        "class_field": "medication_class",
        "unit_field": "medication_unit",
        "value_field": "medication_value",
        "frequency_field": "medication_frequency",
        "days_field": "medication_days",
        "total_amount_field": "medication_total_amount",
    }
    for pattern_key, standard_prefix in medication_fields.items():
        template = medication_pattern.get(pattern_key)
        if not template:
            continue
        for index in range(1, medication_max_index + 1):
            raw_name = template.format(i=index)
            configured_raw_names.add(raw_name)
            rows.append(
                {
                    "source_table": schema_config.get("table_name", "monitoring_table"),
                    "sheet_scope": "all_expected_sheets",
                    "raw_field_name": raw_name,
                    "standard_field_name": f"{standard_prefix}_{index}",
                    "field_group": "medication_pattern",
                    "is_configured": True,
                    "present_in_source": raw_name in presence_map,
                    "presence_count": len(presence_map.get(raw_name, [])),
                    "sheet_list": ",".join(sorted(presence_map.get(raw_name, []))),
                    "notes": "",
                }
            )

    pattern = schema_config.get("follicle_detail_pattern", {})
    max_index = int(pattern.get("max_index", 0))
    pattern_fields = {
        "lateral_field": "follicle_lateral",
        "order_field": "follicle_order",
        "long_axis_field": "follicle_long_axis",
        "short_axis_field": "follicle_short_axis",
        "count_field": "follicle_count",
        "mean_diameter_field": "follicle_mean_diameter",
    }
    for pattern_key, standard_prefix in pattern_fields.items():
        template = pattern.get(pattern_key)
        if not template:
            continue
        for index in range(1, max_index + 1):
            raw_name = template.format(i=index)
            configured_raw_names.add(raw_name)
            rows.append(
                {
                    "source_table": schema_config.get("table_name", "monitoring_table"),
                    "sheet_scope": "all_expected_sheets",
                    "raw_field_name": raw_name,
                    "standard_field_name": f"{standard_prefix}_{index}",
                    "field_group": "follicle_detail_pattern",
                    "is_configured": True,
                    "present_in_source": raw_name in presence_map,
                    "presence_count": len(presence_map.get(raw_name, [])),
                    "sheet_list": ",".join(sorted(presence_map.get(raw_name, []))),
                    "notes": "",
                }
            )

    for raw_name in union_columns:
        if raw_name in configured_raw_names:
            continue
        rows.append(
            {
                "source_table": schema_config.get("table_name", "monitoring_table"),
                "sheet_scope": "sheet_union",
                "raw_field_name": raw_name,
                "standard_field_name": "",
                "field_group": "unmapped",
                "is_configured": False,
                "present_in_source": True,
                "presence_count": len(presence_map[raw_name]),
                "sheet_list": ",".join(sorted(presence_map[raw_name])),
                "notes": "",
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["is_configured", "field_group", "standard_field_name", "raw_field_name"],
        ascending=[False, True, True, True],
    )
