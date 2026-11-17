from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


def _safe_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (int, str, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 6)
    return str(value)


def _sample_values(series: pd.Series, limit: int = 3) -> list[Any]:
    samples: list[Any] = []
    for value in series.dropna().head(100):
        safe = _safe_scalar(value)
        if safe not in samples:
            samples.append(safe)
        if len(samples) >= limit:
            break
    return samples


def _numeric_profile(series: pd.Series) -> dict[str, Any]:
    non_null = series.dropna()
    if non_null.empty:
        return {
            "min": None,
            "q1": None,
            "median": None,
            "q3": None,
            "max": None,
            "outlier_count_iqr": 0,
            "outlier_rate_iqr": 0.0,
        }

    q1 = float(non_null.quantile(0.25))
    q3 = float(non_null.quantile(0.75))
    iqr = q3 - q1
    if iqr > 0:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (non_null < lower) | (non_null > upper)
        outlier_count = int(outlier_mask.sum())
    else:
        outlier_count = 0

    return {
        "min": _safe_scalar(float(non_null.min())),
        "q1": round(q1, 6),
        "median": round(float(non_null.median()), 6),
        "q3": round(q3, 6),
        "max": _safe_scalar(float(non_null.max())),
        "outlier_count_iqr": outlier_count,
        "outlier_rate_iqr": round(outlier_count / len(series), 6),
    }


def _datetime_profile(series: pd.Series) -> dict[str, Any]:
    non_null = series.dropna()
    if non_null.empty:
        return {"min": None, "max": None}
    return {
        "min": _safe_scalar(non_null.min()),
        "max": _safe_scalar(non_null.max()),
    }


def build_column_profile(df: pd.DataFrame, table_name: str, sheet_name: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_rows = len(df)
    for column in df.columns:
        series = df[column]
        missing_count = int(series.isna().sum())
        row: dict[str, Any] = {
            "table_name": table_name,
            "sheet_name": sheet_name or "",
            "column_name": str(column),
            "dtype": str(series.dtype),
            "row_count": total_rows,
            "non_null_count": int(series.notna().sum()),
            "missing_count": missing_count,
            "missing_rate": round(missing_count / total_rows, 6) if total_rows else 0.0,
            "unique_non_null": int(series.nunique(dropna=True)),
            "sample_values": str(_sample_values(series)),
        }
        if is_numeric_dtype(series):
            row["data_kind"] = "numeric"
            row.update(_numeric_profile(pd.to_numeric(series, errors="coerce")))
        elif is_datetime64_any_dtype(series):
            row["data_kind"] = "datetime"
            row.update(_datetime_profile(series))
            row.update({"q1": None, "median": None, "q3": None, "outlier_count_iqr": 0, "outlier_rate_iqr": 0.0})
        else:
            row["data_kind"] = "text"
            row.update(
                {
                    "min": None,
                    "q1": None,
                    "median": None,
                    "q3": None,
                    "max": None,
                    "outlier_count_iqr": 0,
                    "outlier_rate_iqr": 0.0,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_keys(df: pd.DataFrame, key_columns: dict[str, str | None]) -> dict[str, Any]:
    key_summary: dict[str, Any] = {"resolved_keys": {}, "composite_duplicate_count": None}
    resolved_cols = []
    for standard_name, raw_name in key_columns.items():
        if raw_name and raw_name in df.columns:
            missing_count = int(df[raw_name].isna().sum())
            duplicate_count = int(df[raw_name].duplicated().sum())
            key_summary["resolved_keys"][standard_name] = {
                "raw_column": raw_name,
                "missing_count": missing_count,
                "duplicate_count": duplicate_count,
            }
            resolved_cols.append(raw_name)
        else:
            key_summary["resolved_keys"][standard_name] = {
                "raw_column": None,
                "missing_count": None,
                "duplicate_count": None,
            }

    if resolved_cols:
        subset = df[resolved_cols]
        valid_subset = subset.dropna(how="any")
        key_summary["composite_duplicate_count"] = int(valid_subset.duplicated().sum())
    return key_summary


def summarize_dataframe(
    df: pd.DataFrame,
    table_name: str,
    column_profile: pd.DataFrame,
    key_columns: dict[str, str | None],
) -> dict[str, Any]:
    dtype_counts = column_profile["data_kind"].value_counts().to_dict()
    top_missing = (
        column_profile.sort_values(["missing_rate", "column_name"], ascending=[False, True])
        .head(15)[["column_name", "missing_rate"]]
        .to_dict(orient="records")
    )
    top_outliers = (
        column_profile[column_profile["outlier_count_iqr"] > 0]
        .sort_values(["outlier_count_iqr", "column_name"], ascending=[False, True])
        .head(15)[["column_name", "outlier_count_iqr", "outlier_rate_iqr"]]
        .to_dict(orient="records")
    )
    return {
        "table_name": table_name,
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "duplicate_row_count": int(df.duplicated().sum()),
        "data_kind_counts": {key: int(value) for key, value in dtype_counts.items()},
        "top_missing_columns": top_missing,
        "top_outlier_columns": top_outliers,
        "key_summary": summarize_keys(df, key_columns),
    }
