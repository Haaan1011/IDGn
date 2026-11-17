from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT
BASELINE = PROJECT / "data" / "processed" / "baseline_cycle_dataset.csv"
SPLIT = PROJECT / "data" / "splits" / "split_manifest_v1.csv"
STANDARDIZED = PROJECT / "data" / "interim" / "clinical_standardized.csv"
VISITS = PROJECT / "data" / "interim" / "monitoring_visits_standardized.csv"
MEDICATIONS = PROJECT / "data" / "interim" / "monitoring_medications_long.csv"
FOLLICLES = PROJECT / "data" / "interim" / "monitoring_follicles_long.csv"
OUT = ROOT / "outputs/table1"

SPLITS = ["train", "valid", "test"]
DISPLAY_SPLITS = {
    "overall": None,
    "training": "train",
    "validation": "valid",
    "testing": "test",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def key_string(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def max_pairwise_smd_continuous(frame: pd.DataFrame, column: str) -> float:
    values = []
    for left, right in itertools.combinations(SPLITS, 2):
        x = numeric(frame.loc[frame["split"].eq(left), column]).dropna()
        y = numeric(frame.loc[frame["split"].eq(right), column]).dropna()
        pooled = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
        values.append(abs((x.mean() - y.mean()) / pooled) if pooled > 0 else 0.0)
    return float(max(values))


def max_pairwise_smd_binary(frame: pd.DataFrame, mask: pd.Series) -> float:
    values = []
    for left, right in itertools.combinations(SPLITS, 2):
        p1 = float(mask.loc[frame["split"].eq(left)].mean())
        p2 = float(mask.loc[frame["split"].eq(right)].mean())
        pooled = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
        values.append(abs((p1 - p2) / pooled) if pooled > 0 else 0.0)
    return float(max(values))


def format_mean_sd(series: pd.Series) -> str:
    x = numeric(series).dropna()
    if x.empty:
        return "NA"
    return f"{x.mean():,.2f} ± {x.std(ddof=1):,.2f}"


def format_n_pct(mask: pd.Series) -> str:
    n = int(mask.sum())
    denominator = int(len(mask))
    return f"{n:,} ({100 * n / denominator:.1f})"


def raw_clinical_subset() -> pd.DataFrame:
    needed = {
        "总编号",
        "ART号",
        "男年龄(岁)",
        "适应症",
        "诊断1",
        "诊断2",
        "诊断3",
        "诊断4",
        "诊断5",
        "诊断6",
        "治疗方案",
        "授精方式",
        "GN总量",
        "GN天数",
    }
    raw = pd.read_excel(
        RAW_XLSX,
        usecols=lambda column: column in needed,
        dtype={"总编号": "string", "ART号": "string"},
    )
    raw = raw.rename(columns={"总编号": "cycle_id", "ART号": "art_id"})
    raw["cycle_id"] = key_string(raw["cycle_id"])
    raw["art_id"] = key_string(raw["art_id"])
    return raw


def classify_infertility_cause(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if "不明原因" in text:
        return "Unexplained infertility"
    male = "夫" in text or bool(
        re.search(r"(少精|弱精|畸精|无精|精子|精液|输精|射精|勃起功能)", text)
    )
    female = bool(
        re.search(
            r"(输卵|子宫|内膜异位|排卵|卵巢|盆腔|宫腔|多囊|卵泡|高龄|卵巢储备|腺肌)",
            text,
        )
    )
    if male and female:
        return "Combined female and male factors"
    if male:
        return "Male factor only"
    if female:
        return "Female factor only"
    return "Other or unclear indication"


def prepare_follicle_visits(follicles: pd.DataFrame) -> pd.DataFrame:
    frame = follicles.copy()
    for column in [
        "follicle_long_axis",
        "follicle_short_axis",
        "follicle_count",
        "follicle_mean_diameter",
    ]:
        frame[column] = numeric(frame[column])
    frame["follicle_count"] = frame["follicle_count"].fillna(0).clip(lower=0)
    frame["effective_diameter"] = frame["follicle_mean_diameter"].fillna(
        frame[["follicle_long_axis", "follicle_short_axis"]].mean(axis=1)
    )
    laterality = (
        frame["follicle_laterality"]
        .astype("string")
        .str.upper()
        .replace({"LEFT": "L", "RIGHT": "R", "左": "L", "右": "R"})
    )
    frame["laterality_norm"] = laterality
    index = pd.Index(frame["visit_uid"].dropna().unique(), name="visit_uid")
    out = pd.DataFrame(index=index)
    out["total_follicle_count"] = frame.groupby("visit_uid")["follicle_count"].sum()
    side = (
        frame.loc[frame["laterality_norm"].isin(["L", "R"])]
        .pivot_table(
            index="visit_uid",
            columns="laterality_norm",
            values="follicle_count",
            aggfunc="sum",
            fill_value=0,
        )
        .rename(columns={"L": "left_follicle_count", "R": "right_follicle_count"})
    )
    out = out.join(side, how="left")
    out["max_follicle_diameter"] = frame.groupby("visit_uid")[
        "effective_diameter"
    ].max()
    weight = frame["follicle_count"]
    weighted = (frame["effective_diameter"].fillna(0) * weight).groupby(
        frame["visit_uid"]
    ).sum()
    out["mean_follicle_diameter"] = weighted / weight.groupby(
        frame["visit_uid"]
    ).sum().replace(0, np.nan)
    rules = {
        "follicle_count_lt_10": frame["effective_diameter"].lt(10),
        "follicle_count_10_12": frame["effective_diameter"].between(10, 12),
        "follicle_count_13_15": frame["effective_diameter"].between(13, 15),
        "follicle_count_16_18": frame["effective_diameter"].between(16, 18),
        "follicle_count_gt_18": frame["effective_diameter"].gt(18),
    }
    for column, mask in rules.items():
        out[column] = (
            frame.loc[mask].groupby("visit_uid")["follicle_count"].sum()
        )
    for column in [
        "total_follicle_count",
        "left_follicle_count",
        "right_follicle_count",
        *rules,
    ]:
        out[column] = numeric(out[column]).fillna(0)
    return out.reset_index()


def build_cycle_frame() -> tuple[pd.DataFrame, dict[str, object]]:
    baseline = pd.read_csv(
        BASELINE, dtype={"cycle_id": "string", "art_id": "string"}, low_memory=False
    )
    manifest = pd.read_csv(SPLIT, low_memory=False)
    baseline["cycle_id"] = key_string(baseline["cycle_id"])
    baseline["art_id"] = key_string(baseline["art_id"])
    raw = raw_clinical_subset()
    if raw.duplicated(["cycle_id", "art_id"]).any():
        raise RuntimeError("Raw clinical cycle_id + art_id is not unique")
    frame = baseline.merge(
        manifest[["cycle_uid", "split"]],
        on="cycle_uid",
        how="left",
        validate="one_to_one",
    ).merge(
        raw,
        on=["cycle_id", "art_id"],
        how="left",
        validate="one_to_one",
        indicator="raw_match",
    )
    if len(frame) != 4071 or frame["split"].isna().any():
        raise RuntimeError("Frozen 4,071-cycle/split contract is not intact")
    if not frame["raw_match"].eq("both").all():
        raise RuntimeError("Some frozen cycles do not match the raw clinical workbook")
    frame = frame.drop(columns="raw_match")

    frame["female_age"] = numeric(frame["age"])
    frame["male_age"] = numeric(frame["男年龄(岁)"])
    frame["bmi_value"] = numeric(frame["bmi"])
    frame["infertility_duration_value"] = numeric(frame["infertility_duration"])
    frame["amh_value"] = numeric(frame["amh"])
    frame["afc_value"] = numeric(frame["afc"])
    frame["basal_fsh_value"] = numeric(frame["basal_fsh"])
    frame["basal_lh_value"] = numeric(frame["basal_lh"])
    frame["basal_e2_value"] = numeric(frame["basal_e2"])
    frame["basal_p_value"] = numeric(frame["basal_p"])
    frame["initial_gn_dose"] = numeric(frame["observed_initial_gn_dose"])
    frame["stimulation_duration"] = numeric(frame["GN天数"])
    frame["cumulative_gn_dose"] = numeric(frame["GN总量"])
    infertility_type = (
        frame["diagnosis_primary_secondary"]
        .astype("string")
        .str.replace("症", "", regex=False)
        .str.strip()
    )
    frame["infertility_type"] = infertility_type.where(
        infertility_type.isin(["原发不孕", "继发不孕"]), "Missing or other"
    )
    frame["infertility_cause"] = frame["适应症"].map(classify_infertility_cause)
    frame["fertilization_method_group"] = (
        frame["授精方式"].astype("string").fillna("Missing")
    )

    visits = pd.read_csv(VISITS, low_memory=False)
    visits = visits.loc[visits["cycle_uid"].isin(frame["cycle_uid"])].copy()
    visits["monitoring_visits"] = 1
    hormone_columns = ["e2", "lh", "p", "fsh", "endometrium"]
    visit_summary = (
        visits.groupby("cycle_uid")
        .agg(
            monitoring_visits=("monitoring_visits", "sum"),
            mean_serum_e2=("e2", "mean"),
            mean_serum_lh=("lh", "mean"),
            mean_serum_p=("p", "mean"),
            mean_serum_fsh=("fsh", "mean"),
            mean_endometrium=("endometrium", "mean"),
        )
        .reset_index()
    )
    frame = frame.merge(visit_summary, on="cycle_uid", how="left", validate="one_to_one")

    medications = pd.read_csv(
        MEDICATIONS,
        usecols=["visit_uid", "medication_class", "medication_value"],
        low_memory=False,
    )
    medication_map = {
        "人工合成FSH": "fsh",
        "尿源性FSH": "fsh",
        "人工合成LH": "lh",
        "高纯度HMG（FSH+LH）": "hmg",
        "Gn(FSH+LH)": "hmg",
    }
    medications["drug_group"] = medications["medication_class"].map(medication_map)
    medications["medication_value"] = numeric(medications["medication_value"])
    medications = medications.loc[medications["drug_group"].notna()]
    medication_visit = (
        medications.groupby(["visit_uid", "drug_group"])["medication_value"]
        .sum(min_count=1)
        .unstack(fill_value=0)
        .reset_index()
    )
    visit_doses = visits[["visit_uid", "cycle_uid"]].merge(
        medication_visit, on="visit_uid", how="left"
    )
    for column in ["fsh", "lh", "hmg"]:
        visit_doses[column] = numeric(visit_doses[column]).fillna(0)
    cycle_doses = (
        visit_doses.groupby("cycle_uid")[["fsh", "lh", "hmg"]]
        .mean()
        .rename(
            columns={
                "fsh": "mean_daily_fsh_dose",
                "lh": "mean_daily_lh_dose",
                "hmg": "mean_daily_hmg_dose",
            }
        )
        .reset_index()
    )
    frame = frame.merge(cycle_doses, on="cycle_uid", how="left", validate="one_to_one")

    follicles = pd.read_csv(FOLLICLES, low_memory=False)
    follicle_visit = prepare_follicle_visits(follicles)
    visit_follicles = visits[["visit_uid", "cycle_uid"]].merge(
        follicle_visit, on="visit_uid", how="left"
    )
    follicle_columns = [
        "total_follicle_count",
        "left_follicle_count",
        "right_follicle_count",
        "max_follicle_diameter",
        "mean_follicle_diameter",
        "follicle_count_lt_10",
        "follicle_count_10_12",
        "follicle_count_13_15",
        "follicle_count_16_18",
        "follicle_count_gt_18",
    ]
    cycle_follicles = (
        visit_follicles.groupby("cycle_uid")[follicle_columns]
        .mean()
        .add_prefix("mean_")
        .reset_index()
    )
    frame = frame.merge(
        cycle_follicles, on="cycle_uid", how="left", validate="one_to_one"
    )

    standardized = pd.read_csv(STANDARDIZED, low_memory=False)
    strict = standardized.loc[
        standardized["cohort_strict_eligible"].fillna(False).astype(bool), "cycle_uid"
    ]
    treatment_scope = standardized["treatment_plan_is_ivf_icsi_scope"].fillna(
        False
    ).astype(bool)
    protocol_scope = standardized["protocol_matches_official_protocol"].fillna(
        False
    ).astype(bool)
    first_treatment = standardized["treatment_count_eq_1_flag"].fillna(False).astype(
        bool
    )
    audit = {
        "raw_rows": int(load_workbook(RAW_XLSX, read_only=True).active.max_row - 1),
        "baseline_rows": int(len(baseline)),
        "baseline_unique_cycle_uid": int(baseline["cycle_uid"].nunique()),
        "baseline_unique_cycle_id": int(baseline["cycle_id"].nunique()),
        "baseline_unique_art_id": int(baseline["art_id"].nunique()),
        "split_counts": manifest["split"].value_counts().to_dict(),
        "baseline_equals_manifest": bool(
            set(baseline["cycle_uid"]) == set(manifest["cycle_uid"])
        ),
        "strict_flag_count": int(strict.nunique()),
        "strict_flag_equals_baseline": bool(set(strict) == set(baseline["cycle_uid"])),
        "verified_cohort_flow": {
            "raw": int(len(standardized)),
            "allowed_treatment_plan": int(treatment_scope.sum()),
            "allowed_plan_and_protocol": int((treatment_scope & protocol_scope).sum()),
            "allowed_plan_protocol_and_treatment_count_1": int(
                (treatment_scope & protocol_scope & first_treatment).sum()
            ),
        },
        "raw_key_match_count": int(len(frame)),
        "monitoring_rows": int(len(visits)),
        "monitoring_cycles": int(visits["cycle_uid"].nunique()),
        "source_sha256": {
            str(path): sha256(path)
            for path in [
                RAW_XLSX,
                BASELINE,
                SPLIT,
                STANDARDIZED,
                VISITS,
                MEDICATIONS,
                FOLLICLES,
            ]
        },
    }
    return frame, audit


CONTINUOUS = [
    ("Female age, years", "female_age"),
    ("Male age, years", "male_age"),
    ("BMI, kg/m²", "bmi_value"),
    ("Infertility duration, years", "infertility_duration_value"),
    ("AMH, ng/mL", "amh_value"),
    ("AFC, n", "afc_value"),
    ("Basal FSH, IU/L", "basal_fsh_value"),
    ("Basal LH, IU/L", "basal_lh_value"),
    ("Basal E2, pg/mL", "basal_e2_value"),
    ("Basal progesterone, ng/mL", "basal_p_value"),
    ("Initial Gn dose, IU/day", "initial_gn_dose"),
    ("Stimulation duration, days", "stimulation_duration"),
    ("Monitoring visits, n", "monitoring_visits"),
    ("Cumulative Gn dose, IU", "cumulative_gn_dose"),
    ("Mean daily FSH dose, IU/day", "mean_daily_fsh_dose"),
    ("Mean daily LH dose, IU/day", "mean_daily_lh_dose"),
    ("Mean daily HMG dose, IU/day", "mean_daily_hmg_dose"),
    ("Mean serum E2, pg/mL", "mean_serum_e2"),
    ("Mean serum LH, IU/L", "mean_serum_lh"),
    ("Mean serum progesterone, ng/mL", "mean_serum_p"),
    ("Mean serum FSH, IU/L", "mean_serum_fsh"),
    ("Mean endometrial thickness, mm", "mean_endometrium"),
    ("Mean total follicle count, n", "mean_total_follicle_count"),
    ("Mean left follicle count, n", "mean_left_follicle_count"),
    ("Mean right follicle count, n", "mean_right_follicle_count"),
    ("Mean maximum follicle diameter, mm", "mean_max_follicle_diameter"),
    ("Mean follicle diameter, mm", "mean_mean_follicle_diameter"),
    ("Mean follicle count <10 mm, n", "mean_follicle_count_lt_10"),
    ("Mean follicle count 10–12 mm, n", "mean_follicle_count_10_12"),
    ("Mean follicle count 13–15 mm, n", "mean_follicle_count_13_15"),
    ("Mean follicle count 16–18 mm, n", "mean_follicle_count_16_18"),
    ("Mean follicle count >18 mm, n", "mean_follicle_count_gt_18"),
]


def continuous_row(frame: pd.DataFrame, label: str, column: str) -> dict[str, str]:
    row = {"row_type": "data", "label": label}
    for display, split in DISPLAY_SPLITS.items():
        subset = frame if split is None else frame.loc[frame["split"].eq(split)]
        row[display] = format_mean_sd(subset[column])
    row["max_smd"] = f"{max_pairwise_smd_continuous(frame, column):.3f}"
    return row


def category_row(
    frame: pd.DataFrame, label: str, column: str, value: str
) -> dict[str, str]:
    row = {"row_type": "data", "label": label}
    overall_mask = frame[column].astype("string").eq(value)
    for display, split in DISPLAY_SPLITS.items():
        subset = frame if split is None else frame.loc[frame["split"].eq(split)]
        row[display] = format_n_pct(subset[column].astype("string").eq(value))
    row["max_smd"] = f"{max_pairwise_smd_binary(frame, overall_mask):.3f}"
    return row


def build_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def group(label: str, row_type: str = "group") -> None:
        rows.append(
            {
                "row_type": row_type,
                "label": label,
                "overall": "",
                "training": "",
                "validation": "",
                "testing": "",
                "max_smd": "",
            }
        )

    group("Participant and infertility characteristics")
    for label, column in CONTINUOUS[:4]:
        rows.append(continuous_row(frame, label, column))
    group("Infertility type, n (%)", "subgroup")
    for value, label in [
        ("原发不孕", "Primary infertility"),
        ("继发不孕", "Secondary infertility"),
        ("Missing or other", "Missing or other infertility type"),
    ]:
        rows.append(category_row(frame, label, "infertility_type", value))
    group("Cause of infertility, n (%)", "subgroup")
    for value in [
        "Female factor only",
        "Male factor only",
        "Combined female and male factors",
        "Unexplained infertility",
        "Other or unclear indication",
    ]:
        rows.append(category_row(frame, value, "infertility_cause", value))

    group("Ovarian reserve and basal hormones")
    for label, column in CONTINUOUS[4:10]:
        rows.append(continuous_row(frame, label, column))

    group("Fertilization method, n (%)")
    for value, label in [
        ("IVF", "IVF"),
        ("ICSI", "ICSI"),
        ("短时授精", "Short insemination"),
        ("短时授精和补救", "Short insemination with rescue"),
        ("Missing", "Missing fertilization method"),
    ]:
        if value == "Missing" or frame["fertilization_method_group"].eq(value).any():
            rows.append(
                category_row(
                    frame, label, "fertilization_method_group", value
                )
            )

    group("Ovarian stimulation and cycle-level monitoring")
    for label, column in CONTINUOUS[10:]:
        rows.append(continuous_row(frame, label, column))
    return pd.DataFrame(rows)[
        [
            "row_type",
            "label",
            "overall",
            "training",
            "validation",
            "testing",
            "max_smd",
        ]
    ]


def write_workbook(
    table: pd.DataFrame, missing: pd.DataFrame, category_counts: pd.DataFrame, path: Path
) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Table1_Word_ready"
    ws.append(list(table.columns))
    for row in table.itertuples(index=False, name=None):
        ws.append(list(row))
    for cell in ws[1]:
        cell.font = Font(name="Arial", bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(horizontal="center")
    for row_index, row_type in enumerate(table["row_type"], start=2):
        if row_type == "group":
            for cell in ws[row_index]:
                cell.fill = PatternFill("solid", fgColor="D9E2F3")
                cell.font = Font(name="Arial", bold=True)
        elif row_type == "subgroup":
            for cell in ws[row_index]:
                cell.fill = PatternFill("solid", fgColor="EAF0F8")
                cell.font = Font(name="Arial", bold=True)
        else:
            for cell in ws[row_index]:
                cell.font = Font(name="Arial")
    widths = [14, 42, 22, 22, 22, 22, 12]
    for column, width in zip("ABCDEFG", widths):
        ws.column_dimensions[column].width = width
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for title, data in [
        ("Missingness", missing),
        ("Category_counts", category_counts),
    ]:
        sheet = wb.create_sheet(title)
        sheet.append(list(data.columns))
        for row in data.itertuples(index=False, name=None):
            sheet.append(list(row))
        for cell in sheet[1]:
            cell.font = Font(name="Arial", bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1F4E78")
        for column in sheet.columns:
            letter = column[0].column_letter
            sheet.column_dimensions[letter].width = min(
                50, max(len(str(cell.value or "")) for cell in column) + 2
            )
        sheet.freeze_panes = "A2"
    wb.save(path)


def main() -> None:
    global RAW_XLSX
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical", required=True, type=Path)
    RAW_XLSX = parser.parse_args().clinical.resolve(strict=True)
    OUT.mkdir(parents=True, exist_ok=True)
    frame, audit = build_cycle_frame()
    table = build_table(frame)
    table_path = OUT / "table1_word_ready_mean_sd.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    missing_rows = []
    for label, column in CONTINUOUS:
        row = {"label": label, "column": column}
        for display, split in DISPLAY_SPLITS.items():
            subset = frame if split is None else frame.loc[frame["split"].eq(split)]
            row[f"{display}_N"] = int(len(subset))
            row[f"{display}_available"] = int(numeric(subset[column]).notna().sum())
            row[f"{display}_missing"] = int(numeric(subset[column]).isna().sum())
        missing_rows.append(row)
    missing = pd.DataFrame(missing_rows)
    missing.to_csv(OUT / "table1_missingness.csv", index=False, encoding="utf-8-sig")
    category_counts = (
        pd.concat(
            [
                frame.groupby(["split", "infertility_type"]).size().rename("n"),
                frame.groupby(["split", "infertility_cause"]).size().rename("n"),
                frame.groupby(["split", "fertilization_method_group"])
                .size()
                .rename("n"),
            ],
            keys=["infertility_type", "infertility_cause", "fertilization_method"],
            names=["variable"],
        )
        .reset_index()
        .rename(columns={"level_2": "category"})
    )
    category_counts.to_csv(
        OUT / "table1_category_counts.csv", index=False, encoding="utf-8-sig"
    )
    write_workbook(
        table,
        missing,
        category_counts,
        OUT / "table1_data_audit.xlsx",
    )
    max_row = table.loc[
        table["row_type"].eq("data")
        & pd.to_numeric(table["max_smd"], errors="coerce").notna()
    ].copy()
    max_row["max_smd_num"] = pd.to_numeric(max_row["max_smd"])
    maximum = max_row.sort_values("max_smd_num", ascending=False).iloc[0]
    audit["maximum_smd"] = {
        "label": maximum["label"],
        "value": float(maximum["max_smd_num"]),
    }
    (OUT / "source_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "table_rows": int(len(table)),
        "data_rows": int(table["row_type"].eq("data").sum()),
        "max_smd": audit["maximum_smd"],
        "artifacts": {
            "table_csv": str(table_path),
            "workbook": str(OUT / "table1_data_audit.xlsx"),
            "missingness_csv": str(OUT / "table1_missingness.csv"),
            "category_counts_csv": str(OUT / "table1_category_counts.csv"),
            "source_audit_json": str(OUT / "source_audit.json"),
        },
    }
    (OUT / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
