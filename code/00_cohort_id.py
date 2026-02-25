import marimo

__generated_with = "0.20.2"
app = marimo.App()


# ---------------------------------------------------------------------------
# Cell 1: marimo import
# ---------------------------------------------------------------------------
@app.cell
def _():
    import marimo as mo

    return (mo,)


# ---------------------------------------------------------------------------
# Cell 2: Setup — path resolution, all imports, config load
# ---------------------------------------------------------------------------
@app.cell
def _():
    import os
    import sys
    import json
    import warnings
    from pathlib import Path

    _CODE_DIR = os.path.dirname(os.path.abspath(__file__))
    _UTILS_DIR = os.path.join(_CODE_DIR, "..", "utils")
    if _UTILS_DIR not in sys.path:
        sys.path.insert(0, _UTILS_DIR)
    os.chdir(_CODE_DIR)

    import numpy as np
    import pandas as pd
    import polars as pl
    from tqdm import tqdm
    import pyCLIF as pc

    warnings.filterwarnings("ignore")

    # Load outlier config once here so all downstream cells can use it
    _outlier_path = Path("../config/outlier_config.json")
    try:
        with open(_outlier_path, "r") as _f:
            outlier_cfg: dict = json.load(_f)
    except (FileNotFoundError, json.JSONDecodeError):
        outlier_cfg = {}
        print("WARNING: outlier_config.json not found or invalid. No outlier thresholds applied.")

    return np, os, outlier_cfg, pc, pd, pl, tqdm


# ---------------------------------------------------------------------------
# Cell 3: Load base tables — ADT, Hospitalization, Patient
#         Build encounter-block stitching map and aggregate to block level
# ---------------------------------------------------------------------------
@app.cell
def _(os, pc, pd, pl):
    # ---- ADT ---------------------------------------------------------------
    _adt_raw = pc.load_data("clif_adt")
    _adt_raw["hospitalization_id"] = _adt_raw["hospitalization_id"].astype(str)
    _adt_raw = pc.convert_datetime_columns_to_site_tz(_adt_raw, pc.helper["your_site_timezone"])
    _adt_raw["in_dttm"] = pc.getdttm(_adt_raw["in_dttm"])

    # ---- Hospitalization ---------------------------------------------------
    _hosp_raw = pc.load_data("clif_hospitalization")
    _hosp_raw["hospitalization_id"] = _hosp_raw["hospitalization_id"].astype(str)
    if "hospitalization_joined_id" not in _hosp_raw.columns:
        _hosp_raw["hospitalization_joined_id"] = _hosp_raw["hospitalization_id"]
    _hosp_raw["hospitalization_joined_id"] = _hosp_raw["hospitalization_joined_id"].astype(str)
    _hosp_raw = pc.convert_datetime_columns_to_site_tz(_hosp_raw, pc.helper["your_site_timezone"])
    _hosp_raw["admission_dttm"] = pc.getdttm(_hosp_raw["admission_dttm"])
    _hosp_raw["discharge_dttm"] = pc.getdttm(_hosp_raw["discharge_dttm"])
    _hosp_raw["Hosp_key_bkp"] = _hosp_raw["hospitalization_id"]
    _adt_raw["Hosp_key_bkp"] = _adt_raw["hospitalization_id"]

    # ---- Encounter-block stitching -----------------------------------------
    _eblock = pc.stitch_encounters(_hosp_raw, _adt_raw)
    hospitalization_to_block: dict = {
        hosp_id: block
        for block, hosp_list in zip(
            _eblock["encounter_block"].astype(str),
            _eblock["list_hospitalization_id"],
        )
        for hosp_id in hosp_list
    }

    # Persist the mapping for downstream inspection
    _map_df = pd.DataFrame(
        list(hospitalization_to_block.items()),
        columns=["hospitalization_id", "encounter_block"],
    )
    _map_df.to_csv("../output/intermediate/hospitalization_to_block_df.csv", index=False)

    # ---- Aggregate hospitalization to block level ---------------------------
    _agg_rules = {
        "patient_id": "first",
        "admission_dttm": "min",
        "discharge_dttm": "max",
        "age_at_admission": "mean",
        "discharge_category": "last",
        "hospitalization_joined_id": lambda x: ", ".join(x.unique()),
        "Hosp_key_bkp": lambda x: ", ".join(x.unique()),
    }
    if "discharge_name" in _hosp_raw.columns:
        _agg_rules["discharge_name"] = "last"

    _hosp_mapped = _hosp_raw.copy()
    _hosp_mapped["hospitalization_id"] = _hosp_mapped["hospitalization_id"].map(hospitalization_to_block).astype(str)
    hosp = (
        _hosp_mapped
        .sort_values(["hospitalization_id", "admission_dttm"])
        .groupby("hospitalization_id")
        .agg(_agg_rules)
        .reset_index()
    )

    # ---- ADT mapped to blocks ----------------------------------------------
    adt = (
        _adt_raw[["hospitalization_id", "in_dttm", "location_category", "hospital_id"]]
        .assign(hospitalization_id=lambda df: df["hospitalization_id"].map(hospitalization_to_block).astype(str))
        .sort_values(["hospitalization_id", "in_dttm"])
        .reset_index(drop=True)
    )

    # ---- Patient demographics ----------------------------------------------
    _pat_raw = pc.load_data("clif_patient")
    pat = pc.convert_datetime_columns_to_site_tz(
        _pat_raw[["patient_id", "sex_category", "race_category", "ethnicity_category", "language_name"]],
        pc.helper["your_site_timezone"],
    )

    print(f"hosp blocks: {len(hosp):,}  |  adt rows: {len(adt):,}  |  patients: {len(pat):,}")
    return adt, hospitalization_to_block, hosp, pat


# ---------------------------------------------------------------------------
# Cell 4: Load respiratory support — waterfall fill, device/mode cleanup,
#         tracheostomy exclusion (pre- and post-waterfall)
# ---------------------------------------------------------------------------
@app.cell
def _(hospitalization_to_block, np, pc, pl):
    # ---- Load & remap ------------------------------------------------------
    _rst_raw = pc.load_data("clif_respiratory_support")
    _rst_raw["hospitalization_id"] = _rst_raw["hospitalization_id"].astype(str)

    _unmapped = _rst_raw["hospitalization_id"].map(hospitalization_to_block).isna().sum()
    if _unmapped > 0:
        print(f"WARNING: {_unmapped} respiratory_support rows have unmapped hospitalization_id (dropped)")

    _rst_raw["hospitalization_id"] = _rst_raw["hospitalization_id"].map(hospitalization_to_block)
    _rst_mapped = _rst_raw[_rst_raw["hospitalization_id"].notna()].copy()
    _rst_mapped["hospitalization_id"] = _rst_mapped["hospitalization_id"].astype(str)

    # Pre-waterfall trach exclusion
    _trach_pre = _rst_mapped[_rst_mapped["tracheostomy"] == 1]["hospitalization_id"].unique()
    _rst_no_trach = _rst_mapped[~_rst_mapped["hospitalization_id"].isin(_trach_pre)].copy()

    # ---- Site-timezone conversion + UTC for waterfall ----------------------
    _rst_tz = pc.convert_datetime_columns_to_site_tz(_rst_no_trach, pc.helper["your_site_timezone"])
    _rst_tz["recorded_dttm"] = _rst_tz["recorded_dttm"].dt.tz_convert("UTC")

    # ---- Waterfall forward-fill --------------------------------------------
    _rst_wf = pc.process_resp_support_waterfall(_rst_tz)

    # Convert back to site timezone
    _rst_wf["recorded_dttm"] = _rst_wf["recorded_dttm"].dt.tz_convert(pc.helper["your_site_timezone"])
    _rst_wf["recorded_dttm"] = pc.getdttm(_rst_wf["recorded_dttm"])

    # Post-waterfall trach exclusion (waterfall may have propagated the flag)
    _trach_post = _rst_wf[_rst_wf["tracheostomy"] == 1]["hospitalization_id"].unique()
    _before = len(_rst_wf)
    _rst_clean = _rst_wf[~_rst_wf["hospitalization_id"].isin(_trach_post)].reset_index(drop=True)
    print(
        f"Post-waterfall trach exclusion: {_before:,} -> {len(_rst_clean):,} rows "
        f"(excluded {len(_trach_post)} hospitalizations)"
    )

    # ---- Column selection & normalization ----------------------------------
    _rst_cols = [
        "hospitalization_id", "recorded_dttm", "device_category",
        "mode_category", "fio2_set", "peep_set", "resp_rate_set",
        "pressure_support_set", "mode_name",
    ]
    if pc.helper["site_name"] == "RUSH":
        for _extra in ["tube_comp_%", "sbt_timepoint", "vent_brand_name"]:
            if _extra in _rst_clean.columns:
                _rst_cols.append(_extra)
        _rst_clean["device_category"] = _rst_clean["device_category"].replace("nan", np.nan)

    _rst_cols = [c for c in _rst_cols if c in _rst_clean.columns]
    rst = _rst_clean[_rst_cols].copy()
    rst["device_category"] = rst["device_category"].str.lower()
    rst["mode_category"] = rst["mode_category"].str.lower()

    pc.deftime(rst["recorded_dttm"])
    print(f"rst rows after cleaning: {len(rst):,}")
    return (rst,)


# ---------------------------------------------------------------------------
# Cell 5: Extubation flags + IMV episode classification
#         Sequential row-by-row logic — kept in pandas intentionally
# ---------------------------------------------------------------------------
@app.cell
def _(rst, tqdm):
    from definitions_source_of_truth import classify_imv_episodes

    MAX_EXTUBATION_GAP_HOURS = 24

    # ---- Sort and forward-fill device within hospitalization ---------------
    rst_ef = rst.sort_values(["hospitalization_id", "recorded_dttm"]).copy()
    rst_ef["device_category"] = rst_ef.groupby("hospitalization_id")["device_category"].ffill()

    # Device segment IDs (each contiguous block of the same device)
    rst_ef["_device_change"] = (
        rst_ef["device_category"] != rst_ef.groupby("hospitalization_id")["device_category"].shift()
    ).astype(int)
    rst_ef["device_segment_id"] = rst_ef.groupby("hospitalization_id")["_device_change"].cumsum()
    rst_ef["mode_category"] = rst_ef.groupby(["hospitalization_id", "device_segment_id"])["mode_category"].ffill()
    rst_ef["Device_IMV"] = (rst_ef["device_category"] == "imv").astype(int)
    rst_ef.drop(columns=["_device_change"], inplace=True)

    # ---- Sequential extubation flag loop -----------------------------------
    import numpy as _np

    rst_ef["extubated"] = 0
    for _hosp_id, _group in tqdm(rst_ef.groupby("hospitalization_id"), desc="Generating Extubated Flags"):
        _group = _group.sort_values("recorded_dttm")
        _idx = _group.index
        _times = _group["recorded_dttm"].values
        _imv_vals = _group["Device_IMV"].values
        for _i in range(len(_group) - 2):
            if _imv_vals[_i] == 1 and _imv_vals[_i + 1] == 0 and _imv_vals[_i + 2] == 0:
                _gap_h = (_times[_i + 2] - _times[_i]) / _np.timedelta64(1, "h")
                if _gap_h <= MAX_EXTUBATION_GAP_HOURS:
                    rst_ef.at[_idx[_i], "extubated"] = 1

    # ---- IMV episode classification (72-hour gap rule) --------------------
    _imv_records = rst_ef[rst_ef["device_category"] == "imv"][["hospitalization_id", "recorded_dttm"]].copy()
    _imv_classified = classify_imv_episodes(_imv_records)
    rst_final = rst_ef.merge(
        _imv_classified[["hospitalization_id", "recorded_dttm", "imv_episode_id"]],
        on=["hospitalization_id", "recorded_dttm"],
        how="left",
    )
    rst_final["imv_episode_id"] = rst_final.groupby("hospitalization_id")["imv_episode_id"].ffill()
    print(f"IMV episodes identified: {rst_final['imv_episode_id'].nunique():,}")
    return (rst_final,)


# ---------------------------------------------------------------------------
# Cell 6: Cohort filtering — apply IMV + ICU + age + date criteria
# ---------------------------------------------------------------------------
@app.cell
def _(adt, hosp, pat, pc, pd, rst_final):
    import numpy as _np2

    _imv_ids = set(rst_final[rst_final["device_category"] == "imv"]["hospitalization_id"].unique())
    _icu_ids = set(adt[adt["location_category"].str.lower() == "icu"]["hospitalization_id"].dropna().unique())
    _eligible_ids = _imv_ids & _icu_ids

    hosp_filtered = hosp[
        (hosp["admission_dttm"].dt.year >= 2022)
        & (hosp["admission_dttm"].dt.year <= 2024)
        & (hosp["discharge_dttm"].dt.year <= 2024)
        & hosp["hospitalization_id"].isin(_eligible_ids)
        & hosp["age_at_admission"].between(18, 119)
    ].reset_index(drop=True)

    required_id = set(hosp_filtered["hospitalization_id"].unique())
    print(f"{len(required_id):,} : potential cohort count")

    # ---- Base demographic table --------------------------------------------
    _keep_cols = [
        "patient_id", "hospitalization_id", "admission_dttm", "discharge_dttm",
        "age_at_admission", "discharge_category", "sex_category",
        "race_category", "ethnicity_category", "language_name",
    ]
    base = pd.merge(hosp_filtered, pat, on="patient_id", how="inner")[_keep_cols].copy()
    base["admission_dttm"] = pc.getdttm(base["admission_dttm"])

    # ---- Filter downstream tables to cohort --------------------------------
    adt_cohort = adt[adt["hospitalization_id"].isin(required_id)].reset_index(drop=True)
    rst_cohort = rst_final[rst_final["hospitalization_id"].isin(required_id)].reset_index(drop=True)

    print(f"base: {base.shape}  |  adt_cohort: {adt_cohort.shape}  |  rst_cohort: {rst_cohort.shape}")
    return adt_cohort, base, required_id, rst_cohort


# ---------------------------------------------------------------------------
# Cell 7: Load medications — filter, TZ convert, unit filter
# ---------------------------------------------------------------------------
@app.cell
def _(hospitalization_to_block, pc, required_id):
    _MAC_CATEGORIES = [
        "norepinephrine", "epinephrine", "phenylephrine", "angiotensin",
        "vasopressin", "dopamine", "dobutamine", "milrinone", "isoproterenol",
        "cisatracurium", "vecuronium", "rocuronium",
        "fentanyl", "propofol", "lorazepam", "midazolam",
        "hydromorphone", "morphine",
    ]
    _MAC_COLS = [
        "hospitalization_id", "admin_dttm", "med_dose",
        "med_category", "med_dose_unit", "mar_action_group",
    ]

    _mac_raw = pc.load_data("clif_medication_admin_continuous")
    _mac_raw["hospitalization_id"] = (
        _mac_raw["hospitalization_id"].astype(str).map(hospitalization_to_block).astype(str)
    )

    # Filter to cohort + relevant meds
    _mac_filtered = _mac_raw[
        _mac_raw["hospitalization_id"].isin(required_id)
        & _mac_raw["med_category"].isin(_MAC_CATEGORIES)
    ].reset_index(drop=True)

    # Administered-only (CLIF 2.1)
    if "mar_action_group" in _mac_filtered.columns:
        _before = len(_mac_filtered)
        _mac_filtered = _mac_filtered[_mac_filtered["mar_action_group"].str.lower() == "administered"]
        print(f"mar_action_group filter: {_before:,} -> {len(_mac_filtered):,} rows")
    else:
        print("WARNING: mar_action_group column not found. Cannot filter to administered-only records.")

    _mac_filtered = _mac_filtered[[c for c in _MAC_COLS if c in _mac_filtered.columns]].reset_index(drop=True)
    _mac_tz = pc.convert_datetime_columns_to_site_tz(_mac_filtered, pc.helper["your_site_timezone"])
    _mac_tz["admin_dttm"] = pc.getdttm(_mac_tz["admin_dttm"])
    _mac_tz["med_dose_unit"] = _mac_tz["med_dose_unit"].str.lower()

    # Keep only dose-rate units (contains "/" but not "units/hr")
    mac = _mac_tz[
        _mac_tz["med_dose_unit"].str.contains(r"/", na=False)
        & (_mac_tz["med_dose_unit"] != "units/hr")
    ].reset_index(drop=True)

    pc.deftime(mac["admin_dttm"])
    print(f"mac rows: {len(mac):,}")
    return (mac,)


# ---------------------------------------------------------------------------
# Cell 8: Load patient assessments — remap, pivot-ready long format
# ---------------------------------------------------------------------------
@app.cell
def _(hospitalization_to_block, np, pc, pd, required_id):
    _PAT_ASSESS_CATS = [
        "sbt_delivery_pass_fail", "sbt_screen_pass_fail",
        "sat_delivery_pass_fail", "sat_screen_pass_fail",
        "gcs_total", "rass",
    ]
    _CAT_MAP = {
        "negative": 0, "fail": 0, "pass": 1, "positive": 1,
        "yes": 1, "no": 0, None: np.nan, np.nan: np.nan,
    }
    _PA_COLS = ["hospitalization_id", "recorded_dttm", "numerical_value", "categorical_value", "assessment_category"]

    _pa_raw = pc.load_data("clif_patient_assessments", -1)
    _pa_raw["assessment_category"] = _pa_raw["assessment_category"].str.lower()
    _pa_raw = _pa_raw[_pa_raw["assessment_category"].isin(_PAT_ASSESS_CATS)][_PA_COLS].reset_index(drop=True)
    _pa_raw = pc.convert_datetime_columns_to_site_tz(_pa_raw, pc.helper["your_site_timezone"])
    _pa_raw["recorded_dttm"] = pc.getdttm(_pa_raw["recorded_dttm"])

    # Remap to encounter blocks
    _pa_raw["hospitalization_id"] = _pa_raw["hospitalization_id"].astype(str)
    _unmapped = _pa_raw["hospitalization_id"].map(hospitalization_to_block).isna().sum()
    if _unmapped > 0:
        print(f"WARNING: {_unmapped} patient_assessment rows with unmapped hospitalization_id (dropped)")
    _pa_raw["hospitalization_id"] = _pa_raw["hospitalization_id"].map(hospitalization_to_block)
    _pa_mapped = _pa_raw[_pa_raw["hospitalization_id"].notna()].copy()
    _pa_mapped["hospitalization_id"] = _pa_mapped["hospitalization_id"].astype(str)
    _pa_cohort = _pa_mapped[_pa_mapped["hospitalization_id"].isin(required_id)][_PA_COLS].reset_index(drop=True)

    # Map categorical values then resolve assessment_value (category-first for pass/fail)
    _pa_cohort["categorical_value"] = _pa_cohort["categorical_value"].str.lower().map(_CAT_MAP)

    def _compute_value(row):
        if row["assessment_category"].endswith("_pass_fail"):
            return row["categorical_value"] if pd.notnull(row["categorical_value"]) else row["numerical_value"]
        return row["numerical_value"] if pd.notnull(row["numerical_value"]) else row["categorical_value"]

    _pa_cohort["assessment_value"] = _pa_cohort.apply(_compute_value, axis=1)
    pat_at = _pa_cohort.drop(columns=["numerical_value", "categorical_value"])

    pc.deftime(pat_at["recorded_dttm"])
    print(f"pat_at rows: {len(pat_at):,}  |  categories: {sorted(pat_at['assessment_category'].unique())}")
    return (pat_at,)


# ---------------------------------------------------------------------------
# Cell 9: Load vitals — outlier cleaning, weight extraction
# ---------------------------------------------------------------------------
@app.cell
def _(hospitalization_to_block, np, outlier_cfg, pc, required_id):
    _VIT_CATS = ["map", "heart_rate", "sbp", "dbp", "spo2", "respiratory_rate", "weight_kg", "height_cm"]
    _VIT_COLS = ["hospitalization_id", "recorded_dttm_min", "vital_category", "vital_value"]

    _vit_raw = pc.load_data("clif_vitals", -1)
    _vit_raw["hospitalization_id"] = (
        _vit_raw["hospitalization_id"].astype(str).map(hospitalization_to_block).astype(str)
    )
    _vit_raw["vital_category"] = _vit_raw["vital_category"].str.lower()
    _vit_raw = pc.convert_datetime_columns_to_site_tz(_vit_raw, pc.helper["your_site_timezone"])
    _vit_raw["recorded_dttm_min"] = pc.getdttm(_vit_raw["recorded_dttm"])

    _vit_cohort = (
        _vit_raw[
            _vit_raw["hospitalization_id"].isin(required_id)
            & _vit_raw["vital_category"].isin(_VIT_CATS)
        ][_VIT_COLS]
        .sort_values(["hospitalization_id", "recorded_dttm_min"])
        .groupby(["hospitalization_id", "vital_category", "recorded_dttm_min"], as_index=False)
        .agg({"vital_value": "first"})
        .reset_index(drop=True)
    )
    _vit_cohort["vital_value"] = _vit_cohort["vital_value"].astype(float)

    # Apply outlier thresholds
    if outlier_cfg:
        for _vcat in _VIT_CATS:
            if _vcat in outlier_cfg:
                _lo, _hi = outlier_cfg[_vcat]
                _mask = (_vit_cohort["vital_category"] == _vcat) & ~_vit_cohort["vital_value"].between(_lo, _hi)
                _vit_cohort.loc[_mask, "vital_value"] = np.nan
        print(f"Vitals outlier thresholds applied. NaN count: {_vit_cohort['vital_value'].isna().sum():,}")

    vit = _vit_cohort.reset_index(drop=True)
    vit_weight = vit[vit["vital_category"] == "weight_kg"].reset_index(drop=True)

    pc.deftime(vit["recorded_dttm_min"])
    print(f"vit rows: {len(vit):,}")
    return vit, vit_weight


# ---------------------------------------------------------------------------
# Cell 10: Med unit conversion — forward-fill weight, convert to standard units,
#          apply outlier thresholds to doses
# ---------------------------------------------------------------------------
@app.cell
def _(mac, np, outlier_cfg, pd, tqdm, vit_weight):
    # ---- Forward-fill patient weight into mac rows -------------------------
    _weight_for_mac = vit_weight.rename(
        columns={"vital_category": "med_category", "recorded_dttm_min": "admin_dttm"}
    ).copy()
    _mac_with_weight = (
        pd.concat([mac, _weight_for_mac], ignore_index=True)
        .sort_values(["hospitalization_id", "admin_dttm"])
    )
    _mac_with_weight["vital_value"] = _mac_with_weight.groupby("hospitalization_id")["vital_value"].ffill().bfill()
    _mac_no_weight = _mac_with_weight[_mac_with_weight["med_category"] != "weight_kg"].reset_index(drop=True)
    _mac_pos_weight = _mac_no_weight[_mac_no_weight["vital_value"] > 0].reset_index(drop=True)

    # ---- Unit conversion ---------------------------------------------------
    _MED_UNIT_INFO = {
        "norepinephrine": {"required_unit": "mcg/kg/min", "acceptable_units": ["mcg/kg/min", "mcg/kg/hr", "mg/kg/hr", "mcg/min", "mg/hr"]},
        "epinephrine":    {"required_unit": "mcg/kg/min", "acceptable_units": ["mcg/kg/min", "mcg/kg/hr", "mg/kg/hr", "mcg/min", "mg/hr"]},
        "phenylephrine":  {"required_unit": "mcg/kg/min", "acceptable_units": ["mcg/kg/min", "mcg/kg/hr", "mg/kg/hr", "mcg/min", "mg/hr"]},
        "angiotensin":    {"required_unit": "ng/kg/min",  "acceptable_units": ["ng/kg/min", "ng/kg/hr"]},
        "vasopressin":    {"required_unit": "units/min",  "acceptable_units": ["units/min", "units/hr", "milliunits/min", "milliunits/hr"]},
        "dopamine":       {"required_unit": "mcg/kg/min", "acceptable_units": ["mcg/kg/min", "mcg/kg/hr", "mg/kg/hr", "mcg/min", "mg/hr"]},
        "dobutamine":     {"required_unit": "mcg/kg/min", "acceptable_units": ["mcg/kg/min", "mcg/kg/hr", "mg/kg/hr", "mcg/min", "mg/hr"]},
        "milrinone":      {"required_unit": "mcg/kg/min", "acceptable_units": ["mcg/kg/min", "mcg/kg/hr", "mg/kg/hr", "mcg/min", "mg/hr"]},
        "isoproterenol":  {"required_unit": "mcg/kg/min", "acceptable_units": ["mcg/kg/min", "mcg/kg/hr", "mg/kg/hr", "mcg/min", "mg/hr"]},
    }
    _MED_MASS_CONV = {
        ("mg", "mcg"): 1000.0, ("mcg", "mg"): 0.001,
        ("milliunits", "units"): 0.001, ("units", "milliunits"): 1000.0,
    }

    def _convert_med_dose(row):
        cat = row["med_category"]
        if cat not in _MED_UNIT_INFO:
            return row
        info = _MED_UNIT_INFO[cat]
        req_unit = info["required_unit"]
        cur_unit = row["med_dose_unit"]
        if cur_unit == req_unit or cur_unit not in info["acceptable_units"]:
            return row
        weight = row["vital_value"]
        factor = 1.0
        # Weight normalization
        if "kg" in cur_unit and "kg" not in req_unit:
            factor *= weight
        elif "kg" not in cur_unit and "kg" in req_unit:
            factor /= weight
        # Time normalization
        if "hr" in cur_unit and "min" in req_unit:
            factor /= 60.0
        elif "min" in cur_unit and "hr" in req_unit:
            factor *= 60.0
        # Mass unit conversion
        _cur_mass = cur_unit.split("/")[0]
        _req_mass = req_unit.split("/")[0]
        if _cur_mass != _req_mass:
            _mfactor = _MED_MASS_CONV.get((_cur_mass, _req_mass))
            if _mfactor is None:
                return row
            factor *= _mfactor
        row["med_dose"] = row["med_dose"] * factor
        row["med_dose_unit"] = req_unit
        return row

    tqdm.pandas(desc="Converting medication doses")
    _mac_converted = _mac_pos_weight.progress_apply(_convert_med_dose, axis=1)

    # Apply outlier thresholds to converted doses
    if outlier_cfg:
        _before_count = _mac_converted["med_dose"].notna().sum()
        for _mcat in _mac_converted["med_category"].unique():
            if _mcat in outlier_cfg:
                _lo, _hi = outlier_cfg[_mcat]
                _mask = (_mac_converted["med_category"] == _mcat) & ~_mac_converted["med_dose"].between(_lo, _hi)
                _mac_converted.loc[_mask, "med_dose"] = np.nan
        print(f"Med outlier thresholds: {_before_count:,} -> {_mac_converted['med_dose'].notna().sum():,} non-null doses")

    mac_final = _mac_converted.reset_index(drop=True)
    print(f"mac_final rows: {len(mac_final):,}")

    # Summary table for QC
    _summary = (
        mac_final
        .groupby(["med_category", "med_dose_unit"])
        .agg(
            total_N=("med_category", "size"),
            min=("med_dose", "min"),
            max=("med_dose", "max"),
            q25=("med_dose", lambda x: x.quantile(0.25)),
            median=("med_dose", lambda x: x.quantile(0.50)),
            q75=("med_dose", lambda x: x.quantile(0.75)),
            missing=("med_dose", lambda x: x.isna().sum()),
        )
        .reset_index()
    )
    _summary
    return (mac_final,)


# ---------------------------------------------------------------------------
# Cell 11: Build wide cohort — union all event timestamps with polars,
#          then join rst, adt, mac pivot, pat_at pivot, vitals pivot
# ---------------------------------------------------------------------------
@app.cell
def _(adt_cohort, base, mac_final, pat_at, pc, pd, pl, rst_cohort, vit):
    # ---- Build combo_id key helper -----------------------------------------
    def _make_combo(df_pl: pl.DataFrame, id_col: str, time_col: str) -> pl.DataFrame:
        return (
            df_pl
            .filter(pl.col(time_col).is_not_null())
            .select([
                pl.col(id_col).alias("hospitalization_id"),
                pl.col(time_col).alias("event_time"),
            ])
            .with_columns(
                (
                    pl.col("hospitalization_id") + "_"
                    + pl.col("event_time").dt.strftime("%Y%m%d%H%M")
                ).alias("combo_id")
            )
        )

    # ---- Convert each source to polars and collect unique event times ------
    _base_pl  = pl.from_pandas(base)
    _rst_pl   = pl.from_pandas(rst_cohort)
    _adt_pl   = pl.from_pandas(adt_cohort)
    _pat_pl   = pl.from_pandas(pat_at)
    _mac_pl   = pl.from_pandas(mac_final)
    _vit_pl   = pl.from_pandas(vit)

    _uni_times = pl.concat([
        _make_combo(_rst_pl,  "hospitalization_id", "recorded_dttm"),
        _make_combo(_adt_pl,  "hospitalization_id", "in_dttm"),
        _make_combo(_pat_pl,  "hospitalization_id", "recorded_dttm"),
        _make_combo(_mac_pl,  "hospitalization_id", "admin_dttm"),
        _make_combo(_vit_pl,  "hospitalization_id", "recorded_dttm_min"),
    ]).unique(subset=["hospitalization_id", "event_time"])

    # ---- Build base × event_time spine ------------------------------------
    _spine = (
        _base_pl
        .join(
            _uni_times.select(["hospitalization_id", "event_time"]),
            on="hospitalization_id",
            how="left",
        )
        .unique()
        .with_columns(
            (
                pl.col("hospitalization_id") + "_"
                + pl.col("event_time").dt.strftime("%Y%m%d%H%M")
            ).alias("combo_id")
        )
    )

    # ---- Pivot patient assessments -----------------------------------------
    _pat_wide_pl = (
        _pat_pl
        .filter(pl.col("recorded_dttm").is_not_null())
        .with_columns(
            (
                pl.col("hospitalization_id") + "_"
                + pl.col("recorded_dttm").dt.strftime("%Y%m%d%H%M")
            ).alias("combo_id")
        )
        .pivot(
            values="assessment_value",
            index="combo_id",
            on="assessment_category",
            aggregate_function="first",
        )
    )

    # ---- Pivot medications -------------------------------------------------
    _mac_wide_pl = (
        _mac_pl
        .filter(pl.col("admin_dttm").is_not_null())
        .with_columns(
            (
                pl.col("hospitalization_id") + "_"
                + pl.col("admin_dttm").dt.strftime("%Y%m%d%H%M")
            ).alias("combo_id")
        )
        .pivot(
            values="med_dose",
            index="combo_id",
            on="med_category",
            aggregate_function="min",
        )
    )

    # ---- Pivot vitals -------------------------------------------------------
    _vit_wide_pl = (
        _vit_pl
        .filter(pl.col("recorded_dttm_min").is_not_null())
        .with_columns(
            (
                pl.col("hospitalization_id") + "_"
                + pl.col("recorded_dttm_min").dt.strftime("%Y%m%d%H%M")
            ).alias("combo_id")
        )
        .pivot(
            values="vital_value",
            index="combo_id",
            on="vital_category",
            aggregate_function="first",
        )
    )

    # ---- RST with combo_id -------------------------------------------------
    _rst_wide_pl = (
        _rst_pl
        .filter(pl.col("recorded_dttm").is_not_null())
        .with_columns(
            (
                pl.col("hospitalization_id") + "_"
                + pl.col("recorded_dttm").dt.strftime("%Y%m%d%H%M")
            ).alias("combo_id")
        )
    )

    # ---- ADT with combo_id -------------------------------------------------
    _adt_wide_pl = (
        _adt_pl
        .filter(pl.col("in_dttm").is_not_null())
        .with_columns(
            (
                pl.col("hospitalization_id") + "_"
                + pl.col("in_dttm").dt.strftime("%Y%m%d%H%M")
            ).alias("combo_id")
        )
    )

    # ---- Join everything onto the spine ------------------------------------
    # Drop duplicate join keys from right-hand tables before each join
    def _drop_dupe_cols(df: pl.DataFrame, existing_cols: list[str], keep: list[str]) -> pl.DataFrame:
        drop = [c for c in df.columns if c in existing_cols and c not in keep]
        return df.drop(drop)

    _spine_cols = _spine.columns

    _rst_join = _drop_dupe_cols(_rst_wide_pl, _spine_cols, ["combo_id"])
    _adt_join = _drop_dupe_cols(_adt_wide_pl, _spine_cols, ["combo_id"])

    _joined = (
        _spine
        .join(_adt_join, on="combo_id", how="left", suffix="_adt")
        .join(_rst_join, on="combo_id", how="left", suffix="_rst")
        .join(_mac_wide_pl, on="combo_id", how="left", suffix="_mac")
        .join(_pat_wide_pl, on="combo_id", how="left", suffix="_pat")
        .join(_vit_wide_pl, on="combo_id", how="left", suffix="_vit")
        .unique()
    )

    # Convert back to pandas for remaining pandas-dependent steps
    wide_df = _joined.to_pandas()
    wide_df["event_time"] = pd.to_datetime(wide_df["event_time"])
    pc.deftime(wide_df["event_time"])
    print(f"wide_df shape: {wide_df.shape}")
    return (wide_df,)


# ---------------------------------------------------------------------------
# Cell 12: Day numbering + encounter blocks
# ---------------------------------------------------------------------------
@app.cell
def _(wide_df):
    from definitions_source_of_truth import VENT_DAY_ANCHOR_HOUR
    import pandas as _pd2

    _df = wide_df.sort_values(["hospitalization_id", "event_time"]).reset_index(drop=True).copy()

    # Ventilator-day anchor: shift back VENT_DAY_ANCHOR_HOUR hours before flooring to date
    _df["date"] = (_df["event_time"] - _pd2.Timedelta(hours=VENT_DAY_ANCHOR_HOUR)).dt.date

    if "imv_episode_id" in _df.columns:
        _df["day_number"] = (
            _df.groupby(["hospitalization_id", "imv_episode_id"])["date"]
            .rank(method="dense")
            .astype("Int64")
        )
        _df["hosp_id_day_key"] = (
            _df["imv_episode_id"].astype(str) + "_day_" + _df["day_number"].astype(str)
        )
    else:
        print("WARNING: imv_episode_id not found. Using hospitalization-level day numbering.")
        _df["day_number"] = (
            _df.groupby("hospitalization_id")["date"]
            .rank(method="dense")
            .astype("Int64")
        )
        _df["hosp_id_day_key"] = (
            _df["hospitalization_id"].astype(str) + "_day_" + _df["day_number"].astype(str)
        )

    study_cohort = _df
    print(f"study_cohort shape: {study_cohort.shape}")
    return (study_cohort,)


# ---------------------------------------------------------------------------
# Cell 13: Flowsheet flags — ensure all expected columns exist
# ---------------------------------------------------------------------------
@app.cell
def _(np, study_cohort):
    _ASSESSMENT_COLS = [
        "sbt_delivery_pass_fail", "sbt_screen_pass_fail",
        "sat_delivery_pass_fail", "sat_screen_pass_fail",
        "rass", "gcs_total",
    ]
    _MED_COLS = [
        "norepinephrine", "epinephrine", "phenylephrine", "angiotensin",
        "vasopressin", "dopamine", "dobutamine", "milrinone", "isoproterenol",
        "cisatracurium", "vecuronium", "rocuronium",
        "fentanyl", "propofol", "lorazepam", "midazolam", "hydromorphone", "morphine",
    ]

    study_cohort_final = study_cohort.copy()

    for _col in _ASSESSMENT_COLS:
        if _col not in study_cohort_final.columns:
            study_cohort_final[_col] = np.nan
            print(
                f"Column '{_col}' missing for your site. Filled with NaN. "
                "If unintended, verify your data element."
            )

    for _col in _MED_COLS:
        if _col not in study_cohort_final.columns:
            study_cohort_final[_col] = np.nan
            print(f"mCIDE: '{_col}' missing. Check your CLIF Meds table.")

    print(f"study_cohort_final shape: {study_cohort_final.shape}")
    return (study_cohort_final,)


# ---------------------------------------------------------------------------
# Cell 14: Save outputs
# ---------------------------------------------------------------------------
@app.cell
def _(os, pc, study_cohort_final):
    # Intermediate outputs
    study_cohort_final.to_csv("../output/intermediate/study_cohort.csv", index=False)
    study_cohort_final.to_parquet("../output/intermediate/study_cohort.parquet", index=False)
    print("Intermediate files written.")

    # Final per-site directory
    _site_dir = os.path.join("../output/final/", pc.helper["site_name"])
    if not os.path.exists(_site_dir):
        os.makedirs(_site_dir)
        print(f"Directory '{_site_dir}' created.")
    else:
        print(f"Directory '{_site_dir}' already exists.")

    print("Cohort creation completed!")
    return


if __name__ == "__main__":
    app.run()
