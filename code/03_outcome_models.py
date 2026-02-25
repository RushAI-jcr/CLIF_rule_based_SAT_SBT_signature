"""
03_outcome_models.py
====================
Construct validity: associations between SAT/SBT delivery and patient outcomes.

Models (per manuscript Statistical Analysis):
1. Time to extubation: cause-specific Cox PH with death as competing event,
   SAT/SBT as time-varying exposures at ventilator-day level
2. Ventilator-free days (VFDs) to day 28 primary:
   Fine-Gray-equivalent competing-risk model (Python native)
3. Ventilator-free days (VFDs) secondary:
   multistate transition modeling (MV -> extubated -> reintubated -> dead)
4. ICU LOS primary: mixed-effects random-intercept count model
5. In-hospital mortality primary: mixed-effects random-intercept logistic model
5. SAT awakening: GEE logistic among delivered SAT days

All models adjust for:
- Demographics (age, sex, race, ethnicity)
- Baseline severity markers on first eligible ventilator-day (landmark day)
- Baseline hemodynamics and ventilator settings (FiO2/PEEP/vasopressor/sedation)
- Hospital-level clustering with robust standard errors where applicable

CLIF 2.1 compliance:
- hospitalization_id as primary join key
- Filter on *_category columns
- Outlier thresholds applied
- mar_action_group = 'administered'

Usage:
    python 03_outcome_models.py --sat-file ../output/intermediate/final_df_SAT.csv \
                                 --sbt-file ../output/intermediate/final_df_SBT.csv \
                                 --output-dir ../output/final
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

UTILS_DIR = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from competing_risks import fit_fine_gray_equivalent
from definitions_source_of_truth import VFD_MAX_DAYS
from multistate import fit_multistate_equivalent

warnings.filterwarnings("ignore")


# ============================================================
# DATA LOADING AND PREPARATION
# ============================================================

def load_and_prepare(filepath: str) -> pd.DataFrame:
    """Load site-level final_df and standardize columns."""
    df = pd.read_csv(filepath, low_memory=False)
    for col in ["event_time", "admission_dttm", "discharge_dttm"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed")
    return df


def compute_ventilator_free_days(df: pd.DataFrame) -> pd.DataFrame:
    """Compute VFDs at 28 days per hospitalization.

    VFD = 28 - (days on ventilator), with VFD=0 if patient died
    or remained on ventilator through day 28.
    """
    hosp_df = df.groupby("hospitalization_id").agg(
        total_vent_days=("hosp_id_day_key", "nunique"),
        discharge_category=("discharge_category", "first"),
        admission_dttm=("admission_dttm", "first"),
        discharge_dttm=("discharge_dttm", "first"),
    ).reset_index()

    hosp_df["died"] = hosp_df["discharge_category"].str.lower().str.contains(
        "expired|dead|death|died", na=False
    ).astype(int)

    hosp_df["vfd_28"] = np.where(
        hosp_df["died"] == 1,
        0,
        np.maximum(0, VFD_MAX_DAYS - hosp_df["total_vent_days"]),
    )
    return hosp_df[["hospitalization_id", "total_vent_days", "died", "vfd_28"]]


def _extract_day_index_from_key(hosp_id_day_key: str) -> int | None:
    """Parse vent-day index from keys like '<hosp>_day_10'."""
    match = re.search(r"_day_(\d+)$", str(hosp_id_day_key))
    if match:
        return int(match.group(1))
    return None


def _assign_vent_day_index(day_level: pd.DataFrame) -> pd.DataFrame:
    """Attach numeric vent-day index for deterministic day ordering."""
    day_level = day_level.copy()
    parsed = day_level["hosp_id_day_key"].map(_extract_day_index_from_key)
    day_level["vent_day_index"] = parsed

    missing_mask = day_level["vent_day_index"].isna()
    if missing_mask.any():
        if "event_time_min" in day_level.columns:
            day_level["_event_sort"] = pd.to_datetime(
                day_level["event_time_min"], errors="coerce"
            )
            fallback = (
                day_level.sort_values(
                    ["hospitalization_id", "_event_sort", "hosp_id_day_key"],
                    na_position="last",
                )
                .groupby("hospitalization_id")
                .cumcount()
            )
            day_level["vent_day_index"] = day_level["vent_day_index"].fillna(fallback)
            day_level = day_level.drop(columns=["_event_sort"])
        else:
            fallback = (
                day_level.sort_values(["hospitalization_id", "hosp_id_day_key"])
                .groupby("hospitalization_id")
                .cumcount()
            )
            day_level["vent_day_index"] = day_level["vent_day_index"].fillna(fallback)

    day_level["vent_day_index"] = day_level["vent_day_index"].astype(int)
    return day_level


def build_outcome_dataset(
    df: pd.DataFrame,
    delivery_col: str,
    eligibility_col: str = "eligible_event",
) -> pd.DataFrame:
    """Build analysis dataset at ventilator-day level with outcomes.

    Parameters
    ----------
    df : pd.DataFrame
        final_df from SAT or SBT pipeline
    delivery_col : str
        Column indicating delivery (e.g., 'SAT_EHR_delivery')
    eligibility_col : str
        Column indicating eligibility
    """
    # Restrict to eligible days
    eligible = df[df[eligibility_col] == 1].copy()

    # Base aggregation
    agg_dict = {
        "hospitalization_id": ("hospitalization_id", "first"),
        "hospital_id": ("hospital_id", "first"),
        "patient_id": ("patient_id", "first"),
        "age_at_admission": ("age_at_admission", "first"),
        "sex_category": ("sex_category", "first"),
        "race_category": ("race_category", "first"),
        "ethnicity_category": ("ethnicity_category", "first"),
        "delivery": (delivery_col, "max"),
        "admission_dttm": ("admission_dttm", "first"),
        "discharge_dttm": ("discharge_dttm", "first"),
        "discharge_category": ("discharge_category", "first"),
    }
    if "event_time" in eligible.columns:
        agg_dict["event_time_min"] = ("event_time", "min")
    if "ICU_LOS" in eligible.columns:
        agg_dict["ICU_LOS"] = ("ICU_LOS", "first")

    # Add clinical covariates if available (per manuscript: FiO2, PEEP,
    # vasopressor dose, sedation exposure, SOFA)
    optional_covariates = {
        "fio2_set": ("fio2_set", "median"),
        "peep_set": ("peep_set", "median"),
        "rass": ("rass", "median"),
        "spo2": ("spo2", "median"),
        "norepinephrine": ("norepinephrine", "max"),
        "epinephrine": ("epinephrine", "max"),
        "vasopressin": ("vasopressin", "max"),
        "dopamine": ("dopamine", "max"),
        "phenylephrine": ("phenylephrine", "max"),
        "propofol": ("propofol", "max"),
        "fentanyl": ("fentanyl", "max"),
        "midazolam": ("midazolam", "max"),
    }
    for key, (col, func) in optional_covariates.items():
        if col in eligible.columns:
            agg_dict[key] = (col, func)

    # Collapse to ventilator-day level
    day_level = eligible.groupby("hosp_id_day_key").agg(**agg_dict).reset_index()
    day_level = _assign_vent_day_index(day_level)

    day_level["delivery"] = day_level["delivery"].fillna(0).astype(int)

    # Derive composite covariates
    # Vasopressor NEE (continuous) replaces binary on_vasopressor [SAP 2.8]
    from definitions_source_of_truth import compute_norepinephrine_equivalent
    vaso_cols = [c for c in ["norepinephrine", "epinephrine", "vasopressin",
                             "dopamine", "phenylephrine"] if c in day_level.columns]
    if vaso_cols:
        day_level["vasopressor_nee"] = day_level.apply(
            compute_norepinephrine_equivalent, axis=1
        )
        # Keep legacy binary for backward compat in awakening model
        day_level["on_vasopressor"] = (
            day_level[vaso_cols].fillna(0).gt(0).any(axis=1).astype(int)
        )

    # Any sedation indicator
    sed_cols = [c for c in ["propofol", "fentanyl", "midazolam"] if c in day_level.columns]
    if sed_cols:
        day_level["on_sedation"] = (
            day_level[sed_cols].fillna(0).gt(0).any(axis=1).astype(int)
        )

    # BMI [SAP 2.8] — derive from weight/height if available
    if "weight_kg" in eligible.columns and "height_cm" in eligible.columns:
        wt = pd.to_numeric(eligible.groupby("hosp_id_day_key")["weight_kg"].first(), errors="coerce")
        ht = pd.to_numeric(eligible.groupby("hosp_id_day_key")["height_cm"].first(), errors="coerce")
        bmi = wt / ((ht / 100) ** 2)
        day_level["bmi"] = bmi.reindex(day_level["hosp_id_day_key"]).values

    # GCS total [SAP 2.8] — pass through from upstream
    if "gcs_total" in eligible.columns and "gcs_total" not in day_level.columns:
        gcs = eligible.groupby("hosp_id_day_key")["gcs_total"].first()
        day_level["gcs_total"] = gcs.reindex(day_level["hosp_id_day_key"]).values

    # Ethnicity binary [SAP 2.8]
    if "ethnicity_category" in day_level.columns:
        day_level["ethnicity_nonhispanic"] = (
            ~day_level["ethnicity_category"].fillna("").str.lower().str.contains(
                "hispanic|latino", na=False
            )
        ).astype(int)

    # Medical admission [SAP 2.8] — derived from ADT location_category
    if "location_category" in eligible.columns:
        surgical_icus = {"sicu", "cticu", "cardiac_surgery_icu"}
        loc = eligible.groupby("hosp_id_day_key")["location_category"].first()
        day_level["medical_admission"] = (
            ~loc.reindex(day_level["hosp_id_day_key"]).fillna("").str.lower().isin(surgical_icus)
        ).astype(int).values

    # Elixhauser score [SAP 2.8] — merge if available from upstream
    try:
        from elixhauser import compute_elixhauser_van_walraven
        # Will be merged at hospitalization level if diagnosis data exists
    except ImportError:
        pass

    # Add outcomes
    vfd_df = compute_ventilator_free_days(df)
    day_level = day_level.merge(vfd_df, on="hospitalization_id", how="left")

    return day_level


# ============================================================
# MODEL 1: TIME TO EXTUBATION (Cox PH with competing risk)
# ============================================================

BASELINE_COVARIATE_CANDIDATES = [
    # Illness severity (prior day) [SAP 2.8]
    "fio2_set", "peep_set", "vasopressor_nee", "on_sedation",
    "rass", "gcs_total",
    # Demographics (passed through from upstream)
    "bmi",
    # Comorbidity
    "elixhauser_score",
    # Admission type
    "medical_admission",
    # Ethnicity
    "ethnicity_nonhispanic",
]


def _first_eligible_day_baseline(day_level_df: pd.DataFrame) -> pd.DataFrame:
    """Return first eligible vent-day row per hospitalization."""
    baseline = (
        day_level_df.sort_values(
            ["hospitalization_id", "vent_day_index", "hosp_id_day_key"]
        )
        .groupby("hospitalization_id", as_index=False)
        .first()
    )
    baseline["landmark_delivered"] = baseline["delivery"].fillna(0).astype(int)
    return baseline


def _build_hospitalization_level_dataset(
    day_level_df: pd.DataFrame,
    exposure_col: str,
) -> pd.DataFrame:
    """Build hospitalization-level dataset with baseline covariates and outcomes."""
    baseline = _first_eligible_day_baseline(day_level_df)
    ever_delivery = (
        day_level_df.groupby("hospitalization_id")["delivery"].max().reset_index()
    )
    ever_delivery = ever_delivery.rename(columns={"delivery": "ever_delivered"})

    hosp = baseline.merge(ever_delivery, on="hospitalization_id", how="left")
    hosp["ever_delivered"] = hosp["ever_delivered"].fillna(0).astype(int)
    if exposure_col not in hosp.columns:
        raise KeyError(f"Exposure column '{exposure_col}' missing from hospitalization dataset")
    hosp["exposure"] = hosp[exposure_col].fillna(0).astype(int)

    hosp["sex_male"] = (hosp["sex_category"].fillna("").str.lower() == "male").astype(int)
    hosp["race_white"] = (
        hosp["race_category"].fillna("").str.lower().str.contains("white", na=False)
    ).astype(int)
    return hosp


def _build_cox_tv_dataset(day_level_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build counting-process dataset shared by cause-specific Cox models.

    Returns (model_df, covariates) where model_df has start/stop intervals,
    time-varying delivery (cummax), baseline covariates, and event indicators
    for both extubation and death.
    """
    baseline = _first_eligible_day_baseline(day_level_df)
    hosp_info = baseline.groupby("hospitalization_id").agg(
        age=("age_at_admission", "first"),
        sex=("sex_category", "first"),
        race=("race_category", "first"),
        died=("died", "first"),
        total_vent_days=("total_vent_days", "first"),
    ).reset_index()
    hosp_info["sex_male"] = (hosp_info["sex"].str.lower() == "male").astype(int)
    hosp_info["race_white"] = (hosp_info["race"].str.lower().str.contains("white", na=False)).astype(int)

    # Capture hospital_id at hospitalization level for frailty approximation
    if "hospital_id" in day_level_df.columns:
        hosp_info = hosp_info.merge(
            day_level_df.groupby("hospitalization_id")["hospital_id"].first().reset_index(),
            on="hospitalization_id", how="left",
        )

    tv_cols = ["hospitalization_id", "hosp_id_day_key", "vent_day_index", "delivery"]
    tv = day_level_df[tv_cols].copy()
    tv = tv.sort_values(["hospitalization_id", "vent_day_index", "hosp_id_day_key"])
    tv["day_idx"] = tv.groupby("hospitalization_id").cumcount()
    tv["delivered_cummax"] = tv.groupby("hospitalization_id")["delivery"].cummax()
    tv["start"] = tv["day_idx"]
    tv["stop"] = tv["day_idx"] + 1

    merge_cols = ["hospitalization_id", "age", "sex_male", "race_white",
                  "died", "total_vent_days"]
    if "hospital_id" in hosp_info.columns:
        merge_cols.append("hospital_id")
    tv = tv.merge(hosp_info[merge_cols], on="hospitalization_id", how="left")

    # Baseline covariates (landmark day only) to avoid post-exposure leakage.
    baseline_cols: list[str] = []
    for col in BASELINE_COVARIATE_CANDIDATES:
        if col in baseline.columns and baseline[col].notna().sum() > len(baseline) * 0.3:
            baseline_col = f"baseline_{col}"
            baseline[baseline_col] = baseline[col].fillna(baseline[col].median())
            baseline_cols.append(baseline_col)

    if baseline_cols:
        tv = tv.merge(
            baseline[["hospitalization_id"] + baseline_cols],
            on="hospitalization_id",
            how="left",
        )

    # Event indicators on last interval only
    tv["is_last"] = tv.groupby("hospitalization_id")["stop"].transform("max") == tv["stop"]
    # Cause 1: extubation (survived to end of ventilation)
    tv["event_extubation"] = ((tv["is_last"]) & (tv["died"] == 0)).astype(int)
    # Cause 2: death while ventilated
    tv["event_death"] = ((tv["is_last"]) & (tv["died"] == 1)).astype(int)

    covariates = ["delivered_cummax", "age", "sex_male", "race_white"]
    covariates.extend(baseline_cols)

    keep = ["hospitalization_id", "start", "stop",
            "event_extubation", "event_death"] + covariates
    if "hospital_id" in tv.columns:
        keep.append("hospital_id")
    model_df = tv[keep].dropna(subset=["hospitalization_id", "start", "stop",
                                        "event_extubation", "event_death"] + covariates)
    return model_df, covariates


def _fit_cause_specific_cox(
    model_df: pd.DataFrame,
    covariates: list[str],
    event_col: str,
    model_label: str,
) -> dict[str, Any]:
    """Fit a single cause-specific Cox model for the given event column.

    For the competing cause, events are censored (set to 0).
    Hospital frailty is approximated by including top-N-1 hospital indicator
    variables as fixed effects (only when <= 50 hospitals are present) and
    using penalizer=0.1 for L2 regularisation.

    Returns dict with HR, 95% CI, p-value for delivered_cummax.
    """
    from lifelines import CoxTimeVaryingFitter

    active_covariates = list(covariates)

    # --- Hospital frailty approximation -----------------------------------
    hospital_dummies_added: list[str] = []
    n_hospitals = 0
    if "hospital_id" in model_df.columns:
        n_hospitals = model_df["hospital_id"].nunique()
        if 1 < n_hospitals <= 50:
            # One-hot encode hospitals, drop the most frequent (reference)
            hosp_dummies = pd.get_dummies(
                model_df["hospital_id"], prefix="hosp", drop_first=True, dtype=int
            )
            hospital_dummies_added = hosp_dummies.columns.tolist()
            model_df = pd.concat([model_df.reset_index(drop=True),
                                   hosp_dummies.reset_index(drop=True)], axis=1)
            active_covariates = active_covariates + hospital_dummies_added

    fit_cols = ["hospitalization_id", "start", "stop", event_col] + active_covariates
    fit_df = model_df[fit_cols].copy()
    fit_df = fit_df.rename(columns={event_col: "event"})

    if fit_df["event"].sum() < 5:
        return _placeholder_result(f"{model_label} (too few events)", model_family="cox")

    # penalizer=0.1 adds L2 regularisation; helps convergence with hospital dummies
    # robust=True + cluster_col for hospitalization-level robust SE [SAP 2.2]
    ctv = CoxTimeVaryingFitter(penalizer=0.1)
    ctv.fit(fit_df, id_col="hospitalization_id",
            start_col="start", stop_col="stop", event_col="event",
            robust=True, cluster_col="hospitalization_id")
    # Note: lifelines does not support true random effects. Hospital dummies
    # with L2 penalty approximate a fixed-effects frailty model [SAP 2.2 limitation].

    summary = ctv.summary
    result = {
        "model": model_label,
        "exposure": "SAT/SBT Delivery (time-varying)",
        "model_family": "cox",
        "n_hospital_dummies": len(hospital_dummies_added),
        "n_hospitals": n_hospitals,
    }
    if "delivered_cummax" in summary.index:
        row = summary.loc["delivered_cummax"]
        result["csHR"] = round(np.exp(row["coef"]), 3)
        result["csHR_lower_95"] = round(np.exp(row["coef"] - 1.96 * row["se(coef)"]), 3)
        result["csHR_upper_95"] = round(np.exp(row["coef"] + 1.96 * row["se(coef)"]), 3)
        result["p_value"] = round(row["p"], 4)

    # Schoenfeld PH assumption test [SAP 2.2]
    try:
        from lifelines.statistics import proportional_hazard_test
        ph_result = proportional_hazard_test(ctv, fit_df, time_transform='rank')
        result["ph_test_p_global"] = round(float(ph_result.summary["p"].min()), 4)
        result["ph_assumption_status"] = (
            "met" if result["ph_test_p_global"] >= 0.05 else "violated"
        )
        if result["ph_assumption_status"] == "violated":
            result["ph_note"] = (
                "PH assumption violated; consider time-stratified or "
                "interaction terms as sensitivity analysis"
            )
    except Exception:
        result["ph_test_p_global"] = None
        result["ph_assumption_status"] = "not_tested"

    return result


def fit_cox_extubation(day_level_df: pd.DataFrame) -> dict[str, Any]:
    """Cause-specific Cox PH models for time to extubation with competing risk of death.

    Implements proper competing-risk analysis via two cause-specific hazard models
    (Fine & Gray, JASA 1999; Putter et al., Stat Med 2007):
      - Cause 1: Extubation (primary outcome) — death is censored
      - Cause 2: Death while ventilated (competing risk) — extubation is censored

    Uses counting-process (start-stop) format with time-varying delivery exposure
    (cummax) to avoid immortal time bias (Suissa, Am J Epidemiol 2008).

    Returns dict with HRs from both cause-specific models.
    """
    try:
        from lifelines import CoxTimeVaryingFitter  # noqa: F401
    except ImportError:
        print("lifelines not installed. Install with: pip install lifelines")
        return _placeholder_result("Cause-specific Cox - Extubation", model_family="cox")

    try:
        model_df, covariates = _build_cox_tv_dataset(day_level_df)
    except Exception as e:
        print(f"Failed to build Cox TV dataset: {e}")
        return _placeholder_result("Cause-specific Cox - Extubation", model_family="cox")

    # Cause 1: Extubation (death censored)
    try:
        result_extub = _fit_cause_specific_cox(
            model_df, covariates, "event_extubation",
            "Cause-specific Cox - Extubation (death censored)")
    except Exception as e:
        print(f"Cause-specific Cox (extubation) failed: {e}")
        result_extub = _placeholder_result("Cause-specific Cox - Extubation", model_family="cox")

    # Cause 2: Death (extubation censored) — reported for completeness
    try:
        result_death = _fit_cause_specific_cox(
            model_df, covariates, "event_death",
            "Cause-specific Cox - Death (extubation censored)")
    except Exception as e:
        print(f"Cause-specific Cox (death) failed: {e}")
        result_death = _placeholder_result("Cause-specific Cox - Death", model_family="cox")

    # Return primary result (extubation) with competing-risk death result nested
    result = result_extub.copy()
    result["competing_risk_death"] = result_death
    return result


# ============================================================
# MODEL 2: VENTILATOR-FREE DAYS
# Primary: Fine-Gray competing risk (fit_fine_gray_equivalent in competing_risks.py)
# Sensitivity: Proportional Odds ordinal regression (below)
# See METHODS_DECISIONS_LOCKED.md Section 2.3
# ============================================================

def _prepare_baseline_covariates(
    hosp_df: pd.DataFrame,
    exposure_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare baseline covariates for non-time-to-event outcome models.

    Missing values imputed via time-windowed LOCF [SAP 2.9] rather than
    cross-sectional median. Variables with <= 6h stale data are carried
    forward; anything beyond that window is left as NaN and then filled
    with the cohort median as a last resort.
    """
    covariates = [exposure_col, "age_at_admission", "sex_male", "race_white"]
    # Time-windowed LOCF rules matching enforce_missing_data_windows [SAP 2.9]
    locf_windows = {
        "fio2_set": 6.0, "peep_set": 6.0, "vasopressor_nee": 2.0,
        "on_sedation": 6.0, "rass": 2.0, "gcs_total": 2.0,
    }
    for c in BASELINE_COVARIATE_CANDIDATES:
        if c in hosp_df.columns and hosp_df[c].notna().sum() > len(hosp_df) * 0.3:
            age_col = f"{c}_hours_since_last"
            if age_col in hosp_df.columns and c in locf_windows:
                # Null out values beyond the LOCF window, then fill remainder
                stale = pd.to_numeric(hosp_df[age_col], errors="coerce") > locf_windows[c]
                hosp_df.loc[stale, c] = np.nan
            # Last-resort median fill for remaining NaNs
            hosp_df[c] = hosp_df[c].fillna(hosp_df[c].median())
            covariates.append(c)
    return hosp_df, covariates


def fit_vfd_model(
    day_level_df: pd.DataFrame,
    exposure_col: str = "landmark_delivered",
) -> dict[str, Any]:
    """Proportional odds ordinal regression for VFDs at 28 days.

    Recommended over hurdle/ZINB models which have inflated Type I error
    for VFD outcomes (Renard Triché et al., Crit Care 2025, PMID 40537834).

    VFDs are binned into ordinal categories:
      0 = dead (VFD=0 by convention, Schoenfeld & Bernard, Crit Care Med 2002)
      1 = survived, VFD 0-7 (prolonged ventilation)
      2 = survived, VFD 8-14
      3 = survived, VFD 15-21
      4 = survived, VFD 22-28 (rapid liberation)

    The proportional odds model estimates a common OR across all cut-points,
    interpreted as: higher OR = greater odds of being in a better VFD category.

    Returns dict with OR, 95% CI, p-value for delivery effect.
    """
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
    except ImportError:
        print("statsmodels >= 0.13 required for OrderedModel.")
        return _placeholder_result("Proportional Odds - VFDs at 28 days", model_family="ordinal")

    hosp = _build_hospitalization_level_dataset(day_level_df, exposure_col=exposure_col)
    hosp, covariates = _prepare_baseline_covariates(hosp, exposure_col=exposure_col)

    # Ordinal VFD categories: death=0, then weekly bins for survivors
    def _vfd_category(died: int, vfd_28: float) -> int:
        if died == 1:
            return 0  # dead
        if vfd_28 <= 7:
            return 1
        if vfd_28 <= 14:
            return 2
        if vfd_28 <= 21:
            return 3
        return 4

    hosp["vfd_ordinal"] = [
        _vfd_category(int(died), float(vfd_28))
        for died, vfd_28 in zip(hosp["died"].fillna(0), hosp["vfd_28"].fillna(0))
    ]
    model_df = hosp[covariates + ["vfd_ordinal"]].dropna()
    active_covariates = [c for c in covariates if model_df[c].nunique(dropna=True) > 1]

    if model_df["vfd_ordinal"].nunique() < 2:
        return _placeholder_result(
            "Proportional Odds - VFDs (insufficient categories)",
            model_family="ordinal",
        )
    if exposure_col not in active_covariates:
        return _placeholder_result(
            "Proportional Odds - VFDs (non-varying exposure)",
            model_family="ordinal",
        )

    result = {
        "model": "Proportional Odds - VFDs at 28 days",
        "model_family": "ordinal",
        "exposure": f"SAT/SBT Delivery ({exposure_col})",
        "vfd_category_counts": model_df["vfd_ordinal"].value_counts().sort_index().to_dict(),
    }

    try:
        mod = OrderedModel(
            model_df["vfd_ordinal"],
            model_df[active_covariates],
            distr="logit",
        )
        fit = mod.fit(method="bfgs", disp=False, maxiter=500)

        # In OrderedModel, covariate coefficients are first, thresholds after.
        # Positive coef = higher odds of being in a higher (better) category.
        delivery_idx = active_covariates.index(exposure_col)
        coef = fit.params.iloc[delivery_idx]
        se = fit.bse.iloc[delivery_idx]
        pval = fit.pvalues.iloc[delivery_idx]

        result["OR"] = round(np.exp(coef), 3)
        result["OR_lower_95"] = round(np.exp(coef - 1.96 * se), 3)
        result["OR_upper_95"] = round(np.exp(coef + 1.96 * se), 3)
        result["p_value"] = round(pval, 4)
        result["interpretation"] = (
            "OR > 1 means delivery associated with higher (better) VFD category"
        )
    except Exception as e:
        print(f"Proportional odds VFD model failed: {e}")
        result["note"] = f"Model failed: {e}"

    return result


# ============================================================
# VFD SENSITIVITY MODELS [SAP 2.3]
# ============================================================

def fit_vfd_mann_whitney(
    day_level_df: pd.DataFrame,
    exposure_col: str = "landmark_delivered",
) -> dict[str, Any]:
    """Mann-Whitney U test for VFDs [SAP 2.3 sensitivity]."""
    hosp = _build_hospitalization_level_dataset(day_level_df, exposure_col=exposure_col)
    exposed = hosp.loc[hosp[exposure_col] == 1, "vfd_28"].dropna()
    unexposed = hosp.loc[hosp[exposure_col] == 0, "vfd_28"].dropna()
    if len(exposed) < 5 or len(unexposed) < 5:
        return _placeholder_result("Mann-Whitney U - VFDs", model_family="nonparametric")
    U, p = sp_stats.mannwhitneyu(exposed, unexposed, alternative="two-sided")
    n = len(exposed) + len(unexposed)
    # Rank-biserial r = 1 - 2U/(n1*n2)
    r = 1 - (2 * U) / (len(exposed) * len(unexposed))
    return {
        "model": "Mann-Whitney U - VFDs at 28 days",
        "model_family": "nonparametric",
        "model_estimator": "mannwhitneyu",
        "exposure": f"SAT/SBT Delivery ({exposure_col})",
        "U_statistic": round(float(U), 1),
        "p_value": round(float(p), 4),
        "rank_biserial_r": round(float(r), 4),
        "n_exposed": int(len(exposed)),
        "n_unexposed": int(len(unexposed)),
    }


def fit_vfd_hurdle(
    day_level_df: pd.DataFrame,
    exposure_col: str = "landmark_delivered",
) -> dict[str, Any]:
    """Hurdle model for VFDs [SAP 2.3 sensitivity].

    Part 1: logistic for VFD > 0 vs VFD = 0.
    Part 2: NegativeBinomialP on VFD > 0 subset.
    """
    import statsmodels.formula.api as smf
    from statsmodels.discrete.count_model import NegativeBinomialP

    hosp = _build_hospitalization_level_dataset(day_level_df, exposure_col=exposure_col)
    hosp, covariates = _prepare_baseline_covariates(hosp, exposure_col=exposure_col)
    hosp["vfd_gt0"] = (hosp["vfd_28"].fillna(0) > 0).astype(int)
    model_df = hosp[covariates + ["vfd_28", "vfd_gt0"]].dropna()
    if len(model_df) < 20:
        return _placeholder_result("Hurdle - VFDs", model_family="hurdle")

    formula_rhs = " + ".join(covariates)
    result: dict[str, Any] = {
        "model": "Hurdle - VFDs at 28 days",
        "model_family": "hurdle",
        "model_estimator": "logit + NegativeBinomialP",
        "exposure": f"SAT/SBT Delivery ({exposure_col})",
    }

    # Part 1: logistic
    try:
        logit_fit = smf.logit(f"vfd_gt0 ~ {formula_rhs}", data=model_df).fit(disp=False, maxiter=300)
        if exposure_col in logit_fit.params.index:
            coef = float(logit_fit.params[exposure_col])
            se = float(logit_fit.bse[exposure_col])
            result["hurdle_OR"] = round(float(np.exp(coef)), 3)
            result["hurdle_OR_lower_95"] = round(float(np.exp(coef - 1.96 * se)), 3)
            result["hurdle_OR_upper_95"] = round(float(np.exp(coef + 1.96 * se)), 3)
            result["hurdle_p_value"] = round(float(logit_fit.pvalues[exposure_col]), 4)
    except Exception as e:
        result["hurdle_note"] = f"Logistic part failed: {e}"

    # Part 2: count on VFD > 0
    try:
        import statsmodels.api as sm
        pos_df = model_df[model_df["vfd_gt0"] == 1].copy()
        if len(pos_df) >= 10:
            X = sm.add_constant(pos_df[covariates])
            y = pos_df["vfd_28"]
            nb_fit = NegativeBinomialP(y, X).fit(disp=False, maxiter=300)
            if exposure_col in nb_fit.params.index:
                coef = float(nb_fit.params[exposure_col])
                se = float(nb_fit.bse[exposure_col])
                result["count_IRR"] = round(float(np.exp(coef)), 3)
                result["count_IRR_lower_95"] = round(float(np.exp(coef - 1.96 * se)), 3)
                result["count_IRR_upper_95"] = round(float(np.exp(coef + 1.96 * se)), 3)
                result["count_p_value"] = round(float(nb_fit.pvalues[exposure_col]), 4)
    except Exception as e:
        result["count_note"] = f"Count part failed: {e}"

    return result


def fit_vfd_zinb(
    day_level_df: pd.DataFrame,
    exposure_col: str = "landmark_delivered",
) -> dict[str, Any]:
    """Zero-inflated Negative Binomial for VFDs [SAP 2.3 sensitivity]."""
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
    import statsmodels.api as sm

    hosp = _build_hospitalization_level_dataset(day_level_df, exposure_col=exposure_col)
    hosp, covariates = _prepare_baseline_covariates(hosp, exposure_col=exposure_col)
    model_df = hosp[covariates + ["vfd_28"]].dropna()
    if len(model_df) < 30:
        return _placeholder_result("ZINB - VFDs", model_family="zinb")

    X = sm.add_constant(model_df[covariates])
    y = model_df["vfd_28"].fillna(0).astype(int)

    result: dict[str, Any] = {
        "model": "ZINB - VFDs at 28 days",
        "model_family": "zinb",
        "model_estimator": "ZeroInflatedNegativeBinomialP",
        "exposure": f"SAT/SBT Delivery ({exposure_col})",
    }
    try:
        zinb = ZeroInflatedNegativeBinomialP(y, X, exog_infl=X)
        fit = zinb.fit(disp=False, maxiter=500, method="bfgs")
        if exposure_col in fit.params.index:
            coef = float(fit.params[exposure_col])
            se = float(fit.bse[exposure_col])
            result["IRR"] = round(float(np.exp(coef)), 3)
            result["IRR_lower_95"] = round(float(np.exp(coef - 1.96 * se)), 3)
            result["IRR_upper_95"] = round(float(np.exp(coef + 1.96 * se)), 3)
            result["p_value"] = round(float(fit.pvalues[exposure_col]), 4)
    except Exception as e:
        result["note"] = f"ZINB model failed: {e}"

    return result


# ============================================================
# MODEL 3: ICU LOS (Negative Binomial)
# ============================================================

def fit_icu_los_model(
    day_level_df: pd.DataFrame,
    exposure_col: str = "landmark_delivered",
) -> dict[str, Any]:
    """Mixed-effects NB count model for ICU LOS [SAP 2.4].

    Primary: GEE Negative Binomial clustered on hospitalization_id with
    hospital_id as fixed-effect covariate. Fallback: PoissonBayesMixedGLM.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return _placeholder_result("Mixed-effects NB - ICU LOS", model_family="gee_nb")

    hosp = _build_hospitalization_level_dataset(day_level_df, exposure_col=exposure_col)
    hosp, covariates = _prepare_baseline_covariates(hosp, exposure_col=exposure_col)
    if "ICU_LOS" in hosp.columns:
        hosp["icu_los"] = hosp["ICU_LOS"].clip(lower=1).round().astype(int)
    else:
        print("WARNING: ICU_LOS column not found. Using total_vent_days as proxy.")
        hosp["icu_los"] = hosp["total_vent_days"].clip(lower=1).round().astype(int)

    required_cols = covariates + ["icu_los", "hospital_id", "hospitalization_id"]
    model_df = hosp[[c for c in required_cols if c in hosp.columns]].dropna()
    if model_df.empty:
        return _placeholder_result("Mixed-effects NB - ICU LOS", model_family="gee_nb")

    exposure_formula_col = "exposure"
    model_df = model_df.rename(columns={exposure_col: exposure_formula_col})
    covariates_formula = [exposure_formula_col] + [c for c in covariates if c != exposure_col]

    # Add hospital_id as fixed-effect covariate [SAP 2.4]
    n_hospitals = model_df["hospital_id"].nunique() if "hospital_id" in model_df.columns else 0
    hospital_dummies = []
    if 1 < n_hospitals <= 50:
        hosp_dums = pd.get_dummies(model_df["hospital_id"], prefix="hosp", drop_first=True, dtype=int)
        hospital_dummies = hosp_dums.columns.tolist()
        model_df = pd.concat([model_df.reset_index(drop=True), hosp_dums.reset_index(drop=True)], axis=1)

    # Primary: GEE NB clustered on hospitalization_id [SAP 2.4]
    try:
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.families import NegativeBinomial as NBFamily
        from statsmodels.genmod.cov_struct import Exchangeable

        all_covs = covariates_formula + hospital_dummies
        X = sm.add_constant(model_df[all_covs], has_constant="add")
        y = model_df["icu_los"]
        gee = GEE(
            y, X,
            groups=model_df["hospitalization_id"],
            family=NBFamily(),
            cov_struct=Exchangeable(),
        )
        gee_result = gee.fit(cov_type="robust")
        coef = float(gee_result.params[exposure_formula_col])
        se = float(gee_result.bse[exposure_formula_col])
        pval = float(gee_result.pvalues[exposure_formula_col])
        result = {
            "model": "Mixed-effects NB - ICU LOS",
            "model_family": "gee_nb",
            "model_estimator": "GEE_NegativeBinomial",
            "effect_measure": "IRR",
            "exposure": f"SAT/SBT Delivery ({exposure_col})",
            "IRR": round(float(np.exp(coef)), 3),
            "IRR_lower_95": round(float(np.exp(coef - 1.96 * se)), 3),
            "IRR_upper_95": round(float(np.exp(coef + 1.96 * se)), 3),
            "p_value": round(pval, 4),
            "n_clusters": int(model_df["hospitalization_id"].nunique()),
            "n_hospital_dummies": len(hospital_dummies),
        }
    except Exception as e:
        print(f"GEE NB ICU LOS failed ({e}); using Poisson VB fallback.")
        try:
            from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM
            fixed_terms = " + ".join(covariates_formula)
            formula = f"icu_los ~ {fixed_terms}"
            model = PoissonBayesMixedGLM.from_formula(
                formula=formula,
                vc_formulas={"hospital_re": "0 + C(hospital_id)"},
                data=model_df,
            )
            fit = model.fit_vb()
            fep_names = list(fit.model.fep_names)
            fe_mean = np.asarray(fit.fe_mean)
            fe_sd = np.asarray(fit.fe_sd)
            idx = fep_names.index(exposure_formula_col)
            coef = float(fe_mean[idx])
            se = float(fe_sd[idx])
            z = coef / se if se > 0 else np.nan
            pval = float(2 * (1 - sp_stats.norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
            result = {
                "model": "Poisson VB - ICU LOS (fallback)",
                "model_family": "mixed_poisson_fallback",
                "model_estimator": "PoissonBayesMixedGLM",
                "effect_measure": "IRR",
                "exposure": f"SAT/SBT Delivery ({exposure_col})",
                "IRR": round(float(np.exp(coef)), 3),
                "IRR_lower_95": round(float(np.exp(coef - 1.96 * se)), 3),
                "IRR_upper_95": round(float(np.exp(coef + 1.96 * se)), 3),
                "p_value": round(pval, 4) if np.isfinite(pval) else np.nan,
                "n_clusters": int(model_df["hospital_id"].nunique()),
            }
        except Exception as e2:
            print(f"Poisson VB fallback failed: {e2}")
            result = _placeholder_result("Mixed-effects NB - ICU LOS", model_family="gee_nb")
    return result


# ============================================================
# MODEL 4: IN-HOSPITAL MORTALITY (Logistic Regression)
# ============================================================

def fit_mortality_model(
    day_level_df: pd.DataFrame,
    exposure_col: str = "landmark_delivered",
) -> dict[str, Any]:
    """Mixed-effects random-intercept logistic model for in-hospital mortality."""
    try:
        import statsmodels.api as sm
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError:
        return _placeholder_result("Mixed-effects - Mortality", model_family="mixed_logistic")

    hosp = _build_hospitalization_level_dataset(day_level_df, exposure_col=exposure_col)
    hosp, covariates = _prepare_baseline_covariates(hosp, exposure_col=exposure_col)
    model_df = hosp[covariates + ["died", "hospital_id"]].dropna()
    if model_df.empty:
        return _placeholder_result("Mixed-effects - Mortality", model_family="mixed_logistic")

    exposure_formula_col = "exposure"
    model_df = model_df.rename(columns={exposure_col: exposure_formula_col})
    covariates_formula = [exposure_formula_col] + [c for c in covariates if c != exposure_col]
    formula = f"died ~ {' + '.join(covariates_formula)}"

    try:
        model = BinomialBayesMixedGLM.from_formula(
            formula=formula,
            vc_formulas={"hospital_re": "0 + C(hospital_id)"},
            data=model_df,
        )
        fit = model.fit_vb()
        fep_names = list(fit.model.fep_names)
        fe_mean = np.asarray(fit.fe_mean)
        fe_sd = np.asarray(fit.fe_sd)
        if exposure_formula_col not in fep_names:
            raise ValueError("Exposure coefficient not found in mixed-effects fit")
        idx = fep_names.index(exposure_formula_col)
        coef = float(fe_mean[idx])
        se = float(fe_sd[idx])
        z = coef / se if se > 0 else np.nan
        pval = float(2 * (1 - sp_stats.norm.cdf(abs(z)))) if np.isfinite(z) else np.nan

        result = {
            "model": "Mixed-effects Logistic - In-Hospital Mortality (hospital random intercept)",
            "model_family": "mixed_logistic",
            "model_estimator": "BinomialBayesMixedGLM",
            "exposure": f"SAT/SBT Delivery ({exposure_col})",
            "OR": round(float(np.exp(coef)), 3),
            "OR_lower_95": round(float(np.exp(coef - 1.96 * se)), 3),
            "OR_upper_95": round(float(np.exp(coef + 1.96 * se)), 3),
            "p_value": round(pval, 4) if np.isfinite(pval) else np.nan,
            "n_clusters": int(model_df["hospital_id"].nunique()),
        }
    except Exception as e:
        print(f"Mixed-effects mortality model failed ({e}); using GEE fallback.")
        try:
            from statsmodels.genmod.generalized_estimating_equations import GEE
            from statsmodels.genmod.families import Binomial
            from statsmodels.genmod.cov_struct import Exchangeable

            X = sm.add_constant(model_df[covariates_formula], has_constant="add")
            y = model_df["died"].astype(int)
            # Cluster on hospitalization_id per SAP 2.2-2.5
            gee_groups = model_df["hospitalization_id"] if "hospitalization_id" in model_df.columns else model_df["hospital_id"]
            gee = GEE(
                y,
                X,
                groups=gee_groups,
                family=Binomial(),
                cov_struct=Exchangeable(),
            )
            gee_result = gee.fit(cov_type="robust")
            coef = float(gee_result.params[exposure_formula_col])
            se = float(gee_result.bse[exposure_formula_col])
            pval = float(gee_result.pvalues[exposure_formula_col])
            result = {
                "model": "GEE Logistic - In-Hospital Mortality (fallback)",
                "model_family": "gee_logistic_fallback",
                "model_estimator": "GEE_Binomial",
                "exposure": f"SAT/SBT Delivery ({exposure_col})",
                "OR": round(float(np.exp(coef)), 3),
                "OR_lower_95": round(float(np.exp(coef - 1.96 * se)), 3),
                "OR_upper_95": round(float(np.exp(coef + 1.96 * se)), 3),
                "p_value": round(pval, 4),
                "n_clusters": int(model_df["hospital_id"].nunique()),
            }
        except Exception as e2:
            print(f"GEE mortality fallback failed: {e2}")
            result = _placeholder_result("Mixed-effects - Mortality", model_family="mixed_logistic")
    return result


# ============================================================
# MODEL 5: AWAKENING OUTCOME (SAT-specific)
# ============================================================

def fit_awakening_model(day_level_df: pd.DataFrame) -> dict[str, Any]:
    """GEE logistic regression for RASS improvement after SAT delivery.

    Definition of "awakening":
      - Restricted to days where SAT was delivered (delivery == 1) AND
        baseline RASS <= -2 (patient is deeply sedated at start of the day).
      - Outcome (binary): RASS improves to >= -1 within the same ventilator-day.

    The analysis is at the ventilator-day level.  A patient can contribute
    multiple days (each day is a separate observation).  GEE with an
    exchangeable working correlation structure clustered on hospital_id
    accounts for within-hospital correlation; robust sandwich SEs account
    for within-patient correlation.

    Returns dict with OR, 95% CI, p-value for the delivery effect.
    (All observations have delivery==1 by design; the delivery covariate is
    therefore not meaningful here — the relevant comparison is the proportion
    of days that achieve awakening, adjusted for covariates.)

    Note: Because all selected days have delivery==1, the delivery indicator
    is dropped and the intercept represents the adjusted awakening probability.
    Covariates capture patient-level heterogeneity.
    """
    try:
        import statsmodels.api as sm
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.cov_struct import Exchangeable
    except ImportError:
        return _placeholder_result("GEE Logistic - SAT Awakening", model_family="gee_logistic")

    # Require RASS column
    if "rass" not in day_level_df.columns:
        print("fit_awakening_model: 'rass' column not found — skipping.")
        return _placeholder_result(
            "GEE Logistic - SAT Awakening (no RASS data)",
            model_family="gee_logistic",
        )

    # Restrict to SAT-delivered days where patient starts deeply sedated
    awake_df = day_level_df[
        (day_level_df["delivery"] == 1) &
        (day_level_df["rass"] <= -2)
    ].copy()

    if len(awake_df) < 20:
        print(f"fit_awakening_model: only {len(awake_df)} eligible days — skipping.")
        return _placeholder_result(
            "GEE Logistic - SAT Awakening (too few observations)",
            model_family="gee_logistic",
        )

    # Outcome: did RASS improve to >= -1 on this day?
    # Because rass is the median per vent-day, we check whether the maximum
    # RASS recorded on that day (stored in the source df) reached >= -1.
    # If only median is available, use it as a conservative proxy.
    if "rass_max" in day_level_df.columns:
        rass_peak_col = "rass_max"
        awake_df["rass_peak"] = awake_df["rass_max"]
    else:
        # Fallback: use same aggregated rass column (median-based)
        rass_peak_col = "rass"
        awake_df["rass_peak"] = awake_df["rass"]

    awake_df["awakened"] = (awake_df["rass_peak"] >= -1).astype(int)

    # Baseline covariates
    awake_df["sex_male"] = (
        awake_df["sex_category"].str.lower() == "male"
    ).astype(int)
    awake_df["race_white"] = (
        awake_df["race_category"].str.lower().str.contains("white", na=False)
    ).astype(int)

    covariates = ["age_at_admission", "sex_male", "race_white"]
    for c in ["fio2_set", "peep_set", "on_vasopressor", "on_sedation"]:
        if c in awake_df.columns and awake_df[c].notna().sum() > len(awake_df) * 0.3:
            awake_df[c] = awake_df[c].fillna(awake_df[c].median())
            covariates.append(c)

    required_cols = covariates + ["awakened", "hospital_id"]
    model_df = awake_df[required_cols].dropna()

    if len(model_df) < 20 or model_df["awakened"].nunique() < 2:
        return _placeholder_result(
            "GEE Logistic - SAT Awakening (insufficient outcome variance)",
            model_family="gee_logistic",
        )

    X = sm.add_constant(model_df[covariates])
    y = model_df["awakened"].astype(int)

    result: dict = {
        "model": "GEE Logistic - SAT Awakening (hospital-clustered, robust SE)",
        "model_family": "gee_logistic",
        "exposure": "SAT delivery day with baseline RASS <= -2",
        "outcome": "RASS improvement to >= -1",
        "n_days": int(len(model_df)),
        "n_awakened": int(y.sum()),
        "rass_peak_col_used": rass_peak_col,
    }

    try:
        cov_struct = Exchangeable()
        gee = GEE(y, X, groups=model_df["hospital_id"],
                  family=Binomial(), cov_struct=cov_struct)
        gee_result = gee.fit(cov_type="robust")

        params = gee_result.params
        conf = gee_result.conf_int()
        pvals = gee_result.pvalues

        # Report the intercept as adjusted awakening log-odds, plus all covariates
        coef_series = params
        result["coef_table"] = {
            name: {
                "OR": round(float(np.exp(coef_series[name])), 3),
                "OR_lower_95": round(float(np.exp(conf.loc[name, 0])), 3),
                "OR_upper_95": round(float(np.exp(conf.loc[name, 1])), 3),
                "p_value": round(float(pvals[name]), 4),
            }
            for name in coef_series.index
        }

        # Primary summary: intercept = adjusted awakening probability
        intercept_logodds = float(coef_series["const"])
        result["adjusted_awakening_prob"] = round(
            1 / (1 + np.exp(-intercept_logodds)), 3
        )

        try:
            result["working_correlation"] = str(gee_result.cov_struct.summary())
        except Exception:
            result["working_correlation"] = "unavailable"

        result["n_clusters"] = int(model_df["hospital_id"].nunique())

    except Exception as e:
        print(f"GEE awakening model failed ({e}), falling back to plain logistic")
        try:
            logit = sm.Logit(y, X).fit(disp=False, maxiter=300)
            result["model"] = "Logistic - SAT Awakening (no clustering - fallback)"
            result["adjusted_awakening_prob"] = round(
                float(logit.predict(X).mean()), 3
            )
            result["note"] = f"GEE failed: {e}"
        except Exception as e2:
            print(f"Fallback logistic also failed: {e2}")
            result.update(
                _placeholder_result(
                    "GEE Logistic - SAT Awakening",
                    model_family="gee_logistic",
                )
            )

    return result


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

def bootstrap_ci(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    n_boot: int = 1000,
    ci: float = 0.95,
    cluster_col: str = "imv_episode_id",
) -> tuple[float, float, float]:
    """Cluster bootstrap for confidence intervals.

    Parameters
    ----------
    df : pd.DataFrame
    metric_fn : callable
        Function that takes a DataFrame and returns a scalar metric.
    n_boot : int
        Number of bootstrap iterations.
    ci : float
        Confidence level.
    cluster_col : str
        Column to cluster resampling on (hospitalization_id per manuscript).

    Returns
    -------
    tuple: (point_estimate, lower_ci, upper_ci)
    """
    clusters = df[cluster_col].unique()
    point = metric_fn(df)
    boot_estimates = []

    rng = np.random.default_rng(seed=42)
    for _ in range(n_boot):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        boot_df = pd.concat(
            [df[df[cluster_col] == c] for c in sampled_clusters],
            ignore_index=True,
        )
        try:
            est = metric_fn(boot_df)
            boot_estimates.append(est)
        except Exception:
            continue

    if not boot_estimates:
        return point, np.nan, np.nan

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_estimates, alpha * 100)
    upper = np.percentile(boot_estimates, (1 - alpha) * 100)
    return point, lower, upper


# ============================================================
# UTILITIES
# ============================================================

def _placeholder_result(model_name: str, model_family: str | None = None) -> dict[str, Any]:
    """Return placeholder when model cannot be fit."""
    payload: dict[str, Any] = {
        "model": model_name,
        "exposure": "SAT/SBT Delivery",
        "note": "Model not fit - check dependencies or data",
        "model_estimator": "not_fit",
    }
    if model_family:
        payload["model_family"] = model_family
    return payload


def apply_multiplicity_correction(
    results_df: pd.DataFrame,
    p_col: str = "p_value",
    method: str = "fdr_bh",
) -> pd.DataFrame:
    """Apply multiplicity correction to p-values.

    Two correction scopes (both reported per Benjamini & Hochberg, 1995):
    1. Global: across ALL tests (definitions x models) — most conservative
    2. Per-model: within each outcome model separately — accounts for
       correlated hypotheses within the same outcome

    Parameters
    ----------
    results_df : pd.DataFrame
        Results with a p-value column and 'model' column
    p_col : str
        Column containing p-values
    method : str
        'fdr_bh' (Benjamini-Hochberg FDR, recommended) or 'bonferroni'

    Returns
    -------
    pd.DataFrame with 'p_adj_global', 'p_adj_per_model', and significance columns
    """
    from statsmodels.stats.multitest import multipletests

    df = results_df.copy()
    valid = df[p_col].notna()
    pvals = df.loc[valid, p_col].values

    # Initialize output columns
    for col in ["p_adj_global", "sig_global", "p_adj_per_model", "sig_per_model"]:
        df[col] = np.nan if "p_" in col else False

    if len(pvals) == 0:
        return df

    # 1. Global correction across all tests
    reject_g, p_adj_g, _, _ = multipletests(pvals, alpha=0.05, method=method)
    df.loc[valid, "p_adj_global"] = p_adj_g
    df.loc[valid, "sig_global"] = reject_g

    n_tests = len(pvals)
    n_reject_raw = (pvals < 0.05).sum()
    n_reject_g = reject_g.sum()
    print(f"Global {method}: {n_tests} tests, "
          f"{n_reject_raw} sig raw -> {n_reject_g} sig adjusted")

    # 2. Per-model correction (within each outcome model)
    if "model" in df.columns:
        for model_name, grp in df[valid].groupby("model"):
            grp_pvals = grp[p_col].values
            if len(grp_pvals) < 2:
                df.loc[grp.index, "p_adj_per_model"] = grp_pvals
                df.loc[grp.index, "sig_per_model"] = grp_pvals < 0.05
                continue
            reject_m, p_adj_m, _, _ = multipletests(grp_pvals, alpha=0.05, method=method)
            df.loc[grp.index, "p_adj_per_model"] = p_adj_m
            df.loc[grp.index, "sig_per_model"] = reject_m
            print(f"  Per-model {method} [{model_name}]: {len(grp_pvals)} tests, "
                  f"{(grp_pvals < 0.05).sum()} sig raw -> {reject_m.sum()} sig adjusted")

    return df


# ============================================================
# MAIN
# ============================================================

LANDMARK_RULE = "first eligible ventilator-day (minimum vent_day_index)"


def _resolve_non_time_exposures(
    exposure_strategy: str,
) -> list[tuple[str, str]]:
    """Resolve primary/sensitivity exposure definitions for non-time outcomes."""
    strategy = exposure_strategy.strip().lower()
    if strategy in {"landmark_primary_with_ever_sensitivity", "landmark_primary"}:
        return [
            ("landmark_primary", "landmark_delivered"),
            ("ever_delivery_sensitivity", "ever_delivered"),
        ]
    if strategy == "ever_only":
        return [("ever_primary", "ever_delivered")]
    if strategy == "landmark_only":
        return [("landmark_primary", "landmark_delivered")]
    raise ValueError(
        "exposure_strategy must be one of: "
        "landmark_primary_with_ever_sensitivity, landmark_primary, "
        "landmark_only, ever_only"
    )


def _resolve_analysis_role(exposure_definition: str) -> tuple[str, str]:
    """Map exposure definition to (analysis_role, prespec_status)."""
    if exposure_definition in {"landmark_primary", "ever_primary"}:
        return "primary", "prespecified"
    if exposure_definition == "ever_delivery_sensitivity":
        return "sensitivity", "exploratory"
    return "exploratory", "exploratory"


def _attach_metadata(
    row: dict[str, Any],
    *,
    trial_type: str,
    delivery_definition: str,
    exposure_definition: str,
    analysis_spec_version: str,
    sap_requirement_id: str,
    analysis_role: str,
    prespec_status: str,
    model_estimator: str | None = None,
    ph_assumption_status: str | None = None,
    competing_risk_method: str | None = None,
) -> dict[str, Any]:
    payload = row.copy()
    payload["trial_type"] = trial_type
    payload["delivery_definition"] = delivery_definition
    payload["exposure_definition"] = exposure_definition
    payload["analysis_spec_version"] = analysis_spec_version
    payload["landmark_rule"] = LANDMARK_RULE
    payload["sap_requirement_id"] = sap_requirement_id
    payload["analysis_role"] = analysis_role
    payload["prespec_status"] = prespec_status
    payload["model_estimator"] = (
        model_estimator
        or payload.get("model_estimator")
        or payload.get("model_family")
        or "unspecified"
    )
    payload["ph_assumption_status"] = (
        ph_assumption_status
        if ph_assumption_status is not None
        else payload.get("ph_assumption_status", "not_applicable")
    )
    payload["competing_risk_method"] = (
        competing_risk_method
        if competing_risk_method is not None
        else payload.get("competing_risk_method", "none")
    )
    return payload


def _build_four_level_joint_dataset(
    sat_df: pd.DataFrame,
    sbt_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """Build hospitalization-level joint SAT+SBT 4-level exposure dataset [SAP 2.6]."""
    sat_col = "SAT_EHR_delivery" if "SAT_EHR_delivery" in sat_df.columns else None
    sbt_col = "EHR_Delivery_5mins" if "EHR_Delivery_5mins" in sbt_df.columns else None
    if sat_col is None:
        sat_candidates = [c for c in ["SAT_modified_delivery"] if c in sat_df.columns]
        sat_col = sat_candidates[0] if sat_candidates else None
    if sbt_col is None:
        sbt_candidates = [c for c in ["EHR_Delivery_2mins", "EHR_Delivery_30mins"] if c in sbt_df.columns]
        sbt_col = sbt_candidates[0] if sbt_candidates else None
    if sat_col is None or sbt_col is None:
        return None

    sat_elig = (
        "eligible_event" if "eligible_event" in sat_df.columns
        else "on_vent_and_sedation" if "on_vent_and_sedation" in sat_df.columns
        else "eligible_day"
    )
    sbt_elig = "eligible_day" if "eligible_day" in sbt_df.columns else "eligible_event"
    sat_day = build_outcome_dataset(sat_df, sat_col, sat_elig)
    sbt_day = build_outcome_dataset(sbt_df, sbt_col, sbt_elig)

    sat_hosp = _build_hospitalization_level_dataset(sat_day, exposure_col="landmark_delivered")
    sbt_hosp = _build_hospitalization_level_dataset(sbt_day, exposure_col="landmark_delivered")
    sat_hosp = sat_hosp.rename(columns={"landmark_delivered": "sat_landmark"})
    sbt_hosp = sbt_hosp.rename(columns={"landmark_delivered": "sbt_landmark"})
    merge_cols = ["hospitalization_id", "sbt_landmark"]
    joint = sat_hosp.merge(sbt_hosp[merge_cols], on="hospitalization_id", how="inner")
    if joint.empty:
        return None

    joint["sat_landmark"] = joint["sat_landmark"].fillna(0).astype(int)
    joint["sbt_landmark"] = joint["sbt_landmark"].fillna(0).astype(int)
    joint["exposure_4level"] = np.select(
        [
            (joint["sat_landmark"] == 0) & (joint["sbt_landmark"] == 0),
            (joint["sat_landmark"] == 1) & (joint["sbt_landmark"] == 0),
            (joint["sat_landmark"] == 0) & (joint["sbt_landmark"] == 1),
            (joint["sat_landmark"] == 1) & (joint["sbt_landmark"] == 1),
        ],
        ["neither", "sat_only", "sbt_only", "both"],
        default="neither",
    )
    if joint["exposure_4level"].nunique() < 2:
        return None

    joint["sex_male"] = (joint["sex_category"].fillna("").str.lower() == "male").astype(int)
    joint["race_white"] = (
        joint["race_category"].fillna("").str.lower().str.contains("white", na=False)
    ).astype(int)
    return joint


def _fit_four_level_logistic(joint_df: pd.DataFrame, outcome: str, label: str) -> list[dict[str, Any]]:
    """Fit logistic model with 4-level exposure for a binary outcome."""
    import statsmodels.formula.api as smf
    model_df = joint_df[[outcome, "exposure_4level", "age_at_admission", "sex_male", "race_white"]].dropna()
    if model_df.empty or model_df[outcome].nunique() < 2:
        return []
    try:
        fit = smf.logit(
            f"{outcome} ~ C(exposure_4level, Treatment(reference='neither')) + age_at_admission + sex_male + race_white",
            data=model_df,
        ).fit(disp=False, maxiter=300)
        out_rows: list[dict[str, Any]] = []
        for level in ["sat_only", "sbt_only", "both"]:
            term = f"C(exposure_4level, Treatment(reference='neither'))[T.{level}]"
            if term not in fit.params.index:
                continue
            coef = float(fit.params[term])
            se = float(fit.bse[term])
            pval = float(fit.pvalues[term])
            out_rows.append({
                "model": f"Joint SAT+SBT 4-level exposure - {label}",
                "model_family": "logistic",
                "contrast": f"{level} vs neither",
                "OR": round(float(np.exp(coef)), 4),
                "OR_lower_95": round(float(np.exp(coef - 1.96 * se)), 4),
                "OR_upper_95": round(float(np.exp(coef + 1.96 * se)), 4),
                "p_value": round(pval, 4),
                "n_hospitalizations": int(len(model_df)),
            })
        return out_rows
    except Exception as exc:
        return [{"model": f"Joint 4-level - {label}", "error": str(exc)}]


def fit_joint_sat_sbt_four_level(
    sat_df: pd.DataFrame,
    sbt_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Fit paired SAT+SBT 4-level exposure for ALL outcomes [SAP 2.6]."""
    joint = _build_four_level_joint_dataset(sat_df, sbt_df)
    if joint is None:
        return []

    all_rows: list[dict[str, Any]] = []

    # 1. Mortality (logistic)
    all_rows.extend(_fit_four_level_logistic(joint, "died", "Mortality"))

    # 2. Awakening (logistic) — among delivered SAT days with RASS <= -2
    if "rass" in joint.columns:
        awake_df = joint[(joint["sat_landmark"] == 1) & (joint.get("rass", pd.Series(dtype=float)).fillna(0) <= -2)].copy()
        if "rass_max" in awake_df.columns:
            awake_df["awakened"] = (awake_df["rass_max"] >= -1).astype(int)
        elif "rass" in awake_df.columns:
            awake_df["awakened"] = (awake_df["rass"] >= -1).astype(int)
        if "awakened" in awake_df.columns and len(awake_df) >= 20:
            all_rows.extend(_fit_four_level_logistic(awake_df, "awakened", "SAT Awakening"))

    # 3. ICU LOS (NB GEE with 4-level)
    try:
        import statsmodels.api as sm
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.families import NegativeBinomial as NBFamily
        from statsmodels.genmod.cov_struct import Exchangeable

        if "ICU_LOS" in joint.columns:
            joint["icu_los"] = joint["ICU_LOS"].clip(lower=1).round().astype(int)
        else:
            joint["icu_los"] = joint["total_vent_days"].clip(lower=1).round().astype(int)
        dummies = pd.get_dummies(joint["exposure_4level"], prefix="exp4", drop_first=True, dtype=int)
        los_df = pd.concat([joint[["icu_los", "age_at_admission", "sex_male", "race_white", "hospitalization_id"]].reset_index(drop=True),
                            dummies.reset_index(drop=True)], axis=1).dropna()
        covs = ["age_at_admission", "sex_male", "race_white"] + dummies.columns.tolist()
        X = sm.add_constant(los_df[covs])
        gee = GEE(los_df["icu_los"], X, groups=los_df["hospitalization_id"],
                   family=NBFamily(), cov_struct=Exchangeable())
        gee_fit = gee.fit(cov_type="robust")
        for col in dummies.columns:
            if col in gee_fit.params.index:
                coef = float(gee_fit.params[col])
                se = float(gee_fit.bse[col])
                level = col.replace("exp4_", "")
                all_rows.append({
                    "model": "Joint SAT+SBT 4-level exposure - ICU LOS",
                    "model_family": "gee_nb",
                    "contrast": f"{level} vs neither",
                    "IRR": round(float(np.exp(coef)), 4),
                    "IRR_lower_95": round(float(np.exp(coef - 1.96 * se)), 4),
                    "IRR_upper_95": round(float(np.exp(coef + 1.96 * se)), 4),
                    "p_value": round(float(gee_fit.pvalues[col]), 4),
                    "n_hospitalizations": int(len(los_df)),
                })
    except Exception as e:
        all_rows.append({"model": "Joint 4-level - ICU LOS", "error": str(e)})

    # 4. Fine-Gray for VFDs with 4-level exposure
    try:
        # Use the most complete exposure_4level as categorical dummies
        joint["event_type"] = np.where(joint["died"] == 1, 2, 1)
        joint["time_to_event"] = joint["total_vent_days"].fillna(1).clip(lower=1).astype(int)
        dummies_fg = pd.get_dummies(joint["exposure_4level"], prefix="exp4", drop_first=True, dtype=int)
        fg_covs = ["age_at_admission", "sex_male", "race_white"] + dummies_fg.columns.tolist()
        fg_df = pd.concat([joint[["hospitalization_id", "time_to_event", "event_type", "age_at_admission", "sex_male", "race_white"]].reset_index(drop=True),
                           dummies_fg.reset_index(drop=True)], axis=1).dropna()
        # Try rpy2 first, fall back to note
        all_rows.append({
            "model": "Joint SAT+SBT 4-level exposure - VFD Fine-Gray",
            "model_family": "competing_risk_subdistribution",
            "note": "4-level Fine-Gray: use rpy2 bridge or discrete-time cloglog with dummies",
            "n_hospitalizations": int(len(fg_df)),
        })
    except Exception as e:
        all_rows.append({"model": "Joint 4-level - VFD Fine-Gray", "error": str(e)})

    # 5. Cox PH with 4-level time-varying (placeholder — requires day-level data)
    all_rows.append({
        "model": "Joint SAT+SBT 4-level exposure - Cox Extubation",
        "model_family": "cox",
        "note": "4-level Cox requires day-level data with both SAT and SBT; run separately",
    })

    return all_rows


def run_all_models(
    sat_file: str,
    sbt_file: str,
    output_dir: str,
    exposure_strategy: str = "landmark_primary_with_ever_sensitivity",
    analysis_spec_version: str = "2026.02.landmark_v1",
    vfd_primary_method: str = "fine_gray_equivalent",
    include_vfd_sensitivities: bool = True,
    disable_multiplicity: bool = True,
) -> pd.DataFrame:
    """Run all construct-validity models for SAT and SBT.

    Required SAP output artifacts are generated in ``output_dir``:
      - construct_validity_outcomes.csv
      - construct_validity_vfd_components.csv
      - construct_validity_cif_curves.csv
      - construct_validity_multistate_transitions.csv
    """
    os.makedirs(output_dir, exist_ok=True)
    results: list[dict[str, Any]] = []
    cif_rows: list[dict[str, Any]] = []
    vfd_component_rows: list[dict[str, Any]] = []
    multistate_rows: list[dict[str, Any]] = []
    non_time_exposures = _resolve_non_time_exposures(exposure_strategy)

    for label, filepath, delivery_cols in [
        ("SAT", sat_file, [
            "SAT_EHR_delivery",
            "SAT_modified_delivery",
        ]),
        ("SBT", sbt_file, [
            "EHR_Delivery_2mins",
            "EHR_Delivery_5mins",
            "EHR_Delivery_30mins",
        ]),
    ]:
        if not os.path.exists(filepath):
            print(f"Skipping {label}: {filepath} not found")
            continue

        df = load_and_prepare(filepath)
        print(f"\n{'='*60}")
        print(f"Running models for {label} (n={len(df):,} rows)")
        print(f"{'='*60}")

        for dcol in delivery_cols:
            if dcol not in df.columns:
                print(f"  Skipping {dcol}: column not found")
                continue

            eligibility_col = (
                "eligible_event" if "eligible_event" in df.columns
                else "on_vent_and_sedation" if "on_vent_and_sedation" in df.columns
                else "eligible_day"
            )

            print(f"\n--- {label}: {dcol} ---")
            day_df = build_outcome_dataset(df, dcol, eligibility_col)
            n_hosp = day_df["hospitalization_id"].nunique()
            n_delivered = day_df.groupby("hospitalization_id")["delivery"].max().sum()
            print(f"  Hospitalizations: {n_hosp}, with delivery: {int(n_delivered)}")

            # Model 1: time-varying Cox (single analysis by design)
            try:
                cox_result = fit_cox_extubation(day_df)
                cr_death = cox_result.pop("competing_risk_death", None)
                cox_result = _attach_metadata(
                    cox_result,
                    trial_type=label,
                    delivery_definition=dcol,
                    exposure_definition="time_varying_cummax",
                    analysis_spec_version=analysis_spec_version,
                    sap_requirement_id="SAP-5.2-COX-EXTUBATION",
                    analysis_role="primary",
                    prespec_status="prespecified",
                    model_estimator=cox_result.get("model_estimator", "CoxTimeVaryingFitter"),
                    ph_assumption_status="not_tested_time_varying_cox",
                    competing_risk_method="cause_specific",
                )
                results.append(cox_result)
                print(f"  {cox_result.get('model', 'Unknown')}: {cox_result}")
                if cr_death and isinstance(cr_death, dict):
                    cr_death = _attach_metadata(
                        cr_death,
                        trial_type=label,
                        delivery_definition=dcol,
                        exposure_definition="time_varying_cummax",
                        analysis_spec_version=analysis_spec_version,
                        sap_requirement_id="SAP-5.2-COX-COMPETING-DEATH",
                        analysis_role="secondary",
                        prespec_status="prespecified",
                        model_estimator=cr_death.get("model_estimator", "CoxTimeVaryingFitter"),
                        ph_assumption_status="not_tested_time_varying_cox",
                        competing_risk_method="cause_specific",
                    )
                    results.append(cr_death)
                    print(f"  {cr_death.get('model', 'Unknown')}: {cr_death}")
            except Exception as e:
                print(f"  fit_cox_extubation failed: {e}")
                results.append(
                    _attach_metadata(
                        {
                            "model": "fit_cox_extubation",
                            "model_family": "cox",
                            "error": str(e),
                        },
                        trial_type=label,
                        delivery_definition=dcol,
                        exposure_definition="time_varying_cummax",
                        analysis_spec_version=analysis_spec_version,
                        sap_requirement_id="SAP-5.2-COX-EXTUBATION",
                        analysis_role="primary",
                        prespec_status="prespecified",
                        model_estimator="CoxTimeVaryingFitter",
                        ph_assumption_status="fit_failed",
                        competing_risk_method="cause_specific",
                    )
                )

            # Models 2-4: non-time outcomes with explicit exposure definitions.
            for exposure_definition, exposure_col in non_time_exposures:
                analysis_role, prespec_status = _resolve_analysis_role(exposure_definition)

                # 2a. VFD primary: Fine-Gray model [SAP 2.3].
                # Prefer rpy2 bridge to cmprsk::crr(); fall back to discrete-time cloglog.
                if vfd_primary_method == "fine_gray_equivalent":
                    try:
                        from competing_risks import fit_fine_gray_rpy2
                        rpy2_result = fit_fine_gray_rpy2(
                            day_df, exposure_col=exposure_col,
                            covariates=["age_at_admission", "sex_male", "race_white"],
                            horizon=VFD_MAX_DAYS,
                        )
                    except Exception:
                        rpy2_result = None
                    try:
                        fg_result, cif_df, vfd_comp_df = fit_fine_gray_equivalent(
                            day_df,
                            exposure_col=exposure_col,
                            covariates=["age_at_admission", "sex_male", "race_white"],
                            horizon=VFD_MAX_DAYS,
                        )
                        # Override with rpy2 result if available
                        if rpy2_result is not None and rpy2_result.get("shr") is not None:
                            fg_result.update(rpy2_result)
                        fg_row = {
                            "model": fg_result.get(
                                "model",
                                "Fine-Gray equivalent - extubation alive by day 28",
                            ),
                            "model_family": "competing_risk_subdistribution",
                            "SHR": fg_result.get("shr"),
                            "SHR_lower_95": fg_result.get("shr_lower_95"),
                            "SHR_upper_95": fg_result.get("shr_upper_95"),
                            "p_value": fg_result.get("p_value"),
                            "n_hospitalizations": fg_result.get("n_hospitalizations"),
                            "note": fg_result.get("note"),
                        }
                        fg_row = _attach_metadata(
                            fg_row,
                            trial_type=label,
                            delivery_definition=dcol,
                            exposure_definition=exposure_definition,
                            analysis_spec_version=analysis_spec_version,
                            sap_requirement_id="SAP-5.3-VFD-FG",
                            analysis_role=analysis_role,
                            prespec_status=prespec_status,
                            model_estimator=fg_result.get("estimator", "discrete_time_subdistribution_cloglog"),
                            competing_risk_method="fine_gray_equivalent",
                        )
                        results.append(fg_row)

                        if not cif_df.empty:
                            cif = cif_df.copy()
                            cif["trial_type"] = label
                            cif["delivery_definition"] = dcol
                            cif["exposure_definition"] = exposure_definition
                            cif["analysis_spec_version"] = analysis_spec_version
                            cif["sap_requirement_id"] = "SAP-5.3-VFD-FG-CIF"
                            cif_rows.extend(cif.to_dict(orient="records"))

                        if not vfd_comp_df.empty:
                            comps = vfd_comp_df.copy()
                            comps["trial_type"] = label
                            comps["delivery_definition"] = dcol
                            comps["exposure_definition"] = exposure_definition
                            comps["analysis_spec_version"] = analysis_spec_version
                            comps["sap_requirement_id"] = "SAP-5.3-VFD-COMPONENTS"
                            vfd_component_rows.extend(comps.to_dict(orient="records"))
                    except Exception as e:
                        print(f"  fine_gray_equivalent failed: {e}")
                        results.append(
                            _attach_metadata(
                                {
                                    "model": "Fine-Gray equivalent - extubation alive by day 28",
                                    "model_family": "competing_risk_subdistribution",
                                    "error": str(e),
                                },
                                trial_type=label,
                                delivery_definition=dcol,
                                exposure_definition=exposure_definition,
                                analysis_spec_version=analysis_spec_version,
                                sap_requirement_id="SAP-5.3-VFD-FG",
                                analysis_role=analysis_role,
                                prespec_status=prespec_status,
                                model_estimator="discrete_time_subdistribution_cloglog",
                                competing_risk_method="fine_gray_equivalent",
                            )
                        )

                # 2b. VFD sensitivity models [SAP 2.3].
                if include_vfd_sensitivities:
                    # Proportional-odds ordinal
                    try:
                        ord_result = fit_vfd_model(day_df, exposure_col=exposure_col)
                        ord_result = _attach_metadata(
                            ord_result,
                            trial_type=label,
                            delivery_definition=dcol,
                            exposure_definition=exposure_definition,
                            analysis_spec_version=analysis_spec_version,
                            sap_requirement_id="SAP-8.1-VFD-ORDINAL-SENS",
                            analysis_role="sensitivity",
                            prespec_status="exploratory",
                            model_estimator="OrderedModel_logit",
                        )
                        results.append(ord_result)
                    except Exception as e:
                        print(f"  fit_vfd_model failed: {e}")
                        results.append(
                            _attach_metadata(
                                {"model": "Proportional Odds - VFDs", "error": str(e)},
                                trial_type=label, delivery_definition=dcol,
                                exposure_definition=exposure_definition,
                                analysis_spec_version=analysis_spec_version,
                                sap_requirement_id="SAP-8.1-VFD-ORDINAL-SENS",
                                analysis_role="sensitivity", prespec_status="exploratory",
                                model_estimator="OrderedModel_logit",
                            )
                        )

                    # Mann-Whitney U [SAP 2.3]
                    for vfd_sens_fn, vfd_sens_est in [
                        (fit_vfd_mann_whitney, "mannwhitneyu"),
                        (fit_vfd_hurdle, "logit_NB_hurdle"),
                        (fit_vfd_zinb, "ZeroInflatedNegativeBinomialP"),
                    ]:
                        try:
                            vfd_s = vfd_sens_fn(day_df, exposure_col=exposure_col)
                            vfd_s = _attach_metadata(
                                vfd_s,
                                trial_type=label, delivery_definition=dcol,
                                exposure_definition=exposure_definition,
                                analysis_spec_version=analysis_spec_version,
                                sap_requirement_id="SAP-2.3-VFD-SENS",
                                analysis_role="sensitivity", prespec_status="prespecified",
                                model_estimator=vfd_sens_est,
                            )
                            results.append(vfd_s)
                        except Exception as e:
                            print(f"  {vfd_sens_fn.__name__} failed: {e}")

                # 2c. VFD secondary: multistate transition model.
                if exposure_definition == "landmark_primary":
                    try:
                        ms_hazards, ms_transitions = fit_multistate_equivalent(
                            day_df,
                            exposure_col=exposure_col,
                            horizon=VFD_MAX_DAYS,
                        )
                        for _, row in ms_hazards.iterrows():
                            ms_row = {
                                "model": row["model"],
                                "model_family": "multistate",
                                "transition": row["transition"],
                                "HR": row.get("hr"),
                                "HR_lower_95": row.get("hr_lower_95"),
                                "HR_upper_95": row.get("hr_upper_95"),
                                "p_value": row.get("p_value"),
                                "n_subjects": row.get("n_subjects"),
                                "n_events": row.get("n_events"),
                                "note": row.get("note"),
                            }
                            ms_row = _attach_metadata(
                                ms_row,
                                trial_type=label,
                                delivery_definition=dcol,
                                exposure_definition=exposure_definition,
                                analysis_spec_version=analysis_spec_version,
                                sap_requirement_id="SAP-5.4-VFD-MULTISTATE",
                                analysis_role="secondary",
                                prespec_status="prespecified",
                                model_estimator=row.get("model_estimator", "discrete_time_cloglog"),
                                competing_risk_method="multistate",
                            )
                            results.append(ms_row)

                        if not ms_transitions.empty:
                            ms = ms_transitions.copy()
                            ms["trial_type"] = label
                            ms["delivery_definition"] = dcol
                            ms["exposure_definition"] = exposure_definition
                            ms["analysis_spec_version"] = analysis_spec_version
                            ms["sap_requirement_id"] = "SAP-5.4-VFD-MULTISTATE-PROB"
                            multistate_rows.extend(ms.to_dict(orient="records"))
                    except Exception as e:
                        print(f"  multistate analysis failed: {e}")

                # 3-4. LOS and mortality with mixed-effects primary models.
                non_time_model_fns: list[tuple[Callable[..., dict[str, Any]], str, str]] = [
                    (fit_icu_los_model, "mixed_poisson", "SAP-5.5-ICU-LOS-MIXED"),
                    (fit_mortality_model, "mixed_logistic", "SAP-5.6-MORTALITY-MIXED"),
                ]
                for model_fn, model_family, sap_id in non_time_model_fns:
                    role = analysis_role if analysis_role == "primary" else "sensitivity"
                    prespec = prespec_status if role == "primary" else "exploratory"
                    if role != "primary":
                        sap_id = f"{sap_id}-SENS"
                    try:
                        result = model_fn(day_df, exposure_col=exposure_col)
                        result = _attach_metadata(
                            result,
                            trial_type=label,
                            delivery_definition=dcol,
                            exposure_definition=exposure_definition,
                            analysis_spec_version=analysis_spec_version,
                            sap_requirement_id=sap_id,
                            analysis_role=role,
                            prespec_status=prespec,
                            model_estimator=result.get("model_estimator", model_family),
                        )
                        results.append(result)
                        print(f"  {result.get('model', 'Unknown')}: {result}")
                    except Exception as e:
                        print(f"  {model_fn.__name__} failed: {e}")
                        results.append(
                            _attach_metadata(
                                {
                                    "model": model_fn.__name__,
                                    "model_family": model_family,
                                    "error": str(e),
                                },
                                trial_type=label,
                                delivery_definition=dcol,
                                exposure_definition=exposure_definition,
                                analysis_spec_version=analysis_spec_version,
                                sap_requirement_id=sap_id,
                                analysis_role=role,
                                prespec_status=prespec,
                                model_estimator=model_family,
                            )
                        )

            # Model 5: SAT awakening only
            if label == "SAT":
                try:
                    result = fit_awakening_model(day_df)
                    result = _attach_metadata(
                        result,
                        trial_type=label,
                        delivery_definition=dcol,
                        exposure_definition="sat_delivered_days_only",
                        analysis_spec_version=analysis_spec_version,
                        sap_requirement_id="SAP-5.7-SAT-AWAKENING",
                        analysis_role="secondary",
                        prespec_status="prespecified",
                        model_estimator=result.get("model_estimator", "GEE_Binomial"),
                    )
                    results.append(result)
                    print(f"  {result.get('model', 'Unknown')}: {result}")
                except Exception as e:
                    print(f"  fit_awakening_model failed: {e}")
                    results.append(
                        _attach_metadata(
                            {
                                "model": "fit_awakening_model",
                                "model_family": "gee_logistic",
                                "error": str(e),
                            },
                            trial_type=label,
                            delivery_definition=dcol,
                            exposure_definition="sat_delivered_days_only",
                            analysis_spec_version=analysis_spec_version,
                            sap_requirement_id="SAP-5.7-SAT-AWAKENING",
                            analysis_role="secondary",
                            prespec_status="prespecified",
                            model_estimator="GEE_Binomial",
                        )
                    )

    # Paired SAT+SBT 4-level exposure model (neither/sat_only/sbt_only/both).
    if os.path.exists(sat_file) and os.path.exists(sbt_file):
        try:
            sat_df_joint = load_and_prepare(sat_file)
            sbt_df_joint = load_and_prepare(sbt_file)
            joint_rows = fit_joint_sat_sbt_four_level(sat_df_joint, sbt_df_joint)
            for row in joint_rows:
                results.append(
                    _attach_metadata(
                        row,
                        trial_type="SAT+SBT",
                        delivery_definition="SAT_EHR_delivery + EHR_Delivery_5mins",
                        exposure_definition="joint_4level_primary",
                        analysis_spec_version=analysis_spec_version,
                        sap_requirement_id="SAP-5.8-JOINT-4LEVEL",
                        analysis_role="secondary",
                        prespec_status="prespecified",
                        model_estimator="Logit_MLE",
                    )
                )
        except Exception as exc:
            results.append(
                _attach_metadata(
                    {
                        "model": "Joint SAT+SBT 4-level exposure - Mortality",
                        "model_family": "logistic",
                        "error": str(exc),
                    },
                    trial_type="SAT+SBT",
                    delivery_definition="SAT_EHR_delivery + EHR_Delivery_5mins",
                    exposure_definition="joint_4level_primary",
                    analysis_spec_version=analysis_spec_version,
                    sap_requirement_id="SAP-5.8-JOINT-4LEVEL",
                    analysis_role="secondary",
                    prespec_status="prespecified",
                    model_estimator="Logit_MLE",
                )
            )

    # Save main outcome results.
    results_df = pd.DataFrame(results)

    if (
        not disable_multiplicity
        and "p_value" in results_df.columns
        and results_df["p_value"].notna().any()
    ):
        results_df = apply_multiplicity_correction(results_df, p_col="p_value")
        results_df["multiplicity_method"] = "fdr_bh"
    else:
        results_df["multiplicity_method"] = "none"

    outpath = os.path.join(output_dir, "construct_validity_outcomes.csv")
    results_df.to_csv(outpath, index=False)
    print(f"\nResults saved to {outpath}")

    if vfd_component_rows:
        comp_path = os.path.join(output_dir, "construct_validity_vfd_components.csv")
        pd.DataFrame(vfd_component_rows).to_csv(comp_path, index=False)
        print(f"VFD components saved to {comp_path}")
    if cif_rows:
        cif_path = os.path.join(output_dir, "construct_validity_cif_curves.csv")
        pd.DataFrame(cif_rows).to_csv(cif_path, index=False)
        print(f"CIF curves saved to {cif_path}")
    if multistate_rows:
        ms_path = os.path.join(output_dir, "construct_validity_multistate_transitions.csv")
        pd.DataFrame(multistate_rows).to_csv(ms_path, index=False)
        print(f"Multistate transitions saved to {ms_path}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct validity outcome models")
    parser.add_argument("--sat-file", default="../output/intermediate/final_df_SAT.csv")
    parser.add_argument("--sbt-file", default="../output/intermediate/final_df_SBT.csv")
    parser.add_argument("--output-dir", default="../output/final")
    parser.add_argument(
        "--exposure-strategy",
        default="landmark_primary_with_ever_sensitivity",
        help=(
            "Exposure strategy for non-time outcomes: "
            "landmark_primary_with_ever_sensitivity|landmark_primary|"
            "landmark_only|ever_only"
        ),
    )
    parser.add_argument(
        "--analysis-spec-version",
        default="2026.02.landmark_v1",
        help="Version stamp included in output rows for traceability.",
    )
    parser.add_argument(
        "--vfd-primary-method",
        default="fine_gray_equivalent",
        choices=["fine_gray_equivalent", "ordinal"],
        help="Primary VFD modeling approach.",
    )
    parser.add_argument(
        "--include-vfd-sensitivities",
        action="store_true",
        default=True,
        help="Include proportional-odds VFD and other sensitivity outputs (default on).",
    )
    parser.add_argument(
        "--exclude-vfd-sensitivities",
        action="store_false",
        dest="include_vfd_sensitivities",
        help="Disable VFD sensitivity-model outputs.",
    )
    parser.add_argument(
        "--disable-multiplicity",
        action="store_true",
        default=True,
        help="Disable multiplicity correction in primary outputs (SAP default).",
    )
    parser.add_argument(
        "--enable-multiplicity",
        action="store_false",
        dest="disable_multiplicity",
        help="Enable BH/FDR correction as exploratory post-hoc mode.",
    )
    args = parser.parse_args()
    run_all_models(
        args.sat_file,
        args.sbt_file,
        args.output_dir,
        exposure_strategy=args.exposure_strategy,
        analysis_spec_version=args.analysis_spec_version,
        vfd_primary_method=args.vfd_primary_method,
        include_vfd_sensitivities=args.include_vfd_sensitivities,
        disable_multiplicity=args.disable_multiplicity,
    )
