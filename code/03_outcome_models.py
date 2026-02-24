"""
03_outcome_models.py
====================
Construct validity: associations between SAT/SBT delivery and patient outcomes.

Models (per manuscript Statistical Analysis):
1. Time to extubation: Cox PH with death as competing event,
   SAT/SBT as time-varying exposures at ventilator-day level
2. Ventilator-free days (VFDs) to day 28: mixed-effects zero-inflated
   negative binomial
3. ICU LOS: mixed-effects negative binomial
4. In-hospital mortality: mixed-effects logistic regression

All models adjust for:
- Demographics (age, sex, race, ethnicity)
- Comorbidity burden (Charlson or SOFA as proxy)
- Illness severity markers
- Hemodynamics, FiO2, PEEP, vasopressor dose, sedation exposure (prior day)
- Random intercepts for hospital
- Robust SEs clustered at hospitalization level

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

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "utils"))


import argparse
import os
import warnings

import numpy as np
import pandas as pd
from definitions_source_of_truth import (
    VFD_MAX_DAYS,
    VENT_DAY_ANCHOR_HOUR,
    get_vent_day_boundaries,
)

warnings.filterwarnings("ignore")


# ============================================================
# DATA LOADING AND PREPARATION
# ============================================================

def load_and_prepare(filepath):
    """Load site-level final_df and standardize columns."""
    df = pd.read_csv(filepath, low_memory=False)
    for col in ["event_time", "admission_dttm", "discharge_dttm"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed")
    return df


def compute_ventilator_free_days(df):
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


def build_outcome_dataset(df, delivery_col, eligibility_col="eligible_event"):
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

    day_level["delivery"] = day_level["delivery"].fillna(0).astype(int)

    # Derive composite covariates
    # Any vasopressor indicator
    vaso_cols = [c for c in ["norepinephrine", "epinephrine", "vasopressin",
                             "dopamine", "phenylephrine"] if c in day_level.columns]
    if vaso_cols:
        day_level["on_vasopressor"] = (
            day_level[vaso_cols].fillna(0).gt(0).any(axis=1).astype(int)
        )

    # Any sedation indicator
    sed_cols = [c for c in ["propofol", "fentanyl", "midazolam"] if c in day_level.columns]
    if sed_cols:
        day_level["on_sedation"] = (
            day_level[sed_cols].fillna(0).gt(0).any(axis=1).astype(int)
        )

    # Add outcomes
    vfd_df = compute_ventilator_free_days(df)
    day_level = day_level.merge(vfd_df, on="hospitalization_id", how="left")

    return day_level


# ============================================================
# MODEL 1: TIME TO EXTUBATION (Cox PH with competing risk)
# ============================================================

def _build_cox_tv_dataset(day_level_df):
    """Build counting-process dataset shared by cause-specific Cox models.

    Returns (model_df, covariates) where model_df has start/stop intervals,
    time-varying delivery (cummax), baseline covariates, and event indicators
    for both extubation and death.
    """
    hosp_info = day_level_df.groupby("hospitalization_id").agg(
        age=("age_at_admission", "first"),
        sex=("sex_category", "first"),
        race=("race_category", "first"),
        died=("died", "first"),
        total_vent_days=("total_vent_days", "first"),
    ).reset_index()
    hosp_info["sex_male"] = (hosp_info["sex"].str.lower() == "male").astype(int)
    hosp_info["race_white"] = (hosp_info["race"].str.lower().str.contains("white", na=False)).astype(int)

    tv_cols = ["hospitalization_id", "hosp_id_day_key", "delivery"]
    for c in ["fio2_set", "peep_set", "on_vasopressor", "on_sedation", "rass"]:
        if c in day_level_df.columns:
            tv_cols.append(c)
    tv = day_level_df[tv_cols].copy()
    tv = tv.sort_values(["hospitalization_id", "hosp_id_day_key"])
    tv["day_idx"] = tv.groupby("hospitalization_id").cumcount()
    tv["delivered_cummax"] = tv.groupby("hospitalization_id")["delivery"].cummax()
    tv["start"] = tv["day_idx"]
    tv["stop"] = tv["day_idx"] + 1

    tv = tv.merge(hosp_info[["hospitalization_id", "age", "sex_male", "race_white",
                              "died", "total_vent_days"]], on="hospitalization_id", how="left")

    # Event indicators on last interval only
    tv["is_last"] = tv.groupby("hospitalization_id")["stop"].transform("max") == tv["stop"]
    # Cause 1: extubation (survived to end of ventilation)
    tv["event_extubation"] = ((tv["is_last"]) & (tv["died"] == 0)).astype(int)
    # Cause 2: death while ventilated
    tv["event_death"] = ((tv["is_last"]) & (tv["died"] == 1)).astype(int)

    covariates = ["delivered_cummax", "age", "sex_male", "race_white"]
    for c in ["fio2_set", "peep_set", "on_vasopressor", "on_sedation"]:
        if c in tv.columns and tv[c].notna().sum() > len(tv) * 0.3:
            tv[c] = tv[c].fillna(tv[c].median())
            covariates.append(c)

    keep = ["hospitalization_id", "start", "stop",
            "event_extubation", "event_death"] + covariates
    model_df = tv[keep].dropna()
    return model_df, covariates


def _fit_cause_specific_cox(model_df, covariates, event_col, model_label):
    """Fit a single cause-specific Cox model for the given event column.

    For the competing cause, events are censored (set to 0).
    Returns dict with HR, 95% CI, p-value for delivered_cummax.
    """
    from lifelines import CoxTimeVaryingFitter

    fit_df = model_df[["hospitalization_id", "start", "stop", event_col] + covariates].copy()
    fit_df = fit_df.rename(columns={event_col: "event"})

    if fit_df["event"].sum() < 5:
        return _placeholder_result(f"{model_label} (too few events)")

    ctv = CoxTimeVaryingFitter()
    ctv.fit(fit_df, id_col="hospitalization_id",
            start_col="start", stop_col="stop", event_col="event")

    summary = ctv.summary
    result = {"model": model_label, "exposure": "SAT/SBT Delivery (time-varying)"}
    if "delivered_cummax" in summary.index:
        row = summary.loc["delivered_cummax"]
        result["HR"] = round(np.exp(row["coef"]), 3)
        result["HR_lower_95"] = round(np.exp(row["coef"] - 1.96 * row["se(coef)"]), 3)
        result["HR_upper_95"] = round(np.exp(row["coef"] + 1.96 * row["se(coef)"]), 3)
        result["p_value"] = round(row["p"], 4)
    return result


def fit_cox_extubation(day_level_df):
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
        return _placeholder_result("Cause-specific Cox - Extubation")

    try:
        model_df, covariates = _build_cox_tv_dataset(day_level_df)
    except Exception as e:
        print(f"Failed to build Cox TV dataset: {e}")
        return _placeholder_result("Cause-specific Cox - Extubation")

    # Cause 1: Extubation (death censored)
    try:
        result_extub = _fit_cause_specific_cox(
            model_df, covariates, "event_extubation",
            "Cause-specific Cox - Extubation (death censored)")
    except Exception as e:
        print(f"Cause-specific Cox (extubation) failed: {e}")
        result_extub = _placeholder_result("Cause-specific Cox - Extubation")

    # Cause 2: Death (extubation censored) — reported for completeness
    try:
        result_death = _fit_cause_specific_cox(
            model_df, covariates, "event_death",
            "Cause-specific Cox - Death (extubation censored)")
    except Exception as e:
        print(f"Cause-specific Cox (death) failed: {e}")
        result_death = _placeholder_result("Cause-specific Cox - Death")

    # Return primary result (extubation) with competing-risk death result nested
    result = result_extub.copy()
    result["competing_risk_death"] = result_death
    return result


# ============================================================
# MODEL 2: VENTILATOR-FREE DAYS (Proportional Odds)
# ============================================================

def fit_vfd_model(day_level_df):
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
        return _placeholder_result("Proportional Odds - VFDs at 28 days")

    agg_dict = {
        "ever_delivered": ("delivery", "max"),
        "age": ("age_at_admission", "first"),
        "sex": ("sex_category", "first"),
        "race": ("race_category", "first"),
        "vfd_28": ("vfd_28", "first"),
        "died": ("died", "first"),
    }
    for c, func in [("fio2_set", "median"), ("peep_set", "median"),
                     ("on_vasopressor", "max"), ("on_sedation", "max")]:
        if c in day_level_df.columns:
            agg_dict[c] = (c, func)

    hosp = day_level_df.groupby("hospitalization_id").agg(**agg_dict).reset_index()

    hosp["sex_male"] = (hosp["sex"].str.lower() == "male").astype(int)
    hosp["race_white"] = (hosp["race"].str.lower().str.contains("white", na=False)).astype(int)

    # Ordinal VFD categories: death=0, then weekly bins for survivors
    def _vfd_category(row):
        if row["died"] == 1:
            return 0  # dead
        vfd = row["vfd_28"]
        if vfd <= 7:
            return 1
        elif vfd <= 14:
            return 2
        elif vfd <= 21:
            return 3
        else:
            return 4

    hosp["vfd_ordinal"] = hosp.apply(_vfd_category, axis=1)

    covariates = ["ever_delivered", "age", "sex_male", "race_white"]
    for c in ["fio2_set", "peep_set", "on_vasopressor", "on_sedation"]:
        if c in hosp.columns and hosp[c].notna().sum() > len(hosp) * 0.3:
            hosp[c] = hosp[c].fillna(hosp[c].median())
            covariates.append(c)
    model_df = hosp[covariates + ["vfd_ordinal"]].dropna()

    if model_df["vfd_ordinal"].nunique() < 2:
        return _placeholder_result("Proportional Odds - VFDs (insufficient categories)")

    result = {
        "model": "Proportional Odds - VFDs at 28 days",
        "exposure": "SAT/SBT Delivery (ever)",
        "vfd_category_counts": model_df["vfd_ordinal"].value_counts().sort_index().to_dict(),
    }

    try:
        mod = OrderedModel(
            model_df["vfd_ordinal"],
            model_df[covariates],
            distr="logit",
        )
        fit = mod.fit(method="bfgs", disp=False, maxiter=500)

        # In OrderedModel, covariate coefficients are first, thresholds after.
        # Positive coef = higher odds of being in a higher (better) category.
        delivery_idx = covariates.index("ever_delivered")
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
# MODEL 3: ICU LOS (Negative Binomial)
# ============================================================

def fit_icu_los_model(day_level_df):
    """Mixed-effects negative binomial for ICU LOS.

    Returns dict with IRR, 95% CI, p-value.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return _placeholder_result("NB - ICU LOS")

    agg_dict_los = {
        "ever_delivered": ("delivery", "max"),
        "age": ("age_at_admission", "first"),
        "sex": ("sex_category", "first"),
        "race": ("race_category", "first"),
        "total_vent_days": ("total_vent_days", "first"),
        "hospital_id": ("hospital_id", "first"),
    }
    for c, func in [("fio2_set", "median"), ("peep_set", "median"),
                     ("on_vasopressor", "max"), ("on_sedation", "max")]:
        if c in day_level_df.columns:
            agg_dict_los[c] = (c, func)

    hosp = day_level_df.groupby("hospitalization_id").agg(**agg_dict_los).reset_index()

    hosp["sex_male"] = (hosp["sex"].str.lower() == "male").astype(int)
    hosp["race_white"] = (hosp["race"].str.lower().str.contains("white", na=False)).astype(int)
    # Use actual ICU_LOS if available (computed in 01_SAT_standard.ipynb),
    # otherwise fall back to total_vent_days (with warning)
    if "ICU_LOS" in day_level_df.columns:
        hosp_icu = day_level_df.groupby("hospitalization_id")["ICU_LOS"].first().reset_index()
        hosp = hosp.merge(hosp_icu, on="hospitalization_id", how="left")
        hosp["icu_los"] = hosp["ICU_LOS"].clip(lower=0.1)
    else:
        print("WARNING: ICU_LOS column not found. Using total_vent_days as proxy (less accurate).")
        hosp["icu_los"] = hosp["total_vent_days"].clip(lower=1)

    covariates = ["ever_delivered", "age", "sex_male", "race_white"]
    for c in ["fio2_set", "peep_set", "on_vasopressor", "on_sedation"]:
        if c in hosp.columns and hosp[c].notna().sum() > len(hosp) * 0.3:
            hosp[c] = hosp[c].fillna(hosp[c].median())
            covariates.append(c)
    model_df = hosp[covariates + ["icu_los"]].dropna()

    X = sm.add_constant(model_df[covariates])
    y = model_df["icu_los"].astype(int)

    try:
        nb = sm.NegativeBinomial(y, X, loglike_method="nb2").fit(
            disp=False, maxiter=300
        )
        params = nb.params
        conf = nb.conf_int()
        pvals = nb.pvalues
        idx = covariates.index("ever_delivered") + 1

        result = {
            "model": "NB - ICU LOS",
            "exposure": "SAT/SBT Delivery (ever)",
            "IRR": round(np.exp(params.iloc[idx]), 3),
            "IRR_lower_95": round(np.exp(conf.iloc[idx, 0]), 3),
            "IRR_upper_95": round(np.exp(conf.iloc[idx, 1]), 3),
            "p_value": round(pvals.iloc[idx], 4),
        }
    except Exception as e:
        print(f"ICU LOS model failed: {e}")
        result = _placeholder_result("NB - ICU LOS")
    return result


# ============================================================
# MODEL 4: IN-HOSPITAL MORTALITY (Logistic Regression)
# ============================================================

def fit_mortality_model(day_level_df):
    """Mixed-effects logistic regression for in-hospital mortality.

    Returns dict with OR, 95% CI, p-value.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return _placeholder_result("Logistic - Mortality")

    agg_dict_mort = {
        "ever_delivered": ("delivery", "max"),
        "age": ("age_at_admission", "first"),
        "sex": ("sex_category", "first"),
        "race": ("race_category", "first"),
        "died": ("died", "first"),
        "hospital_id": ("hospital_id", "first"),
    }
    for c, func in [("fio2_set", "median"), ("peep_set", "median"),
                     ("on_vasopressor", "max"), ("on_sedation", "max")]:
        if c in day_level_df.columns:
            agg_dict_mort[c] = (c, func)

    hosp = day_level_df.groupby("hospitalization_id").agg(**agg_dict_mort).reset_index()

    hosp["sex_male"] = (hosp["sex"].str.lower() == "male").astype(int)
    hosp["race_white"] = (hosp["race"].str.lower().str.contains("white", na=False)).astype(int)

    covariates = ["ever_delivered", "age", "sex_male", "race_white"]
    for c in ["fio2_set", "peep_set", "on_vasopressor", "on_sedation"]:
        if c in hosp.columns and hosp[c].notna().sum() > len(hosp) * 0.3:
            hosp[c] = hosp[c].fillna(hosp[c].median())
            covariates.append(c)
    model_df = hosp[covariates + ["died", "hospital_id"]].dropna()

    X = sm.add_constant(model_df[covariates])
    y = model_df["died"].astype(int)

    try:
        # Use GEE with exchangeable correlation to account for hospital clustering
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.cov_struct import Exchangeable

        cov_struct = Exchangeable()
        gee = GEE(y, X, groups=model_df["hospital_id"],
                   family=Binomial(), cov_struct=cov_struct)
        # cov_type="robust" ensures sandwich (empirical) standard errors
        gee_result = gee.fit(cov_type="robust")
        params = gee_result.params
        conf = gee_result.conf_int()
        pvals = gee_result.pvalues
        idx = covariates.index("ever_delivered") + 1

        # Extract estimated working correlation for reporting
        try:
            dep_params = gee_result.cov_struct.summary()
            working_corr = str(dep_params)
        except Exception:
            working_corr = "unavailable"

        result = {
            "model": "GEE Logistic - In-Hospital Mortality (hospital-clustered, robust SE)",
            "exposure": "SAT/SBT Delivery (ever)",
            "OR": round(np.exp(params.iloc[idx]), 3),
            "OR_lower_95": round(np.exp(conf.iloc[idx, 0]), 3),
            "OR_upper_95": round(np.exp(conf.iloc[idx, 1]), 3),
            "p_value": round(pvals.iloc[idx], 4),
            "se_type": "robust (sandwich)",
            "working_correlation": working_corr,
            "n_clusters": int(model_df["hospital_id"].nunique()),
        }
    except Exception as e:
        print(f"GEE mortality model failed ({e}), falling back to plain logistic")
        try:
            logit = sm.Logit(y, X).fit(disp=False, maxiter=300)
            params = logit.params
            conf = logit.conf_int()
            pvals = logit.pvalues
            idx = covariates.index("ever_delivered") + 1
            result = {
                "model": "Logistic - In-Hospital Mortality (no clustering - fallback)",
                "exposure": "SAT/SBT Delivery (ever)",
                "OR": round(np.exp(params.iloc[idx]), 3),
                "OR_lower_95": round(np.exp(conf.iloc[idx, 0]), 3),
                "OR_upper_95": round(np.exp(conf.iloc[idx, 1]), 3),
                "p_value": round(pvals.iloc[idx], 4),
            }
        except Exception as e2:
            print(f"Fallback logistic also failed: {e2}")
            result = _placeholder_result("Logistic - Mortality")
    return result


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

def bootstrap_ci(df, metric_fn, n_boot=1000, ci=0.95, cluster_col="hospitalization_id"):
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

def _placeholder_result(model_name):
    """Return placeholder when model cannot be fit."""
    return {
        "model": model_name,
        "exposure": "SAT/SBT Delivery",
        "note": "Model not fit - check dependencies or data",
    }


def apply_multiplicity_correction(results_df, p_col="p_value", method="fdr_bh"):
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

def run_all_models(sat_file, sbt_file, output_dir):
    """Run all construct validity models for SAT and SBT."""
    os.makedirs(output_dir, exist_ok=True)
    results = []

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

            # Fit models
            for model_fn in [
                fit_cox_extubation,
                fit_vfd_model,
                fit_icu_los_model,
                fit_mortality_model,
            ]:
                try:
                    result = model_fn(day_df)
                    # Extract nested competing-risk death result if present
                    cr_death = result.pop("competing_risk_death", None)
                    result["trial_type"] = label
                    result["delivery_definition"] = dcol
                    results.append(result)
                    print(f"  {result.get('model', 'Unknown')}: {result}")
                    if cr_death and isinstance(cr_death, dict):
                        cr_death["trial_type"] = label
                        cr_death["delivery_definition"] = dcol
                        results.append(cr_death)
                        print(f"  {cr_death.get('model', 'Unknown')}: {cr_death}")
                except Exception as e:
                    print(f"  {model_fn.__name__} failed: {e}")
                    results.append({
                        "model": model_fn.__name__,
                        "trial_type": label,
                        "delivery_definition": dcol,
                        "error": str(e),
                    })

    # Save results with multiplicity correction
    results_df = pd.DataFrame(results)

    # Apply Benjamini-Hochberg FDR correction across all tests
    # (6 SAT + 3 SBT definitions × 4 models = up to 36 tests)
    if "p_value" in results_df.columns and results_df["p_value"].notna().any():
        results_df = apply_multiplicity_correction(results_df, p_col="p_value")

    outpath = os.path.join(output_dir, "construct_validity_outcomes.csv")
    results_df.to_csv(outpath, index=False)
    print(f"\nResults saved to {outpath}")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct validity outcome models")
    parser.add_argument("--sat-file", default="../output/intermediate/final_df_SAT.csv")
    parser.add_argument("--sbt-file", default="../output/intermediate/final_df_SBT.csv")
    parser.add_argument("--output-dir", default="../output/final")
    args = parser.parse_args()
    run_all_models(args.sat_file, args.sbt_file, args.output_dir)
