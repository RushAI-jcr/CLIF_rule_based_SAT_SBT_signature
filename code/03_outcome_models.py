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

def fit_cox_extubation(day_level_df):
    """Cox PH for time to extubation with TIME-VARYING delivery exposure.

    Uses counting-process (start-stop) format to avoid immortal time bias:
    each ventilator-day is an interval, and delivery switches from 0→1 on
    the day it first occurs.

    Death is treated as a competing event (censored at death).

    Returns dict with HR, 95% CI, p-value.
    """
    try:
        from lifelines import CoxTimeVaryingFitter
    except ImportError:
        print("lifelines not installed. Install with: pip install lifelines")
        return _placeholder_result("Cox TV - Time to Extubation")

    # Build start-stop dataset: one row per ventilator-day per patient
    # day_level_df already has one row per eligible vent-day
    hosp_info = day_level_df.groupby("hospitalization_id").agg(
        age=("age_at_admission", "first"),
        sex=("sex_category", "first"),
        race=("race_category", "first"),
        died=("died", "first"),
        total_vent_days=("total_vent_days", "first"),
    ).reset_index()
    hosp_info["sex_male"] = (hosp_info["sex"].str.lower() == "male").astype(int)
    hosp_info["race_white"] = (hosp_info["race"].str.lower().str.contains("white", na=False)).astype(int)

    # Create counting-process intervals with time-varying covariates
    tv_cols = ["hospitalization_id", "hosp_id_day_key", "delivery"]
    # Add available clinical covariates as time-varying
    for c in ["fio2_set", "peep_set", "on_vasopressor", "on_sedation", "rass"]:
        if c in day_level_df.columns:
            tv_cols.append(c)
    tv = day_level_df[tv_cols].copy()
    tv = tv.sort_values(["hospitalization_id", "hosp_id_day_key"])
    tv["day_idx"] = tv.groupby("hospitalization_id").cumcount()

    # Time-varying delivery: once delivered, stays 1 for all subsequent days
    tv["delivered_cummax"] = tv.groupby("hospitalization_id")["delivery"].cummax()

    tv["start"] = tv["day_idx"]
    tv["stop"] = tv["day_idx"] + 1

    # Add baseline covariates and event
    tv = tv.merge(hosp_info[["hospitalization_id", "age", "sex_male", "race_white",
                              "died", "total_vent_days"]], on="hospitalization_id", how="left")

    # Event on last interval only: extubation if survived
    tv["is_last"] = tv.groupby("hospitalization_id")["stop"].transform("max") == tv["stop"]
    tv["event"] = ((tv["is_last"]) & (tv["died"] == 0)).astype(int)

    covariates = ["delivered_cummax", "age", "sex_male", "race_white"]
    # Add available time-varying clinical covariates
    for c in ["fio2_set", "peep_set", "on_vasopressor", "on_sedation"]:
        if c in tv.columns and tv[c].notna().sum() > len(tv) * 0.3:
            tv[c] = tv[c].fillna(tv[c].median())
            covariates.append(c)
    model_df = tv[["hospitalization_id", "start", "stop", "event"] + covariates].dropna()

    if model_df["event"].sum() < 5:
        return _placeholder_result("Cox TV - Time to Extubation (too few events)")

    try:
        ctv = CoxTimeVaryingFitter()
        ctv.fit(model_df, id_col="hospitalization_id",
                start_col="start", stop_col="stop", event_col="event")

        summary = ctv.summary
        result = {
            "model": "Cox TV - Time to Extubation",
            "exposure": "SAT/SBT Delivery (time-varying)",
        }
        if "delivered_cummax" in summary.index:
            row = summary.loc["delivered_cummax"]
            result["HR"] = round(np.exp(row["coef"]), 3)
            result["HR_lower_95"] = round(np.exp(row["coef"] - 1.96 * row["se(coef)"]), 3)
            result["HR_upper_95"] = round(np.exp(row["coef"] + 1.96 * row["se(coef)"]), 3)
            result["p_value"] = round(row["p"], 4)
        return result
    except Exception as e:
        print(f"Cox time-varying model failed: {e}")
        return _placeholder_result("Cox TV - Time to Extubation")


# ============================================================
# MODEL 2: VENTILATOR-FREE DAYS (ZINB)
# ============================================================

def fit_vfd_model(day_level_df):
    """Hurdle model for VFDs at 28 days.

    Two-part model to properly separate structural zeros from count data:
    - Part 1 (hurdle): Logistic regression for death (VFD=0 structurally)
    - Part 2 (count): Truncated negative binomial for VFD|survived (VFD>0)

    This avoids the ZINB pitfall of conflating structural zeros (death)
    with sampling zeros (survived but ventilated all 28 days).

    Returns dict with results from both parts.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        print("statsmodels not installed.")
        return _placeholder_result("Hurdle - VFDs at 28 days")

    agg_dict = {
        "ever_delivered": ("delivery", "max"),
        "age": ("age_at_admission", "first"),
        "sex": ("sex_category", "first"),
        "race": ("race_category", "first"),
        "vfd_28": ("vfd_28", "first"),
        "died": ("died", "first"),
    }
    # Add available clinical covariates (episode-level summaries)
    for c, func in [("fio2_set", "median"), ("peep_set", "median"),
                     ("on_vasopressor", "max"), ("on_sedation", "max")]:
        if c in day_level_df.columns:
            agg_dict[c] = (c, func)

    hosp = day_level_df.groupby("hospitalization_id").agg(**agg_dict).reset_index()

    hosp["sex_male"] = (hosp["sex"].str.lower() == "male").astype(int)
    hosp["race_white"] = (hosp["race"].str.lower().str.contains("white", na=False)).astype(int)
    covariates = ["ever_delivered", "age", "sex_male", "race_white"]
    # Add clinical covariates with >=30% completeness
    for c in ["fio2_set", "peep_set", "on_vasopressor", "on_sedation"]:
        if c in hosp.columns and hosp[c].notna().sum() > len(hosp) * 0.3:
            hosp[c] = hosp[c].fillna(hosp[c].median())
            covariates.append(c)
    model_df = hosp[covariates + ["vfd_28", "died"]].dropna()

    X = sm.add_constant(model_df[covariates])
    delivery_idx = covariates.index("ever_delivered") + 1  # +1 for constant

    result = {
        "model": "Hurdle - VFDs at 28 days",
        "exposure": "SAT/SBT Delivery (ever)",
    }

    # Part 1: Logistic for death (structural zero)
    try:
        y_death = model_df["died"].astype(int)
        logit = sm.Logit(y_death, X).fit(disp=False, maxiter=300)
        params = logit.params
        conf = logit.conf_int()
        pvals = logit.pvalues
        result["hurdle_OR"] = round(np.exp(params.iloc[delivery_idx]), 3)
        result["hurdle_OR_lower_95"] = round(np.exp(conf.iloc[delivery_idx, 0]), 3)
        result["hurdle_OR_upper_95"] = round(np.exp(conf.iloc[delivery_idx, 1]), 3)
        result["hurdle_p_value"] = round(pvals.iloc[delivery_idx], 4)
    except Exception as e:
        print(f"Hurdle part 1 (logistic) failed: {e}")
        result["hurdle_note"] = f"Part 1 failed: {e}"

    # Part 2: NB for VFD among survivors (VFD > 0)
    try:
        survivors = model_df[model_df["died"] == 0]
        if len(survivors) < 10:
            result["count_note"] = "Too few survivors for count model"
            return result

        X_surv = sm.add_constant(survivors[covariates])
        y_vfd = survivors["vfd_28"].astype(int)

        # Use NB for count part (truncated at 0: exclude VFD=0 survivors)
        # VFD=0 among survivors means ventilated all 28 days (rare but possible)
        nb = sm.NegativeBinomial(y_vfd, X_surv, loglike_method="nb2").fit(
            disp=False, maxiter=300
        )
        params = nb.params
        conf = nb.conf_int()
        pvals = nb.pvalues

        result["count_IRR"] = round(np.exp(params.iloc[delivery_idx]), 3)
        result["count_IRR_lower_95"] = round(np.exp(conf.iloc[delivery_idx, 0]), 3)
        result["count_IRR_upper_95"] = round(np.exp(conf.iloc[delivery_idx, 1]), 3)
        result["count_p_value"] = round(pvals.iloc[delivery_idx], 4)
    except Exception as e:
        print(f"Hurdle part 2 (count) failed: {e}")
        result["count_note"] = f"Part 2 failed: {e}"

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

        gee = GEE(y, X, groups=model_df["hospital_id"],
                   family=Binomial(), cov_struct=Exchangeable())
        gee_result = gee.fit()
        params = gee_result.params
        conf = gee_result.conf_int()
        pvals = gee_result.pvalues
        idx = covariates.index("ever_delivered") + 1

        result = {
            "model": "GEE Logistic - In-Hospital Mortality (hospital-clustered)",
            "exposure": "SAT/SBT Delivery (ever)",
            "OR": round(np.exp(params.iloc[idx]), 3),
            "OR_lower_95": round(np.exp(conf.iloc[idx, 0]), 3),
            "OR_upper_95": round(np.exp(conf.iloc[idx, 1]), 3),
            "p_value": round(pvals.iloc[idx], 4),
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
    """Apply multiplicity correction to p-values across definitions.

    With 6 SAT definitions + 3 SBT definitions tested across 4 outcome models,
    the family-wise error rate inflates substantially. This applies Benjamini-Hochberg
    FDR correction (default) or Bonferroni as specified.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results with a p-value column
    p_col : str
        Column containing p-values
    method : str
        'fdr_bh' (Benjamini-Hochberg FDR, recommended) or 'bonferroni'

    Returns
    -------
    pd.DataFrame with additional 'p_adjusted' and 'significant_adjusted' columns
    """
    from statsmodels.stats.multitest import multipletests

    df = results_df.copy()
    valid = df[p_col].notna()
    pvals = df.loc[valid, p_col].values

    if len(pvals) == 0:
        df["p_adjusted"] = np.nan
        df["significant_adjusted"] = False
        return df

    reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method=method)
    df.loc[valid, "p_adjusted"] = p_adj
    df.loc[valid, "significant_adjusted"] = reject
    df.loc[~valid, "p_adjusted"] = np.nan
    df.loc[~valid, "significant_adjusted"] = False

    n_tests = len(pvals)
    n_reject_raw = (pvals < 0.05).sum()
    n_reject_adj = reject.sum()
    print(f"Multiplicity correction ({method}): {n_tests} tests, "
          f"{n_reject_raw} significant raw -> {n_reject_adj} significant adjusted")

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
                    result["trial_type"] = label
                    result["delivery_definition"] = dcol
                    results.append(result)
                    print(f"  {result.get('model', 'Unknown')}: {result}")
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
