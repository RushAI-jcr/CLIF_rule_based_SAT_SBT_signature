"""
06_sensitivity_analyses.py
==========================
Sensitivity analyses per manuscript Methods section:
1. Alternative SAT interruption durations (15, 30, 45, 60 min)
2. Alternative SBT durations (2, 5, 30 min) and PS thresholds (8, 10, 12)
3. Exclude cardiac arrest, TTM, comfort care, IMV day 0
4. Restrict to first IMV episode only
5. Restrict to sites/periods meeting data completeness thresholds
6. Data completeness assessment by hospital and over time

CLIF 2.1 compliance:
- hospitalization_id as join key
- Filter on *_category columns
- Outlier thresholds applied

Usage:
    python 06_sensitivity_analyses.py \
        --sat-file  ../output/intermediate/final_df_SAT.csv \
        --sbt-file  ../output/intermediate/final_df_SBT.csv \
        --cohort    ../output/intermediate/study_cohort.parquet \
        --output-dir ../output/final/sensitivity

    # Legacy single-cohort mode (completeness + exclusion analyses only):
    python 06_sensitivity_analyses.py --cohort ../output/intermediate/study_cohort.parquet \
                                       --output-dir ../output/final/sensitivity
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "utils"))


import argparse
import warnings

import numpy as np
import pandas as pd
from definitions_source_of_truth import (
    SENSITIVITY_SAT_DURATIONS_MIN,
    SENSITIVITY_SBT_DURATIONS_MIN,
    SENSITIVITY_SBT_PS_THRESHOLDS,
    SENSITIVITY_SBT_CPAP_THRESHOLDS,
    SBT_MODIFIED_DURATION_MIN,
)
from pySAT import detect_sat_delivery
from pySBT import process_diagnostic_flip_sbt_optimized_v2

warnings.filterwarnings("ignore")


# ============================================================
# HELPERS
# ============================================================

def _load_df(path: str) -> pd.DataFrame:
    """Load a CSV or Parquet file into a DataFrame."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def enforce_missing_data_windows(cohort_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Apply SAP missing-data windows.

    Rules:
    - FiO2 / PEEP windows: <= 6 hours
    - SpO2 / hemodynamics windows: <= 2 hours
    """
    df = cohort_df.copy()
    rules = {
        "fio2_set": 6.0,
        "peep_set": 6.0,
        "spo2": 2.0,
        "norepinephrine": 2.0,
        "epinephrine": 2.0,
        "vasopressin": 2.0,
        "dopamine": 2.0,
        "phenylephrine": 2.0,
    }
    audit_rows: list[dict[str, object]] = []

    def _find_age_col(var: str) -> str | None:
        candidates = [
            f"{var}_hours_since_last",
            f"{var}_hours_since_measurement",
            f"{var}_age_hours",
            f"{var}_age_hr",
        ]
        for c in candidates:
            if c in df.columns:
                return c
        for c in df.columns:
            low = c.lower()
            if var.lower() in low and "hour" in low and ("since" in low or "age" in low):
                return c
        return None

    for var, max_hours in rules.items():
        if var not in df.columns:
            audit_rows.append(
                {"variable": var, "window_hours": max_hours, "status": "column_missing", "n_staled": 0}
            )
            continue
        age_col = _find_age_col(var)
        if age_col is None:
            audit_rows.append(
                {"variable": var, "window_hours": max_hours, "status": "age_col_missing", "n_staled": 0}
            )
            continue
        ages = pd.to_numeric(df[age_col], errors="coerce")
        stale_mask = ages > max_hours
        n_staled = int(stale_mask.fillna(False).sum())
        df.loc[stale_mask, var] = np.nan
        audit_rows.append(
            {
                "variable": var,
                "window_hours": max_hours,
                "status": "applied",
                "age_col_used": age_col,
                "n_staled": n_staled,
            }
        )

    pd.DataFrame(audit_rows).to_csv(
        os.path.join(output_dir, "missing_data_window_audit.csv"),
        index=False,
    )
    return df


def apply_eligibility_missing_source_exclusions(
    cohort_df: pd.DataFrame, output_dir: str
) -> pd.DataFrame:
    """Exclude records missing required source fields before trial classification."""
    df = cohort_df.copy()
    required_cols = [c for c in ["fio2_set", "peep_set", "spo2"] if c in df.columns]
    if not required_cols:
        pd.DataFrame(
            [
                {
                    "status": "skipped",
                    "reason": "required source columns not available",
                }
            ]
        ).to_csv(os.path.join(output_dir, "eligibility_missing_source_exclusions.csv"), index=False)
        return df

    missing_required = df[required_cols].isna().any(axis=1)
    eligible_mask = pd.Series(False, index=df.index)
    if "eligible_event" in df.columns:
        eligible_mask = eligible_mask | (pd.to_numeric(df["eligible_event"], errors="coerce") == 1)
    if "eligible_day" in df.columns:
        eligible_mask = eligible_mask | (pd.to_numeric(df["eligible_day"], errors="coerce") == 1)
    excluded_mask = eligible_mask & missing_required
    summary = pd.DataFrame(
        [
            {
                "n_rows_total": int(len(df)),
                "n_eligible_rows": int(eligible_mask.sum()),
                "n_excluded_missing_sources": int(excluded_mask.sum()),
                "required_cols": ",".join(required_cols),
            }
        ]
    )
    summary.to_csv(
        os.path.join(output_dir, "eligibility_missing_source_exclusions.csv"),
        index=False,
    )
    df["excluded_missing_sources"] = excluded_mask.astype(int)
    df = df.loc[~excluded_mask].copy()
    return df


# ============================================================
# DATA COMPLETENESS ASSESSMENT
# ============================================================

def assess_data_completeness(cohort_df, output_dir):
    """Quantify completeness of each required data element by hospital and time.

    Per manuscript: "We quantified completeness of each required data element
    by hospital and over time."
    """
    os.makedirs(output_dir, exist_ok=True)

    # Key data elements to check
    elements = {
        "device_category": "Ventilator mode",
        "mode_category": "Ventilator mode detail",
        "fio2_set": "FiO2",
        "peep_set": "PEEP",
        "pressure_support_set": "Pressure support",
        "rass": "RASS score",
        "gcs_total": "GCS total",
        "norepinephrine": "Norepinephrine",
        "fentanyl": "Fentanyl",
        "propofol": "Propofol",
        "spo2": "SpO2",
        "sat_delivery_pass_fail": "SAT flowsheet",
        "sbt_delivery_pass_fail": "SBT flowsheet",
    }

    results = []
    for col, label in elements.items():
        if col not in cohort_df.columns:
            continue

        # By hospital
        if "hospital_id" in cohort_df.columns:
            by_hosp = cohort_df.groupby("hospital_id").apply(
                lambda g: g[col].notna().mean()
            ).reset_index()
            by_hosp.columns = ["hospital_id", "completeness"]
            by_hosp["data_element"] = label
            by_hosp["data_element_col"] = col
            results.append(by_hosp)

    if results:
        completeness_df = pd.concat(results, ignore_index=True)
        completeness_df.to_csv(
            os.path.join(output_dir, "data_completeness_by_hospital.csv"),
            index=False,
        )
        print(f"  Data completeness by hospital saved")

        # Pivot for readability
        pivot = completeness_df.pivot_table(
            index="data_element", columns="hospital_id",
            values="completeness", aggfunc="first",
        ).round(3)
        pivot.to_csv(
            os.path.join(output_dir, "data_completeness_matrix.csv"),
        )
        print(f"  Data completeness matrix saved")

    # By time (monthly)
    if "event_time" in cohort_df.columns:
        cohort_df["month"] = pd.to_datetime(
            cohort_df["event_time"], format="mixed"
        ).dt.to_period("M")
        time_results = []
        for col, label in elements.items():
            if col not in cohort_df.columns:
                continue
            by_month = cohort_df.groupby("month").apply(
                lambda g: g[col].notna().mean()
            ).reset_index()
            by_month.columns = ["month", "completeness"]
            by_month["data_element"] = label
            time_results.append(by_month)

        if time_results:
            time_df = pd.concat(time_results, ignore_index=True)
            time_df["month"] = time_df["month"].astype(str)
            time_df.to_csv(
                os.path.join(output_dir, "data_completeness_by_month.csv"),
                index=False,
            )
            print(f"  Data completeness over time saved")


# ============================================================
# SENSITIVITY: ALTERNATIVE SAT DURATIONS
# ============================================================

def sensitivity_sat_durations(sat_file: str, output_dir: str) -> None:
    """Re-run SAT delivery detection with alternative interruption durations.

    Loads the SAT-eligible intermediate dataset, calls ``detect_sat_delivery``
    for each duration in ``SENSITIVITY_SAT_DURATIONS_MIN``, and writes a
    summary CSV with EHR and modified delivery rates per duration.

    Parameters
    ----------
    sat_file : str
        Path to CSV/Parquet containing the SAT-eligible rows
        (output of 01_SAT_standard pipeline). Must include the columns
        expected by ``detect_sat_delivery``.
    output_dir : str
        Directory where ``sensitivity_sat_durations.csv`` will be written.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Loading SAT file: {sat_file}")
    sat_df = _load_df(sat_file)
    n_days = sat_df["hosp_id_day_key"].nunique() if "hosp_id_day_key" in sat_df.columns else len(sat_df)
    print(f"  SAT dataset: {len(sat_df):,} rows, {n_days:,} unique vent-days")

    results = []

    for duration in SENSITIVITY_SAT_DURATIONS_MIN:
        print(f"  Running SAT detection: duration={duration} min ...")
        result_df = detect_sat_delivery(sat_df.copy(), duration_min=duration)

        if "hosp_id_day_key" in result_df.columns:
            ehr_day_agg = result_df.groupby("hosp_id_day_key")["SAT_EHR_delivery"].max()
            n_vent_days = len(ehr_day_agg)
            ehr_delivered = int(ehr_day_agg.sum())
        else:
            n_vent_days = len(result_df)
            ehr_delivered = int(result_df["SAT_EHR_delivery"].sum())

        if "hosp_id_day_key" in result_df.columns and "SAT_modified_delivery" in result_df.columns:
            mod_day_agg = result_df.groupby("hosp_id_day_key")["SAT_modified_delivery"].max()
            mod_delivered = int(mod_day_agg.sum())
        else:
            mod_delivered = int(result_df["SAT_modified_delivery"].sum()) if "SAT_modified_delivery" in result_df.columns else 0

        ehr_rate = ehr_delivered / n_vent_days if n_vent_days > 0 else np.nan
        mod_rate = mod_delivered / n_vent_days if n_vent_days > 0 else np.nan

        results.append({
            "analysis": "SAT duration threshold",
            "duration_min": duration,
            "n_eligible_vent_days": n_vent_days,
            "SAT_EHR_delivered_days": ehr_delivered,
            "SAT_EHR_delivery_rate": round(ehr_rate, 4) if not np.isnan(ehr_rate) else np.nan,
            "SAT_modified_delivered_days": mod_delivered,
            "SAT_modified_delivery_rate": round(mod_rate, 4) if not np.isnan(mod_rate) else np.nan,
        })
        print(
            f"    duration={duration} min | EHR rate={ehr_rate:.1%} "
            f"({ehr_delivered}/{n_vent_days}) | "
            f"Modified rate={mod_rate:.1%} ({mod_delivered}/{n_vent_days})"
        )

    out_path = os.path.join(output_dir, "sensitivity_sat_durations.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"  SAT duration sensitivity results saved -> {out_path}")


# ============================================================
# SENSITIVITY: ALTERNATIVE SBT PARAMETERS
# ============================================================

def sensitivity_sbt_parameters(sbt_file: str, output_dir: str) -> None:
    """Re-run SBT delivery with alternative durations and PS thresholds.

    Loads the SBT-eligible intermediate dataset, calls
    ``process_diagnostic_flip_sbt_optimized_v2`` for each combination of
    ``SENSITIVITY_SBT_DURATIONS_MIN`` x ``SENSITIVITY_SBT_PS_THRESHOLDS``,
    and writes a summary CSV with delivery rates per parameter combination.

    Parameters
    ----------
    sbt_file : str
        Path to CSV/Parquet containing the SBT-eligible rows
        (output of 02_SBT_standard pipeline). Must include the columns
        expected by ``process_diagnostic_flip_sbt_optimized_v2``.
    output_dir : str
        Directory where ``sensitivity_sbt_parameters.csv`` will be written.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Loading SBT file: {sbt_file}")
    sbt_df = _load_df(sbt_file)
    n_days = sbt_df["hosp_id_day_key"].nunique() if "hosp_id_day_key" in sbt_df.columns else len(sbt_df)
    print(f"  SBT dataset: {len(sbt_df):,} rows, {n_days:,} unique vent-days")

    results = []

    for duration in SENSITIVITY_SBT_DURATIONS_MIN:
        for ps in SENSITIVITY_SBT_PS_THRESHOLDS:
            print(f"  Running SBT detection: duration={duration} min, PS<={ps} cmH2O ...")

            # durations_min list: use the sensitivity duration as primary,
            # keep the standard secondary duration (5 min) and 30 min.
            durations_list = sorted(set([duration, SBT_MODIFIED_DURATION_MIN, 30]))

            result_df = process_diagnostic_flip_sbt_optimized_v2(
                sbt_df.copy(),
                duration_min=duration,
                ps_threshold=ps,
                durations_min=durations_list,
            )

            primary_col = f"EHR_Delivery_{duration}mins"
            if "hosp_id_day_key" in result_df.columns and primary_col in result_df.columns:
                sbt_day_agg = result_df.groupby("hosp_id_day_key")[primary_col].max()
                n_vent_days = len(sbt_day_agg)
                delivered = int(sbt_day_agg.sum())
            elif "hosp_id_day_key" in result_df.columns:
                n_vent_days = result_df["hosp_id_day_key"].nunique()
                delivered = 0
            else:
                n_vent_days = len(result_df)
                delivered = int(result_df[primary_col].sum()) if primary_col in result_df.columns else 0
            rate = delivered / n_vent_days if n_vent_days > 0 else np.nan

            results.append({
                "analysis": "SBT parameter sweep",
                "duration_min": duration,
                "ps_threshold_cmH2O": ps,
                "n_eligible_vent_days": n_vent_days,
                "SBT_delivered_days": delivered,
                "SBT_delivery_rate": round(rate, 4) if not np.isnan(rate) else np.nan,
            })
            print(
                f"    duration={duration} min, PS<={ps} | "
                f"rate={rate:.1%} ({delivered}/{n_vent_days})"
            )

    out_path = os.path.join(output_dir, "sensitivity_sbt_parameters.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"  SBT parameter sensitivity results saved -> {out_path}")


# ============================================================
# SENSITIVITY: CLINICAL EXCLUSIONS
# ============================================================

def sensitivity_exclusions(cohort_df, output_dir):
    """Exclude specific clinical populations and re-run analyses.

    Exclusions per manuscript:
    1. Cardiac arrest admission diagnosis
    2. Targeted temperature management during prior day
    3. Comfort care status
    4. Day 0 of IMV (day of intubation)
    """
    results = []
    n_before = cohort_df["hosp_id_day_key"].nunique() if "hosp_id_day_key" in cohort_df.columns else len(cohort_df)
    working = cohort_df.copy()

    def _find_flag_col(keywords: list[str]) -> str | None:
        for col in working.columns:
            low = col.lower()
            if all(k in low for k in keywords):
                return col
        return None

    def _to_bool(s: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(s):
            return s.fillna(False)
        if pd.api.types.is_numeric_dtype(s):
            return s.fillna(0).astype(float).gt(0)
        return s.astype(str).str.lower().isin(
            {"1", "true", "yes", "y", "present", "positive", "active"}
        )

    # IMV day 0 / day 1 exclusion.
    if "day_number" in working.columns:
        pre = len(working)
        working = working[pd.to_numeric(working["day_number"], errors="coerce") > 1]
        results.append(
            {
                "exclusion": "IMV day 0/day 1",
                "status": "computed",
                "rows_before": pre,
                "rows_after": len(working),
                "rows_excluded": pre - len(working),
            }
        )
    else:
        results.append(
            {
                "exclusion": "IMV day 0/day 1",
                "status": "needs_data",
                "note": "day_number column unavailable",
            }
        )

    exclusion_specs = [
        ("Cardiac arrest diagnosis", ["cardiac", "arrest"]),
        ("Targeted temperature management", ["ttm"]),
        ("Comfort care status", ["comfort", "care"]),
    ]
    for label, keywords in exclusion_specs:
        col = _find_flag_col(keywords)
        if col is None and label == "Targeted temperature management":
            col = _find_flag_col(["temperature", "management"])
        if col is None and label == "Comfort care status":
            col = _find_flag_col(["code", "status"])
        if col is None:
            results.append(
                {
                    "exclusion": label,
                    "status": "needs_data",
                    "note": f"No matching columns for keywords={keywords}",
                }
            )
            continue
        flag = _to_bool(working[col])
        pre = len(working)
        working = working[~flag]
        results.append(
            {
                "exclusion": label,
                "status": "computed",
                "source_col": col,
                "rows_before": pre,
                "rows_after": len(working),
                "rows_excluded": pre - len(working),
            }
        )

    n_after = working["hosp_id_day_key"].nunique() if "hosp_id_day_key" in working.columns else len(working)
    summary = {
        "exclusion": "Population restriction summary",
        "status": "computed",
        "days_before": int(n_before),
        "days_after": int(n_after),
        "days_excluded": int(n_before - n_after),
    }
    results.append(summary)

    pd.DataFrame(results).to_csv(
        os.path.join(output_dir, "sensitivity_exclusions.csv"), index=False
    )
    working.to_csv(
        os.path.join(output_dir, "sensitivity_population_restricted_cohort.csv"),
        index=False,
    )
    print("  Clinical exclusion sensitivity saved")


# ============================================================
# SENSITIVITY: FIRST EPISODE ONLY
# ============================================================

def sensitivity_first_episode(cohort_df, output_dir):
    """Restrict to first IMV episode per hospitalization."""
    work = cohort_df.copy()
    episode_col = None
    for c in work.columns:
        if c.lower() == "imv_episode_id":
            episode_col = c
            break
    if episode_col is None:
        print("  Cannot identify first episode without imv_episode_id")
        pd.DataFrame(
            [
                {
                    "analysis": "First IMV episode only",
                    "status": "needs_data",
                    "note": "imv_episode_id column not available",
                }
            ]
        ).to_csv(os.path.join(output_dir, "sensitivity_first_episode.csv"), index=False)
        return

    pre_rows = len(work)
    first = work[work[episode_col].astype(str).str.endswith("_ep_1", na=False)].copy()
    if first.empty:
        # Fallback: first unique episode label per hospitalization.
        first_episode = (
            work.groupby("hospitalization_id")[episode_col].first().rename("first_episode").reset_index()
        )
        first = work.merge(first_episode, on="hospitalization_id", how="left")
        first = first[first[episode_col] == first["first_episode"]].drop(columns=["first_episode"])

    results = {
        "analysis": "First IMV episode only",
        "status": "computed",
        "total_rows_before": int(pre_rows),
        "total_rows_after": int(len(first)),
        "total_hospitalizations_before": int(work["hospitalization_id"].nunique()),
        "total_hospitalizations_after": int(first["hospitalization_id"].nunique()),
    }

    pd.DataFrame([results]).to_csv(
        os.path.join(output_dir, "sensitivity_first_episode.csv"), index=False
    )
    first.to_csv(
        os.path.join(output_dir, "sensitivity_first_episode_cohort.csv"),
        index=False,
    )
    print("  First episode sensitivity saved")


def completeness_threshold_reruns(cohort_df: pd.DataFrame, output_dir: str) -> None:
    """Run >=90% completeness restrictions and emit side-by-side counts."""
    work = cohort_df.copy()
    if work.empty:
        return

    vent_cols = [c for c in ["fio2_set", "peep_set", "mode_category"] if c in work.columns]
    mar_cols = [c for c in ["propofol", "fentanyl", "midazolam", "sat_delivery_pass_fail", "sbt_delivery_pass_fail"] if c in work.columns]

    if vent_cols:
        work["vent_complete"] = work[vent_cols].notna().mean(axis=1)
    else:
        work["vent_complete"] = np.nan
    if mar_cols:
        work["mar_complete"] = work[mar_cols].notna().mean(axis=1)
    else:
        work["mar_complete"] = np.nan

    base_rows = len(work)
    vent_restricted = work[work["vent_complete"] >= 0.9] if vent_cols else work.iloc[0:0]
    mar_restricted = work[work["mar_complete"] >= 0.9] if mar_cols else work.iloc[0:0]
    both_restricted = work[
        (work["vent_complete"] >= 0.9) & (work["mar_complete"] >= 0.9)
    ] if (vent_cols and mar_cols) else work.iloc[0:0]

    summary = pd.DataFrame(
        [
            {
                "restriction": "none",
                "rows": int(base_rows),
                "fraction": 1.0,
            },
            {
                "restriction": "vent_settings_ge_90pct",
                "rows": int(len(vent_restricted)),
                "fraction": float(len(vent_restricted) / base_rows) if base_rows else np.nan,
            },
            {
                "restriction": "mar_ge_90pct",
                "rows": int(len(mar_restricted)),
                "fraction": float(len(mar_restricted) / base_rows) if base_rows else np.nan,
            },
            {
                "restriction": "vent_and_mar_ge_90pct",
                "rows": int(len(both_restricted)),
                "fraction": float(len(both_restricted) / base_rows) if base_rows else np.nan,
            },
        ]
    )
    summary.to_csv(
        os.path.join(output_dir, "sensitivity_completeness_threshold_reruns.csv"),
        index=False,
    )
    print("  Completeness-threshold reruns saved")


# ============================================================
# SENSITIVITY: OUTCOME MODEL RE-RUNS [SAP 2.10]
# ============================================================

def run_outcome_sensitivity(
    restricted_df: pd.DataFrame,
    label: str,
    output_dir: str,
) -> None:
    """Re-run all outcome models on a restricted population [SAP 2.10].

    After each population restriction (day 0-1 exclusion, cardiac arrest,
    TTM, comfort care, first episode), outcome models are re-fit.
    """
    from pathlib import Path
    sens_dir = os.path.join(output_dir, f"outcomes_{label}")
    os.makedirs(sens_dir, exist_ok=True)

    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from importlib import import_module
        om = import_module("03_outcome_models")

        # Write restricted data to temp files for run_all_models
        sat_path = os.path.join(sens_dir, "restricted_sat.csv")
        sbt_path = os.path.join(sens_dir, "restricted_sbt.csv")
        restricted_df.to_csv(sat_path, index=False)
        restricted_df.to_csv(sbt_path, index=False)

        om.run_all_models(
            sat_file=sat_path,
            sbt_file=sbt_path,
            output_dir=sens_dir,
            include_vfd_sensitivities=False,
            disable_multiplicity=True,
        )
        print(f"  Outcome sensitivity '{label}' saved to {sens_dir}")
    except Exception as e:
        print(f"  Outcome sensitivity '{label}' failed: {e}")
        pd.DataFrame([{
            "analysis": f"outcome_sensitivity_{label}",
            "status": "failed",
            "error": str(e),
        }]).to_csv(os.path.join(sens_dir, "outcome_sensitivity_error.csv"), index=False)


# ============================================================
# SENSITIVITY: CPAP THRESHOLD SWEEP [SAP 2.11]
# ============================================================

def sensitivity_cpap_thresholds(sbt_file: str, output_dir: str) -> None:
    """Re-run SBT delivery with alternative CPAP thresholds [SAP 2.11]."""
    os.makedirs(output_dir, exist_ok=True)
    sbt_df = _load_df(sbt_file)
    results = []
    for cpap in SENSITIVITY_SBT_CPAP_THRESHOLDS:
        print(f"  Running SBT detection: CPAP<={cpap} cmH2O ...")
        try:
            result_df = process_diagnostic_flip_sbt_optimized_v2(
                sbt_df.copy(),
                cpap_threshold=cpap,
            )
            n_vent_days = result_df["hosp_id_day_key"].nunique() if "hosp_id_day_key" in result_df.columns else len(result_df)
            primary_col = "EHR_Delivery_2mins"
            delivered = int(result_df[primary_col].sum()) if primary_col in result_df.columns else 0
            rate = delivered / n_vent_days if n_vent_days > 0 else np.nan
            results.append({
                "analysis": "SBT CPAP threshold sweep",
                "cpap_threshold_cmH2O": cpap,
                "n_eligible_vent_days": n_vent_days,
                "SBT_delivered_days": delivered,
                "SBT_delivery_rate": round(rate, 4) if not np.isnan(rate) else np.nan,
            })
        except Exception as e:
            results.append({
                "analysis": "SBT CPAP threshold sweep",
                "cpap_threshold_cmH2O": cpap,
                "error": str(e),
            })
    pd.DataFrame(results).to_csv(
        os.path.join(output_dir, "sensitivity_cpap_thresholds.csv"), index=False
    )
    print(f"  CPAP threshold sensitivity saved")


# ============================================================
# MAIN
# ============================================================

def run_all_sensitivity(
    cohort_path: str,
    output_dir: str,
    sat_file: str | None = None,
    sbt_file: str | None = None,
    run_population_restrictions: bool = True,
    run_completeness_threshold_reruns: bool = True,
) -> None:
    """Run all sensitivity analyses.

    Parameters
    ----------
    cohort_path : str
        Path to the main study cohort (parquet or CSV). Used for data
        completeness assessment and exclusion sensitivity analyses.
    output_dir : str
        Root output directory; subdirectories are created as needed.
    sat_file : str or None
        Path to the SAT-eligible intermediate file. When provided, the SAT
        duration sweep is run with actual detection logic. When None, a
        placeholder is written instead.
    sbt_file : str or None
        Path to the SBT-eligible intermediate file. When provided, the SBT
        parameter sweep is run with actual detection logic. When None, a
        placeholder is written instead.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading cohort from {cohort_path}...")
    cohort = _load_df(cohort_path)

    print(f"Cohort: {len(cohort):,} rows, "
          f"{cohort['hospitalization_id'].nunique():,} hospitalizations")

    print("\n0. Applying SAP missing-data windows...")
    cohort = enforce_missing_data_windows(cohort, output_dir)

    print("\n0b. Applying eligibility missing-source exclusions...")
    cohort = apply_eligibility_missing_source_exclusions(cohort, output_dir)

    print("\n1. Data completeness assessment...")
    assess_data_completeness(cohort, output_dir)

    print("\n2. SAT duration sensitivity...")
    if sat_file is not None:
        sensitivity_sat_durations(sat_file, output_dir)
    else:
        print("  No --sat-file provided; writing placeholder.")
        placeholder = [
            {
                "analysis": "SAT duration threshold",
                "duration_min": d,
                "note": "Pass --sat-file to compute actual rates",
                "status": "needs_sat_file",
            }
            for d in SENSITIVITY_SAT_DURATIONS_MIN
        ]
        pd.DataFrame(placeholder).to_csv(
            os.path.join(output_dir, "sensitivity_sat_durations.csv"), index=False
        )

    print("\n3. SBT parameter sensitivity...")
    if sbt_file is not None:
        sensitivity_sbt_parameters(sbt_file, output_dir)
    else:
        print("  No --sbt-file provided; writing placeholder.")
        placeholder = [
            {
                "analysis": "SBT parameter sweep",
                "duration_min": d,
                "ps_threshold_cmH2O": ps,
                "note": "Pass --sbt-file to compute actual rates",
                "status": "needs_sbt_file",
            }
            for d in SENSITIVITY_SBT_DURATIONS_MIN
            for ps in SENSITIVITY_SBT_PS_THRESHOLDS
        ]
        pd.DataFrame(placeholder).to_csv(
            os.path.join(output_dir, "sensitivity_sbt_parameters.csv"), index=False
        )

    print("\n4. Clinical exclusions...")
    if run_population_restrictions:
        sensitivity_exclusions(cohort, output_dir)
    else:
        print("  Skipped population restrictions by CLI option.")

    print("\n5. First episode restriction...")
    if run_population_restrictions:
        sensitivity_first_episode(cohort, output_dir)
    else:
        print("  Skipped first-episode restriction by CLI option.")

    print("\n6. Completeness-threshold reruns...")
    if run_completeness_threshold_reruns:
        completeness_threshold_reruns(cohort, output_dir)
    else:
        print("  Skipped completeness reruns by CLI option.")

    # 7. CPAP threshold sweep [SAP 2.11]
    print("\n7. CPAP threshold sensitivity...")
    if sbt_file is not None:
        sensitivity_cpap_thresholds(sbt_file, output_dir)
    else:
        print("  No --sbt-file; skipping CPAP sweep.")

    # 8. Outcome model re-runs on restricted populations [SAP 2.10]
    print("\n8. Outcome sensitivity re-runs...")
    if run_population_restrictions:
        restricted_path = os.path.join(output_dir, "sensitivity_population_restricted_cohort.csv")
        if os.path.exists(restricted_path):
            restricted = pd.read_csv(restricted_path, low_memory=False)
            run_outcome_sensitivity(restricted, "population_restricted", output_dir)

        first_ep_path = os.path.join(output_dir, "sensitivity_first_episode_cohort.csv")
        if os.path.exists(first_ep_path):
            first_ep = pd.read_csv(first_ep_path, low_memory=False)
            run_outcome_sensitivity(first_ep, "first_episode_only", output_dir)

    print(f"\nAll sensitivity analyses saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SAT/SBT sensitivity analyses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cohort",
        default="../output/intermediate/study_cohort.parquet",
        help="Main study cohort file (parquet or CSV)",
    )
    parser.add_argument(
        "--sat-file",
        default=None,
        help="SAT-eligible intermediate file (final_df_SAT.csv or parquet). "
             "Required to compute actual SAT duration sensitivity rates.",
    )
    parser.add_argument(
        "--sbt-file",
        default=None,
        help="SBT-eligible intermediate file (final_df_SBT.csv or parquet). "
             "Required to compute actual SBT parameter sensitivity rates.",
    )
    parser.add_argument(
        "--output-dir",
        default="../output/final/sensitivity",
        help="Directory to write all sensitivity output CSVs",
    )
    parser.add_argument(
        "--run-population-restrictions",
        action="store_true",
        default=True,
        help="Run day0/day1 + diagnosis/status population restrictions (default on).",
    )
    parser.add_argument(
        "--skip-population-restrictions",
        action="store_false",
        dest="run_population_restrictions",
        help="Skip population-restriction sensitivity analyses.",
    )
    parser.add_argument(
        "--run-completeness-threshold-reruns",
        action="store_true",
        default=True,
        help="Run >=90% vent-settings/MAR completeness threshold reruns (default on).",
    )
    parser.add_argument(
        "--skip-completeness-threshold-reruns",
        action="store_false",
        dest="run_completeness_threshold_reruns",
        help="Skip completeness-threshold reruns.",
    )
    args = parser.parse_args()
    run_all_sensitivity(
        cohort_path=args.cohort,
        output_dir=args.output_dir,
        sat_file=args.sat_file,
        sbt_file=args.sbt_file,
        run_population_restrictions=args.run_population_restrictions,
        run_completeness_threshold_reruns=args.run_completeness_threshold_reruns,
    )
