"""
07_aggregate_sites.py
=====================
Aggregate per-site outputs into pooled manuscript-ready numbers.

This is the missing orchestration layer between per-site pipeline outputs
(00-02 notebooks) and manuscript figures/tables (03-06 scripts).

Reads:
    ../output/final/{SITE}/SAT_standard/  - SAT concordance, stats, table1
    ../output/final/{SITE}/SBT_standard/  - SBT concordance, stats, table1
    ../output/intermediate/final_df_SAT.csv
    ../output/intermediate/final_df_SBT.csv
    ../output/intermediate/study_cohort.parquet

Produces:
    ../output/final/pooled/consort_numbers.json      - CONSORT flow numbers
    ../output/final/pooled/site_summary.csv           - site_output_schema format
    ../output/final/pooled/pooled_delivery_rates.csv  - Logit-pooled rates with CIs
    ../output/final/pooled/pooled_concordance.csv     - Pooled concordance metrics
    ../output/final/pooled/table1_operational_definitions.csv
    ../output/final/pooled/manuscript_numbers.json    - All numbers for filling manuscript

Usage:
    python 07_aggregate_sites.py --data-dir ../output --output-dir ../output/final/pooled
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "utils"))


import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

from definitions_source_of_truth import (
    SAT_SEDATIVES, SAT_OPIOIDS, PARALYTICS,
    SAT_COMPLETE_DURATION_MIN, SAT_MODIFIED_DURATION_MIN,
    SAT_MIN_SEDATION_HOURS, SAT_ELIGIBILITY_WINDOW_START_HOUR,
    SAT_ELIGIBILITY_WINDOW_END_HOUR, RASS_AGITATION_THRESHOLD,
    SBT_CONTROLLED_MODES, SBT_MIN_CONTROLLED_MODE_HOURS,
    SBT_FIO2_MAX, SBT_PEEP_MAX, SBT_SPO2_MIN,
    SBT_PRIMARY_DURATION_MIN, SBT_MODIFIED_DURATION_MIN,
    SBT_PS_MAX, SBT_CPAP_MAX, SBT_SUPPORT_MODES,
    SBT_VASOPRESSOR_LIMITS, SBT_MIN_STABILITY_HOURS,
    VENT_DAY_ANCHOR_HOUR, IMV_EPISODE_GAP_HOURS,
    MIN_IMV_HOURS, STUDY_PERIOD_START, STUDY_PERIOD_END,
    MIN_AGE, MAX_AGE,
)
from site_output_schema import compute_site_se
from meta_analysis import cluster_bootstrap_concordance

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONSORT NUMBERS
# ============================================================

def compute_consort_numbers(data_dir):
    """Extract CONSORT flow diagram numbers from pipeline outputs."""
    consort = {}

    # Try to load study cohort for raw counts
    cohort_path = os.path.join(data_dir, "intermediate", "study_cohort.parquet")
    if not os.path.exists(cohort_path):
        cohort_path = os.path.join(data_dir, "intermediate", "study_cohort.csv")

    if os.path.exists(cohort_path):
        if cohort_path.endswith(".parquet"):
            cohort = pd.read_parquet(cohort_path)
        else:
            cohort = pd.read_csv(cohort_path, low_memory=False)

        consort["n_patients"] = int(cohort["patient_id"].nunique())
        consort["n_hospitalizations"] = int(cohort["hospitalization_id"].nunique())
        consort["n_vent_days"] = int(cohort["hosp_id_day_key"].nunique())

        if "hospital_id" in cohort.columns:
            consort["n_hospitals"] = int(cohort["hospital_id"].nunique())

    # SAT-specific
    sat_path = os.path.join(data_dir, "intermediate", "final_df_SAT.csv")
    if os.path.exists(sat_path):
        sat = pd.read_csv(sat_path, low_memory=False)
        consort["n_sat_eligible_days"] = int(
            sat[sat.get("eligible_event", sat.get("on_vent_and_sedation", pd.Series())) == 1]["hosp_id_day_key"].nunique()
        ) if "eligible_event" in sat.columns else int(
            sat[sat.get("on_vent_and_sedation", pd.Series()) == 1]["hosp_id_day_key"].nunique()
        ) if "on_vent_and_sedation" in sat.columns else 0
        consort["n_sat_patients"] = int(sat["patient_id"].nunique())

        # Flowsheet availability
        if "sat_delivery_pass_fail" in sat.columns:
            sat_flow_hosps = sat[sat["sat_delivery_pass_fail"].notna()]["hospital_id"].nunique()
            consort["n_sat_flowsheet_hospitals"] = int(sat_flow_hosps)

    # SBT-specific
    sbt_path = os.path.join(data_dir, "intermediate", "final_df_SBT.csv")
    if os.path.exists(sbt_path):
        sbt = pd.read_csv(sbt_path, low_memory=False)
        elig_col = "eligible_day" if "eligible_day" in sbt.columns else "eligible_event"
        if elig_col in sbt.columns:
            consort["n_sbt_eligible_days"] = int(
                sbt[sbt[elig_col] == 1]["hosp_id_day_key"].nunique()
            )
        consort["n_sbt_patients"] = int(sbt["patient_id"].nunique())

        if "sbt_delivery_pass_fail" in sbt.columns:
            sbt_flow_hosps = sbt[sbt["sbt_delivery_pass_fail"].notna()]["hospital_id"].nunique()
            consort["n_sbt_flowsheet_hospitals"] = int(sbt_flow_hosps)

    return consort


# ============================================================
# 2. POOLED DELIVERY RATES
# ============================================================

def compute_pooled_delivery_rates(data_dir):
    """Compute per-hospital delivery rates and pool across sites."""
    from meta_analysis import run_proportion_meta_analysis

    results = []

    for trial, filepath, elig_col, delivery_cols in [
        ("SAT", os.path.join(data_dir, "intermediate", "final_df_SAT.csv"),
         "eligible_event",
         ["SAT_EHR_delivery", "SAT_modified_delivery"]),
        ("SBT", os.path.join(data_dir, "intermediate", "final_df_SBT.csv"),
         "eligible_day",
         ["EHR_Delivery_2mins", "EHR_Delivery_5mins", "EHR_Delivery_30mins"]),
    ]:
        if not os.path.exists(filepath):
            continue

        df = pd.read_csv(filepath, low_memory=False)
        # Fallback eligibility column
        if elig_col not in df.columns:
            elig_col = "on_vent_and_sedation" if "on_vent_and_sedation" in df.columns else "eligible_event"

        eligible = df[df.get(elig_col, pd.Series(dtype=float)) == 1] if elig_col in df.columns else df

        for dcol in delivery_cols:
            if dcol not in df.columns:
                continue

            # Per-hospital rates
            hosp_rates = eligible.groupby("hospital_id").agg(
                n_eligible=("hosp_id_day_key", "nunique"),
                n_delivered=(dcol, lambda x: (x == 1).sum()),
            ).reset_index()
            hosp_rates["rate"] = hosp_rates["n_delivered"] / hosp_rates["n_eligible"]
            hosp_rates["se"] = hosp_rates.apply(
                lambda r: compute_site_se(int(r["n_delivered"]), int(r["n_eligible"])), axis=1
            )
            hosp_rates = hosp_rates[hosp_rates["n_eligible"] >= 10]  # min sample

            # Overall crude rate
            total_elig = hosp_rates["n_eligible"].sum()
            total_deliv = hosp_rates["n_delivered"].sum()
            overall_rate = total_deliv / total_elig if total_elig > 0 else 0

            result = {
                "trial_type": trial,
                "delivery_definition": dcol,
                "overall_rate": round(overall_rate, 4),
                "overall_rate_pct": f"{overall_rate:.1%}",
                "n_eligible_days": int(total_elig),
                "n_delivered_days": int(total_deliv),
                "n_hospitals": len(hosp_rates),
            }

            # Meta-analytic pooling via logit transform
            if len(hosp_rates) >= 3:
                try:
                    hosp_rates_clean = hosp_rates[hosp_rates["rate"].between(0.001, 0.999)]
                    if len(hosp_rates_clean) >= 3:
                        _, summary = run_proportion_meta_analysis(
                            hosp_rates_clean,
                            rate_col="rate",
                            n_col="n_eligible",
                            label_col="hospital_id",
                        )
                        pooled_row = summary[summary["label"].str.startswith("Pooled")].iloc[0]
                        result["pooled_rate_re"] = round(pooled_row["eff_prop"], 4)
                        result["pooled_ci_low"] = round(pooled_row["ci_low_prop"], 4)
                        result["pooled_ci_high"] = round(pooled_row["ci_upp_prop"], 4)
                        result["pooled_rate_pct"] = (
                            f"{pooled_row['eff_prop']:.1%} "
                            f"({pooled_row['ci_low_prop']:.1%}-{pooled_row['ci_upp_prop']:.1%})"
                        )
                except Exception as e:
                    print(f"  Meta-analysis failed for {dcol}: {e}")

            # Hospital variation summary
            result["median_rate"] = round(hosp_rates["rate"].median(), 4)
            result["iqr_low"] = round(hosp_rates["rate"].quantile(0.25), 4)
            result["iqr_high"] = round(hosp_rates["rate"].quantile(0.75), 4)
            result["range_low"] = round(hosp_rates["rate"].min(), 4)
            result["range_high"] = round(hosp_rates["rate"].max(), 4)

            results.append(result)
            print(f"  {trial} {dcol}: {result['overall_rate_pct']} "
                  f"(n={total_elig} days, {len(hosp_rates)} hospitals)")

    return pd.DataFrame(results)


# ============================================================
# 3. POOLED CONCORDANCE
# ============================================================

def _load_day_level_df(data_dir: str, trial_type: str) -> pd.DataFrame | None:
    """Attempt to load the raw day-level DataFrame for ``trial_type`` (SAT|SBT).

    Checks the canonical intermediate paths produced by the per-site pipeline.
    Returns None if no file is found so callers can degrade gracefully.
    """
    candidates = [
        os.path.join(data_dir, "intermediate", f"final_df_{trial_type}.csv"),
        os.path.join(data_dir, "intermediate", f"final_df_{trial_type}.parquet"),
        os.path.join(data_dir, "intermediate", f"day_level_{trial_type.lower()}.csv"),
        os.path.join(data_dir, "intermediate", f"day_level_{trial_type.lower()}.parquet"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            if path.endswith(".parquet"):
                return pd.read_parquet(path)
            return pd.read_csv(path, low_memory=False)
        except Exception as exc:
            print(f"  Warning: could not read {path}: {exc}")
    return None


def _bootstrap_ci_for_definition(
    day_df: pd.DataFrame,
    ehr_col: str,
    flowsheet_col: str,
) -> dict | None:
    """Run ``cluster_bootstrap_concordance`` for one concordance definition.

    Returns the full result dict on success, or None when the required
    columns are absent or the bootstrap cannot be run.
    """
    if ehr_col not in day_df.columns or flowsheet_col not in day_df.columns:
        return None
    try:
        return cluster_bootstrap_concordance(
            day_level_df=day_df,
            ehr_col=ehr_col,
            flowsheet_col=flowsheet_col,
            cluster_col="imv_episode_id",
            n_boot=1000,
            seed=42,
        )
    except Exception as exc:
        print(f"  Bootstrap failed for ({ehr_col}, {flowsheet_col}): {exc}")
        return None


def compute_pooled_concordance(data_dir):
    """Pool concordance metrics across sites from concordance_summary CSVs.

    After computing aggregate point estimates from the confusion-matrix
    summaries, the function attempts to load the raw day-level intermediate
    data to run ``cluster_bootstrap_concordance`` and append 95% percentile
    CI columns to each pooled row.
    """
    concordance_files = []
    for root, dirs, files in os.walk(os.path.join(data_dir, "final")):
        for f in files:
            if "concordance_summary" in f and f.endswith(".csv"):
                concordance_files.append(os.path.join(root, f))

    # Also check the SAT/SBT delivery_concordance_summary files
    for root, dirs, files in os.walk(os.path.join(data_dir, "final")):
        for f in files:
            if "delivery_concordance_summary" in f and f.endswith(".csv"):
                fpath = os.path.join(root, f)
                if fpath not in concordance_files:
                    concordance_files.append(fpath)

    if not concordance_files:
        print("  No concordance files found")
        return pd.DataFrame()

    all_metrics = []
    for fpath in concordance_files:
        try:
            df = pd.read_csv(fpath)
            # Determine trial type from path
            if "SAT" in fpath.upper():
                df["trial_type"] = "SAT"
            elif "SBT" in fpath.upper():
                df["trial_type"] = "SBT"
            else:
                df["trial_type"] = "Unknown"
            df["source_file"] = os.path.basename(fpath)
            all_metrics.append(df)
        except Exception as e:
            print(f"  Error reading {fpath}: {e}")

    if not all_metrics:
        return pd.DataFrame()

    metrics_df = pd.concat(all_metrics, ignore_index=True)

    # Pool by Column (definition)
    pooled = []
    group_col = "Column" if "Column" in metrics_df.columns else metrics_df.columns[0]
    for col_name, group in metrics_df.groupby(group_col):
        tp = group["TP"].sum() if "TP" in group.columns else 0
        fp = group["FP"].sum() if "FP" in group.columns else 0
        fn = group["FN"].sum() if "FN" in group.columns else 0
        tn = group["TN"].sum() if "TN" in group.columns else 0
        total = tp + fp + fn + tn

        if total == 0:
            continue

        accuracy = (tp + tn) / total
        sensitivity = tp / (tp + fn) if (tp + fn) else 0
        specificity = tn / (tn + fp) if (tn + fp) else 0
        ppv = tp / (tp + fp) if (tp + fp) else 0
        npv = tn / (tn + fn) if (tn + fn) else 0
        f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) else 0

        # Pooled kappa (if available)
        kappa_mean = group["Cohen_Kappa"].mean() if "Cohen_Kappa" in group.columns else np.nan

        row: dict = {
            "definition": col_name,
            "trial_type": group["trial_type"].iloc[0] if "trial_type" in group.columns else "",
            "n_hospitals": len(group),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "Accuracy": round(accuracy, 3),
            "Sensitivity": round(sensitivity, 3),
            "Specificity": round(specificity, 3),
            "PPV": round(ppv, 3),
            "NPV": round(npv, 3),
            "F1": round(f1, 3),
            "Cohen_Kappa_mean": round(kappa_mean, 3) if not np.isnan(kappa_mean) else np.nan,
        }

        # CI columns default to NaN; filled in below if bootstrap succeeds
        for metric in ("Sensitivity", "Specificity", "PPV", "NPV", "F1",
                        "Accuracy", "Kappa"):
            row[f"{metric}_CI_low"] = np.nan
            row[f"{metric}_CI_high"] = np.nan

        pooled.append(row)

    pooled_df = pd.DataFrame(pooled)
    if pooled_df.empty:
        return pooled_df

    # ------------------------------------------------------------------ #
    # Bootstrap CIs: attempt to load raw day-level data per trial type    #
    # ------------------------------------------------------------------ #
    # Build a mapping: definition column name -> (ehr_col, flowsheet_col).
    # The concordance_summary files store the column pair in the "Column"
    # field; by convention the EHR column is the reference and is paired
    # with a flowsheet column.  We look for an "EHR_col" / "Flowsheet_col"
    # metadata column in the summary, otherwise we try the definition name
    # itself as both (i.e. the concordance was computed on a single binary
    # column compared against itself - unlikely but safe to attempt).
    def _col_pair_from_group(grp: pd.DataFrame, definition: str):
        """Extract (ehr_col, flowsheet_col) from a summary group."""
        if "EHR_col" in grp.columns and "Flowsheet_col" in grp.columns:
            ehr = grp["EHR_col"].iloc[0]
            fs  = grp["Flowsheet_col"].iloc[0]
            return ehr, fs
        # Fallback: treat definition as the column name pair separated by
        # " vs " or "__vs__"
        for sep in (" vs ", "__vs__", " VS "):
            if sep in definition:
                parts = definition.split(sep, 1)
                return parts[0].strip(), parts[1].strip()
        return definition, definition

    trial_day_cache: dict[str, pd.DataFrame | None] = {}

    for idx, row in pooled_df.iterrows():
        trial = row["trial_type"]
        definition = row["definition"]

        # Load (and cache) the day-level DataFrame for this trial type
        if trial not in trial_day_cache:
            print(f"  Loading day-level data for bootstrap CIs ({trial})...")
            trial_day_cache[trial] = _load_day_level_df(data_dir, trial)

        day_df = trial_day_cache.get(trial)
        if day_df is None:
            continue  # No raw data available; leave CI columns as NaN

        # Recover the concordance column pair for this definition
        grp = metrics_df[metrics_df[group_col] == definition]
        ehr_col, flowsheet_col = _col_pair_from_group(grp, definition)

        boot = _bootstrap_ci_for_definition(day_df, ehr_col, flowsheet_col)
        if boot is None:
            continue

        # Merge CI values back into the pooled row
        metric_map = {
            "Sensitivity": "sensitivity",
            "Specificity": "specificity",
            "PPV":         "ppv",
            "NPV":         "npv",
            "F1":          "f1",
            "Accuracy":    "accuracy",
            "Kappa":       "kappa",
        }
        for col_prefix, boot_key in metric_map.items():
            if boot_key in boot:
                pooled_df.at[idx, f"{col_prefix}_CI_low"]  = boot[boot_key]["ci_low"]
                pooled_df.at[idx, f"{col_prefix}_CI_high"] = boot[boot_key]["ci_high"]

        print(f"  Bootstrap CIs added for '{definition}' ({trial}): "
              f"n={boot['n_rows']} rows, {boot['n_clusters']} clusters "
              f"[{boot['cluster_col_used']}]")

    return pooled_df


# ============================================================
# 4. TABLE 1: OPERATIONAL DEFINITIONS
# ============================================================

def generate_table1_operational_definitions(output_path):
    """Generate Table 1: operational definitions from source of truth.

    This is a descriptive table mapping each phenotype component to its
    data element, threshold, and CLIF source.
    """
    rows = [
        # SAT Eligibility
        {
            "Domain": "SAT Eligibility",
            "Component": "Assessment window",
            "Operational Definition": f"{SAT_ELIGIBILITY_WINDOW_START_HOUR}:00 prior day to {SAT_ELIGIBILITY_WINDOW_END_HOUR:02d}:00 index day",
            "CLIF Data Element": "respiratory_support.recorded_dttm, medication_admin_continuous.admin_dttm",
            "Threshold/Value": f"10 PM - 6 AM",
        },
        {
            "Domain": "SAT Eligibility",
            "Component": "Continuous sedatives",
            "Operational Definition": f"Receiving continuous infusion of {', '.join(SAT_SEDATIVES)} for >= {SAT_MIN_SEDATION_HOURS} hours",
            "CLIF Data Element": "medication_admin_continuous.med_category, .med_dose",
            "Threshold/Value": f">= {SAT_MIN_SEDATION_HOURS} hours, dose > 0",
        },
        {
            "Domain": "SAT Eligibility",
            "Component": "Continuous opioids",
            "Operational Definition": f"Receiving continuous infusion of {', '.join(SAT_OPIOIDS)} for >= {SAT_MIN_SEDATION_HOURS} hours",
            "CLIF Data Element": "medication_admin_continuous.med_category, .med_dose",
            "Threshold/Value": f">= {SAT_MIN_SEDATION_HOURS} hours, dose > 0",
        },
        {
            "Domain": "SAT Eligibility",
            "Component": "Paralytic exclusion",
            "Operational Definition": f"Exclude if receiving {', '.join(PARALYTICS)}",
            "CLIF Data Element": "medication_admin_continuous.med_category",
            "Threshold/Value": "Any dose > 0",
        },
        {
            "Domain": "SAT Eligibility",
            "Component": "Agitation exclusion",
            "Operational Definition": f"Exclude if RASS >= {RASS_AGITATION_THRESHOLD}",
            "CLIF Data Element": "patient_assessments.assessment_category = 'rass'",
            "Threshold/Value": f"RASS >= {RASS_AGITATION_THRESHOLD}",
        },
        # SAT Delivery
        {
            "Domain": "SAT Delivery",
            "Component": "Complete SAT",
            "Operational Definition": f"Discontinuation of ALL sedatives + opioids for >= {SAT_COMPLETE_DURATION_MIN} min",
            "CLIF Data Element": "medication_admin_continuous.med_dose = 0 or absent",
            "Threshold/Value": f">= {SAT_COMPLETE_DURATION_MIN} min",
        },
        {
            "Domain": "SAT Delivery",
            "Component": "Modified SAT",
            "Operational Definition": f"Discontinuation of sedatives only (opioids may continue) for >= {SAT_MODIFIED_DURATION_MIN} min",
            "CLIF Data Element": "medication_admin_continuous.med_dose",
            "Threshold/Value": f">= {SAT_MODIFIED_DURATION_MIN} min",
        },
        # SBT Eligibility
        {
            "Domain": "SBT Eligibility",
            "Component": "Controlled mode",
            "Operational Definition": f"On controlled ventilation mode for >= {SBT_MIN_CONTROLLED_MODE_HOURS} hours",
            "CLIF Data Element": "respiratory_support.mode_category",
            "Threshold/Value": f">= {SBT_MIN_CONTROLLED_MODE_HOURS} hours; modes: {', '.join(SBT_CONTROLLED_MODES)}",
        },
        {
            "Domain": "SBT Eligibility",
            "Component": "Respiratory stability",
            "Operational Definition": f"FiO2 <= {SBT_FIO2_MAX:.0%}, PEEP <= {SBT_PEEP_MAX} cmH2O, SpO2 >= {SBT_SPO2_MIN}%",
            "CLIF Data Element": "respiratory_support.fio2_set, .peep_set; vitals.vital_category = 'spo2'",
            "Threshold/Value": f"FiO2 <= {SBT_FIO2_MAX}, PEEP <= {SBT_PEEP_MAX}, SpO2 >= {SBT_SPO2_MIN}",
        },
        {
            "Domain": "SBT Eligibility",
            "Component": "Hemodynamic stability",
            "Operational Definition": "Norepinephrine equivalent <= 0.2 mcg/kg/min",
            "CLIF Data Element": "medication_admin_continuous.med_category, .med_dose",
            "Threshold/Value": "NEE <= 0.2 mcg/kg/min for >= 2 hours",
        },
        # SBT Delivery
        {
            "Domain": "SBT Delivery",
            "Component": "Primary SBT",
            "Operational Definition": f"Transition from controlled to support mode for >= {SBT_PRIMARY_DURATION_MIN} min",
            "CLIF Data Element": "respiratory_support.mode_category",
            "Threshold/Value": f">= {SBT_PRIMARY_DURATION_MIN} min on {', '.join(SBT_SUPPORT_MODES)} or T-piece; PS <= {SBT_PS_MAX}, CPAP <= {SBT_CPAP_MAX}",
        },
        {
            "Domain": "SBT Delivery",
            "Component": "Modified SBT",
            "Operational Definition": f"Transition sustained for >= {SBT_MODIFIED_DURATION_MIN} min",
            "CLIF Data Element": "respiratory_support.mode_category",
            "Threshold/Value": f">= {SBT_MODIFIED_DURATION_MIN} min",
        },
        # Ventilator-day definition
        {
            "Domain": "General",
            "Component": "Ventilator-day",
            "Operational Definition": f"24-hour period anchored at {VENT_DAY_ANCHOR_HOUR:02d}:00 local time",
            "CLIF Data Element": "respiratory_support.recorded_dttm",
            "Threshold/Value": f"{VENT_DAY_ANCHOR_HOUR:02d}:00 to {VENT_DAY_ANCHOR_HOUR-1:02d}:59 next day",
        },
        {
            "Domain": "General",
            "Component": "IMV episode",
            "Operational Definition": f"Consecutive IMV records separated by < {IMV_EPISODE_GAP_HOURS} hours",
            "CLIF Data Element": "respiratory_support.device_category = 'imv'",
            "Threshold/Value": f"{IMV_EPISODE_GAP_HOURS}-hour gap defines new episode",
        },
        {
            "Domain": "General",
            "Component": "Study period",
            "Operational Definition": f"{STUDY_PERIOD_START} to {STUDY_PERIOD_END}",
            "CLIF Data Element": "hospitalization.admission_dttm",
            "Threshold/Value": f"Age {MIN_AGE}-{MAX_AGE}, >= {MIN_IMV_HOURS}h IMV",
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Table 1 (operational definitions) saved: {output_path}")
    return df


# ============================================================
# 5. MANUSCRIPT NUMBERS JSON
# ============================================================

def compile_manuscript_numbers(consort, rates_df, concordance_df):
    """Compile all numbers needed for manuscript text into a single dict."""
    numbers = {}

    # CONSORT
    numbers["consort"] = consort

    # Delivery rates
    for _, row in rates_df.iterrows():
        key = f"{row['trial_type']}_{row['delivery_definition']}"
        numbers[key] = {
            "overall_rate": row.get("overall_rate_pct", ""),
            "pooled_rate": row.get("pooled_rate_pct", ""),
            "n_eligible": int(row.get("n_eligible_days", 0)),
            "n_delivered": int(row.get("n_delivered_days", 0)),
            "median_rate": row.get("median_rate", ""),
            "iqr": f"[{row.get('iqr_low', '')}, {row.get('iqr_high', '')}]",
            "range": f"[{row.get('range_low', '')}, {row.get('range_high', '')}]",
        }

    # Concordance — include bootstrap CI columns when present
    _ci_metric_map = {
        "accuracy":    ("Accuracy",    "Accuracy_CI_low",    "Accuracy_CI_high"),
        "sensitivity": ("Sensitivity", "Sensitivity_CI_low", "Sensitivity_CI_high"),
        "specificity": ("Specificity", "Specificity_CI_low", "Specificity_CI_high"),
        "ppv":         ("PPV",         "PPV_CI_low",         "PPV_CI_high"),
        "npv":         ("NPV",         "NPV_CI_low",         "NPV_CI_high"),
        "f1":          ("F1",          "F1_CI_low",          "F1_CI_high"),
        "kappa":       ("Cohen_Kappa_mean", "Kappa_CI_low",  "Kappa_CI_high"),
    }

    for _, row in concordance_df.iterrows():
        key = f"concordance_{row.get('trial_type', '')}_{row['definition']}"
        entry: dict = {}

        for out_key, (point_col, ci_low_col, ci_high_col) in _ci_metric_map.items():
            point_val = row.get(point_col, "")
            ci_low    = row.get(ci_low_col, np.nan)
            ci_high   = row.get(ci_high_col, np.nan)

            # Format as "0.XXX (95% CI 0.XXX-0.XXX)" when CIs are available
            if (pd.notna(ci_low) and pd.notna(ci_high)
                    and ci_low != "" and ci_high != ""):
                try:
                    entry[out_key] = (
                        f"{float(point_val):.3f} "
                        f"(95% CI {float(ci_low):.3f}-{float(ci_high):.3f})"
                    )
                except (TypeError, ValueError):
                    entry[out_key] = point_val
            else:
                entry[out_key] = point_val

            # Also expose raw numeric CI values for downstream use
            entry[f"{out_key}_ci_low"]  = ci_low  if pd.notna(ci_low)  else None
            entry[f"{out_key}_ci_high"] = ci_high if pd.notna(ci_high) else None

        numbers[key] = entry

    return numbers


# ============================================================
# 6. ESM TABLES
# ============================================================

def generate_esm_tables(data_dir, output_dir):
    """Generate Electronic Supplementary Material tables for ICM submission.

    eTable 1: Construct validity outcome models
    eTable 2: Site-stratified confusion matrices
    eTable 3: Sensitivity analysis results
    """
    esm_dir = os.path.join(output_dir, "..", "esm")
    os.makedirs(esm_dir, exist_ok=True)

    # eTable 1: Construct validity outcomes
    outcomes_path = os.path.join(data_dir, "final", "construct_validity_outcomes.csv")
    if os.path.exists(outcomes_path):
        try:
            df = pd.read_csv(outcomes_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            df = pd.DataFrame()
        # Format for publication: round, add CI strings
        for est_col, lo_col, hi_col in [
            ("HR", "HR_lower_95", "HR_upper_95"),
            ("OR", "OR_lower_95", "OR_upper_95"),
            ("IRR", "IRR_lower_95", "IRR_upper_95"),
        ]:
            if est_col in df.columns:
                mask = df[est_col].notna()
                df.loc[mask, f"{est_col}_formatted"] = df.loc[mask].apply(
                    lambda r: f"{r[est_col]:.2f} ({r.get(lo_col, 'NA'):.2f}-{r.get(hi_col, 'NA'):.2f})"
                    if pd.notna(r.get(lo_col)) else f"{r[est_col]:.2f}",
                    axis=1,
                )
        df.to_csv(os.path.join(esm_dir, "etable1_construct_validity.csv"), index=False)
        print(f"  eTable 1 (construct validity) saved")
    else:
        print(f"  eTable 1 skipped: {outcomes_path} not found")

    # eTable 2: Site-stratified concordance (full confusion matrices)
    concordance_files = []
    for root, dirs, files in os.walk(os.path.join(data_dir, "final")):
        for f in files:
            if "concordance_summary" in f and f.endswith(".csv"):
                concordance_files.append(os.path.join(root, f))

    if concordance_files:
        all_conc = []
        for fpath in concordance_files:
            try:
                cdf = pd.read_csv(fpath)
                # Extract site from path
                parts = fpath.split(os.sep)
                site_parts = [p for p in parts if p not in ["output", "final", "intermediate"]]
                cdf["site"] = site_parts[-2] if len(site_parts) >= 2 else "Unknown"
                all_conc.append(cdf)
            except Exception:
                pass
        if all_conc:
            full_conc = pd.concat(all_conc, ignore_index=True)
            full_conc.to_csv(os.path.join(esm_dir, "etable2_site_concordance.csv"), index=False)
            print(f"  eTable 2 (site-stratified concordance) saved: {len(concordance_files)} files")
    else:
        print(f"  eTable 2 skipped: no concordance files found")

    # eTable 3: Sensitivity analyses
    sensitivity_dir = os.path.join(data_dir, "final", "sensitivity")
    if os.path.exists(sensitivity_dir):
        sens_frames = []
        for f in sorted(os.listdir(sensitivity_dir)):
            if f.startswith("sensitivity_") and f.endswith(".csv"):
                sdf = pd.read_csv(os.path.join(sensitivity_dir, f))
                sdf["source_file"] = f
                sens_frames.append(sdf)
        if sens_frames:
            all_sens = pd.concat(sens_frames, ignore_index=True)
            all_sens.to_csv(os.path.join(esm_dir, "etable3_sensitivity_analyses.csv"), index=False)
            print(f"  eTable 3 (sensitivity analyses) saved: {len(sens_frames)} files")
    else:
        print(f"  eTable 3 skipped: sensitivity dir not found")


# ============================================================
# MAIN
# ============================================================

def run_aggregation(data_dir, output_dir):
    """Run all site aggregation steps."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("SITE AGGREGATION FOR MANUSCRIPT")
    print("=" * 60)

    # 1. CONSORT numbers
    print("\n1. Computing CONSORT numbers...")
    consort = compute_consort_numbers(data_dir)
    consort_path = os.path.join(output_dir, "consort_numbers.json")
    with open(consort_path, "w") as f:
        json.dump(consort, f, indent=2)
    print(f"  CONSORT: {consort}")

    # 2. Pooled delivery rates
    print("\n2. Computing pooled delivery rates...")
    rates_df = compute_pooled_delivery_rates(data_dir)
    if not rates_df.empty:
        rates_df.to_csv(os.path.join(output_dir, "pooled_delivery_rates.csv"), index=False)

    # 3. Pooled concordance
    print("\n3. Computing pooled concordance...")
    concordance_df = compute_pooled_concordance(data_dir)
    if not concordance_df.empty:
        concordance_df.to_csv(os.path.join(output_dir, "pooled_concordance.csv"), index=False)
        print(f"  Pooled concordance:\n{concordance_df.to_string()}")

    # 4. Table 1 operational definitions
    print("\n4. Generating Table 1 (operational definitions)...")
    generate_table1_operational_definitions(
        os.path.join(output_dir, "table1_operational_definitions.csv")
    )

    # 5. ESM tables
    print("\n5. Generating ESM tables...")
    generate_esm_tables(data_dir, output_dir)

    # 6. Manuscript numbers JSON
    print("\n6. Compiling manuscript numbers...")
    if not rates_df.empty or not concordance_df.empty:
        numbers = compile_manuscript_numbers(
            consort,
            rates_df if not rates_df.empty else pd.DataFrame(),
            concordance_df if not concordance_df.empty else pd.DataFrame(),
        )
        numbers_path = os.path.join(output_dir, "manuscript_numbers.json")
        with open(numbers_path, "w") as f:
            json.dump(numbers, f, indent=2, default=str)
        print(f"  Manuscript numbers saved: {numbers_path}")

    print(f"\nAll aggregation complete. Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate site outputs for manuscript")
    parser.add_argument("--data-dir", default="../output")
    parser.add_argument("--output-dir", default="../output/final/pooled")
    args = parser.parse_args()
    run_aggregation(args.data_dir, args.output_dir)
