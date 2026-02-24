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
    python 06_sensitivity_analyses.py --cohort ../output/intermediate/study_cohort.parquet \
                                       --output-dir ../output/final/sensitivity
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "utils"))


import argparse
import os
import warnings

import numpy as np
import pandas as pd
from definitions_source_of_truth import (
    SENSITIVITY_SAT_DURATIONS_MIN,
    SENSITIVITY_SBT_DURATIONS_MIN,
    SENSITIVITY_SBT_PS_THRESHOLDS,
    SENSITIVITY_EXCLUSIONS,
    CLIF_TABLES_REQUIRED,
)

warnings.filterwarnings("ignore")


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

def sensitivity_sat_durations(cohort_df, output_dir):
    """Re-run SAT delivery detection with alternative durations."""
    results = []

    for duration in SENSITIVITY_SAT_DURATIONS_MIN:
        # This would re-run the SAT delivery algorithm from 01_SAT_standard
        # with different duration thresholds. For now, collect the framework.
        results.append({
            "analysis": "SAT duration threshold",
            "parameter": f"{duration} min",
            "note": f"Re-run SAT delivery with {duration}-min discontinuation threshold",
            "status": "framework_ready",
        })

    pd.DataFrame(results).to_csv(
        os.path.join(output_dir, "sensitivity_sat_durations.csv"), index=False
    )
    print(f"  SAT duration sensitivity framework saved")


# ============================================================
# SENSITIVITY: ALTERNATIVE SBT PARAMETERS
# ============================================================

def sensitivity_sbt_parameters(cohort_df, output_dir):
    """Re-run SBT delivery with alternative durations and PS thresholds."""
    results = []

    for duration in SENSITIVITY_SBT_DURATIONS_MIN:
        results.append({
            "analysis": "SBT duration threshold",
            "parameter": f"{duration} min",
            "note": f"Re-run SBT delivery with {duration}-min transition threshold",
            "status": "framework_ready",
        })

    for ps in SENSITIVITY_SBT_PS_THRESHOLDS:
        results.append({
            "analysis": "SBT PS threshold",
            "parameter": f"PS <= {ps} cmH2O",
            "note": f"Re-run SBT with PS threshold of {ps} cmH2O",
            "status": "framework_ready",
        })

    pd.DataFrame(results).to_csv(
        os.path.join(output_dir, "sensitivity_sbt_parameters.csv"), index=False
    )
    print(f"  SBT parameter sensitivity framework saved")


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

    # Day 0 exclusion (most straightforward)
    if "day_number" in cohort_df.columns:
        n_before = cohort_df["hosp_id_day_key"].nunique()
        excluded = cohort_df[cohort_df["day_number"] > 1]
        n_after = excluded["hosp_id_day_key"].nunique()
        results.append({
            "exclusion": "IMV day 0 (intubation day)",
            "days_before": n_before,
            "days_after": n_after,
            "days_excluded": n_before - n_after,
            "status": "computed",
        })
    else:
        results.append({
            "exclusion": "IMV day 0 (intubation day)",
            "note": "day_number column not available",
            "status": "needs_data",
        })

    # Other exclusions need ICD codes or additional CLIF tables
    for excl in ["Cardiac arrest diagnosis", "Targeted temperature management",
                 "Comfort care status"]:
        results.append({
            "exclusion": excl,
            "note": "Requires hospital_diagnosis or code_status CLIF table",
            "status": "needs_data",
        })

    pd.DataFrame(results).to_csv(
        os.path.join(output_dir, "sensitivity_exclusions.csv"), index=False
    )
    print(f"  Clinical exclusion sensitivity saved")


# ============================================================
# SENSITIVITY: FIRST EPISODE ONLY
# ============================================================

def sensitivity_first_episode(cohort_df, output_dir):
    """Restrict to first IMV episode per hospitalization."""
    if "day_number" not in cohort_df.columns:
        print("  Cannot identify first episode without episode tracking")
        return

    # Identify first episode (lowest day numbers per hospitalization)
    # This is an approximation; true episode identification requires
    # classify_imv_episodes() from definitions_source_of_truth.py
    results = {
        "analysis": "First IMV episode only",
        "total_hospitalizations": cohort_df["hospitalization_id"].nunique(),
        "note": "Framework ready - requires IMV episode classification",
        "status": "framework_ready",
    }

    pd.DataFrame([results]).to_csv(
        os.path.join(output_dir, "sensitivity_first_episode.csv"), index=False
    )
    print(f"  First episode sensitivity framework saved")


# ============================================================
# MAIN
# ============================================================

def run_all_sensitivity(cohort_path, output_dir):
    """Run all sensitivity analyses."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading cohort from {cohort_path}...")
    if cohort_path.endswith(".parquet"):
        cohort = pd.read_parquet(cohort_path)
    else:
        cohort = pd.read_csv(cohort_path, low_memory=False)

    print(f"Cohort: {len(cohort):,} rows, "
          f"{cohort['hospitalization_id'].nunique():,} hospitalizations")

    print("\n1. Data completeness assessment...")
    assess_data_completeness(cohort, output_dir)

    print("\n2. SAT duration sensitivity...")
    sensitivity_sat_durations(cohort, output_dir)

    print("\n3. SBT parameter sensitivity...")
    sensitivity_sbt_parameters(cohort, output_dir)

    print("\n4. Clinical exclusions...")
    sensitivity_exclusions(cohort, output_dir)

    print("\n5. First episode restriction...")
    sensitivity_first_episode(cohort, output_dir)

    print(f"\nAll sensitivity analyses saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="../output/intermediate/study_cohort.parquet")
    parser.add_argument("--output-dir", default="../output/final/sensitivity")
    args = parser.parse_args()
    run_all_sensitivity(args.cohort, args.output_dir)
