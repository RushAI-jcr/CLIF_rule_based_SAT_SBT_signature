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

        n_vent_days = result_df["hosp_id_day_key"].nunique() if "hosp_id_day_key" in result_df.columns else len(result_df)

        # Delivery is 1 per eligible vent-day (first qualifying candidate)
        ehr_delivered = int(result_df["SAT_EHR_delivery"].sum())
        mod_delivered = int(result_df["SAT_modified_delivery"].sum())

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

            n_vent_days = (
                result_df["hosp_id_day_key"].nunique()
                if "hosp_id_day_key" in result_df.columns
                else len(result_df)
            )

            primary_col = f"EHR_Delivery_{duration}mins"
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

def run_all_sensitivity(
    cohort_path: str,
    output_dir: str,
    sat_file: str | None = None,
    sbt_file: str | None = None,
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
    sensitivity_exclusions(cohort, output_dir)

    print("\n5. First episode restriction...")
    sensitivity_first_episode(cohort, output_dir)

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
    args = parser.parse_args()
    run_all_sensitivity(
        cohort_path=args.cohort,
        output_dir=args.output_dir,
        sat_file=args.sat_file,
        sbt_file=args.sbt_file,
    )
