"""
Site-Level Output Schema for Multi-Site CLIF Federated Analysis
===============================================================
Each CLIF site runs the SAT/SBT pipeline locally and exports CSVs
matching these schemas. Only aggregate data leaves each site.

Usage:
    from site_output_schema import validate_site_summary, SITE_SUMMARY_COLUMNS
"""

import pandas as pd
import numpy as np

# ============================================================
# SITE SUMMARY SCHEMA (one row per site)
# ============================================================

SITE_SUMMARY_COLUMNS = {
    # Identifiers
    "site_id": str,
    "site_name": str,
    "hospital_id": str,

    # Cohort counts
    "n_hospitalizations": int,
    "n_patients": int,
    "n_vent_days": int,

    # Primary outcome: SAT delivery rate (proportion)
    "sat_ehr_delivery_rate": float,
    "sat_ehr_delivery_se": float,
    "sat_modified_delivery_rate": float,
    "sat_modified_delivery_se": float,

    # SBT outcomes
    "sbt_eligible_days": int,
    "sbt_ehr_delivery_2min_rate": float,
    "sbt_ehr_delivery_2min_se": float,
    "sbt_ehr_delivery_5min_rate": float,
    "sbt_ehr_delivery_5min_se": float,
    "sbt_ehr_delivery_30min_rate": float,
    "sbt_ehr_delivery_30min_se": float,

    # Concordance metrics (vs flowsheet)
    "sat_concordance_accuracy": float,
    "sat_concordance_precision": float,
    "sat_concordance_recall": float,
    "sat_concordance_f1": float,
    "sbt_concordance_accuracy": float,
    "sbt_concordance_precision": float,
    "sbt_concordance_recall": float,
    "sbt_concordance_f1": float,
}

# ============================================================
# TABLE 1 SCHEMA (one row per site for pooling)
# ============================================================

TABLE1_COLUMNS = {
    "site_id": str,
    "n_total": int,
    "n_male": int,
    "n_female": int,
    "age_mean": float,
    "age_sd": float,
    "n_race_white": int,
    "n_race_black": int,
    "n_race_asian": int,
    "n_race_other": int,
    "n_hispanic": int,
    "icu_los_median": float,
    "icu_los_iqr_lower": float,
    "icu_los_iqr_upper": float,
    "inpatient_los_median": float,
    "inpatient_los_iqr_lower": float,
    "inpatient_los_iqr_upper": float,
    "n_discharged_alive": int,
    "n_discharged_expired": int,
    "n_discharged_hospice": int,
}

# ============================================================
# VALIDATION
# ============================================================

def validate_site_summary(df: pd.DataFrame) -> list[str]:
    """Validate a site summary DataFrame against the schema.
    Returns list of error messages (empty = valid)."""
    errors = []
    for col, dtype in SITE_SUMMARY_COLUMNS.items():
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    for col in df.columns:
        if col in SITE_SUMMARY_COLUMNS:
            if SITE_SUMMARY_COLUMNS[col] == float:
                if df[col].dtype not in [np.float64, np.float32, float]:
                    try:
                        df[col] = df[col].astype(float)
                    except (ValueError, TypeError):
                        errors.append(f"Column {col} cannot be cast to float")
    # Check SE > 0 where applicable
    se_cols = [c for c in df.columns if c.endswith("_se")]
    for c in se_cols:
        if (df[c].dropna() <= 0).any():
            errors.append(f"Non-positive standard error in {c}")
    return errors


def validate_table1(df: pd.DataFrame) -> list[str]:
    """Validate a Table 1 DataFrame against the schema."""
    errors = []
    for col in TABLE1_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    return errors


def compute_site_se(n_events: int, n_total: int) -> float:
    """Compute standard error for a proportion (site-level delivery rate)."""
    if n_total == 0:
        return np.nan
    p = n_events / n_total
    return np.sqrt(p * (1 - p) / n_total)
