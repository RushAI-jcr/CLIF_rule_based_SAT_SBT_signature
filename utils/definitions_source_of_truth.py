"""
Single Source of Truth: SAT/SBT Definitions and Study Parameters
================================================================
All phenotyping definitions, eligibility criteria, and study parameters
are defined here. All analysis scripts must import from this module.

References:
- Manuscript: "SAT SBT Manuscript_02192026.docx"
- CLIF 2.1 Data Dictionary: https://clif-icu.com/data-dictionary
- CLIF controlled vocabularies (mCIDE)

CLIF 2.1 Compliance Notes:
- All joins on hospitalization_id (not patient_id) for longitudinal tables
- Filter on *_category columns, never *_name
- mar_action_group = 'administered' for drug exposure
- Outlier thresholds from outlier-handling/ applied before statistics
- Timestamps in UTC; site timezone conversion via config
"""

import pandas as pd
import numpy as np

# ============================================================
# STUDY PARAMETERS
# ============================================================

STUDY_PERIOD_START = "2022-01-01"
STUDY_PERIOD_END = "2024-12-31"
MIN_AGE = 18
MAX_AGE = 119  # CLIF convention for age cap
MIN_IMV_HOURS = 6  # Minimum hours of IMV for episode inclusion

# Ventilator-day anchor: 06:00 local time (06:00 to 05:59 next day)
# Per manuscript Statistical Analysis section
VENT_DAY_ANCHOR_HOUR = 6

# IMV episode separation: 72 hours without ventilator support
IMV_EPISODE_GAP_HOURS = 72

# Maximum forward-fill window (in observations, not hours) for carried-forward data
# Typical ICU charting interval is 1-2 hours; 6 observations ≈ 6-12 hours max carry
MAX_FFILL_OBSERVATIONS = 6  # limit for medications, vitals, RASS forward-fill

# ============================================================
# SAT DEFINITIONS
# ============================================================

# --- SAT Eligibility ---
# Window: 10 PM prior day to 6 AM index day
SAT_ELIGIBILITY_WINDOW_START_HOUR = 22  # prior day
SAT_ELIGIBILITY_WINDOW_END_HOUR = 6    # index day

# Continuous sedatives (CLIF med_category values)
# Dexmedetomidine EXCLUDED per manuscript: "allows for awake status"
SAT_SEDATIVES = ["propofol", "lorazepam", "midazolam"]

# Continuous opioids (CLIF med_category values)
SAT_OPIOIDS = ["fentanyl", "morphine", "hydromorphone"]

# All sedation meds (sedatives + opioids)
SAT_ALL_SEDATION_MEDS = SAT_SEDATIVES + SAT_OPIOIDS

# Minimum duration on continuous sedatives/opioids for SAT eligibility
SAT_MIN_SEDATION_HOURS = 4

# Paralytics that exclude eligibility (CLIF med_category values)
PARALYTICS = ["cisatracurium", "rocuronium", "vecuronium"]

# RASS agitation exclusion threshold
RASS_AGITATION_THRESHOLD = 2  # Exclude if RASS >= 2

# --- SAT Delivery (Primary) ---
# Complete SAT: discontinuation of ALL sedatives + opioids >= 30 min
SAT_COMPLETE_DURATION_MIN = 30

# --- SAT Delivery (Modified) ---
# Modified SAT: discontinuation of sedatives only (opioids may continue)
SAT_MODIFIED_DURATION_MIN = 30

# --- SAT Delivery (RASS-Enhanced) ---
# Alertness following discontinuation: RASS 0 to +1 within 45 min
SAT_RASS_ALERTNESS_WINDOW_MIN = 45
SAT_RASS_ALERTNESS_RANGE = (0, 1)  # inclusive

# ============================================================
# SBT DEFINITIONS
# ============================================================

# --- SBT Eligibility ---
# Window: same as SAT (10 PM prior day to 6 AM index day)
SBT_ELIGIBILITY_WINDOW_START_HOUR = 22
SBT_ELIGIBILITY_WINDOW_END_HOUR = 6

# Controlled modes of ventilation (CLIF mode_category values)
SBT_CONTROLLED_MODES = [
    "assist control-volume control",
    "pressure control",
    "pressure-regulated volume control",
]

# Minimum hours on controlled mode for SBT eligibility
SBT_MIN_CONTROLLED_MODE_HOURS = 12  # Manuscript specifies 12 hours

# Respiratory stability thresholds
SBT_FIO2_MAX = 0.50       # FiO2 <= 50%
SBT_PEEP_MAX = 8          # PEEP <= 8 cmH2O
SBT_SPO2_MIN = 88         # SpO2 >= 88%

# Hemodynamic stability thresholds (vasopressor dose limits)
# Norepinephrine equivalent <= 0.2 mcg/kg/min
SBT_VASOPRESSOR_LIMITS = {
    "norepinephrine": 0.2,     # mcg/kg/min
    "dopamine": 5.0,           # mcg/kg/min
    "dobutamine": 5.0,         # mcg/kg/min
    "vasopressin": np.inf,     # any dose allowed
    "milrinone": np.inf,       # any dose allowed
}

# Minimum stability duration
SBT_MIN_STABILITY_HOURS = 2

# --- SBT Delivery (Primary) ---
# Transition from controlled to support mode >= 2 min
SBT_PRIMARY_DURATION_MIN = 2

# Support modes (CLIF mode_category values)
SBT_SUPPORT_MODES = ["pressure support/cpap"]
# Also T-piece (checked via mode_name)

# Support mode pressure limits
SBT_PS_MAX = 8    # Pressure support <= 8 cmH2O
SBT_CPAP_MAX = 8  # CPAP <= 8 cmH2O

# --- SBT Delivery (Modified) ---
# Transition sustained >= 5 min
SBT_MODIFIED_DURATION_MIN = 5

# ============================================================
# OUTCOME DEFINITIONS
# ============================================================

# Ventilator-free days: up to day 28
VFD_MAX_DAYS = 28

# Extubation: transition from IMV to 2 consecutive non-IMV observations
# (already implemented in 00_cohort_id.ipynb)

# Death as competing event for time-to-extubation

# ============================================================
# CLIF 2.1 TABLE AND COLUMN REFERENCES
# ============================================================

CLIF_TABLES_REQUIRED = {
    "patient": ["patient_id", "sex_category", "race_category",
                "ethnicity_category"],
    "hospitalization": ["patient_id", "hospitalization_id",
                        "admission_dttm", "discharge_dttm",
                        "age_at_admission"],
    "adt": ["hospitalization_id", "in_dttm", "location_category",
            "hospital_id"],
    "medication_admin_continuous": ["hospitalization_id", "admin_dttm",
                                    "med_category", "med_dose",
                                    "med_dose_unit", "mar_action_group"],
    "respiratory_support": ["hospitalization_id", "recorded_dttm",
                            "device_category", "mode_category",
                            "fio2_set", "peep_set",
                            "pressure_support_set", "mode_name",
                            "tracheostomy"],
    "patient_assessments": ["hospitalization_id", "recorded_dttm",
                            "assessment_category", "numerical_value",
                            "categorical_value"],
    "vitals": ["hospitalization_id", "recorded_dttm",
               "vital_category", "vital_value"],
}

# CLIF 2.1 patient_assessments categories for SAT/SBT flowsheets
FLOWSHEET_ASSESSMENT_CATEGORIES = [
    "sat_screen_pass_fail",
    "sat_delivery_pass_fail",
    "sbt_screen_pass_fail",
    "sbt_delivery_pass_fail",
    "sbt_failure_reason",
    "rass",
    "gcs_total",
]

# Vasopressor med_category values (CLIF 2.1 mCIDE)
VASOPRESSORS = [
    "norepinephrine", "epinephrine", "phenylephrine",
    "angiotensin", "vasopressin", "dopamine",
    "dobutamine", "milrinone", "isoproterenol",
]

# ============================================================
# SENSITIVITY ANALYSIS PARAMETERS
# ============================================================

SENSITIVITY_SAT_DURATIONS_MIN = [15, 30, 60]
SENSITIVITY_SBT_DURATIONS_MIN = [2, 5, 30]
SENSITIVITY_SBT_PS_THRESHOLDS = [5, 8, 10]  # PS ≤5, ≤8 (primary), ≤10
SENSITIVITY_SBT_CPAP_THRESHOLDS = [5, 8]    # CPAP ≤5, ≤8 (primary)

# Exclusion conditions for sensitivity analyses
SENSITIVITY_EXCLUSIONS = [
    "cardiac_arrest_diagnosis",       # ICD codes for cardiac arrest
    "targeted_temperature_management", # TTM during prior day
    "comfort_care_status",            # Code status = comfort
    "imv_day_0",                      # Day of intubation
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_vent_day_boundaries(date, anchor_hour=VENT_DAY_ANCHOR_HOUR):
    """Return (start, end) for a ventilator-day anchored at anchor_hour.

    Per manuscript: ventilator-days are 24-hour periods anchored at 06:00
    local time (06:00 to 05:59 next day).
    """
    start = pd.Timestamp(str(date)).normalize() + pd.Timedelta(hours=anchor_hour)
    end = start + pd.Timedelta(hours=24)  # half-open interval [start, end)
    return start, end


def get_eligibility_window(date,
                           start_hour=SAT_ELIGIBILITY_WINDOW_START_HOUR,
                           end_hour=SAT_ELIGIBILITY_WINDOW_END_HOUR):
    """Return (start, end) for eligibility assessment window.

    10 PM prior day to 6 AM index day.
    """
    day = pd.Timestamp(date).normalize()
    start = day - pd.Timedelta(days=1) + pd.Timedelta(hours=start_hour)
    end = day + pd.Timedelta(hours=end_hour)
    return start, end


def classify_imv_episodes(resp_df, gap_hours=IMV_EPISODE_GAP_HOURS):
    """Assign IMV episode IDs based on gaps in ventilator support.

    Per manuscript: 72 hours of absence of ventilator support to distinguish
    episodes.

    Parameters
    ----------
    resp_df : pd.DataFrame
        respiratory_support table filtered to device_category == 'IMV',
        must have hospitalization_id and recorded_dttm columns.
    gap_hours : int
        Hours of gap to define new episode.

    Returns
    -------
    pd.DataFrame with additional 'imv_episode_id' column.
    """
    df = resp_df.sort_values(
        ["hospitalization_id", "recorded_dttm"]
    ).copy()
    gap = pd.Timedelta(hours=gap_hours)

    df["time_diff"] = df.groupby("hospitalization_id")["recorded_dttm"].diff()
    df["new_episode"] = (df["time_diff"] > gap) | df["time_diff"].isna()
    df["imv_episode_id"] = (
        df.groupby("hospitalization_id")["new_episode"].cumsum()
    )
    # Create globally unique episode ID
    df["imv_episode_id"] = (
        df["hospitalization_id"].astype(str)
        + "_ep_"
        + df["imv_episode_id"].astype(str)
    )
    df.drop(columns=["time_diff", "new_episode"], inplace=True)
    return df


# Surgical ICU location categories for medical_admission derivation [SAP 2.8]
SURGICAL_ICU_LOCATIONS = {"sicu", "cticu", "cardiac_surgery_icu"}


def compute_norepinephrine_equivalent(row):
    """Compute norepinephrine equivalent dose in mcg/kg/min.

    Conversion factors per Goradia et al. and CLIF convention:
    - norepinephrine: 1x
    - epinephrine: 1x
    - dopamine: 0.01x (dose in mcg/kg/min * 0.01)
    - phenylephrine: 0.1x
    - vasopressin: 2.5 units/hr = 0.1 mcg/kg/min NE equiv (fixed)
    """
    nee = 0.0
    if pd.notna(row.get("norepinephrine")):
        nee += row["norepinephrine"]
    if pd.notna(row.get("epinephrine")):
        nee += row["epinephrine"]
    if pd.notna(row.get("dopamine")):
        nee += row["dopamine"] * 0.01
    if pd.notna(row.get("phenylephrine")):
        nee += row["phenylephrine"] * 0.1
    if pd.notna(row.get("vasopressin")):
        # vasopressin in units/min; 0.04 units/min ~ 0.1 mcg/kg/min NE
        nee += (row["vasopressin"] / 0.04) * 0.1
    return nee
