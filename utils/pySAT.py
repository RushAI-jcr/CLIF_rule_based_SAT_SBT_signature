"""
pySAT.py
========
Reusable SAT (Spontaneous Awakening Trial) delivery detection functions.

Extracted from 01_SAT_standard.ipynb (Cell 13) to support parameterized
sensitivity analyses with alternative interruption durations.

CLIF 2.1 compliance:
- Joins on hospitalization_id
- Filters on *_category columns (device_category_ffill, location_category_ffill)
- med_category columns: fentanyl, propofol, lorazepam, midazolam, hydromorphone, morphine
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional

from definitions_source_of_truth import (
    SAT_COMPLETE_DURATION_MIN,
    SAT_SEDATIVES,
    SAT_OPIOIDS,
    SAT_ALL_SEDATION_MEDS,
)

# Medication column groups
_ALL_SEDATION_COLS = ["fentanyl", "propofol", "lorazepam", "midazolam", "hydromorphone", "morphine"]
_SEDATIVE_ONLY_COLS = ["propofol", "lorazepam", "midazolam"]


def _all_zero_or_nan(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Return True for each row where all specified columns are 0 or NaN."""
    present = [c for c in cols if c in df.columns]
    if not present:
        raise KeyError(
            f"_all_zero_or_nan: none of the expected columns {cols} found in DataFrame"
        )
    return df[present].apply(
        lambda row: all(pd.isna(v) or v == 0 for v in row), axis=1
    )


def detect_sat_delivery(
    df: pd.DataFrame,
    duration_min: int = SAT_COMPLETE_DURATION_MIN,
) -> pd.DataFrame:
    """Detect SAT delivery on eligible ventilator-days.

    For each ``hosp_id_day_key`` in rows where ``on_vent_and_sedation == 1``,
    this function examines candidate rows (those where ``rank_sedation`` is not
    NaN, indicating zero sedation dose) and checks whether a forward window of
    ``duration_min`` minutes satisfies all SAT delivery conditions.

    Two delivery columns are added to the returned DataFrame:

    - ``SAT_EHR_delivery``: All sedatives **and** opioids
      (fentanyl, propofol, lorazepam, midazolam, hydromorphone, morphine) are
      0 or NaN for the full ``duration_min`` window, AND the patient remains on
      IMV in the ICU throughout that window.

    - ``SAT_modified_delivery``: Non-opioid sedatives only
      (propofol, lorazepam, midazolam) are 0 or NaN for the full
      ``duration_min`` window, with the same device/location requirements.

    Parameters
    ----------
    df : pd.DataFrame
        Row-level time-series DataFrame. Required columns:

        - ``hosp_id_day_key`` — unique key for hospitalization-day
        - ``event_time`` — datetime of each observation
        - ``on_vent_and_sedation`` — 1 if the row is an eligible vent-day row
        - ``rank_sedation`` — non-NaN indicates a zero sedation dose observation
        - ``rank_sedation_non_ops`` — non-NaN indicates zero non-opioid sedation
        - ``device_category_ffill`` — forward-filled device category
        - ``location_category_ffill`` — forward-filled location category
        - medication columns: fentanyl, propofol, lorazepam, midazolam,
          hydromorphone, morphine (0 or NaN for zero/absent dose)

    duration_min : int, optional
        Minimum duration in minutes of sedation discontinuation required to
        count as SAT delivery. Default is ``SAT_COMPLETE_DURATION_MIN`` (30).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with two new columns appended:

        - ``SAT_EHR_delivery`` : float (1.0 = delivered, NaN = not delivered)
        - ``SAT_modified_delivery`` : float (1.0 = delivered, NaN = not delivered)

    Notes
    -----
    Only the **first** qualifying candidate row per ``hosp_id_day_key`` is
    flagged; subsequent rows for the same day are not re-evaluated once
    delivery is confirmed.

    Examples
    --------
    >>> result = detect_sat_delivery(sat_df, duration_min=30)
    >>> delivery_rate = result["SAT_EHR_delivery"].sum() / result["hosp_id_day_key"].nunique()
    """
    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values(["hosp_id_day_key", "event_time"]).reset_index(drop=True)

    # Initialize output columns
    df["SAT_EHR_delivery"] = np.nan
    df["SAT_modified_delivery"] = np.nan

    delta = pd.Timedelta(minutes=duration_min)

    # Work only on eligible rows
    eligible_mask = df["on_vent_and_sedation"] == 1
    eligible_df = df[eligible_mask].copy()

    if eligible_df.empty:
        return df

    # Group by ventilator-day
    for day_key, day_group in tqdm(
        eligible_df.groupby("hosp_id_day_key"),
        desc=f"Detecting SAT delivery (duration={duration_min} min)",
        leave=False,
    ):
        day_group = day_group.sort_values("event_time")
        times = day_group["event_time"].values
        orig_indices = day_group.index

        # Candidate rows: rank_sedation is not NaN (zero sedation dose)
        candidate_mask_ehr = day_group["rank_sedation"].notna()
        # Candidate rows for modified: rank_sedation_non_ops is not NaN
        candidate_mask_mod = day_group["rank_sedation_non_ops"].notna()

        # --- SAT_EHR_delivery ---
        for pos, (idx, row) in enumerate(day_group[candidate_mask_ehr].iterrows()):
            t0 = row["event_time"]
            t_end = t0 + delta

            # Slice the forward window
            window = day_group[
                (day_group["event_time"] >= t0) & (day_group["event_time"] < t_end)
            ]
            # Require at least 50% observation coverage of the requested duration
            if window.empty or window["event_time"].max() < t0 + delta * 0.5:
                continue

            # Verify all rows in window: IMV + ICU
            if not (
                (window["device_category_ffill"].fillna("").str.lower() == "imv").all()
                and (window["location_category_ffill"].fillna("").str.lower() == "icu").all()
            ):
                continue

            # Verify all rows: ALL sedatives + opioids are 0 or NaN
            if _all_zero_or_nan(window, _ALL_SEDATION_COLS).all():
                df.at[idx, "SAT_EHR_delivery"] = 1.0
                break  # First qualifying candidate only

        # --- SAT_modified_delivery ---
        for pos, (idx, row) in enumerate(day_group[candidate_mask_mod].iterrows()):
            t0 = row["event_time"]
            t_end = t0 + delta

            window = day_group[
                (day_group["event_time"] >= t0) & (day_group["event_time"] < t_end)
            ]
            # Require at least 50% observation coverage of the requested duration
            if window.empty or window["event_time"].max() < t0 + delta * 0.5:
                continue

            # Verify all rows in window: IMV + ICU
            if not (
                (window["device_category_ffill"].fillna("").str.lower() == "imv").all()
                and (window["location_category_ffill"].fillna("").str.lower() == "icu").all()
            ):
                continue

            # Verify all rows: only non-opioid sedatives are 0 or NaN
            if _all_zero_or_nan(window, _SEDATIVE_ONLY_COLS).all():
                df.at[idx, "SAT_modified_delivery"] = 1.0
                break  # First qualifying candidate only

    return df


