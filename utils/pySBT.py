import numpy as np
import pandas as pd
import re
import pyCLIF as pc
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
from definitions_source_of_truth import (
    SBT_MIN_CONTROLLED_MODE_HOURS,
    SBT_PRIMARY_DURATION_MIN,
    SBT_MODIFIED_DURATION_MIN,
    SBT_PS_MAX,
    SBT_CPAP_MAX,
    VENT_DAY_ANCHOR_HOUR,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import warnings
from typing import Optional

warnings.filterwarnings("ignore")
from tableone import TableOne

def process_cohort_conditions(cohort, How ):
    # --- Preliminary processing ---
    # Ensure event_time is datetime and sort the dataframe
    cohort['event_time'] = pd.to_datetime(cohort['event_time'])
    cohort = cohort.sort_values(['hospitalization_id', 'event_time']).reset_index(drop=True)
    
    # IMV flag
    if How == 'Standard':
        cohort['IMV_flag'] = (
            (cohort['device_category'] == 'imv') &
            (cohort['location_category'] == 'icu')
        )
        print('analysis by:',How)
    elif How == 'Respiratory_Stability':
        cohort['IMV_flag'] = (
            (cohort['device_category'] == 'imv') &
            (cohort['location_category'] == 'icu') &
            (cohort['Respiratory_Stability'] == 1)
        )
        print('analysis by:',How)
    elif How == 'Hemodynamic_Stability':
        cohort['IMV_flag'] = (
            (cohort['device_category'] == 'imv') &
            (cohort['location_category'] == 'icu') &
            (cohort['Hemodynamic_Stability_by_NEE'] == 1)
        )
        print('analysis by:',How)
    elif How == 'Both_stabilities':
        cohort['IMV_flag'] = (
            (cohort['device_category'] == 'imv') &
            (cohort['location_category'] == 'icu') &
            (cohort['Respiratory_Stability'] == 1) &
            (cohort['Hemodynamic_Stability_by_NEE'] == 1)
        )
        print('analysis by:',How)
    else:
        raise ValueError("Invalid `How` parameter: choose one of "
                        "['Standard', 'Respiratory_Stability', "
                        "'Hemodynamic_Stability', 'Both_stabilities'].")
    
    # --- Prepare new flag columns ---
    # For Condition 1, record the event_time when the threshold is reached.
    cohort['IMV_Controlled_met_time'] = pd.NaT
    # New flag for eligible day (1 if condition 1 is met that day, else 0)
    cohort['eligible_day'] = 0
    
    # For grouping by day, use the normalized event_time (midnight)
    cohort['current_day'] = (cohort['event_time'] - pd.Timedelta(hours=VENT_DAY_ANCHOR_HOUR)).dt.normalize()
    
    # Build a dictionary of full hospitalization data to avoid repeated filtering.
    hosp_groups = {
        hosp_id: df.copy().sort_values('event_time')
        for hosp_id, df in cohort.groupby('hospitalization_id')
    }
    
    # --- Define thresholds and time windows ---
    cond1_threshold = pd.Timedelta(hours=SBT_MIN_CONTROLLED_MODE_HOURS)  # Manuscript: 12 hours

    # For Condition 1: look back 24 hours from the start of the index vent-day (06:00)
    # to allow accumulating 12h of controlled mode within a feasible window.
    # NOTE: The SAT eligibility uses a narrower 10PM-6AM window for 4h sedation check.
    # SBT requires 12h controlled mode, so we must look back a full 24h.
    cond1_window_start_offset = pd.Timedelta(hours=VENT_DAY_ANCHOR_HOUR) - pd.Timedelta(days=1)  # prior day 06:00
    cond1_window_end_offset = pd.Timedelta(hours=VENT_DAY_ANCHOR_HOUR)  # current day 06:00
    
    # --- Process each hospitalization and day ---
    # --- vented days
    vented_day = cohort[(cohort['device_category'] == 'imv')]['hosp_id_day_key'].unique()

    # Group by hospitalization and current day
    groups = cohort[cohort['hosp_id_day_key'].isin(vented_day)].groupby(['hospitalization_id', 'current_day'])

    
    
    for (hosp_id, curr_day), day_group in tqdm(groups, desc="Evaluating SBT eligibility per hospital-day group"):

        cohort.loc[day_group.index, 'vent_day'] = 1
        # --- Condition 1: IMV in controlled mode ---
        # Define window for condition 1 based on the current day
        cond1_start = curr_day + cond1_window_start_offset
        cond1_end = curr_day + cond1_window_end_offset
        
        # Use full hospitalization data so events before midnight can contribute.
        hosp_df = hosp_groups[hosp_id]
        cond1_df = hosp_df[(hosp_df['event_time'] >= cond1_start) & (hosp_df['event_time'] <= cond1_end)].copy()
        

        if cond1_df.empty:
            continue  # no events in this window

        if cond1_df['max_paralytics'].max() > 0:
            continue

        cohort.loc[day_group.index, 'vent_day_without_paralytics'] = 1

        if not cond1_df['IMV_flag'].any():
            continue
    

        # Identify contiguous segments where IMV_flag is True.
        cond1_df['seg'] = (cond1_df['IMV_flag'] != cond1_df['IMV_flag'].shift()).cumsum()
        valid_segs = cond1_df[cond1_df['IMV_flag']].groupby('seg')
        
        cond1_met = False  # flag indicating if condition 1 was met
        for seg_id, seg_df in valid_segs:
            seg_df = seg_df.sort_values('event_time')
            seg_df['duration'] = seg_df['event_time'].diff().fillna(pd.Timedelta(seconds=0))
            seg_df['cum_duration'] = seg_df['duration'].cumsum()
            if seg_df['cum_duration'].iloc[-1] >= cond1_threshold:
                # Find the first row where the cumulative duration reaches the threshold.
                flag_row = seg_df[seg_df['cum_duration'] >= cond1_threshold].iloc[0]
                flag_idx = flag_row.name  # this is the original index in hosp_df (and cohort)
                flag_time = flag_row['event_time']
                cohort.loc[flag_idx, 'IMV_Controlled_met_time'] = flag_time
                cond1_met = True
                break  # Only the first qualifying segment for this day is flagged.
        
        # --- Eligible Day Flag ---
        # If condition 1 is met for the day, mark all rows of this day as eligible_day = 1.
        if cond1_met:
            cohort.loc[day_group.index, 'eligible_day'] = 1
    
    return cohort


def process_diagnostic_flip_sbt_optimized_v2(cohort):
    # Ensure event_time is datetime.
    cohort['event_time'] = pd.to_datetime(cohort['event_time'])
    
    # Preinitialize diagnostic and flip evaluation columns.
    diag_cols = ['cond_device_imv', 'cond_location_icu', 'cond_mode_ps_cpap',
                 'cond_ps_set_le8', 'cond_peep_set_le8', 'cond_mode_tpiece',
                 'flip_skip_reason', 'first_flip_time']
    for col in diag_cols:
        cohort[col] = None
        
    # Initialize EHR delivery columns.
    for mins in [2, 5, 30]:
        cohort[f"EHR_Delivery_{mins}mins"] = np.nan

    # --- Precompute diagnostic flags (vectorized) ---
    mask_eligible = cohort['eligible_day'] == 1
    
    # Normalize and compare strings.
    cond_imv = cohort['device_category'].fillna('').str.strip().str.lower() == 'imv'
    cond_icu = cohort['location_category'].fillna('').str.strip().str.lower() == 'icu'
    
    mode_cat_lower = cohort['mode_category'].fillna('').str.lower()
    cond_mode_ps = mode_cat_lower.str.contains('pressure support|cpap', regex=True)
    cond_ps_le8 = cohort['pressure_support_set'] <= 8
    cond_peep_le8 = cohort['peep_set'] <= 8
    conditionA = cond_mode_ps & cond_ps_le8 & cond_peep_le8
    mode_name_lower = cohort['mode_name'].fillna('').str.strip().str.lower()
    cond_mode_tpiece = mode_name_lower.str.match(r'^t[-]?piece$', na=False)
    composite = conditionA | cond_mode_tpiece
    passed = cond_imv & cond_icu & composite

    # Set diagnostic columns for eligible rows.
    cohort.loc[mask_eligible & (~cond_imv), 'cond_device_imv'] = \
        cohort.loc[mask_eligible & (~cond_imv), 'device_category']
    cohort.loc[mask_eligible & cond_imv & (~cond_icu), 'cond_location_icu'] = \
        cohort.loc[mask_eligible & cond_imv & (~cond_icu), 'location_category']
    
    mask_composite_fail = mask_eligible & cond_imv & cond_icu & (~composite)
    cohort.loc[mask_composite_fail & (~cond_mode_ps), 'cond_mode_ps_cpap'] = \
        cohort.loc[mask_composite_fail & (~cond_mode_ps), 'mode_category']
    mask_ps_fail = cohort['pressure_support_set'].isnull() | (cohort['pressure_support_set'] > 8)
    cohort.loc[mask_composite_fail & mask_ps_fail, 'cond_ps_set_le8'] = \
        cohort.loc[mask_composite_fail & mask_ps_fail, 'pressure_support_set']
    mask_peep_fail = cohort['peep_set'].isnull() | (cohort['peep_set'] > 8)
    cohort.loc[mask_composite_fail & mask_peep_fail, 'cond_peep_set_le8'] = \
        cohort.loc[mask_composite_fail & mask_peep_fail, 'peep_set']
    cohort.loc[mask_composite_fail & (~cond_mode_tpiece), 'cond_mode_tpiece'] = \
        cohort.loc[mask_composite_fail & (~cond_mode_tpiece), 'mode_name']
    
    # Mark candidate rows.
    cohort['flip_check_flag'] = False
    cohort.loc[mask_eligible, 'flip_check_flag'] = passed[mask_eligible]
    
    # Compute the minimum IMV_Controlled_met_time per eligible group.
    cohort.loc[mask_eligible, 'min_met_time'] = (
        cohort.loc[mask_eligible]
        .groupby(['hospitalization_id', 'current_day'])['IMV_Controlled_met_time']
        .transform('min')
    )
    
    # --- Process each eligible group using vectorized operations ---
    def process_group(group):
        # Work on a copy sorted by event_time.
        group = group.sort_values('event_time').copy()
        n = len(group)
        if n == 0:
            return group
        
        # Convert event_time to numpy array.
        times = group['event_time'].values.astype('datetime64[ns]')
        flip_int = group['flip_check_flag'].astype(int).values

        def compute_sustained(delta_minutes):
            delta = np.timedelta64(delta_minutes, 'm')
            boundaries = np.searchsorted(times, times + delta, side='right')
            cnt_total = boundaries - np.arange(n)
            cumsum = np.cumsum(flip_int)
            cnt_pass = np.empty(n, dtype=int)
            for i in range(n):
                start = i
                end = boundaries[i] - 1
                cnt_pass[i] = cumsum[end] - (cumsum[start-1] if start > 0 else 0)
            return (cnt_total == cnt_pass) & group['flip_check_flag'], cnt_total, cnt_pass

        # Compute sustained flags for 2 mins, 5 mins, and 30 mins
        group['sustained_2min'], group['cnt_total_2'], group['cnt_pass_2'] = compute_sustained(SBT_PRIMARY_DURATION_MIN)
        group['sustained_5min'], group['cnt_total_5'], group['cnt_pass_5'] = compute_sustained(SBT_MODIFIED_DURATION_MIN)
        group['sustained_30min'], group['cnt_total_30'], group['cnt_pass_30'] = compute_sustained(30)

        # Apply 2-min logic
        candidate_indices = group.index[group['flip_check_flag']].tolist()
        for idx in candidate_indices:
            group.at[idx, 'first_flip_time'] = group.at[idx, 'event_time']
            if group.at[idx, 'event_time'] < group.at[idx, 'min_met_time']:
                group.at[idx, 'flip_skip_reason'] = "Flip before IMV_Controlled_met_time"
                continue
            else:
                if group.at[idx, 'sustained_2min']:
                    group.at[idx, 'EHR_Delivery_2mins'] = 1
                    group.at[idx, 'flip_skip_reason'] = None
                    break
                else:
                    group.at[idx, 'flip_skip_reason'] = "ehr_delivery_2min not possible"
                    continue

        # Apply 5-min logic (modified SBT per manuscript)
        for idx in candidate_indices:
            if group.at[idx, 'event_time'] < group.at[idx, 'min_met_time']:
                continue
            if group.at[idx, 'sustained_5min']:
                group.at[idx, 'EHR_Delivery_5mins'] = 1
                break

        # Apply 30-min logic (independently)
        for idx in candidate_indices:
            if group.at[idx, 'event_time'] < group.at[idx, 'min_met_time']:
                continue
            if group.at[idx, 'sustained_30min']:
                group.at[idx, 'EHR_Delivery_30mins'] = 1
                break

        return group

    # Apply the per-group processing only on eligible rows.
    eligible_df = cohort[mask_eligible].copy()
    processed = eligible_df.groupby(['hospitalization_id', 'current_day'], group_keys=False).apply(process_group)
    
    # Update only the eligible rows in the original DataFrame.
    cohort.update(processed)
    
    # Remove helper columns.
    helper_cols = ['cnt_total_2', 'cnt_pass_2', 'sustained_2min',
                   'cnt_total_5', 'cnt_pass_5', 'sustained_5min',
                   'cnt_total_30', 'cnt_pass_30', 'sustained_30min', 'min_met_time']
    cohort.drop(columns=[col for col in helper_cols if col in cohort.columns], inplace=True)
    
    return cohort


def apply_2_45_extubated_flag(cohort):
    # Ensure time columns are datetime
    cohort['event_time'] = pd.to_datetime(cohort['event_time'])
    cohort['first_flip_time'] = pd.to_datetime(cohort['first_flip_time'])

    # Initialize flag column
    cohort['flag_2_45_extubated'] = np.nan

    # Loop over each group
    group_cols = ['hospitalization_id', 'current_day']
    for (hosp_id, day), group in cohort.groupby(group_cols):
        flip_row = group[(group['EHR_Delivery_2mins'] == 1) & (~group['first_flip_time'].isna())]
        if flip_row.empty:
            continue

        flip_time = flip_row.iloc[0]['first_flip_time']
        time_window_end = flip_time + pd.Timedelta(minutes=45)

        # Look for extubation within time window
        extubation_mask = (group['event_time'] > flip_time) & \
                          (group['event_time'] <= time_window_end) & \
                          (group['extubated'] == 1)

        if extubation_mask.any():
            cohort.loc[flip_row.index[0], 'flag_2_45_extubated'] = 1

    return cohort

def compute_time_to_extubation(cohort):
    # Ensure time columns are datetime
    cohort['event_time'] = pd.to_datetime(cohort['event_time'])
    cohort['first_flip_time'] = pd.to_datetime(cohort['first_flip_time'])

    # Initialize new column
    cohort['delta_to_extubation_mins'] = np.nan

    # Grouping by patient and day
    group_cols = ['hospitalization_id', 'current_day']
    for (hosp_id, day), group in cohort.groupby(group_cols):
        group = group.sort_values('event_time')

        flip_row = group[(group['EHR_Delivery_30mins'] == 1) & (~group['first_flip_time'].isna())]
        if flip_row.empty:
            continue

        flip_time = flip_row.iloc[0]['first_flip_time']
        flip_index = flip_row.index[0]

        # Find first extubation event *after* flip_time
        post_extubated = group[(group['event_time'] > flip_time) & (group['extubated'] == 1)]
        if not post_extubated.empty:
            extubation_time = post_extubated.iloc[0]['event_time']
            delta = (extubation_time - flip_time).total_seconds() / 60.0
            cohort.loc[flip_index, 'delta_to_extubation_mins'] = delta

    return cohort

def manual_tableone(df, continuous_cols):
    summary = []
    for col in continuous_cols:
        # coerce to numeric, count non‐missing, and count missing
        col_data = pd.to_numeric(df[col], errors='coerce')
        n = col_data.count()
        missing = col_data.isna().sum()
        
        # compute median and IQR
        median = col_data.median()
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        
        # format IQR and round numeric summaries to 2 decimal places
        iqr = f"[{q1:.2f}, {q3:.2f}]"
        median = round(median, 2) if pd.notnull(median) else median
        
        summary.append({
            "Variable": col,
            "Total" : n+missing,
            "Has Value": n,
            "Missing": missing,
            "Median": median,
            "IQR": iqr
        })
    
    summary_df = pd.DataFrame(summary, 
                              columns=["Variable", "Total", "Has Value", "Missing", "Median", "IQR"])
    
    return summary_df

def manual_categorical_tableone(df, categorical_cols):

    summary = []
    for col in categorical_cols:
        # get counts for each category (including NaNs)
        value_counts = df[col].value_counts(dropna=False)
        
        # total observations for this variable (should equal len(df) if no filtering)
        n = value_counts.sum()
        
        for category, count in value_counts.items():
            summary.append({
                "Variable": col,
                "Category": category,
                "N": n,
                "Count": count,
                # percent of non‐missing obs per category, sums to 100 per variable
                "Percent": round((count / n) * 100, 2)
            })
    summary_df = pd.DataFrame(summary, 
                              columns=["Variable", "Category", "N", "Count", "Percent"])
    return summary_df


def apply_outlier_thresholds(df, col_name, min_val, max_val):
    """
    Helper function to clamp column values between min and max thresholds, 
    setting values outside range to NaN.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the column to process
        col_name (str): Name of the column to apply thresholds to
        min_val (float): Minimum allowed value (inclusive)
        max_val (float): Maximum allowed value (inclusive)
        
    Returns:
        None: Modifies the DataFrame in place by updating the specified column
    """
    df[col_name] = df[col_name].where(df[col_name].between(min_val, 
                                                           max_val, 
                                                           inclusive='both'), 
                                                           np.nan)
    
print('Imported SBT Helper!')


