import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import sys
    sys.path.insert(0, os.path.join(os.pardir, 'utils'))
    import pandas as pd
    import numpy as np
    import re
    import os
    import matplotlib.pyplot as plt
    import duckdb
    import pyCLIF as pc
    import pySBT as t1code
    from tqdm import tqdm
    from datetime import datetime
    from tableone import TableOne, load_dataset
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    con = pc.load_config()
    return TableOne, confusion_matrix, np, os, pc, pd, plt, re, t1code, tqdm


@app.cell
def _(pd):
    # cohort = pd.read_csv("../output/intermediate/study_cohort.csv")
    cohort = pd.read_parquet('../output/intermediate/study_cohort.parquet')
    cohort['hospital_id'] = cohort['hospital_id'].str.replace(r'[^a-zA-Z]', '', regex=True)
    t1_cohort = cohort.copy()
    return cohort, t1_cohort


@app.cell
def _(os, pc):
    # Construct the full directory path
    directory_path = os.path.join("../output/final/", pc.helper["site_name"], "SAT_standard")

    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    return (directory_path,)


@app.cell
def _(cohort, np):
    # Convert pass_fail columns to binary indicators:
    # 1 = pass/positive (flowsheet documented delivery), 0 = fail/negative, NaN = not documented
    # NOTE: The original {0:1, 1:1} mapping was incorrect — it treated "fail" as "delivery"
    # which inflated the reference positive rate. Correct mapping: 1 → 1 (pass), 0 → 0 (fail).
    cohort['sat_delivery_pass_fail_documented'] = cohort['sat_delivery_pass_fail'].notna().astype(float)
    cohort['sat_delivery_pass_fail_documented'] = cohort['sat_delivery_pass_fail_documented'].replace(0, np.nan)
    # Keep the actual pass/fail values for concordance (1=pass, 0=fail, NaN=not documented)
    cohort['sat_screen_pass_fail_documented'] = cohort['sat_screen_pass_fail'].notna().astype(float)
    cohort['sat_screen_pass_fail_documented'] = cohort['sat_screen_pass_fail_documented'].replace(0, np.nan)
    return


@app.cell
def _(cohort, np, pd):
    from definitions_source_of_truth import MAX_FFILL_OBSERVATIONS
    cohort['event_time'] = pd.to_datetime(cohort['event_time'])
    cohort['admission_dttm'] = pd.to_datetime(cohort['admission_dttm'], format='mixed')
    cohort['discharge_dttm'] = pd.to_datetime(cohort['discharge_dttm'], format='mixed')
    cohort_1 = cohort.sort_values(by=['hospitalization_id', 'event_time']).reset_index(drop=True)
    cohort_1['device_category_ffill'] = cohort_1.groupby('hospitalization_id')['device_category'].ffill()
    cohort_1['location_category_ffill'] = cohort_1.groupby('hospitalization_id')['location_category'].ffill()
    active_sedation_n_col = ['fentanyl', 'propofol', 'lorazepam', 'midazolam', 'hydromorphone', 'morphine']
    for _col in active_sedation_n_col:
        if _col not in cohort_1.columns:
            cohort_1[_col] = np.nan
            print(f"Column '{_col}' is missing. Please check your CLIF Meds table — it might be missing, or it's okay if your site doesn't use it.")
    cohort_1[['fentanyl', 'propofol', 'lorazepam', 'midazolam', 'hydromorphone', 'morphine']] = cohort_1.groupby('hospitalization_id')[['fentanyl', 'propofol', 'lorazepam', 'midazolam', 'hydromorphone', 'morphine']].ffill(limit=MAX_FFILL_OBSERVATIONS)
    cohort_1['min_sedation_dose'] = cohort_1[['fentanyl', 'propofol', 'lorazepam', 'midazolam', 'hydromorphone', 'morphine']].min(axis=1, skipna=True)
    cohort_1['min_sedation_dose_2'] = cohort_1[['fentanyl', 'propofol', 'lorazepam', 'midazolam', 'hydromorphone', 'morphine']].where(cohort_1[['fentanyl', 'propofol', 'lorazepam', 'midazolam', 'hydromorphone', 'morphine']] > 0).min(axis=1, skipna=True)
    cohort_1['min_sedation_dose_non_ops'] = cohort_1[['propofol', 'lorazepam', 'midazolam']].min(axis=1, skipna=True)
    cohort_1['min_sedation_dose_non_ops'] = cohort_1['min_sedation_dose_non_ops'].fillna(0)
    cohort_1[['cisatracurium', 'vecuronium', 'rocuronium']] = cohort_1.groupby('hospitalization_id')[['cisatracurium', 'vecuronium', 'rocuronium']].ffill(limit=MAX_FFILL_OBSERVATIONS)
    cohort_1['max_paralytics'] = cohort_1[['cisatracurium', 'vecuronium', 'rocuronium']].max(axis=1, skipna=True).fillna(0)
    cohort_1[['rass']] = cohort_1.groupby('hospitalization_id')[['rass']].ffill(limit=MAX_FFILL_OBSERVATIONS)
    cohort_1 = cohort_1.sort_values(by=['hospitalization_id', 'event_time']).reset_index(drop=True)
    return (cohort_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Identify eligible days
    """)
    return


@app.cell
def _(cohort_1, pd, tqdm):
    def process_cohort(df):
        df = df.sort_values(by=['hospitalization_id', 'event_time']).reset_index(drop=True)  # Preprocessing
        df['device_category_ffill'] = df.groupby('hospitalization_id')['device_category'].ffill()
        df['location_category_ffill'] = df.groupby('hospitalization_id')['location_category'].ffill()
        df['event_time'] = pd.to_datetime(df['event_time'])
        df['all_conditions_check'] = ((df['device_category_ffill'].str.lower() == 'imv') & (df['min_sedation_dose_2'] > 0) & (df['location_category_ffill'].str.lower() == 'icu') & (df['max_paralytics'] <= 0)).astype(int)
        result = []
        vented_days = df[df['device_category'] == 'imv']['hosp_id_day_key'].unique()  # Precompute condition flags
        df = df[df['hosp_id_day_key'].isin(vented_days)]
        _shifted = df['event_time'] - pd.Timedelta(hours=6)
        hosp_grouped = df.groupby(['hospitalization_id', _shifted.dt.normalize()])
        window_start_offset = pd.Timedelta(hours=22) - pd.Timedelta(days=1)
        window_end_offset = pd.Timedelta(hours=6)
        for (hosp_id, date), group in tqdm(hosp_grouped, desc='Evaluating hospitalizations by day for SAT eligibility'):
            temp_df = group.sort_values('event_time')
            start_time = date + window_start_offset
            end_time = date + window_end_offset
            mask = (temp_df['event_time'] >= start_time) & (temp_df['event_time'] <= end_time)
            window_df = temp_df.loc[mask].copy()
            if window_df.empty or not window_df['all_conditions_check'].any():  # Fix 3: Group by 06:00-anchored day (shift back 6h before normalizing)
                continue
            window_df['condition_met_group'] = (window_df['all_conditions_check'] != window_df['all_conditions_check'].shift()).cumsum()
            valid_segments = window_df[window_df['all_conditions_check'] == 1].groupby('condition_met_group')
            for _, segment in valid_segments:  # Time window offsets
                segment = segment.sort_values('event_time')
                segment['duration'] = segment['event_time'].diff().fillna(pd.Timedelta(seconds=0))
                segment['cumulative_duration'] = segment['duration'].cumsum()
                if segment['cumulative_duration'].iloc[-1] >= pd.Timedelta(hours=4):
                    event_time_at_4_hours = segment[segment['cumulative_duration'] >= pd.Timedelta(hours=4)].iloc[0]['event_time']
                    result.append({'hospitalization_id': hosp_id, 'current_day_key': date, 'event_time_at_4_hours': event_time_at_4_hours})
                    break
        return pd.DataFrame(result)
    result_df = process_cohort(cohort_1)
    print('Encounter days with at least 4 hours of conditions met from 10 PM to 6 AM:', len(result_df))  # Get the event_time when duration crosses 4 hours  # Move to next hospitalization-day
    return (result_df,)


@app.cell
def _(cohort_1, np, pd, result_df):
    # Ensure proper datetime
    cohort_1['event_time'] = pd.to_datetime(cohort_1['event_time'])
    cohort_1['event_date'] = (cohort_1['event_time'] - pd.Timedelta(hours=6)).dt.normalize()
    cohort_2 = cohort_1.merge(result_df[['hospitalization_id', 'current_day_key', 'event_time_at_4_hours']], how='left', left_on=['hospitalization_id', 'event_date'], right_on=['hospitalization_id', 'current_day_key'])
    # Merge SAT result
    mask_valid = cohort_2['event_time_at_4_hours'].notna()
    mask_after_time = cohort_2['event_time'] >= cohort_2['event_time_at_4_hours']
    eligible_rows = cohort_2[mask_valid & mask_after_time].copy()
    first_eligible_idx = eligible_rows.groupby(['hospitalization_id', 'event_date'])['event_time'].idxmin()
    cohort_2['eligible_event'] = np.nan
    cohort_2.loc[first_eligible_idx, 'eligible_event'] = 1
    last_idxs = cohort_2.groupby('hospitalization_id')['event_time'].idxmax()
    # Only keep rows where a SAT event time is present and event_time >= that time
    cohort_2.loc[last_idxs, 'eligible_event'] = np.nan
    eligible_days = cohort_2.loc[cohort_2['eligible_event'] == 1, 'hosp_id_day_key'].unique()
    cohort_2['on_vent_and_sedation'] = cohort_2['hosp_id_day_key'].isin(eligible_days).astype(int)
    cohort_2.drop(columns=['event_date', 'current_day_key', 'event_time_at_4_hours'], inplace=True)
    # Find first event per hospitalization where conditions are met
    # Initialize eligible_event column
    # Fix: Remove flag on last event per hospitalization
    # Mark all rows of eligible days
    # Cleanup
    del result_df
    return (cohort_2,)


@app.cell
def _(cohort_2):
    merged_cohort = cohort_2.copy()
    merged_cohort['eligible_event'].value_counts()
    return (merged_cohort,)


@app.cell
def _(merged_cohort):
    merged_cohort[merged_cohort['on_vent_and_sedation']==1]['hosp_id_day_key'].nunique()
    return


@app.cell
def _(merged_cohort, np, tqdm):
    df = merged_cohort[merged_cohort['on_vent_and_sedation'] == 1].sort_values(by=['hospitalization_id', 'event_time']).reset_index(drop=True)
    df['rank_sedation'] = np.nan
    for hosp_id_day_key, hosp_data in tqdm(df[df['on_vent_and_sedation'] == 1].groupby('hosp_id_day_key'), desc='Detecting weaning or tapering behavior for sedation meds'):
        zero_mask = hosp_data['min_sedation_dose'] == 0
        _ranks = zero_mask.cumsum() * zero_mask
        df.loc[hosp_data.index, 'rank_sedation'] = _ranks.replace(0, np.nan)
    df['rank_sedation_non_ops'] = np.nan
    for hosp_id_day_key, hosp_data in tqdm(df[df['on_vent_and_sedation'] == 1].groupby('hosp_id_day_key'), desc='Detecting weaning or tapering behavior for non opioids sedation meds'):
        zero_mask = hosp_data['min_sedation_dose_non_ops'] == 0
        _ranks = zero_mask.cumsum() * zero_mask
        df.loc[hosp_data.index, 'rank_sedation_non_ops'] = _ranks.replace(0, np.nan)
    return (df,)


@app.cell
def _(df):
    #quick rass check
    df['rass'] = df['rass'].astype(float)
    df['rass'].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### SAT EHR Flags
    """)
    return


@app.cell
def _(df, np, pd, tqdm):
    # Setup
    med_columns = ['fentanyl', 'propofol', 'lorazepam', 'midazolam', 'hydromorphone', 'morphine']
    med_columns2 = ['propofol', 'lorazepam', 'midazolam']
    flags = ['SAT_EHR_delivery', 'SAT_modified_delivery', 'SAT_rass_nonneg_30', 'SAT_med_halved_rass_pos', 'SAT_no_meds_rass_pos_45', 'SAT_rass_first_neg_30_last45_nonneg']
    for f in flags:
        df[f] = np.nan
    vent_df = df[df['on_vent_and_sedation'] == 1]
    delta30 = pd.Timedelta(minutes=30)
    delta45 = pd.Timedelta(minutes=45)  # new: no meds & rass >=0 in 45 min
    for key, group in tqdm(vent_df.groupby('hosp_id_day_key'), desc='Evaluating SAT using RASS, meds, and ventilation criteria. (6 Ways)'):
        grp = group.sort_values('event_time')
        times = grp['event_time'].values
        _ranks = grp['rank_sedation'].values
        idxs = grp.index
        for idx, current_time, rank in zip(idxs, times, _ranks):
            if pd.isna(rank):
                continue
            fw30 = grp[(grp['event_time'] >= current_time) & (grp['event_time'] <= current_time + delta30)]
    # Main loop
            fw45 = grp[(grp['event_time'] >= current_time) & (grp['event_time'] <= current_time + delta45)]
            pr30 = grp[(grp['event_time'] >= current_time - delta30) & (grp['event_time'] < current_time)]
            imv_ok = (fw30['device_category_ffill'] == 'imv').all()
            icu_ok = (fw30['location_category_ffill'] == 'icu').all()
            flags_set = {}
            if imv_ok and icu_ok:
                meds_ok = (fw30[med_columns].isna() | (fw30[med_columns] == 0)).all().all()
                if meds_ok:
                    flags_set['SAT_EHR_delivery'] = 1
                meds2_ok = (fw30[med_columns2].isna() | (fw30[med_columns2] == 0)).all().all()
                if meds2_ok:  # Define windows
                    flags_set['SAT_modified_delivery'] = 1
                rass30 = fw30['rass'].dropna()
                rass45 = fw45['rass'].dropna()
                rass45_ok = not rass45.empty and rass45.iloc[-1] >= 0
                rass30_pre = pr30['rass'].dropna()  # ICU+IMV check
                if not rass30.empty and (rass30 >= 0).all():
                    flags_set['SAT_rass_nonneg_30'] = 1
                if not pr30.empty and (not fw30.empty):
                    half_max = pr30[med_columns].max() * 0.5
                    halved_ok = True
                    for med in med_columns:  # No meds at all in next 30 min
                        vals = fw30[med].dropna()
                        vals = vals[vals != 0]
                        if not vals.empty and (not (vals <= half_max[med]).all()):
                            halved_ok = False
                            break  # No subset meds
                    if halved_ok and rass45_ok:
                        flags_set['SAT_med_halved_rass_pos'] = 1
                if meds_ok:
                    if rass45_ok:
                        flags_set['SAT_no_meds_rass_pos_45'] = 1
                if not rass30_pre.empty and (not rass45.empty):
                    first_rass30 = rass30_pre.iloc[0]
                    last_rass45 = rass45.iloc[-1]
                    if first_rass30 < 0 and last_rass45 >= 0:
                        flags_set['SAT_rass_first_neg_30_last45_nonneg'] = 1  # RASS >= 0 in next 30 min
                for f, val in flags_set.items():
                    df.at[idx, f] = val  # New: meds halved & last RASS45 >= 0  # Compute 50% thresholds  # Only check non-NaN, non-zero meds in forward window  # ignore zero doses  # New: no meds & last RASS45 >= 0  # --- new flag: first RASS <0 in next 30 & last RASS ≥0 in next 45 ---
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PAtch for non RASS sites
    """)
    return


@app.cell
def _(cohort_2, df):
    if cohort_2['rass'].nunique() <= 1:
        for _x in ['SAT_rass_nonneg_30', 'SAT_med_halved_rass_pos', 'SAT_no_meds_rass_pos_45', 'SAT_rass_first_neg_30_last45_nonneg']:
            df[_x] = 0
            print('Your SITE CLIF dont have RASS!! making all RASS Flags to Zero')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Delta plot
    """)
    return


@app.cell
def _(df, directory_path, pd):
    # 1. Identify initial sat failure events and delivery events
    mask_initial = (df['sat_delivery_pass_fail'] == 1) | (df['sat_screen_pass_fail'] == 1)
    mask_ehr = df['SAT_EHR_delivery'] == 1
    mask_mod = df['SAT_modified_delivery'] == 1
    initial_times = df[mask_initial].groupby('hosp_id_day_key')['event_time'].min().rename('initial_time')
    ehr_times = df[mask_ehr].groupby('hosp_id_day_key')['event_time'].min().rename('ehr_time')
    mod_times = df[mask_mod].groupby('hosp_id_day_key')['event_time'].min().rename('mod_time')
    times_df = pd.concat([initial_times, ehr_times, mod_times], axis=1).dropna()
    for _col in ['initial_time', 'ehr_time', 'mod_time']:
    # 2. Merge into a single DataFrame and drop incomplete cases
        times_df[_col] = pd.to_datetime(times_df[_col])
    times_df['delta_to_ehr'] = (times_df['ehr_time'] - times_df['initial_time']).dt.total_seconds() / 60
    # 3. Convert event_time columns to datetime
    times_df['delta_to_mod'] = (times_df['mod_time'] - times_df['initial_time']).dt.total_seconds() / 60
    times_df = times_df[(times_df['delta_to_ehr'] >= 0) & (times_df['delta_to_ehr'] <= 1440) & (times_df['delta_to_mod'] >= 0) & (times_df['delta_to_mod'] <= 1440)]
    bins = list(range(0, 24 * 60 + 1, 60))
    # 4. Compute deltas in minutes
    _labels = [f'{i}-{i + 1}hr' for i in range(24)]
    ehr_binned = pd.cut(times_df['delta_to_ehr'], bins=bins, labels=_labels, right=False)
    mod_binned = pd.cut(times_df['delta_to_mod'], bins=bins, labels=_labels, right=False)
    # 5. Filter deltas to positive values within 24 hours (0–1440 minutes)
    _ehr_counts = ehr_binned.value_counts().sort_index()
    mod_counts = mod_binned.value_counts().sort_index()
    binned_df = pd.DataFrame({'hour_bin': _labels, 'count_to_SAT_EHR_delivery': _ehr_counts.values, 'count_to_SAT_modified_delivery': mod_counts.values})
    # 6. Bin into hourly intervals
    # # 7. Plot both lines hour-wise
    # plt.figure(figsize=(12, 6))
    # plt.plot(labels, ehr_counts.values, marker='o', label='To SAT_EHR_delivery')
    # plt.plot(labels, mod_counts.values, marker='s', label='To SAT_modified_delivery')
    # plt.xlabel('Hours since initial failure event')
    # plt.ylabel('Count of Hospital-Day Keys')
    # plt.title('Hourly Distribution of Time to EHR and Modified Deliveries')
    # plt.xticks(rotation=45)
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    binned_df.to_csv(f'{directory_path}/binned_delta_counts.csv', index=False)  # [0, 60, 120, ..., 1440]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Icu los calculation
    """)
    return


@app.cell
def _(cohort_2):
    icu_los = cohort_2[['hospitalization_id', 'event_time', 'location_category_ffill']]
    icu_los = icu_los.sort_values(by=['hospitalization_id', 'event_time']).reset_index(drop=True)
    icu_los['segment'] = (icu_los['location_category_ffill'] != icu_los['location_category_ffill'].shift()).cumsum()
    icu_segments = icu_los[icu_los['location_category_ffill'].str.lower() == 'icu'].groupby(['hospitalization_id', 'segment']).agg(location_start=('event_time', 'first'), location_end=('event_time', 'last')).reset_index()
    icu_segments['los_days'] = (icu_segments['location_end'] - icu_segments['location_start']).dt.total_seconds() / (24 * 3600)
    icu_los_per_encounter = icu_segments[['hospitalization_id', 'los_days']]
    total_icu_los_per_hosp = icu_los_per_encounter.groupby('hospitalization_id', as_index=False).agg(ICU_LOS=('los_days', 'sum'))
    total_icu_los_per_hosp.shape
    return (total_icu_los_per_hosp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### last dishcharge hosptial_id
    """)
    return


@app.cell
def _(cohort_2):
    last_hosp = cohort_2[['hospitalization_id', 'event_time', 'hospital_id']]
    last_hosp = last_hosp.sort_values(by=['hospitalization_id', 'event_time'], ascending=False).groupby(['hospitalization_id'], as_index=False).agg({'hospital_id': 'first'}).reset_index(drop=True)
    last_hosp.shape
    return (last_hosp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Table one df
    """)
    return


@app.cell
def _(df):
    main = df[['patient_id', 'hospitalization_id', 'admission_dttm', 'discharge_dttm',
           'age_at_admission', 'discharge_category', 'sex_category',
           'race_category', 'ethnicity_category','hosp_id_day_key']].drop_duplicates()
    main.shape
    return (main,)


@app.cell
def _(last_hosp, main, pd, total_icu_los_per_hosp):
    main_1 = pd.merge(main, total_icu_los_per_hosp, on='hospitalization_id', how='left')
    main_1 = pd.merge(main_1, last_hosp, on='hospitalization_id', how='left')
    main_1.shape
    return (main_1,)


@app.cell
def _(df, main_1, np):
    group_cols = ['hosp_id_day_key']
    max_cols = ['sat_screen_pass_fail', 'sat_delivery_pass_fail', 'SAT_EHR_delivery', 'SAT_modified_delivery', 'eligible_event', 'SAT_EHR_delivery', 'SAT_modified_delivery', 'SAT_rass_nonneg_30', 'SAT_med_halved_rass_pos', 'SAT_no_meds_rass_pos_45', 'SAT_rass_first_neg_30_last45_nonneg']
    agg_dict = {_col: 'max' for _col in max_cols}
    df_grouped = df.groupby(group_cols).agg(agg_dict).reset_index()
    df_grouped = df_grouped.sort_values('hosp_id_day_key').reset_index(drop=True)
    df_grouped['sat_flowsheet_delivery_flag'] = np.where(((df_grouped['sat_screen_pass_fail'] == 1) | (df_grouped['sat_delivery_pass_fail'] == 1)) & (df_grouped['eligible_event'] == 1), 1, np.nan)
    final_df = main_1.merge(df_grouped, on='hosp_id_day_key', how='inner')
    final_df.shape
    return (final_df,)


@app.cell
def _(final_df):
    for _x in ['sat_delivery_pass_fail', 'sat_screen_pass_fail', 'SAT_EHR_delivery', 'SAT_modified_delivery', 'eligible_event', 'sat_flowsheet_delivery_flag', 'SAT_rass_first_neg_30_last45_nonneg', 'SAT_rass_nonneg_30', 'SAT_med_halved_rass_pos', 'SAT_no_meds_rass_pos_45']:
        print(final_df[_x].value_counts())
        print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### concordance
    """)
    return


@app.cell
def _(cohort_2, confusion_matrix, directory_path, final_df, np, os, pd, plt):
    from sklearn.metrics import cohen_kappa_score

    def _landis_koch(kappa):
        """Landis & Koch (1977) interpretation of kappa values."""
        if kappa < 0:
            return 'Poor'
        elif kappa < 0.21:
            return 'Slight'
        elif kappa < 0.41:
            return 'Fair'
        elif kappa < 0.61:
            return 'Moderate'
        elif kappa < 0.81:
            return 'Substantial'
        else:
            return 'Almost Perfect'

    def _kappa_ci_bootstrap(y_true, y_pred, n_boot=2000, ci=0.95, seed=42):
        """Bootstrap 95% CI for Cohen's kappa."""
        rng = np.random.default_rng(seed)
        n = len(y_true)
        kappas = []
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            try:
                kappas.append(cohen_kappa_score(y_true.iloc[idx], y_pred.iloc[idx]))
            except Exception:
                continue
        alpha = (1 - ci) / 2
        return (np.percentile(kappas, alpha * 100), np.percentile(kappas, (1 - alpha) * 100))
    con_df = final_df.copy()
    for _col in ['SAT_EHR_delivery', 'SAT_modified_delivery', 'sat_flowsheet_delivery_flag', 'SAT_rass_nonneg_30', 'SAT_med_halved_rass_pos', 'SAT_no_meds_rass_pos_45', 'SAT_rass_first_neg_30_last45_nonneg']:
        con_df[_col] = con_df[_col].fillna(0)
    metrics_list = []
    for _col in ['SAT_EHR_delivery', 'SAT_modified_delivery', 'SAT_rass_nonneg_30', 'SAT_med_halved_rass_pos', 'SAT_no_meds_rass_pos_45', 'SAT_rass_first_neg_30_last45_nonneg']:
        y_true = con_df['sat_flowsheet_delivery_flag']
        y_pred = con_df[_col]
        if 'rass' in _col and cohort_2['rass'].nunique() <= 1:
            continue
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        total = cm.sum()
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        specificity = tn / (tn + fp) if tn + fp else 0
        kappa = cohen_kappa_score(y_true, y_pred)
        kappa_lo, kappa_hi = _kappa_ci_bootstrap(y_true, y_pred)
        kappa_interp = _landis_koch(kappa)
        cm_pct = cm / total * 100
        _labels = ['No Delivery', 'Delivery']
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(_labels)
        ax.set_yticklabels(_labels)
        ax.set_xlabel(f'{_col} Flag')
        ax.set_ylabel('Flowsheet Delivery Flag')
        ax.set_title(f'Concordance: flowsheet vs {_col}')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = cm_pct[i, j]
                ax.text(j, i, f'{count}\n({pct:.1f}%)', ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plot_path = os.path.join(directory_path, f'confusion_matrix_{_col}.png')
        fig.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        print(f'--- Concordance for {_col} ---')
        print(f'Accuracy    : {accuracy:.3f}')
        print(f'Precision   : {precision:.3f}')
        print(f'Recall      : {recall:.3f}')
        print(f'F1 Score    : {f1:.3f}')
        print(f'Specificity : {specificity:.3f}')
        print(f'Cohen kappa : {kappa:.3f} (95% CI: {kappa_lo:.3f}-{kappa_hi:.3f}), {kappa_interp}\n')
        metrics_list.append({'Column': _col, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'Specificity': specificity, 'Cohen_Kappa': kappa, 'Kappa_CI_lower': round(kappa_lo, 3), 'Kappa_CI_upper': round(kappa_hi, 3), 'Kappa_Interpretation': kappa_interp})
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(os.path.join(directory_path, 'delivery_concordance_summary.csv'), index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(final_df):
    final_df.to_csv('../output/intermediate/final_df_SAT.csv', index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### table one print
    """)
    return


@app.cell
def _(final_df, pd):
    categorical_columns = ['sex_category', 'race_category', 'ethnicity_category','discharge_category']
    non_categorical_columns = ['age_at_admission',  'ICU_LOS', 'Inpatient_LOS']

    final_df['admission_dttm'] = pd.to_datetime(final_df['admission_dttm'], format='mixed')
    final_df['discharge_dttm'] = pd.to_datetime(final_df['discharge_dttm'], format='mixed')
    return categorical_columns, non_categorical_columns


@app.cell
def _(
    TableOne,
    categorical_columns,
    directory_path,
    final_df,
    non_categorical_columns,
    pc,
):
    ### SAT FLAG Table 1


    sat_flow_t1 = final_df[final_df['sat_flowsheet_delivery_flag'] == 1][[ 'hospitalization_id', 'admission_dttm', 'discharge_dttm', 'age_at_admission', 'discharge_category', 'sex_category','race_category', 'ethnicity_category','ICU_LOS']].drop_duplicates()
    sat_flow_t1['Inpatient_LOS'] = (sat_flow_t1['discharge_dttm'] - sat_flow_t1['admission_dttm']).dt.total_seconds() / (24 * 3600)

    if len(sat_flow_t1)>1:
        table1 = TableOne(sat_flow_t1, categorical=categorical_columns, nonnormal=non_categorical_columns, columns=categorical_columns+non_categorical_columns )

        table1.to_csv(f'{directory_path}/table1_sat_flowhseet_{pc.helper["site_name"]}.csv')
        print(table1)
    return


@app.cell
def _(
    TableOne,
    categorical_columns,
    directory_path,
    final_df,
    non_categorical_columns,
    pc,
):
    ### SAT EHR FLAG Table 1

    sat_ehr_t1 = final_df[(final_df['SAT_EHR_delivery'] == 1) | (final_df['SAT_modified_delivery'] == 1)][[ 'hospitalization_id', 'admission_dttm', 'discharge_dttm', 'age_at_admission', 'discharge_category', 'sex_category','race_category', 'ethnicity_category','ICU_LOS']].drop_duplicates()
    sat_ehr_t1['Inpatient_LOS'] = (sat_ehr_t1['discharge_dttm'] - sat_ehr_t1['admission_dttm']).dt.total_seconds() / (24 * 3600)

    if len(sat_ehr_t1)>1:
        table2 = TableOne(sat_ehr_t1, categorical=categorical_columns, nonnormal=non_categorical_columns, columns=categorical_columns+non_categorical_columns )

        table2.to_csv(f'{directory_path}/table1_sat_ehr_{pc.helper["site_name"]}.csv')
        print(table2)
    return


@app.cell
def _(
    TableOne,
    categorical_columns,
    directory_path,
    final_df,
    non_categorical_columns,
    pc,
):
    ### all Table 1

    all_t1 = final_df[[ 'hospitalization_id', 'admission_dttm', 'discharge_dttm', 'age_at_admission', 'discharge_category', 'sex_category','race_category', 'ethnicity_category','ICU_LOS']].drop_duplicates()
    all_t1['Inpatient_LOS'] = (all_t1['discharge_dttm'] - all_t1['admission_dttm']).dt.total_seconds() / (24 * 3600)

    if len(all_t1)>1:
        table3 = TableOne(all_t1, categorical=categorical_columns, nonnormal=non_categorical_columns, columns=categorical_columns+non_categorical_columns )

        table3.to_csv(f'{directory_path}/table1_all_t1_{pc.helper["site_name"]}.csv')
        print(table3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Per hospital stats
    """)
    return


@app.cell
def _(directory_path, final_df, pc, pd):
    # Initialize an empty list to store each hospital's data
    data_list = []
    for _x in final_df['hospital_id'].astype(str).unique():
    # Iterate over unique hospital IDs as strings
        eligible_event_count = final_df[(final_df['eligible_event'] == 1) & (final_df['hospital_id'].astype(str) == _x)].shape[0]
        sat_flowsheet_delivery_flag_count = final_df[(final_df['sat_flowsheet_delivery_flag'] == 1) & (final_df['hospital_id'].astype(str) == _x)].shape[0]  # Count of eligible events for this site
        SAT_modified_delivery_count = final_df[(final_df['SAT_modified_delivery'] == 1) & (final_df['hospital_id'].astype(str) == _x)].shape[0]
        SAT_EHR_delivery_count = final_df[(final_df['SAT_EHR_delivery'] == 1) & (final_df['hospital_id'].astype(str) == _x)].shape[0]
        SAT_rass_nonneg_30_count = final_df[(final_df['SAT_rass_nonneg_30'] == 1) & (final_df['hospital_id'].astype(str) == _x)].shape[0]
        SAT_med_halved_rass_pos_count = final_df[(final_df['SAT_med_halved_rass_pos'] == 1) & (final_df['hospital_id'].astype(str) == _x)].shape[0]
        SAT_no_meds_rass_pos_45_count = final_df[(final_df['SAT_no_meds_rass_pos_45'] == 1) & (final_df['hospital_id'].astype(str) == _x)].shape[0]
        SAT_rass_first_neg_30_last45_nonneg_count = final_df[(final_df['SAT_rass_first_neg_30_last45_nonneg'] == 1) & (final_df['hospital_id'].astype(str) == _x)].shape[0]  # Existing SAT flags counts
        SAT_EHR_uni_pats = final_df[(final_df['SAT_EHR_delivery'] == 1) & (final_df['hospital_id'].astype(str) == _x)]['patient_id'].nunique()
        SAT_EHR_hosp = final_df[(final_df['SAT_EHR_delivery'] == 1) & (final_df['hospital_id'].astype(str) == _x)]['hospitalization_id'].nunique()
        SAT_modified_uni_pats = final_df[(final_df['SAT_modified_delivery'] == 1) & (final_df['hospital_id'].astype(str) == _x)]['patient_id'].nunique()
        SAT_modified_hosp = final_df[(final_df['SAT_modified_delivery'] == 1) & (final_df['hospital_id'].astype(str) == _x)]['hospitalization_id'].nunique()
        SAT_rass_nonneg_30_uni_pats = final_df[(final_df['SAT_rass_nonneg_30'] == 1) & (final_df['hospital_id'].astype(str) == _x)]['patient_id'].nunique()
        SAT_rass_nonneg_30_hosp = final_df[(final_df['SAT_rass_nonneg_30'] == 1) & (final_df['hospital_id'].astype(str) == _x)]['hospitalization_id'].nunique()
        SAT_med_halved_rass_pos_uni_pats = final_df[(final_df['SAT_med_halved_rass_pos'] == 1) & (final_df['hospital_id'].astype(str) == _x)]['patient_id'].nunique()
        SAT_med_halved_rass_pos_hosp = final_df[(final_df['SAT_med_halved_rass_pos'] == 1) & (final_df['hospital_id'].astype(str) == _x)]['hospitalization_id'].nunique()
        SAT_no_meds_rass_pos_45_uni_pats = final_df[(final_df['SAT_no_meds_rass_pos_45'] == 1) & (final_df['hospital_id'].astype(str) == _x)]['patient_id'].nunique()
        SAT_no_meds_rass_pos_45_hosp = final_df[(final_df['SAT_no_meds_rass_pos_45'] == 1) & (final_df['hospital_id'].astype(str) == _x)]['hospitalization_id'].nunique()
        SAT_rass_first_neg_30_last45_nonneg_uni_pats = final_df[(final_df['SAT_rass_first_neg_30_last45_nonneg'] == 1) & (final_df['hospital_id'].astype(str) == _x)]['patient_id'].nunique()
        SAT_rass_first_neg_30_last45_nonneg_hosp = final_df[(final_df['SAT_rass_first_neg_30_last45_nonneg'] == 1) & (final_df['hospital_id'].astype(str) == _x)]['hospitalization_id'].nunique()
        if eligible_event_count > 0:
            percent_sat_flowsheet_delivery_flag = sat_flowsheet_delivery_flag_count / eligible_event_count * 100  # New SAT flags counts
            percent_SAT_modified_delivery = SAT_modified_delivery_count / eligible_event_count * 100
            percent_SAT_EHR_delivery = SAT_EHR_delivery_count / eligible_event_count * 100
            percent_SAT_rass_nonneg_30 = SAT_rass_nonneg_30_count / eligible_event_count * 100
            percent_SAT_med_halved_rass_pos = SAT_med_halved_rass_pos_count / eligible_event_count * 100
            percent_SAT_no_meds_rass_pos_45 = SAT_no_meds_rass_pos_45_count / eligible_event_count * 100
            percent_SAT_rass_first_neg_30_last45_nonneg = SAT_rass_first_neg_30_last45_nonneg_count / eligible_event_count * 100
        else:
            percent_sat_flowsheet_delivery_flag = percent_SAT_modified_delivery = 0
            percent_SAT_EHR_delivery = percent_SAT_rass_nonneg_30 = 0
            percent_SAT_med_halved_rass_pos = percent_SAT_no_meds_rass_pos_45 = 0
            percent_SAT_rass_first_neg_30_last45_nonneg = 0
        data_list.append({'Site_Name_Hosp': pc.helper['site_name'] + '_' + _x, '%_of_SAT_flowsheet_delivery_flag': percent_sat_flowsheet_delivery_flag, '%_of_SAT_modified_delivery': percent_SAT_modified_delivery, '%_of_SAT_EHR_delivery': percent_SAT_EHR_delivery, '%_of_SAT_rass_nonneg_30': percent_SAT_rass_nonneg_30, '%_of_SAT_med_halved_rass_pos': percent_SAT_med_halved_rass_pos, '%_of_SAT_no_meds_rass_pos_45': percent_SAT_no_meds_rass_pos_45, '%_of_SAT_rass_first_neg_30_last45_nonneg': percent_SAT_rass_first_neg_30_last45_nonneg, 'eligible_event_count': eligible_event_count, 'sat_flowsheet_delivery_flag_count': sat_flowsheet_delivery_flag_count, 'SAT_modified_delivery_count': SAT_modified_delivery_count, 'SAT_EHR_delivery_count': SAT_EHR_delivery_count, 'SAT_rass_nonneg_30_count': SAT_rass_nonneg_30_count, 'SAT_med_halved_rass_pos_count': SAT_med_halved_rass_pos_count, 'SAT_no_meds_rass_pos_45_count': SAT_no_meds_rass_pos_45_count, 'SAT_rass_first_neg_30_last45_nonneg_count': SAT_rass_first_neg_30_last45_nonneg_count, 'SAT_EHR_unique_patients': SAT_EHR_uni_pats, 'SAT_EHR_unique_hospitalizations': SAT_EHR_hosp, 'SAT_modified_unique_patients': SAT_modified_uni_pats, 'SAT_modified_unique_hospitalizations': SAT_modified_hosp, 'SAT_rass_nonneg_30_unique_patients': SAT_rass_nonneg_30_uni_pats, 'SAT_rass_nonneg_30_unique_hospitalizations': SAT_rass_nonneg_30_hosp, 'SAT_med_halved_rass_pos_unique_patients': SAT_med_halved_rass_pos_uni_pats, 'SAT_med_halved_rass_pos_unique_hospitalizations': SAT_med_halved_rass_pos_hosp, 'SAT_no_meds_rass_pos_45_unique_patients': SAT_no_meds_rass_pos_45_uni_pats, 'SAT_no_meds_rass_pos_45_unique_hospitalizations': SAT_no_meds_rass_pos_45_hosp, 'SAT_rass_first_neg_30_last45_nonneg_unique_patients': SAT_rass_first_neg_30_last45_nonneg_uni_pats, 'SAT_rass_first_neg_30_last45_nonneg_unique_hospitalizations': SAT_rass_first_neg_30_last45_nonneg_hosp})
    final_data_df = pd.DataFrame(data_list)  # New flag: first negative RASS within 30 min, last 45 min non-negative
    final_data_df.to_csv(f'{directory_path}/sat_stats_{pc.helper['site_name']}.csv', index=False)
    # Create a DataFrame from the list and export
    # Display the final DataFrame
    final_data_df.T  # Unique patients and hospitalizations for each flag  # Unique for the new flag  # Calculate percentages safely  # Append the metrics for this hospital
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot
    """)
    return


@app.cell
def _(df):
    df[["hospital_id"]] = (
        df.groupby("hospitalization_id")[["hospital_id"]].ffill().bfill()
    )
    return


@app.cell
def _(df, directory_path, pd, plt):
    hospital_ids = df['hospital_id'].dropna().unique()
    hospital_summary_list = []
    # This list will hold the summary data for each hospital
    for _hosp in hospital_ids:
        final_hosp = df[df['hospital_id'] == _hosp]
        sat_d_time = final_hosp[(final_hosp['sat_delivery_pass_fail'] == 1) | (final_hosp['sat_screen_pass_fail'] == 1)].sort_values(['hosp_id_day_key', 'event_time']).groupby('hosp_id_day_key', as_index=False).first()[['hosp_id_day_key', 'event_time']]
        ehr_d_time = final_hosp[final_hosp['SAT_EHR_delivery'] == 1].sort_values(['hosp_id_day_key', 'event_time']).groupby('hosp_id_day_key', as_index=False).first()[['hosp_id_day_key', 'event_time']]  # Filter final_df for the current hospital
        sat_hours = sat_d_time['event_time'].dt.hour
        ehr_hours = ehr_d_time['event_time'].dt.hour
        plt.figure(figsize=(10, 6))  # Extract event times for SBT delivery (pass) and EHR delivery (within 2 mins)
        plt.hist(sat_hours, bins=range(0, 25), alpha=0.5, label='SAT Delivery Time', edgecolor='black')
        plt.hist(ehr_hours, bins=range(0, 25), alpha=0.5, label='EHR Delivery Time', edgecolor='black')
        plt.xlabel('Hour of Day')
        plt.ylabel('Frequency')
        plt.title(f'Event Time Distribution (Hourly) - Hospital {_hosp}')
        plt.legend()  # ensure order
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f'{directory_path}/event_time_distribution_hospital_{_hosp}.png')
        plt.close()
        sat_hours = sat_hours.value_counts().sort_index()
        _ehr_counts = ehr_hours.value_counts().sort_index()
        hours_df = pd.DataFrame({'hour': range(24)})
        hours_df['SAT_Delivery'] = hours_df['hour'].map(sat_hours).fillna(0).astype(int)
        hours_df['EHR_Delivery'] = hours_df['hour'].map(_ehr_counts).fillna(0).astype(int)
        hours_df['hospital_id'] = _hosp
        hospital_summary_list.append(hours_df)  # ensure order
    combined_summary_df = pd.concat(hospital_summary_list, ignore_index=True)
    combined_summary_df.to_csv(f'{directory_path}/event_time_distribution_summary.csv', index=False)
    # Combine the summary data for all hospitals into one DataFrame
    print('Overlay plots created and summary CSV saved.')  # Convert event_time to hour values  # Create overlay histogram plot for the current hospital  # Use bins from 0 to 24 (24 bins) to capture each hour of the day  # Save the plot for the current hospital  # Build a summary DataFrame for the current hospital:  # Get counts per hour for each event type  # Create a DataFrame with all hours 0-23, merging the counts (fill missing with 0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # New Table 1 Code
    """)
    return


@app.cell
def _(np, pd, re, t1_cohort):
    # Aggregate functions
    def documented(series):
        return 'Documented' if series.notna().any() else 'Not Documented'

    def age_bucket(mean_age):
        if pd.isna(mean_age):
            return None
        elif mean_age < 40:
            return '18-39'
        elif mean_age < 60:
            return '40-59'
        elif mean_age < 80:
            return '60-79'
        else:
            return '80+'

    # Clean 'language_name' to only "English", "Spanish", or "Other"
    def categorize_language(lang):
        if re.search('english', str(lang), re.IGNORECASE):
            return 'English'
        elif re.search('spanish', str(lang), re.IGNORECASE):
            return 'Spanish'
        else:
            return 'Other'
    t1_col = ['patient_id', 'hospitalization_id', 'hosp_id_day_key', 'age_at_admission', 'sex_category', 'race_category', 'ethnicity_category', 'language_name', 'weight_kg', 'height_cm', 'cisatracurium', 'vecuronium', 'rocuronium', 'dobutamine', 'dopamine', 'epinephrine', 'fentanyl', 'hydromorphone', 'isoproterenol', 'lorazepam', 'midazolam', 'milrinone', 'morphine', 'norepinephrine', 'phenylephrine', 'propofol', 'vasopressin', 'angiotensin', 'rass', 'gcs_total']
    medication_columns = ['rass', 'gcs_total', 'cisatracurium', 'vecuronium', 'rocuronium', 'dobutamine', 'dopamine', 'epinephrine', 'fentanyl', 'hydromorphone', 'isoproterenol', 'lorazepam', 'midazolam', 'milrinone', 'morphine', 'norepinephrine', 'phenylephrine', 'propofol', 'vasopressin', 'angiotensin']
    demographic_columns = ['sex_category', 'race_category', 'ethnicity_category', 'language_name']
    continuous_cols = ['rass', 'gcs_total', 'cisatracurium', 'vecuronium', 'rocuronium', 'dobutamine', 'dopamine', 'epinephrine', 'fentanyl', 'hydromorphone', 'isoproterenol', 'lorazepam', 'midazolam', 'milrinone', 'morphine', 'norepinephrine', 'phenylephrine', 'propofol', 'vasopressin', 'angiotensin', 'bmi']
    drugs = ['cisatracurium', 'vecuronium', 'rocuronium', 'dobutamine', 'dopamine', 'epinephrine', 'fentanyl', 'hydromorphone', 'isoproterenol', 'lorazepam', 'midazolam', 'milrinone', 'morphine', 'norepinephrine', 'phenylephrine', 'propofol', 'vasopressin', 'angiotensin']
    drugs_present = [d for d in drugs if d in t1_cohort.columns]
    t1_cohort[drugs_present] = t1_cohort[drugs_present].apply(lambda col: _col.map(lambda x: _x if _x > 0 else np.nan))
    t1_cohort['bmi'] = t1_cohort['weight_kg'] / (t1_cohort['height_cm'] / 100) ** 2
    if 'language_name' in t1_cohort.columns:
        t1_cohort['language_name'] = t1_cohort['language_name'].apply(categorize_language)
    else:
        t1_cohort['language_name'] = 'Other'
    # Defensive: only process drug columns that exist
    # Apply the transformation (pandas 2.x compatible: use .map instead of .applymap)
    # Apply the function to 'language_name'
    t1_cohort[continuous_cols] = t1_cohort[[c for c in continuous_cols if c in t1_cohort.columns]].astype(float)
    return (
        age_bucket,
        continuous_cols,
        demographic_columns,
        documented,
        medication_columns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Table 1 By ID for Categorical
    """)
    return


@app.cell
def _(
    age_bucket,
    demographic_columns,
    directory_path,
    documented,
    medication_columns,
    t1_cohort,
    t1code,
):
    for _x in ['hospitalization_id', 'patient_id']:
        _t1_summary = t1_cohort.groupby(_x).agg({'age_at_admission': 'mean', **{_col: documented for _col in medication_columns}, **{_col: 'first' for _col in demographic_columns}})
        _t1_summary['age_bucket'] = _t1_summary['age_at_admission'].apply(age_bucket)
        _t1_summary = _t1_summary.drop(columns=['age_at_admission'])
        _t1_summary = _t1_summary.reset_index()
        _summary_df = t1code.manual_categorical_tableone(_t1_summary, medication_columns + demographic_columns + ['age_bucket'])
        if _x == 'hospitalization_id':
            _summary_df.to_csv(f'{directory_path}/table1_hospitalization_id_categorical.csv', index=False)
        else:
            _summary_df.to_csv(f'{directory_path}/table1_patient_id_categorical.csv', index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Table 1 By ID for Continuous
    """)
    return


@app.cell
def _(continuous_cols, directory_path, t1_cohort, t1code):
    hospitalization_summary = None
    patient_summary = None
    _hosp = t1_cohort.groupby('hospitalization_id').agg({**{c: 'median' for c in continuous_cols}}).reset_index()
    patient = t1_cohort.groupby('patient_id').agg({**{c: 'median' for c in continuous_cols}}).reset_index()
    hospitalization_summary = t1code.manual_tableone(_hosp, continuous_cols)
    patient_summary = t1code.manual_tableone(patient, continuous_cols)
    hospitalization_summary.to_csv(f'{directory_path}/table1_hospitalization_id_continuous.csv', index=False)
    # Build for hospitalization level and patient level
    patient_summary.to_csv(f'{directory_path}/table1_patient_id_continuous.csv', index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Table 1 By Days for Categorical
    """)
    return


@app.cell
def _(
    age_bucket,
    demographic_columns,
    directory_path,
    documented,
    final_df,
    medication_columns,
    t1_cohort,
    t1code,
    tqdm,
):
    for _x in tqdm(['eligible_event', 'SAT_EHR_delivery', 'SAT_modified_delivery'], desc='Generating table 1 for categorical variables for each flags'):
        ids_to_use = final_df[final_df[_x] == 1].hosp_id_day_key.unique()
        _t1_summary = t1_cohort[t1_cohort['hosp_id_day_key'].isin(ids_to_use)].groupby('hosp_id_day_key').agg({'age_at_admission': 'mean', **{_col: documented for _col in medication_columns}, **{_col: 'first' for _col in demographic_columns}})
        _t1_summary['age_bucket'] = _t1_summary['age_at_admission'].apply(age_bucket)
        _t1_summary = _t1_summary.drop(columns=['age_at_admission'])
        _t1_summary = _t1_summary.reset_index()
        _summary_df = t1code.manual_categorical_tableone(_t1_summary, medication_columns + demographic_columns + ['age_bucket'])
        _summary_df.to_csv(f'{directory_path}/table1_{_x}_categorical.csv', index=False)  # Groupby aggregation by hospitalization_id  # Apply age bucketing  # Drop raw age if you don't need it  # Reset index if needed
    return


@app.cell
def _(continuous_cols, directory_path, final_df, t1_cohort, t1code, tqdm):
    for _x in tqdm(['eligible_event', 'SAT_EHR_delivery', 'SAT_modified_delivery', 'SAT_rass_nonneg_30', 'SAT_med_halved_rass_pos', 'SAT_no_meds_rass_pos_45', 'SAT_rass_first_neg_30_last45_nonneg'], desc='Generating Table 1 for continuous variables each flag'):
        ids = final_df.loc[final_df[_x] == 1, 'hosp_id_day_key'].unique()
        sub = t1_cohort[t1_cohort['hosp_id_day_key'].isin(ids)]
        day_summary = sub.groupby('hosp_id_day_key').agg({**{c: 'median' for c in continuous_cols}}).reset_index()
        _summary_df = t1code.manual_tableone(day_summary, continuous_cols)
        _summary_df.to_csv(f'{directory_path}/table1_{_x}_continuous.csv', index=False)  # --- filter to only the days in this subcohort  # --- 1) Day-level medians + flags + demographics
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sofa T1's
    """)
    return


@app.cell
def _():
    import pySofa as sofa
    sofa.pyCLIF2.helper

    continuous_cols_sofa = [ 'sofa_cv_97', 'sofa_coag',
           'sofa_liver', 'sofa_resp_pf', 'sofa_resp_pf_imp', 'sofa_resp',
           'sofa_cns', 'sofa_renal', 'sofa_total']
    return continuous_cols_sofa, sofa


@app.cell
def _(pd):
    mapping_ids = pd.read_csv('../output/intermediate/hospitalization_to_block_df.csv')
    encounter_dict = dict(zip(mapping_ids['hospitalization_id'].astype(str), mapping_ids['encounter_block'].astype(str)))
    mapping_ids[['hospitalization_id','encounter_block']]=mapping_ids[['hospitalization_id','encounter_block']].astype(str)
    mapping_ids.head()
    return (mapping_ids,)


@app.cell
def _(cohort_2, pc):
    encounter_level_sofa = cohort_2[['hospitalization_id', 'admission_dttm', 'discharge_dttm']].drop_duplicates().rename(columns={'admission_dttm': 'start_dttm', 'discharge_dttm': 'stop_dttm'})
    encounter_level_sofa = pc.convert_datetime_columns_to_site_tz(encounter_level_sofa, pc.helper['your_site_timezone'])
    encounter_level_sofa.head()
    return (encounter_level_sofa,)


@app.cell
def _(
    continuous_cols_sofa,
    directory_path,
    encounter_level_sofa,
    mapping_ids,
    sofa,
    t1code,
):
    sout = sofa.compute_sofa(
        encounter_level_sofa,
        tables_path=None,
        use_hospitalization_id = False,
        id_mapping = mapping_ids,
        group_by_id = "encounter_block"
    )
    encounter_level_sofa_t1 = t1code.manual_tableone(sout, continuous_cols_sofa)
    encounter_level_sofa_t1.to_csv(f'{directory_path}/encounter_level_sofa_t1.csv',index=False)
    return


@app.cell
def _(
    continuous_cols_sofa,
    df,
    directory_path,
    mapping_ids,
    pc,
    pd,
    sofa,
    t1code,
    tqdm,
):
    for _x in tqdm(['on_vent_and_sedation', 'sat_delivery_pass_fail', 'SAT_EHR_delivery', 'SAT_modified_delivery', 'SAT_rass_nonneg_30', 'SAT_med_halved_rass_pos', 'SAT_no_meds_rass_pos_45', 'SAT_rass_first_neg_30_last45_nonneg'], desc='Generating Sofa table 1 for each Flags'):
        df['event_time'] = pd.to_datetime(df['event_time']).dt.normalize()
        day_df = df[df[_x] == 1][['hospitalization_id', 'hosp_id_day_key', 'event_time']].drop_duplicates()
        if day_df.empty:
            continue
        day_df['start_dttm'] = pd.to_datetime(day_df['event_time']).dt.normalize()
        day_df['stop_dttm'] = day_df['start_dttm'] + pd.Timedelta(hours=23, minutes=59, seconds=59)
        day_df = pc.convert_datetime_columns_to_site_tz(day_df, pc.helper['your_site_timezone'])
        day_sofa = sofa.compute_sofa(day_df, tables_path=None, use_hospitalization_id=False, id_mapping=mapping_ids, group_by_id='hosp_id_day_key')
        day_sofa_t1 = t1code.manual_tableone(day_sofa, continuous_cols_sofa)
        day_sofa_t1.to_csv(f'{directory_path}/{_x}_sofa_t1.csv', index=False)  # Create start_dttm as current_day at 00:00:00  # Create end_dttm as current_day at 23:59:59
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Thank You!!! upload to box :)
    """)
    return


if __name__ == "__main__":
    app.run()
