import marimo

__generated_with = "0.20.2"
app = marimo.App()


# ---------------------------------------------------------------------------
# Cell 1: marimo import
# ---------------------------------------------------------------------------
@app.cell
def _():
    import marimo as mo

    return (mo,)


# ---------------------------------------------------------------------------
# Cell 2: Setup + config — path resolution, imports, pyCLIF, pySBT
# ---------------------------------------------------------------------------
@app.cell
def _():
    import os
    import sys

    _CODE_DIR = os.path.dirname(os.path.abspath(__file__))
    _UTILS_DIR = os.path.join(_CODE_DIR, '..', 'utils')
    if _UTILS_DIR not in sys.path:
        sys.path.insert(0, _UTILS_DIR)
    os.chdir(_CODE_DIR)

    import numpy as np
    import pandas as pd
    import polars as pl
    import re
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from tqdm import tqdm
    from tableone import TableOne

    import pyCLIF as pc
    import pySBT as t1code
    from sklearn.metrics import confusion_matrix

    # JAMA figure style defaults (Arial, 8pt minimum)
    rcParams['font.family'] = 'Arial'
    rcParams['font.size'] = 8

    pc.load_config()

    return (
        TableOne,
        confusion_matrix,
        np,
        os,
        pc,
        pd,
        pl,
        plt,
        re,
        t1code,
        tqdm,
    )


# ---------------------------------------------------------------------------
# Cell 3: Load cohort + prep — read parquet with polars, convert flags,
#          add sedation dose columns. All mutations consolidated here.
# ---------------------------------------------------------------------------
@app.cell
def _(np, os, pc, pd, pl):
    from definitions_source_of_truth import MAX_FFILL_OBSERVATIONS

    # --- output directory setup ---
    _dir = os.path.join("../output/final/", pc.helper["site_name"], "SAT_standard")
    os.makedirs(_dir, exist_ok=True)
    print(f"Output directory: '{_dir}'")
    output_dir = _dir

    # --- read with polars, clean hospital_id, convert to pandas ---
    _raw = (
        pl.read_parquet('../output/intermediate/study_cohort.parquet')
        .with_columns(
            pl.col('hospital_id').str.replace_all(r'[^a-zA-Z]', '')
        )
    )
    # Snapshot for Table 1 (pre-mutation, pandas)
    t1_cohort = _raw.to_pandas()

    # --- work in pandas from here (downstream functions require it) ---
    cohort = t1_cohort.copy()

    # Convert pass_fail columns to binary documentation indicators.
    # 1 = documented (pass), 0 = documented (fail), NaN = not documented.
    cohort['sat_delivery_pass_fail_documented'] = (
        cohort['sat_delivery_pass_fail'].notna().astype(float).replace(0.0, np.nan)
    )
    cohort['sat_screen_pass_fail_documented'] = (
        cohort['sat_screen_pass_fail'].notna().astype(float).replace(0.0, np.nan)
    )

    # --- datetime parsing ---
    cohort['event_time'] = pd.to_datetime(cohort['event_time'])
    cohort['admission_dttm'] = pd.to_datetime(cohort['admission_dttm'], format='mixed')
    cohort['discharge_dttm'] = pd.to_datetime(cohort['discharge_dttm'], format='mixed')

    # --- sort + forward-fill device/location ---
    cohort = cohort.sort_values(
        by=['hospitalization_id', 'event_time']
    ).reset_index(drop=True)
    cohort['device_category_ffill'] = (
        cohort.groupby('hospitalization_id')['device_category'].ffill()
    )
    cohort['location_category_ffill'] = (
        cohort.groupby('hospitalization_id')['location_category'].ffill()
    )

    # --- sedation dose columns ---
    _sed_cols = ['fentanyl', 'propofol', 'lorazepam', 'midazolam', 'hydromorphone', 'morphine']
    for _col in _sed_cols:
        if _col not in cohort.columns:
            cohort[_col] = np.nan
            print(
                f"Column '{_col}' is missing. Check your CLIF Meds table — "
                "it might be missing, or it's okay if your site doesn't use it."
            )

    cohort[_sed_cols] = (
        cohort.groupby('hospitalization_id')[_sed_cols]
        .ffill(limit=MAX_FFILL_OBSERVATIONS)
    )
    cohort['min_sedation_dose'] = cohort[_sed_cols].min(axis=1, skipna=True)
    cohort['min_sedation_dose_2'] = (
        cohort[_sed_cols]
        .where(cohort[_sed_cols] > 0)
        .min(axis=1, skipna=True)
    )
    _non_ops = ['propofol', 'lorazepam', 'midazolam']
    cohort['min_sedation_dose_non_ops'] = (
        cohort[_non_ops].min(axis=1, skipna=True).fillna(0)
    )

    # --- paralytic columns ---
    _par_cols = ['cisatracurium', 'vecuronium', 'rocuronium']
    cohort[_par_cols] = (
        cohort.groupby('hospitalization_id')[_par_cols]
        .ffill(limit=MAX_FFILL_OBSERVATIONS)
    )
    cohort['max_paralytics'] = (
        cohort[_par_cols].max(axis=1, skipna=True).fillna(0)
    )

    # --- RASS forward-fill ---
    cohort[['rass']] = (
        cohort.groupby('hospitalization_id')[['rass']]
        .ffill(limit=MAX_FFILL_OBSERVATIONS)
    )

    cohort = cohort.sort_values(
        by=['hospitalization_id', 'event_time']
    ).reset_index(drop=True)

    return MAX_FFILL_OBSERVATIONS, cohort, output_dir, t1_cohort


# ---------------------------------------------------------------------------
# Cell 4: SAT eligibility — process_cohort() + call
#          Returns (result_df, cohort_with_eligibility)
# ---------------------------------------------------------------------------
@app.cell
def _(cohort, np, pd, tqdm):

    def process_cohort(df: pd.DataFrame) -> pd.DataFrame:
        """Identify encounter-days where conditions for SAT eligibility are met
        for at least 4 continuous hours in the 22:00-06:00 window.

        Returns a DataFrame with columns:
            hospitalization_id, current_day_key, event_time_at_4_hours
        """
        df = df.sort_values(
            by=['hospitalization_id', 'event_time']
        ).reset_index(drop=True)

        df = df.copy()
        df['device_category_ffill'] = (
            df.groupby('hospitalization_id')['device_category'].ffill()
        )
        df['location_category_ffill'] = (
            df.groupby('hospitalization_id')['location_category'].ffill()
        )
        df['event_time'] = pd.to_datetime(df['event_time'])
        df['all_conditions_check'] = (
            (df['device_category_ffill'].str.lower() == 'imv')
            & (df['min_sedation_dose_2'] > 0)
            & (df['location_category_ffill'].str.lower() == 'icu')
            & (df['max_paralytics'] <= 0)
        ).astype(int)

        # Restrict to days that have at least one IMV observation
        vented_days = df[df['device_category'] == 'imv']['hosp_id_day_key'].unique()
        df = df[df['hosp_id_day_key'].isin(vented_days)]

        # Group by 06:00-anchored day (shift event_time back 6 h before normalizing)
        _shifted = df['event_time'] - pd.Timedelta(hours=6)
        hosp_grouped = df.groupby(['hospitalization_id', _shifted.dt.normalize()])

        window_start_offset = pd.Timedelta(hours=22) - pd.Timedelta(days=1)
        window_end_offset = pd.Timedelta(hours=6)

        result = []
        for (hosp_id, date), group in tqdm(
            hosp_grouped,
            desc='Evaluating hospitalizations by day for SAT eligibility',
        ):
            temp_df = group.sort_values('event_time')
            start_time = date + window_start_offset
            end_time = date + window_end_offset
            mask = (
                (temp_df['event_time'] >= start_time)
                & (temp_df['event_time'] <= end_time)
            )
            window_df = temp_df.loc[mask].copy()

            if window_df.empty or not window_df['all_conditions_check'].any():
                continue

            window_df['condition_met_group'] = (
                (window_df['all_conditions_check'] != window_df['all_conditions_check'].shift())
                .cumsum()
            )
            valid_segments = (
                window_df[window_df['all_conditions_check'] == 1]
                .groupby('condition_met_group')
            )
            for _, segment in valid_segments:
                segment = segment.sort_values('event_time')
                segment = segment.copy()
                segment['duration'] = segment['event_time'].diff().fillna(
                    pd.Timedelta(seconds=0)
                )
                segment['cumulative_duration'] = segment['duration'].cumsum()
                if segment['cumulative_duration'].iloc[-1] >= pd.Timedelta(hours=4):
                    event_time_at_4_hours = (
                        segment[segment['cumulative_duration'] >= pd.Timedelta(hours=4)]
                        .iloc[0]['event_time']
                    )
                    result.append({
                        'hospitalization_id': hosp_id,
                        'current_day_key': date,
                        'event_time_at_4_hours': event_time_at_4_hours,
                    })
                    break

        return pd.DataFrame(result)

    result_df = process_cohort(cohort)
    print(
        'Encounter days with at least 4 hours of conditions met from 10 PM to 6 AM:',
        len(result_df),
    )

    # --- merge eligibility back into cohort ---
    cohort_work = cohort.copy()
    cohort_work['event_time'] = pd.to_datetime(cohort_work['event_time'])
    cohort_work['event_date'] = (
        cohort_work['event_time'] - pd.Timedelta(hours=6)
    ).dt.normalize()

    cohort_elig = cohort_work.merge(
        result_df[['hospitalization_id', 'current_day_key', 'event_time_at_4_hours']],
        how='left',
        left_on=['hospitalization_id', 'event_date'],
        right_on=['hospitalization_id', 'current_day_key'],
    )

    mask_valid = cohort_elig['event_time_at_4_hours'].notna()
    mask_after = cohort_elig['event_time'] >= cohort_elig['event_time_at_4_hours']
    eligible_rows = cohort_elig[mask_valid & mask_after].copy()
    first_eligible_idx = (
        eligible_rows.groupby(['hospitalization_id', 'event_date'])['event_time']
        .idxmin()
    )

    cohort_elig['eligible_event'] = np.nan
    cohort_elig.loc[first_eligible_idx, 'eligible_event'] = 1

    # Remove flag on the last event per hospitalization to avoid edge artefacts
    last_idxs = cohort_elig.groupby('hospitalization_id')['event_time'].idxmax()
    cohort_elig.loc[last_idxs, 'eligible_event'] = np.nan

    eligible_days = cohort_elig.loc[
        cohort_elig['eligible_event'] == 1, 'hosp_id_day_key'
    ].unique()
    cohort_elig['on_vent_and_sedation'] = (
        cohort_elig['hosp_id_day_key'].isin(eligible_days).astype(int)
    )
    cohort_with_eligibility = cohort_elig.drop(
        columns=['event_date', 'current_day_key', 'event_time_at_4_hours']
    )

    print(
        'Eligible hosp-day keys (on_vent_and_sedation=1):',
        cohort_with_eligibility[cohort_with_eligibility['on_vent_and_sedation'] == 1][
            'hosp_id_day_key'
        ].nunique(),
    )

    return cohort_with_eligibility, process_cohort, result_df


# ---------------------------------------------------------------------------
# Cell 5: SAT delivery detection — rank sedation weaning, compute 6 EHR flags
#          Returns (cohort_with_delivery,)
# ---------------------------------------------------------------------------
@app.cell
def _(cohort_with_eligibility, np, pd, tqdm):
    _MED_COLS = [
        'fentanyl', 'propofol', 'lorazepam', 'midazolam', 'hydromorphone', 'morphine',
    ]
    _MED_COLS2 = ['propofol', 'lorazepam', 'midazolam']
    _FLAG_COLS = [
        'SAT_EHR_delivery',
        'SAT_modified_delivery',
        'SAT_rass_nonneg_30',
        'SAT_med_halved_rass_pos',
        'SAT_no_meds_rass_pos_45',
        'SAT_rass_first_neg_30_last45_nonneg',
    ]

    vent_eligible = (
        cohort_with_eligibility[cohort_with_eligibility['on_vent_and_sedation'] == 1]
        .sort_values(by=['hospitalization_id', 'event_time'])
        .reset_index(drop=True)
        .copy()
    )

    # Initialise flag columns
    for _f in _FLAG_COLS:
        vent_eligible[_f] = np.nan

    # --- rank_sedation: cumulative zeros per hosp-day-key ---
    vent_eligible['rank_sedation'] = np.nan
    for _key, _grp in tqdm(
        vent_eligible.groupby('hosp_id_day_key'),
        desc='Detecting weaning/tapering for sedation meds',
    ):
        _zero_mask = _grp['min_sedation_dose'] == 0
        _ranks = _zero_mask.cumsum() * _zero_mask
        vent_eligible.loc[_grp.index, 'rank_sedation'] = _ranks.replace(0, np.nan)

    vent_eligible['rank_sedation_non_ops'] = np.nan
    for _key, _grp in tqdm(
        vent_eligible.groupby('hosp_id_day_key'),
        desc='Detecting weaning/tapering for non-opioid sedation meds',
    ):
        _zero_mask = _grp['min_sedation_dose_non_ops'] == 0
        _ranks = _zero_mask.cumsum() * _zero_mask
        vent_eligible.loc[_grp.index, 'rank_sedation_non_ops'] = (
            _ranks.replace(0, np.nan)
        )

    # Ensure RASS is float
    vent_eligible['rass'] = vent_eligible['rass'].astype(float)

    # --- main SAT flag loop ---
    _delta30 = pd.Timedelta(minutes=30)
    _delta45 = pd.Timedelta(minutes=45)

    for _key, _grp in tqdm(
        vent_eligible.groupby('hosp_id_day_key'),
        desc='Evaluating SAT using RASS, meds, and ventilation criteria (6 ways)',
    ):
        _grp = _grp.sort_values('event_time')
        _idxs = _grp.index
        _times = _grp['event_time'].values
        _ranks = _grp['rank_sedation'].values

        for _idx, _cur_time, _rank in zip(_idxs, _times, _ranks):
            if pd.isna(_rank):
                continue

            _fw30 = _grp[
                (_grp['event_time'] >= _cur_time)
                & (_grp['event_time'] <= _cur_time + _delta30)
            ]
            _fw45 = _grp[
                (_grp['event_time'] >= _cur_time)
                & (_grp['event_time'] <= _cur_time + _delta45)
            ]
            _pr30 = _grp[
                (_grp['event_time'] >= _cur_time - _delta30)
                & (_grp['event_time'] < _cur_time)
            ]

            _imv_ok = (_fw30['device_category_ffill'] == 'imv').all()
            _icu_ok = (_fw30['location_category_ffill'] == 'icu').all()
            if not (_imv_ok and _icu_ok):
                continue

            _flags_set: dict = {}

            # SAT_EHR_delivery: no meds in next 30 min
            _meds_ok = (
                _fw30[_MED_COLS].isna() | (_fw30[_MED_COLS] == 0)
            ).all().all()
            if _meds_ok:
                _flags_set['SAT_EHR_delivery'] = 1

            # SAT_modified_delivery: no non-opioid meds in next 30 min
            _meds2_ok = (
                _fw30[_MED_COLS2].isna() | (_fw30[_MED_COLS2] == 0)
            ).all().all()
            if _meds2_ok:
                _flags_set['SAT_modified_delivery'] = 1

            _rass30 = _fw30['rass'].dropna()
            _rass45 = _fw45['rass'].dropna()
            _rass45_ok = not _rass45.empty and _rass45.iloc[-1] >= 0
            _rass30_pre = _pr30['rass'].dropna()

            # SAT_rass_nonneg_30: all RASS >= 0 in next 30 min
            if not _rass30.empty and (_rass30 >= 0).all():
                _flags_set['SAT_rass_nonneg_30'] = 1

            # SAT_med_halved_rass_pos: all meds <= 50 % of prior 30-min max,
            #   AND last RASS in next 45 min >= 0
            if not _pr30.empty and not _fw30.empty:
                _half_max = _pr30[_MED_COLS].max() * 0.5
                _halved_ok = True
                for _med in _MED_COLS:
                    _vals = _fw30[_med].dropna()
                    _vals = _vals[_vals != 0]
                    if not _vals.empty and not (_vals <= _half_max[_med]).all():
                        _halved_ok = False
                        break
                if _halved_ok and _rass45_ok:
                    _flags_set['SAT_med_halved_rass_pos'] = 1

            # SAT_no_meds_rass_pos_45: no meds in 30 min AND last RASS45 >= 0
            if _meds_ok and _rass45_ok:
                _flags_set['SAT_no_meds_rass_pos_45'] = 1

            # SAT_rass_first_neg_30_last45_nonneg: first prior RASS < 0,
            #   last RASS in next 45 min >= 0
            if not _rass30_pre.empty and not _rass45.empty:
                if _rass30_pre.iloc[0] < 0 and _rass45.iloc[-1] >= 0:
                    _flags_set['SAT_rass_first_neg_30_last45_nonneg'] = 1

            for _f, _val in _flags_set.items():
                vent_eligible.at[_idx, _f] = _val

    # Patch: sites without RASS — zero out RASS-based flags
    if cohort_with_eligibility['rass'].nunique() <= 1:
        for _f in [
            'SAT_rass_nonneg_30',
            'SAT_med_halved_rass_pos',
            'SAT_no_meds_rass_pos_45',
            'SAT_rass_first_neg_30_last45_nonneg',
        ]:
            vent_eligible[_f] = 0
        print('Site CLIF has no RASS — all RASS-based flags set to 0.')

    # Fill hospital_id forward/back within hospitalization
    vent_eligible[['hospital_id']] = (
        vent_eligible.groupby('hospitalization_id')[['hospital_id']]
        .ffill()
        .bfill()
    )

    cohort_with_delivery = vent_eligible
    return (cohort_with_delivery,)


# ---------------------------------------------------------------------------
# Cell 6: Delta-time plot — time from first flowsheet event to EHR detection
# ---------------------------------------------------------------------------
@app.cell
def _(cohort_with_delivery, output_dir, pd, pl):
    _mask_initial = (
        (cohort_with_delivery['sat_delivery_pass_fail'] == 1)
        | (cohort_with_delivery['sat_screen_pass_fail'] == 1)
    )
    _mask_ehr = cohort_with_delivery['SAT_EHR_delivery'] == 1
    _mask_mod = cohort_with_delivery['SAT_modified_delivery'] == 1

    _initial_times = (
        cohort_with_delivery[_mask_initial]
        .groupby('hosp_id_day_key')['event_time'].min()
        .rename('initial_time')
    )
    _ehr_times = (
        cohort_with_delivery[_mask_ehr]
        .groupby('hosp_id_day_key')['event_time'].min()
        .rename('ehr_time')
    )
    _mod_times = (
        cohort_with_delivery[_mask_mod]
        .groupby('hosp_id_day_key')['event_time'].min()
        .rename('mod_time')
    )

    _times_df = pd.concat([_initial_times, _ehr_times, _mod_times], axis=1).dropna()
    for _c in ['initial_time', 'ehr_time', 'mod_time']:
        _times_df[_c] = pd.to_datetime(_times_df[_c])

    _times_df['delta_to_ehr'] = (
        (_times_df['ehr_time'] - _times_df['initial_time']).dt.total_seconds() / 60
    )
    _times_df['delta_to_mod'] = (
        (_times_df['mod_time'] - _times_df['initial_time']).dt.total_seconds() / 60
    )
    _times_df = _times_df[
        (_times_df['delta_to_ehr'] >= 0) & (_times_df['delta_to_ehr'] <= 1440)
        & (_times_df['delta_to_mod'] >= 0) & (_times_df['delta_to_mod'] <= 1440)
    ]

    _bins = list(range(0, 24 * 60 + 1, 60))
    _labels = [f'{i}-{i + 1}hr' for i in range(24)]
    _ehr_binned = pd.cut(_times_df['delta_to_ehr'], bins=_bins, labels=_labels, right=False)
    _mod_binned = pd.cut(_times_df['delta_to_mod'], bins=_bins, labels=_labels, right=False)

    # Use polars for final aggregation
    _binned_pl = pl.DataFrame({
        'hour_bin': _labels,
        'count_to_SAT_EHR_delivery': _ehr_binned.value_counts().sort_index().values,
        'count_to_SAT_modified_delivery': _mod_binned.value_counts().sort_index().values,
    })
    _binned_pl.write_csv(f'{output_dir}/binned_delta_counts.csv')
    print('Delta plot counts saved.')
    return


# ---------------------------------------------------------------------------
# Cell 7: Day-level aggregation — groupby hosp_id_day_key, build final_df
#          Returns (final_df,)
# ---------------------------------------------------------------------------
@app.cell
def _(cohort_with_delivery, cohort_with_eligibility, np, pd, pl):
    # --- ICU LOS (polars for speed) ---
    _icu_pl = (
        pl.from_pandas(
            cohort_with_eligibility[
                ['hospitalization_id', 'event_time', 'location_category_ffill']
            ].sort_values(['hospitalization_id', 'event_time'])
        )
        .with_columns(
            (pl.col('location_category_ffill').str.to_lowercase() == 'icu')
            .alias('is_icu')
        )
        .filter(pl.col('is_icu'))
        .group_by('hospitalization_id')
        .agg([
            pl.col('event_time').min().alias('location_start'),
            pl.col('event_time').max().alias('location_end'),
        ])
        .with_columns(
            (
                (pl.col('location_end') - pl.col('location_start')).dt.total_seconds()
                / 86400
            ).alias('ICU_LOS')
        )
        .select(['hospitalization_id', 'ICU_LOS'])
    )
    total_icu_los_per_hosp = _icu_pl.to_pandas()

    # --- last hospital_id per hospitalization ---
    _last_hosp = (
        cohort_with_eligibility[['hospitalization_id', 'event_time', 'hospital_id']]
        .sort_values(['hospitalization_id', 'event_time'], ascending=False)
        .groupby('hospitalization_id', as_index=False)
        .agg({'hospital_id': 'first'})
        .reset_index(drop=True)
    )

    # --- demographics base (unique rows per hosp) ---
    _demo_cols = [
        'patient_id', 'hospitalization_id', 'admission_dttm', 'discharge_dttm',
        'age_at_admission', 'discharge_category', 'sex_category',
        'race_category', 'ethnicity_category', 'hosp_id_day_key',
    ]
    _main = (
        cohort_with_delivery[_demo_cols]
        .drop_duplicates()
    )
    _main = _main.merge(total_icu_los_per_hosp, on='hospitalization_id', how='left')
    _main = _main.merge(_last_hosp, on='hospitalization_id', how='left')

    # --- day-level flag aggregation (polars for groupby performance) ---
    _max_cols = [
        'sat_screen_pass_fail', 'sat_delivery_pass_fail',
        'SAT_EHR_delivery', 'SAT_modified_delivery',
        'eligible_event',
        'SAT_rass_nonneg_30', 'SAT_med_halved_rass_pos',
        'SAT_no_meds_rass_pos_45', 'SAT_rass_first_neg_30_last45_nonneg',
    ]
    _agg_pl = (
        pl.from_pandas(
            cohort_with_delivery[['hosp_id_day_key'] + _max_cols]
        )
        .group_by('hosp_id_day_key')
        .agg([pl.col(c).max().alias(c) for c in _max_cols])
        .sort('hosp_id_day_key')
    )
    _df_grouped = _agg_pl.to_pandas()

    _df_grouped['sat_flowsheet_delivery_flag'] = np.where(
        (
            (_df_grouped['sat_screen_pass_fail'] == 1)
            | (_df_grouped['sat_delivery_pass_fail'] == 1)
        )
        & (_df_grouped['eligible_event'] == 1),
        1,
        np.nan,
    )

    final_df = _main.merge(_df_grouped, on='hosp_id_day_key', how='inner')
    final_df['admission_dttm'] = pd.to_datetime(final_df['admission_dttm'], format='mixed')
    final_df['discharge_dttm'] = pd.to_datetime(final_df['discharge_dttm'], format='mixed')

    print('final_df shape:', final_df.shape)
    for _x in [
        'sat_delivery_pass_fail', 'sat_screen_pass_fail', 'SAT_EHR_delivery',
        'SAT_modified_delivery', 'eligible_event', 'sat_flowsheet_delivery_flag',
        'SAT_rass_first_neg_30_last45_nonneg', 'SAT_rass_nonneg_30',
        'SAT_med_halved_rass_pos', 'SAT_no_meds_rass_pos_45',
    ]:
        print(final_df[_x].value_counts())
        print()

    return final_df, total_icu_los_per_hosp


# ---------------------------------------------------------------------------
# Cell 8: Concordance analysis — confusion matrices, kappa, metrics
# ---------------------------------------------------------------------------
@app.cell
def _(cohort_with_eligibility, confusion_matrix, final_df, np, os, output_dir, pd, plt):
    from sklearn.metrics import cohen_kappa_score
    import matplotlib
    from matplotlib import rcParams

    # Enforce JAMA style
    rcParams['font.family'] = 'Arial'
    rcParams['font.size'] = 8

    def _landis_koch(kappa: float) -> str:
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
        return 'Almost Perfect'

    def _kappa_ci_bootstrap(
        y_true: pd.Series,
        y_pred: pd.Series,
        n_boot: int = 2000,
        ci: float = 0.95,
        seed: int = 42,
    ) -> tuple[float, float]:
        """Bootstrap 95 % CI for Cohen's kappa."""
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
        return (
            float(np.percentile(kappas, alpha * 100)),
            float(np.percentile(kappas, (1 - alpha) * 100)),
        )

    _con_df = final_df.copy()
    _fill_cols = [
        'SAT_EHR_delivery', 'SAT_modified_delivery', 'sat_flowsheet_delivery_flag',
        'SAT_rass_nonneg_30', 'SAT_med_halved_rass_pos',
        'SAT_no_meds_rass_pos_45', 'SAT_rass_first_neg_30_last45_nonneg',
    ]
    for _c in _fill_cols:
        _con_df[_c] = _con_df[_c].fillna(0)

    _eval_cols = [
        'SAT_EHR_delivery', 'SAT_modified_delivery',
        'SAT_rass_nonneg_30', 'SAT_med_halved_rass_pos',
        'SAT_no_meds_rass_pos_45', 'SAT_rass_first_neg_30_last45_nonneg',
    ]
    _has_rass = cohort_with_eligibility['rass'].nunique() > 1
    _metrics_list = []

    for _col in _eval_cols:
        if 'rass' in _col and not _has_rass:
            continue

        _y_true = _con_df['sat_flowsheet_delivery_flag']
        _y_pred = _con_df[_col]

        _cm = confusion_matrix(_y_true, _y_pred)
        _tn, _fp, _fn, _tp = _cm.ravel()
        _total = _cm.sum()
        _accuracy = (_tp + _tn) / _total
        _precision = _tp / (_tp + _fp) if _tp + _fp else 0
        _recall = _tp / (_tp + _fn) if _tp + _fn else 0
        _f1 = (
            2 * _precision * _recall / (_precision + _recall)
            if _precision + _recall else 0
        )
        _specificity = _tn / (_tn + _fp) if _tn + _fp else 0
        _kappa = cohen_kappa_score(_y_true, _y_pred)
        _kappa_lo, _kappa_hi = _kappa_ci_bootstrap(_y_true, _y_pred)
        _kappa_interp = _landis_koch(_kappa)

        # --- JAMA-style confusion matrix plot ---
        _cm_pct = _cm / _total * 100
        _tick_labels = ['No Delivery', 'Delivery']
        _fig, _ax = plt.subplots(figsize=(3.5, 3.0))
        _im = _ax.imshow(_cm, cmap=plt.cm.Blues)
        _ax.set_xticks([0, 1])
        _ax.set_yticks([0, 1])
        _ax.set_xticklabels(_tick_labels, fontsize=8)
        _ax.set_yticklabels(_tick_labels, fontsize=8)
        _ax.set_xlabel(f'{_col} flag', fontsize=9)
        _ax.set_ylabel('Flowsheet delivery flag', fontsize=9)
        _ax.set_title(f'Concordance: flowsheet vs {_col}', fontsize=10, fontweight='bold')
        for _i in range(_cm.shape[0]):
            for _j in range(_cm.shape[1]):
                _ax.text(
                    _j, _i,
                    f'{_cm[_i, _j]}\n({_cm_pct[_i, _j]:.1f}%)',
                    ha='center', va='center', fontsize=8,
                    color='white' if _cm[_i, _j] > _cm.max() / 2 else 'black',
                )
        _fig.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
        _fig.tight_layout()
        _fig.savefig(
            os.path.join(output_dir, f'confusion_matrix_{_col}.png'),
            bbox_inches='tight', dpi=300,
        )
        plt.close(_fig)

        print(f'--- Concordance for {_col} ---')
        print(f'Accuracy    : {_accuracy:.3f}')
        print(f'Precision   : {_precision:.3f}')
        print(f'Recall      : {_recall:.3f}')
        print(f'F1 Score    : {_f1:.3f}')
        print(f'Specificity : {_specificity:.3f}')
        print(
            f'Cohen kappa : {_kappa:.3f} '
            f'(95% CI: {_kappa_lo:.3f}-{_kappa_hi:.3f}), {_kappa_interp}\n'
        )

        _metrics_list.append({
            'Column': _col,
            'TP': _tp, 'FP': _fp, 'FN': _fn, 'TN': _tn,
            'Accuracy': _accuracy,
            'Precision': _precision,
            'Recall': _recall,
            'F1 Score': _f1,
            'Specificity': _specificity,
            'Cohen_Kappa': _kappa,
            'Kappa_CI_lower': round(_kappa_lo, 3),
            'Kappa_CI_upper': round(_kappa_hi, 3),
            'Kappa_Interpretation': _kappa_interp,
        })

    concordance_df = pd.DataFrame(_metrics_list)
    concordance_df.to_csv(
        os.path.join(output_dir, 'delivery_concordance_summary.csv'), index=False
    )
    return (concordance_df,)


# ---------------------------------------------------------------------------
# Cell 9: Hospital-level summary — rates per hospital (polars aggregation)
# ---------------------------------------------------------------------------
@app.cell
def _(final_df, os, output_dir, pc, pd, pl):

    _flag_cols = [
        'eligible_event',
        'sat_flowsheet_delivery_flag',
        'SAT_modified_delivery',
        'SAT_EHR_delivery',
        'SAT_rass_nonneg_30',
        'SAT_med_halved_rass_pos',
        'SAT_no_meds_rass_pos_45',
        'SAT_rass_first_neg_30_last45_nonneg',
    ]

    def _build_hospital_summary(df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-hospital flag counts, rates, and unique patient/hosp counts."""
        rows = []
        for _hosp in df['hospital_id'].astype(str).unique():
            _sub = df[df['hospital_id'].astype(str) == _hosp]
            _eligible_n = int((_sub['eligible_event'] == 1).sum())
            _row: dict = {
                'Site_Name_Hosp': f"{pc.helper['site_name']}_{_hosp}",
                'eligible_event_count': _eligible_n,
            }
            for _flag in _flag_cols[1:]:  # skip eligible_event itself
                _n = int((_sub[_flag] == 1).sum())
                _row[f'{_flag}_count'] = _n
                _row[f'pct_{_flag}'] = (
                    round(_n / _eligible_n * 100, 2) if _eligible_n > 0 else 0.0
                )
                _row[f'{_flag}_unique_patients'] = (
                    _sub[_sub[_flag] == 1]['patient_id'].nunique()
                )
                _row[f'{_flag}_unique_hospitalizations'] = (
                    _sub[_sub[_flag] == 1]['hospitalization_id'].nunique()
                )
            rows.append(_row)
        return pd.DataFrame(rows)

    hospital_summary = _build_hospital_summary(final_df)
    hospital_summary.to_csv(
        os.path.join(output_dir, f"sat_stats_{pc.helper['site_name']}.csv"),
        index=False,
    )
    print(hospital_summary.T)
    return (hospital_summary,)


# ---------------------------------------------------------------------------
# Cell 10: Hourly delivery time distribution plot per hospital
# ---------------------------------------------------------------------------
@app.cell
def _(cohort_with_delivery, os, output_dir, pd, plt):
    from matplotlib import rcParams as _rcParams
    _rcParams['font.family'] = 'Arial'
    _rcParams['font.size'] = 8

    _hospital_summary_list = []
    for _hosp in cohort_with_delivery['hospital_id'].dropna().unique():
        _hosp_df = cohort_with_delivery[cohort_with_delivery['hospital_id'] == _hosp]

        _sat_d_time = (
            _hosp_df[
                (_hosp_df['sat_delivery_pass_fail'] == 1)
                | (_hosp_df['sat_screen_pass_fail'] == 1)
            ]
            .sort_values(['hosp_id_day_key', 'event_time'])
            .groupby('hosp_id_day_key', as_index=False)
            .first()[['hosp_id_day_key', 'event_time']]
        )
        _ehr_d_time = (
            _hosp_df[_hosp_df['SAT_EHR_delivery'] == 1]
            .sort_values(['hosp_id_day_key', 'event_time'])
            .groupby('hosp_id_day_key', as_index=False)
            .first()[['hosp_id_day_key', 'event_time']]
        )

        _sat_hours = _sat_d_time['event_time'].dt.hour
        _ehr_hours = _ehr_d_time['event_time'].dt.hour

        # JAMA-style histogram
        _fig, _ax = plt.subplots(figsize=(5.0, 3.5))
        _ax.hist(
            _sat_hours, bins=range(0, 25), alpha=0.5,
            label='SAT delivery time', edgecolor='black',
        )
        _ax.hist(
            _ehr_hours, bins=range(0, 25), alpha=0.5,
            label='EHR delivery time', edgecolor='black',
        )
        _ax.set_xlabel('Hour of day', fontsize=9)
        _ax.set_ylabel('Frequency', fontsize=9)
        _ax.set_title(
            f'Event time distribution — hospital {_hosp}',
            fontsize=10, fontweight='bold',
        )
        _ax.tick_params(labelsize=8)
        _ax.grid(axis='y', linestyle='--', alpha=0.7)
        # Legend placed outside plot area (below)
        _ax.legend(
            fontsize=8, loc='upper center',
            bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False,
        )
        _fig.tight_layout()
        _fig.savefig(
            os.path.join(output_dir, f'event_time_distribution_hospital_{_hosp}.png'),
            bbox_inches='tight', dpi=300,
        )
        plt.close(_fig)

        _sat_counts = _sat_hours.value_counts().sort_index()
        _ehr_counts = _ehr_hours.value_counts().sort_index()
        _hours_df = pd.DataFrame({'hour': range(24)})
        _hours_df['SAT_Delivery'] = (
            _hours_df['hour'].map(_sat_counts).fillna(0).astype(int)
        )
        _hours_df['EHR_Delivery'] = (
            _hours_df['hour'].map(_ehr_counts).fillna(0).astype(int)
        )
        _hours_df['hospital_id'] = _hosp
        _hospital_summary_list.append(_hours_df)

    _combined = pd.concat(_hospital_summary_list, ignore_index=True)
    _combined.to_csv(
        os.path.join(output_dir, 'event_time_distribution_summary.csv'), index=False
    )
    print('Overlay plots created and summary CSV saved.')
    return


# ---------------------------------------------------------------------------
# Cell 11: Table 1 generation — demographics and medications by flag
# ---------------------------------------------------------------------------
@app.cell
def _(TableOne, final_df, np, output_dir, pc, pd, re, t1_cohort, t1code, tqdm):

    # --- helper functions ---
    def _documented(series: pd.Series) -> str:
        return 'Documented' if series.notna().any() else 'Not Documented'

    def _age_bucket(mean_age: float) -> str | None:
        if pd.isna(mean_age):
            return None
        elif mean_age < 40:
            return '18-39'
        elif mean_age < 60:
            return '40-59'
        elif mean_age < 80:
            return '60-79'
        return '80+'

    def _categorize_language(lang: str) -> str:
        if re.search('english', str(lang), re.IGNORECASE):
            return 'English'
        elif re.search('spanish', str(lang), re.IGNORECASE):
            return 'Spanish'
        return 'Other'

    # --- column lists ---
    _medication_columns = [
        'rass', 'gcs_total',
        'cisatracurium', 'vecuronium', 'rocuronium',
        'dobutamine', 'dopamine', 'epinephrine',
        'fentanyl', 'hydromorphone', 'isoproterenol',
        'lorazepam', 'midazolam', 'milrinone', 'morphine',
        'norepinephrine', 'phenylephrine', 'propofol',
        'vasopressin', 'angiotensin',
    ]
    _demographic_columns = [
        'sex_category', 'race_category', 'ethnicity_category', 'language_name',
    ]
    _continuous_cols = [
        'rass', 'gcs_total',
        'cisatracurium', 'vecuronium', 'rocuronium',
        'dobutamine', 'dopamine', 'epinephrine',
        'fentanyl', 'hydromorphone', 'isoproterenol',
        'lorazepam', 'midazolam', 'milrinone', 'morphine',
        'norepinephrine', 'phenylephrine', 'propofol',
        'vasopressin', 'angiotensin', 'bmi',
    ]
    _drugs = [
        'cisatracurium', 'vecuronium', 'rocuronium',
        'dobutamine', 'dopamine', 'epinephrine',
        'fentanyl', 'hydromorphone', 'isoproterenol',
        'lorazepam', 'midazolam', 'milrinone', 'morphine',
        'norepinephrine', 'phenylephrine', 'propofol',
        'vasopressin', 'angiotensin',
    ]

    # --- prepare t1_cohort_prep (no mutation of t1_cohort) ---
    t1_cohort_prep = t1_cohort.copy()
    _drugs_present = [d for d in _drugs if d in t1_cohort_prep.columns]
    t1_cohort_prep[_drugs_present] = t1_cohort_prep[_drugs_present].apply(
        lambda col: col.map(lambda x: x if x > 0 else np.nan)
    )
    t1_cohort_prep['bmi'] = (
        t1_cohort_prep['weight_kg'] / (t1_cohort_prep['height_cm'] / 100) ** 2
    )
    t1_cohort_prep['language_name'] = (
        t1_cohort_prep['language_name'].apply(_categorize_language)
        if 'language_name' in t1_cohort_prep.columns
        else 'Other'
    )
    _cont_present = [c for c in _continuous_cols if c in t1_cohort_prep.columns]
    t1_cohort_prep[_cont_present] = t1_cohort_prep[_cont_present].astype(float)

    # --- Table 1: pySBT categorical by hospitalization_id and patient_id ---
    for _grp_col in ['hospitalization_id', 'patient_id']:
        _t1_summ = t1_cohort_prep.groupby(_grp_col).agg(
            age_at_admission=('age_at_admission', 'mean'),
            **{c: (_c := c, _documented)[1] for c in _medication_columns
               if c in t1_cohort_prep.columns},
            **{c: (c, 'first')[1] for c in _demographic_columns
               if c in t1_cohort_prep.columns},
        ).reset_index()
        # Rebuild with explicit agg to avoid comprehension lambda scoping issues
        _agg_dict: dict = {'age_at_admission': 'mean'}
        for _mc in _medication_columns:
            if _mc in t1_cohort_prep.columns:
                _agg_dict[_mc] = _documented
        for _dc in _demographic_columns:
            if _dc in t1_cohort_prep.columns:
                _agg_dict[_dc] = 'first'
        _t1_summ = (
            t1_cohort_prep.groupby(_grp_col)
            .agg(_agg_dict)
            .reset_index()
        )
        _t1_summ['age_bucket'] = _t1_summ['age_at_admission'].apply(_age_bucket)
        _t1_summ = _t1_summ.drop(columns=['age_at_admission'])
        _cat_cols_present = [
            c for c in _medication_columns + _demographic_columns + ['age_bucket']
            if c in _t1_summ.columns
        ]
        _summary_df = t1code.manual_categorical_tableone(_t1_summ, _cat_cols_present)
        _fname = (
            'table1_hospitalization_id_categorical.csv'
            if _grp_col == 'hospitalization_id'
            else 'table1_patient_id_categorical.csv'
        )
        _summary_df.to_csv(f'{output_dir}/{_fname}', index=False)

    # --- Table 1: pySBT continuous ---
    _hosp_agg = t1_cohort_prep.groupby('hospitalization_id').agg(
        {c: 'median' for c in _cont_present}
    ).reset_index()
    _pat_agg = t1_cohort_prep.groupby('patient_id').agg(
        {c: 'median' for c in _cont_present}
    ).reset_index()
    t1code.manual_tableone(_hosp_agg, _cont_present).to_csv(
        f'{output_dir}/table1_hospitalization_id_continuous.csv', index=False
    )
    t1code.manual_tableone(_pat_agg, _cont_present).to_csv(
        f'{output_dir}/table1_patient_id_continuous.csv', index=False
    )

    # --- Table 1: per flag (categorical) ---
    for _flag in tqdm(
        ['eligible_event', 'SAT_EHR_delivery', 'SAT_modified_delivery'],
        desc='Generating categorical Table 1 for each flag',
    ):
        _ids = final_df[final_df[_flag] == 1]['hosp_id_day_key'].unique()
        _sub = t1_cohort_prep[t1_cohort_prep['hosp_id_day_key'].isin(_ids)]
        _agg2: dict = {'age_at_admission': 'mean'}
        for _mc in _medication_columns:
            if _mc in _sub.columns:
                _agg2[_mc] = _documented
        for _dc in _demographic_columns:
            if _dc in _sub.columns:
                _agg2[_dc] = 'first'
        _t1f = _sub.groupby('hosp_id_day_key').agg(_agg2).reset_index()
        _t1f['age_bucket'] = _t1f['age_at_admission'].apply(_age_bucket)
        _t1f = _t1f.drop(columns=['age_at_admission'])
        _cat_f = [
            c for c in _medication_columns + _demographic_columns + ['age_bucket']
            if c in _t1f.columns
        ]
        t1code.manual_categorical_tableone(_t1f, _cat_f).to_csv(
            f'{output_dir}/table1_{_flag}_categorical.csv', index=False
        )

    # --- Table 1: per flag (continuous) ---
    for _flag in tqdm(
        [
            'eligible_event', 'SAT_EHR_delivery', 'SAT_modified_delivery',
            'SAT_rass_nonneg_30', 'SAT_med_halved_rass_pos',
            'SAT_no_meds_rass_pos_45', 'SAT_rass_first_neg_30_last45_nonneg',
        ],
        desc='Generating continuous Table 1 for each flag',
    ):
        _ids = final_df.loc[final_df[_flag] == 1, 'hosp_id_day_key'].unique()
        _sub = t1_cohort_prep[t1_cohort_prep['hosp_id_day_key'].isin(_ids)]
        _day_agg = _sub.groupby('hosp_id_day_key').agg(
            {c: 'median' for c in _cont_present}
        ).reset_index()
        t1code.manual_tableone(_day_agg, _cont_present).to_csv(
            f'{output_dir}/table1_{_flag}_continuous.csv', index=False
        )

    # --- TableOne summaries (pySBT-style) for SAT subgroups ---
    _cat_t1_cols = [
        'sex_category', 'race_category', 'ethnicity_category', 'discharge_category',
    ]
    _non_cat_cols = ['age_at_admission', 'ICU_LOS', 'Inpatient_LOS']

    def _make_tableone_subset(
        df: pd.DataFrame,
        mask: pd.Series,
        fname: str,
    ) -> None:
        _t1_sub = df[mask][[
            'hospitalization_id', 'admission_dttm', 'discharge_dttm',
            'age_at_admission', 'discharge_category', 'sex_category',
            'race_category', 'ethnicity_category', 'ICU_LOS',
        ]].drop_duplicates().copy()
        _t1_sub['Inpatient_LOS'] = (
            (_t1_sub['discharge_dttm'] - _t1_sub['admission_dttm'])
            .dt.total_seconds() / 86400
        )
        if len(_t1_sub) > 1:
            _cols_present = [c for c in _cat_t1_cols + _non_cat_cols
                             if c in _t1_sub.columns]
            _cat_present = [c for c in _cat_t1_cols if c in _t1_sub.columns]
            _non_present = [c for c in _non_cat_cols if c in _t1_sub.columns]
            _tbl = TableOne(
                _t1_sub,
                categorical=_cat_present,
                nonnormal=_non_present,
                columns=_cols_present,
            )
            _tbl.to_csv(fname)
            print(_tbl)

    _make_tableone_subset(
        final_df,
        final_df['sat_flowsheet_delivery_flag'] == 1,
        f"{output_dir}/table1_sat_flowsheet_{pc.helper['site_name']}.csv",
    )
    _make_tableone_subset(
        final_df,
        (final_df['SAT_EHR_delivery'] == 1) | (final_df['SAT_modified_delivery'] == 1),
        f"{output_dir}/table1_sat_ehr_{pc.helper['site_name']}.csv",
    )
    _make_tableone_subset(
        final_df,
        pd.Series([True] * len(final_df), index=final_df.index),
        f"{output_dir}/table1_all_{pc.helper['site_name']}.csv",
    )

    return (t1_cohort_prep,)


# ---------------------------------------------------------------------------
# Cell 12: SOFA computation (pySofa optional)
# ---------------------------------------------------------------------------
@app.cell
def _(cohort_with_eligibility, cohort_with_delivery, output_dir, pc, pd, t1code, tqdm):
    try:
        import pySofa as sofa

        _continuous_cols_sofa = [
            'sofa_cv_97', 'sofa_coag', 'sofa_liver',
            'sofa_resp_pf', 'sofa_resp_pf_imp', 'sofa_resp',
            'sofa_cns', 'sofa_renal', 'sofa_total',
        ]

        _mapping_ids = pd.read_csv('../output/intermediate/hospitalization_to_block_df.csv')
        _mapping_ids[['hospitalization_id', 'encounter_block']] = (
            _mapping_ids[['hospitalization_id', 'encounter_block']].astype(str)
        )

        # Encounter-level SOFA
        _enc_sofa_in = (
            cohort_with_eligibility[
                ['hospitalization_id', 'admission_dttm', 'discharge_dttm']
            ]
            .drop_duplicates()
            .rename(columns={'admission_dttm': 'start_dttm', 'discharge_dttm': 'stop_dttm'})
        )
        _enc_sofa_in = pc.convert_datetime_columns_to_site_tz(
            _enc_sofa_in, pc.helper['your_site_timezone']
        )
        _sout = sofa.compute_sofa(
            _enc_sofa_in,
            tables_path=None,
            use_hospitalization_id=False,
            id_mapping=_mapping_ids,
            group_by_id='encounter_block',
        )
        t1code.manual_tableone(_sout, _continuous_cols_sofa).to_csv(
            f'{output_dir}/encounter_level_sofa_t1.csv', index=False
        )

        # Per-flag day-level SOFA
        _sofa_flags = [
            'on_vent_and_sedation', 'sat_delivery_pass_fail',
            'SAT_EHR_delivery', 'SAT_modified_delivery',
            'SAT_rass_nonneg_30', 'SAT_med_halved_rass_pos',
            'SAT_no_meds_rass_pos_45', 'SAT_rass_first_neg_30_last45_nonneg',
        ]
        _df_sofa = cohort_with_delivery.copy()
        _df_sofa['event_time'] = pd.to_datetime(_df_sofa['event_time']).dt.normalize()

        for _flag in tqdm(_sofa_flags, desc='Generating SOFA Table 1 for each flag'):
            _day_df = (
                _df_sofa[_df_sofa[_flag] == 1][
                    ['hospitalization_id', 'hosp_id_day_key', 'event_time']
                ]
                .drop_duplicates()
            )
            if _day_df.empty:
                continue
            _day_df = _day_df.copy()
            _day_df['start_dttm'] = pd.to_datetime(_day_df['event_time']).dt.normalize()
            _day_df['stop_dttm'] = (
                _day_df['start_dttm'] + pd.Timedelta(hours=23, minutes=59, seconds=59)
            )
            _day_df = pc.convert_datetime_columns_to_site_tz(
                _day_df, pc.helper['your_site_timezone']
            )
            _day_sofa = sofa.compute_sofa(
                _day_df,
                tables_path=None,
                use_hospitalization_id=False,
                id_mapping=_mapping_ids,
                group_by_id='hosp_id_day_key',
            )
            t1code.manual_tableone(_day_sofa, _continuous_cols_sofa).to_csv(
                f'{output_dir}/{_flag}_sofa_t1.csv', index=False
            )

        print('SOFA computation complete.')

    except ImportError:
        print('pySofa not available — skipping SOFA computation.')
    except FileNotFoundError as _e:
        print(f'SOFA mapping file not found — skipping SOFA computation. ({_e})')

    return


# ---------------------------------------------------------------------------
# Cell 13: Save outputs — write CSVs and parquet
# ---------------------------------------------------------------------------
@app.cell
def _(final_df, output_dir, pl):
    # Save final_df as CSV (pandas) and parquet (polars for compression)
    final_df.to_csv('../output/intermediate/final_df_SAT.csv', index=False)

    (
        pl.from_pandas(final_df)
        .write_parquet('../output/intermediate/final_df_SAT.parquet')
    )

    print(f'Outputs written to {output_dir} and ../output/intermediate/')
    return


if __name__ == "__main__":
    app.run()
