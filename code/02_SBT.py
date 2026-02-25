import marimo

__generated_with = "0.20.2"
app = marimo.App()


# ---------------------------------------------------------------------------
# Cell 1 — marimo import
# ---------------------------------------------------------------------------
@app.cell
def _():
    import marimo as mo

    return (mo,)


# ---------------------------------------------------------------------------
# Cell 2 — Path setup + all library imports
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
    import pyCLIF as pc
    from tqdm import tqdm
    from datetime import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
        cohen_kappa_score,
    )
    import warnings
    warnings.filterwarnings('ignore')
    from tableone import TableOne
    import pySBT as sbt

    return (
        accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        datetime,
        f1_score,
        np,
        os,
        pc,
        pd,
        pl,
        plt,
        precision_score,
        re,
        recall_score,
        sbt,
        sns,
        sys,
        tqdm,
    )


# ---------------------------------------------------------------------------
# Cell 3 — Variant selector (interactive in notebook; defaults to Standard
#           when executed as a plain script via `python 02_SBT.py`)
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    variant = mo.ui.dropdown(
        options=[
            "Standard",
            "Respiratory_Stability",
            "Hemodynamic_Stability",
            "Both_stabilities",
        ],
        value="Standard",
        label="SBT Variant",
    )
    mo.md(f"**Selected variant: {variant.value}**")
    return (variant,)


# ---------------------------------------------------------------------------
# Cell 4 — Output directory setup
# ---------------------------------------------------------------------------
@app.cell
def _(os, pc, variant):
    _by = variant.value
    directory_path = os.path.join(
        "../output/final/", pc.helper["site_name"], f"SBT_{_by}"
    )
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    return (directory_path,)


# ---------------------------------------------------------------------------
# Cell 5 — Load cohort (polars read → pandas for pySBT compatibility)
# ---------------------------------------------------------------------------
@app.cell
def _(pd, pl):
    # Use polars for the initial read (fast, memory-efficient)
    _cohort_pl = pl.read_parquet('../output/intermediate/study_cohort.parquet')

    # Convert to pandas for downstream pySBT functions
    cohort = _cohort_pl.to_pandas()
    cohort['hospital_id'] = cohort['hospital_id'].str.replace(
        '[^a-zA-Z]', '', regex=True
    )
    # Keep a pristine copy for Table 1 (before any forward-filling mutations)
    t1_cohort = cohort.copy()
    return cohort, t1_cohort


# ---------------------------------------------------------------------------
# Cell 6 — RUSH-specific SBT timepoint adjustment (site-defensive)
# ---------------------------------------------------------------------------
@app.cell
def _(cohort, pc):
    if pc.helper.get("site_name") == "RUSH" and "sbt_timepoint" in cohort.columns:
        cohort.loc[
            cohort["sbt_timepoint"] == "3-5 minute evaluation",
            "pressure_support_set",
        ] = 6.1
        cohort.loc[
            cohort["sbt_timepoint"] == "3-5 minute evaluation",
            "mode_category",
        ] = "pressure support/cpap"
        print("RUSH-specific SBT timepoint adjustment applied.")
    return


# ---------------------------------------------------------------------------
# Cell 7 — Eligibility flag computation
# ---------------------------------------------------------------------------
@app.cell
def _(cohort, pc, pd):
    cohort['event_time'] = pd.to_datetime(cohort['event_time'])
    cohort['admission_dttm'] = pc.getdttm(cohort['admission_dttm'])
    cohort['discharge_dttm'] = pc.getdttm(cohort['discharge_dttm'])

    cohort_1 = (
        cohort
        .sort_values(by=['hospitalization_id', 'event_time'])
        .reset_index(drop=True)
    )
    cohort_1['device_category'] = cohort_1['device_category'].str.lower()
    cohort_1['mode_category'] = cohort_1['mode_category'].str.lower()

    # Forward-fill grouped columns
    _ffill_cols = [
        'device_category', 'mode_category', 'mode_name',
        'location_category', 'hospital_id',
    ]
    cohort_1[_ffill_cols] = (
        cohort_1.groupby('hospitalization_id')[_ffill_cols].ffill()
    )
    cohort_1[['weight_kg', 'height_cm']] = (
        cohort_1.groupby('hospitalization_id')[['weight_kg', 'height_cm']]
        .ffill()
        .bfill()
    )

    _vaso_cols = [
        'norepinephrine', 'epinephrine', 'phenylephrine', 'angiotensin',
        'vasopressin', 'dopamine', 'dobutamine', 'milrinone', 'isoproterenol',
    ]
    cohort_1[_vaso_cols] = (
        cohort_1.groupby('hospitalization_id')[_vaso_cols].ffill()
    )
    cohort_1[['fio2_set', 'peep_set', 'spo2', 'pressure_support_set']] = (
        cohort_1.groupby('hospitalization_id')[
            ['fio2_set', 'peep_set', 'spo2', 'pressure_support_set']
        ].ffill()
    )

    # Zero-fill vasopressors before NEE calculation
    _nee_cols = [
        'norepinephrine', 'epinephrine', 'phenylephrine',
        'dopamine', 'angiotensin', 'vasopressin',
    ]
    cohort_1[_nee_cols] = cohort_1[_nee_cols].fillna(0)

    cohort_1['NEE'] = (
        cohort_1['norepinephrine']
        + cohort_1['epinephrine']
        + cohort_1['phenylephrine'] / 10
        + cohort_1['vasopressin'] * 2.5
        + cohort_1['dopamine'] / 100
        + cohort_1['angiotensin'] * 10
    )
    cohort_1['Hemodynamic_Stability_by_NEE'] = (cohort_1['NEE'] <= 0.2).astype(int)
    cohort_1['Respiratory_Stability'] = (
        (cohort_1['fio2_set'] <= 0.5)
        & (cohort_1['peep_set'] <= 8)
        & (cohort_1['spo2'] >= 88)
    ).astype(int)

    _paralytic_cols = ['cisatracurium', 'vecuronium', 'rocuronium']
    cohort_1[_paralytic_cols] = (
        cohort_1.groupby('hospitalization_id')[_paralytic_cols].ffill()
    )
    cohort_1['max_paralytics'] = (
        cohort_1[_paralytic_cols].max(axis=1, skipna=True).fillna(0)
    )
    return (cohort_1,)


# ---------------------------------------------------------------------------
# Cell 8 — Process cohort conditions (THE only branching point)
# ---------------------------------------------------------------------------
@app.cell
def _(cohort_1, sbt, variant):
    final_df = sbt.process_cohort_conditions(cohort_1, variant.value)

    # Print eligibility statistics
    print('By n = Days')
    _total_days = final_df['hosp_id_day_key'].nunique()
    print('Total number of days for eval in cohort:', _total_days)
    total_vent_days = final_df[final_df['vent_day'] == 1]['hosp_id_day_key'].nunique()
    print('Total vent days (at least one IMV event):', total_vent_days)
    total_vent_days_wo_paralytics = (
        final_df[final_df['vent_day_without_paralytics'] == 1]['hosp_id_day_key'].nunique()
    )
    print('Total vent days without paralytics:', total_vent_days_wo_paralytics)
    eligible_days = final_df[final_df['eligible_day'] == 1]['hosp_id_day_key'].nunique()
    _pct = (
        eligible_days / total_vent_days_wo_paralytics * 100
        if total_vent_days_wo_paralytics > 0 else 0
    )
    print(f'Eligible days: {eligible_days} / {total_vent_days_wo_paralytics} ({_pct:.2f}%)')
    print(
        'Days with at least one IMV event:',
        final_df[final_df['device_category'] == 'imv']['hosp_id_day_key'].nunique(),
    )
    print(
        'Days with at least one IMV & ICU event:',
        final_df[
            (final_df['device_category'] == 'imv')
            & (final_df['location_category'] == 'icu')
        ]['hosp_id_day_key'].nunique(),
    )
    print('\nBy n = Encounter')
    _h_total = final_df['hospitalization_id'].nunique()
    _h_eligible = final_df[final_df['eligible_day'] == 1]['hospitalization_id'].nunique()
    _h_pct = _h_eligible / _h_total * 100 if _h_total > 0 else 0
    print(f'Eligible encounters: {_h_eligible} / {_h_total} ({_h_pct:.2f}%)')

    return eligible_days, final_df, total_vent_days


# ---------------------------------------------------------------------------
# Cell 9 — SBT delivery detection (FLIP check) + extubation flags
# ---------------------------------------------------------------------------
@app.cell
def _(final_df, sbt):
    final_df_1 = sbt.process_diagnostic_flip_sbt_optimized_v2(final_df)
    return (final_df_1,)


@app.cell
def _(final_df_1, sbt):
    final_df_2 = sbt.apply_2_45_extubated_flag(final_df_1)
    return (final_df_2,)


@app.cell
def _(final_df_2, sbt):
    final_df_3 = sbt.compute_time_to_extubation(final_df_2)
    return (final_df_3,)


# ---------------------------------------------------------------------------
# Cell 10 — Delta-to-extubation binning (per hospital + overall plot)
# ---------------------------------------------------------------------------
@app.cell
def _(directory_path, final_df_3, pd):
    _hospital_ids = final_df_3['hospital_id'].dropna().unique()
    _bins = list(range(0, 24 * 60 + 1, 60))
    _labels = [f'{i}-{i + 1}hr' for i in range(24)]
    _rush_summary = []
    for _hosp in _hospital_ids:
        _delta_series = (
            final_df_3[final_df_3['hospital_id'] == _hosp]['delta_to_extubation_mins']
            .dropna()
        )
        _delta_binned = pd.cut(_delta_series, bins=_bins, labels=_labels, right=False)
        _rush_counts = _delta_binned.value_counts().sort_index()
        _row = _rush_counts.to_dict()
        _row['hospital_id'] = _hosp
        _rush_summary.append(_row)
        pd.DataFrame(
            final_df_3[final_df_3['hospital_id'] == _hosp]['delta_to_extubation_mins']
            .describe()
        ).to_csv(
            f'{directory_path}/delta_stats_between_EHR30Min_Extubated_{_hosp}.csv'
        )
    _rush_df = pd.DataFrame(_rush_summary)
    _rush_df = _rush_df.fillna(0).astype(
        {col: 'int' for col in _rush_df.columns if col != 'hospital_id'}
    )
    _rush_df.to_csv(
        f'{directory_path}/rush_counts_by_hour_per_hospital.csv', index=False
    )
    return


@app.cell
def _(final_df_3, pd, plt):
    _delta_series = final_df_3.delta_to_extubation_mins.dropna()
    _bins = list(range(0, 24 * 60 + 1, 60))
    _labels = [f'{i}-{i + 1}hr' for i in range(24)]
    _delta_binned = pd.cut(_delta_series, bins=_bins, labels=_labels, right=False)
    _rush_counts = _delta_binned.value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    plt.plot(_rush_counts.index, _rush_counts.values, marker='o')
    plt.title('Overall: Count of Extubation Events per Hour Bin after EHR signature (30 mins)')
    plt.xlabel('Hours since event (binned)')
    plt.ylabel('Number of Extubations')
    plt.xticks(rotation=45)
    _ax = plt.gca()
    _ax.spines['top'].set_visible(False)
    _ax.spines['right'].set_visible(False)
    _ax.tick_params(direction='out')
    plt.tight_layout()
    plt.show()
    return


# ---------------------------------------------------------------------------
# Cell 11 — Day-level aggregation + documented-column normalization
# ---------------------------------------------------------------------------
@app.cell
def _(final_df_3, np):
    # Corrected _documented column logic (Standard notebook pattern):
    # sbt_delivery_pass_fail_documented == 1.0 when a value is present; NaN otherwise.
    # This avoids the old {0:1, 1:1} mapping that collapsed pass/fail into one category.
    final_df_3['sbt_bkp'] = final_df_3['sbt_delivery_pass_fail']
    final_df_3['sbt_delivery_pass_fail_documented'] = (
        final_df_3['sbt_delivery_pass_fail'].notna().astype(float)
    )
    final_df_3['sbt_delivery_pass_fail_documented'] = (
        final_df_3['sbt_delivery_pass_fail_documented'].replace(0, np.nan)
    )
    final_df_3['sbt_screen_pass_fail_documented'] = (
        final_df_3['sbt_screen_pass_fail'].notna().astype(float)
    )
    final_df_3['sbt_screen_pass_fail_documented'] = (
        final_df_3['sbt_screen_pass_fail_documented'].replace(0, np.nan)
    )
    final_df_3['flip_skip_reason'] = (
        final_df_3.groupby('hosp_id_day_key')['flip_skip_reason']
        .transform(lambda x: x.ffill().bfill())
    )
    return


@app.cell
def _(final_df_3):
    # Convert EHR delivery window columns to binary presence flags
    _datetime_cols = ['EHR_Delivery_2mins', 'EHR_Delivery_30mins']
    for _col in _datetime_cols:
        if _col in final_df_3.columns:
            final_df_3[_col] = final_df_3[_col].notna().astype(int)
    return


@app.cell
def _(final_df_3, pl):
    _pl_df = pl.from_pandas(final_df_3[
        ['hosp_id_day_key', 'hospitalization_id', 'hospital_id', 'eligible_day',
         'EHR_Delivery_2mins', 'EHR_Delivery_30mins', 'sbt_screen_pass_fail',
         'sbt_delivery_pass_fail', 'flag_2_45_extubated', 'flip_skip_reason',
         'extubated']
    ])
    _grouped_pl = (
        _pl_df
        .group_by('hosp_id_day_key')
        .agg(
            pl.col('hospitalization_id').first(),
            pl.col('hospital_id').drop_nulls().last(),
            pl.col('eligible_day').max(),
            pl.col('EHR_Delivery_2mins').max(),
            pl.col('EHR_Delivery_30mins').max(),
            pl.col('sbt_screen_pass_fail').max(),
            pl.col('sbt_delivery_pass_fail').max(),
            pl.col('flag_2_45_extubated').max(),
            pl.col('flip_skip_reason').drop_nulls().last(),
            pl.col('extubated').max(),
        )
    )
    grouped_df = _grouped_pl.to_pandas()
    mat_df = _grouped_pl.filter(pl.col('eligible_day') == 1).to_pandas()
    return grouped_df, mat_df


# ---------------------------------------------------------------------------
# Cell 12 — Hospital-level summary
# ---------------------------------------------------------------------------
@app.cell
def _(directory_path, grouped_df, pl):
    _gpl = pl.from_pandas(grouped_df)
    _eligible_pl = _gpl.filter(pl.col('eligible_day') == 1)

    def _count_flag(col: str) -> "pl.Expr":
        """Count distinct hosp_id_day_key where col equals 1."""
        return (
            pl.col('hosp_id_day_key')
            .filter(pl.col(col) == 1)
            .n_unique()
            .alias(col)
        )

    _summary_pl = _eligible_pl.group_by('hospital_id').agg(
        _count_flag('sbt_screen_pass_fail'),
        _count_flag('sbt_delivery_pass_fail'),
        _count_flag('EHR_Delivery_2mins'),
        _count_flag('EHR_Delivery_30mins'),
        _count_flag('extubated'),
        _count_flag('flag_2_45_extubated'),
        pl.col('hosp_id_day_key').n_unique().alias('df_eligible'),
    ).rename({
        'sbt_screen_pass_fail': 'sbt_screen_pass',
        'sbt_delivery_pass_fail': 'sbt_delivery_pass',
        'EHR_Delivery_2mins': 'ehr_2min',
        'EHR_Delivery_30mins': 'ehr_30min',
        'flag_2_45_extubated': 'ehr_2min_45min_extubated',
    })

    for _row in _summary_pl.iter_rows(named=True):
        print(f"\nHospital ID: {_row['hospital_id']}")
        print(f"  SBT Screen Pass              : {_row['sbt_screen_pass']}")
        print(f"  SBT Delivery Pass            : {_row['sbt_delivery_pass']}")
        print(f"  EHR 2-min Delivery           : {_row['ehr_2min']}")
        print(f"  EHR 30-min Delivery          : {_row['ehr_30min']}")
        print(f"  Extubated                    : {_row['extubated']}")
        print(f"  EHR 2min + 45min extubated   : {_row['ehr_2min_45min_extubated']}")
        print(f"  Eligible days                : {_row['df_eligible']}")

    _summary_pl.write_csv(
        f'{directory_path}/hospital_sbt_ehr_summary_within_eligible_day.csv',
    )
    return


# ---------------------------------------------------------------------------
# Cell 13 — Concordance analysis (confusion matrices + Cohen kappa)
# ---------------------------------------------------------------------------

# --- EHR 2-min vs SBT delivery flag ---
@app.cell
def _(
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    directory_path,
    f1_score,
    mat_df,
    pd,
    plt,
    precision_score,
    recall_score,
    sns,
):
    _hospital_ids = mat_df['hospital_id'].unique()
    mat_df['sbt_delivery_pass_fail'] = mat_df['sbt_delivery_pass_fail'].fillna(0)
    metrics_list_2min = []
    for _hosp in _hospital_ids:
        _df_hosp = mat_df[mat_df['hospital_id'] == _hosp]
        if _df_hosp['sbt_delivery_pass_fail'].nunique() <= 1:
            continue
        _conf_matrix = pd.crosstab(
            _df_hosp['EHR_Delivery_2mins'], _df_hosp['sbt_delivery_pass_fail']
        )
        _conf_matrix_percent = _conf_matrix / _conf_matrix.values.sum() * 100
        _annot = (
            _conf_matrix.astype(str) + '\n'
            + _conf_matrix_percent.round(1).astype(str) + '%'
        )
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            _conf_matrix, annot=_annot, fmt='', cmap='Blues',
            xticklabels=['0', '1'], yticklabels=['0', '1'],
        )
        plt.xlabel('SBT Delivery in Flowsheet')
        plt.ylabel('EHR Delivery in 2 minutes')
        plt.title(f'Confusion Matrix for Hospital {_hosp}')
        plt.savefig(f'{directory_path}/confusion_matrix_{_hosp}_by_SBT.png')
        plt.close()
        _y_true = _df_hosp['EHR_Delivery_2mins']
        _y_pred = _df_hosp['sbt_delivery_pass_fail']
        _tn, _fp, _fn, _tp = confusion_matrix(_y_true, _y_pred).ravel()
        _accuracy = accuracy_score(_y_true, _y_pred)
        _precision = precision_score(_y_true, _y_pred, zero_division=0)
        _recall = recall_score(_y_true, _y_pred, zero_division=0)
        _f1 = f1_score(_y_true, _y_pred, zero_division=0)
        _specificity = _tn / (_tn + _fp) if (_tn + _fp) != 0 else 0
        _kappa = cohen_kappa_score(_y_true, _y_pred)
        print(f'Hospital ID : {_hosp}')
        print(f'Accuracy    : {_accuracy:.3f}')
        print(f'Precision   : {_precision:.3f}')
        print(f'Recall      : {_recall:.3f}')
        print(f'F1 Score    : {_f1:.3f}')
        print(f'Specificity : {_specificity:.3f}')
        print(f'Cohen Kappa : {_kappa:.3f}\n')
        _metrics_dict = {
            'True Positives (TP)': _tp, 'False Positives (FP)': _fp,
            'False Negatives (FN)': _fn, 'True Negatives (TN)': _tn,
            'Accuracy': _accuracy, 'Precision': _precision,
            'Recall': _recall, 'F1 Score': _f1,
            'Specificity': _specificity, 'Cohen_Kappa': _kappa,
        }
        _df_metrics = pd.DataFrame(
            list(_metrics_dict.items()), columns=['Metric', 'Value']
        )
        _df_metrics.to_csv(
            f'{directory_path}/EHR_2min_vs_SBT_metrics_{_hosp}.csv', index=False
        )
        metrics_list_2min.append({
            'Column': 'EHR_Delivery_2mins', 'hospital_id': _hosp,
            'TP': _tp, 'FP': _fp, 'FN': _fn, 'TN': _tn,
            'Accuracy': _accuracy, 'Precision': _precision,
            'Recall': _recall, 'F1 Score': _f1,
            'Specificity': _specificity, 'Cohen_Kappa': _kappa,
        })
    if metrics_list_2min:
        pd.DataFrame(metrics_list_2min).to_csv(
            f'{directory_path}/delivery_concordance_summary_2min.csv', index=False
        )
    return (metrics_list_2min,)


# --- EHR 2-min vs Extubated flag ---
@app.cell
def _(
    accuracy_score,
    confusion_matrix,
    directory_path,
    f1_score,
    mat_df,
    pd,
    plt,
    precision_score,
    recall_score,
    sns,
):
    _hospital_ids = mat_df['hospital_id'].unique()
    mat_df['extubated'] = mat_df['extubated'].fillna(0)
    for _hosp in _hospital_ids:
        _df_hosp = mat_df[mat_df['hospital_id'] == _hosp]
        _conf_matrix = pd.crosstab(_df_hosp['EHR_Delivery_2mins'], _df_hosp['extubated'])
        _conf_matrix_percent = _conf_matrix / _conf_matrix.values.sum() * 100
        _annot = (
            _conf_matrix.astype(str) + '\n'
            + _conf_matrix_percent.round(1).astype(str) + '%'
        )
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            _conf_matrix, annot=_annot, fmt='', cmap='Blues',
            xticklabels=['0', '1'], yticklabels=['0', '1'],
        )
        plt.xlabel('Extubated')
        plt.ylabel('EHR Delivery in 2 minutes')
        plt.title(f'Confusion Matrix for Hospital {_hosp}')
        plt.savefig(f'{directory_path}/confusion_matrix_{_hosp}_by_extubated.png')
        plt.close()
        _y_true = _df_hosp['EHR_Delivery_2mins']
        _y_pred = _df_hosp['extubated']
        _tn, _fp, _fn, _tp = confusion_matrix(_y_true, _y_pred).ravel()
        _accuracy = accuracy_score(_y_true, _y_pred)
        _precision = precision_score(_y_true, _y_pred, zero_division=0)
        _recall = recall_score(_y_true, _y_pred, zero_division=0)
        _f1 = f1_score(_y_true, _y_pred, zero_division=0)
        _specificity = _tn / (_tn + _fp) if (_tn + _fp) != 0 else 0
        print(f'Hospital ID : {_hosp}')
        print(f'Accuracy    : {_accuracy:.3f}')
        print(f'Precision   : {_precision:.3f}')
        print(f'Recall      : {_recall:.3f}')
        print(f'F1 Score    : {_f1:.3f}')
        print(f'Specificity : {_specificity:.3f}\n')
        _metrics_dict = {
            'True Positives (TP)': _tp, 'False Positives (FP)': _fp,
            'False Negatives (FN)': _fn, 'True Negatives (TN)': _tn,
            'Accuracy': _accuracy, 'Precision': _precision,
            'Recall': _recall, 'F1 Score': _f1, 'Specificity': _specificity,
        }
        _df_metrics = pd.DataFrame(
            list(_metrics_dict.items()), columns=['Metric', 'Value']
        )
        _df_metrics.to_csv(
            f'{directory_path}/EHR_2min_VS_EXTUBATED_metrics_{_hosp}.csv', index=False
        )
    return


# --- EHR 30-min vs Extubated flag ---
@app.cell
def _(
    accuracy_score,
    confusion_matrix,
    directory_path,
    f1_score,
    mat_df,
    pd,
    plt,
    precision_score,
    recall_score,
    sns,
):
    _hospital_ids = mat_df['hospital_id'].unique()
    mat_df['extubated'] = mat_df['extubated'].fillna(0)
    for _hosp in _hospital_ids:
        _df_hosp = mat_df[mat_df['hospital_id'] == _hosp]
        _conf_matrix = pd.crosstab(_df_hosp['EHR_Delivery_30mins'], _df_hosp['extubated'])
        _conf_matrix_percent = _conf_matrix / _conf_matrix.values.sum() * 100
        _annot = (
            _conf_matrix.astype(str) + '\n'
            + _conf_matrix_percent.round(1).astype(str) + '%'
        )
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            _conf_matrix, annot=_annot, fmt='', cmap='Blues',
            xticklabels=['0', '1'], yticklabels=['0', '1'],
        )
        plt.xlabel('Extubated')
        plt.ylabel('EHR Delivery in 30 minutes')
        plt.title(f'Confusion Matrix for Hospital {_hosp}')
        plt.savefig(f'{directory_path}/ehr_30_confusion_matrix_{_hosp}_by_extubated.png')
        plt.close()
        _y_true = _df_hosp['EHR_Delivery_30mins']
        _y_pred = _df_hosp['extubated']
        _tn, _fp, _fn, _tp = confusion_matrix(_y_true, _y_pred).ravel()
        _accuracy = accuracy_score(_y_true, _y_pred)
        _precision = precision_score(_y_true, _y_pred, zero_division=0)
        _recall = recall_score(_y_true, _y_pred, zero_division=0)
        _f1 = f1_score(_y_true, _y_pred, zero_division=0)
        _specificity = _tn / (_tn + _fp) if (_tn + _fp) != 0 else 0
        print(f'Hospital ID : {_hosp}')
        print(f'Accuracy    : {_accuracy:.3f}')
        print(f'Precision   : {_precision:.3f}')
        print(f'Recall      : {_recall:.3f}')
        print(f'F1 Score    : {_f1:.3f}')
        print(f'Specificity : {_specificity:.3f}\n')
        _metrics_dict = {
            'True Positives (TP)': _tp, 'False Positives (FP)': _fp,
            'False Negatives (FN)': _fn, 'True Negatives (TN)': _tn,
            'Accuracy': _accuracy, 'Precision': _precision,
            'Recall': _recall, 'F1 Score': _f1, 'Specificity': _specificity,
        }
        _df_metrics = pd.DataFrame(
            list(_metrics_dict.items()), columns=['Metric', 'Value']
        )
        _df_metrics.to_csv(
            f'{directory_path}/EHR_30_VS_EXTUBATED_metrics_{_hosp}.csv', index=False
        )
    return


# --- EHR 30-min vs SBT delivery flag ---
@app.cell
def _(
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    directory_path,
    f1_score,
    mat_df,
    pd,
    plt,
    precision_score,
    recall_score,
    sns,
):
    _hospital_ids = mat_df['hospital_id'].unique()
    mat_df['sbt_delivery_pass_fail'] = mat_df['sbt_delivery_pass_fail'].fillna(0)
    metrics_list_30min = []
    for _hosp in _hospital_ids:
        _df_hosp = mat_df[mat_df['hospital_id'] == _hosp]
        if _df_hosp['sbt_delivery_pass_fail'].nunique() <= 1:
            continue
        _conf_matrix = pd.crosstab(
            _df_hosp['EHR_Delivery_30mins'], _df_hosp['sbt_delivery_pass_fail']
        )
        _conf_matrix_percent = _conf_matrix / _conf_matrix.values.sum() * 100
        _annot = (
            _conf_matrix.astype(str) + '\n'
            + _conf_matrix_percent.round(1).astype(str) + '%'
        )
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            _conf_matrix, annot=_annot, fmt='', cmap='Blues',
            xticklabels=['0', '1'], yticklabels=['0', '1'],
        )
        plt.xlabel('SBT Delivery in Flowsheet')
        plt.ylabel('EHR Delivery in 30 minutes')
        plt.title(f'Confusion Matrix for Hospital {_hosp}')
        plt.savefig(f'{directory_path}/ehr_30_confusion_matrix_{_hosp}_by_SBT.png')
        plt.close()
        _y_true = _df_hosp['EHR_Delivery_30mins']
        _y_pred = _df_hosp['sbt_delivery_pass_fail']
        _tn, _fp, _fn, _tp = confusion_matrix(_y_true, _y_pred).ravel()
        _accuracy = accuracy_score(_y_true, _y_pred)
        _precision = precision_score(_y_true, _y_pred, zero_division=0)
        _recall = recall_score(_y_true, _y_pred, zero_division=0)
        _f1 = f1_score(_y_true, _y_pred, zero_division=0)
        _specificity = _tn / (_tn + _fp) if (_tn + _fp) != 0 else 0
        _kappa = cohen_kappa_score(_y_true, _y_pred)
        print(f'Hospital ID : {_hosp}')
        print(f'Accuracy    : {_accuracy:.3f}')
        print(f'Precision   : {_precision:.3f}')
        print(f'Recall      : {_recall:.3f}')
        print(f'F1 Score    : {_f1:.3f}')
        print(f'Specificity : {_specificity:.3f}')
        print(f'Cohen Kappa : {_kappa:.3f}\n')
        _metrics_dict = {
            'True Positives (TP)': _tp, 'False Positives (FP)': _fp,
            'False Negatives (FN)': _fn, 'True Negatives (TN)': _tn,
            'Accuracy': _accuracy, 'Precision': _precision,
            'Recall': _recall, 'F1 Score': _f1,
            'Specificity': _specificity, 'Cohen_Kappa': _kappa,
        }
        _df_metrics = pd.DataFrame(
            list(_metrics_dict.items()), columns=['Metric', 'Value']
        )
        _df_metrics.to_csv(
            f'{directory_path}/EHR_30_vs_SBT_metrics_{_hosp}.csv', index=False
        )
        metrics_list_30min.append({
            'Column': 'EHR_Delivery_30mins', 'hospital_id': _hosp,
            'TP': _tp, 'FP': _fp, 'FN': _fn, 'TN': _tn,
            'Accuracy': _accuracy, 'Precision': _precision,
            'Recall': _recall, 'F1 Score': _f1,
            'Specificity': _specificity, 'Cohen_Kappa': _kappa,
        })
    if metrics_list_30min:
        pd.DataFrame(metrics_list_30min).to_csv(
            f'{directory_path}/delivery_concordance_summary_30min.csv', index=False
        )
    return (metrics_list_30min,)


# ---------------------------------------------------------------------------
# Cell 14 — Failure-to-detect analysis (FLIP vs SBT flag + vs Extubated)
# ---------------------------------------------------------------------------

def _build_failure_summary(
    final_df_3_arg,
    mat_df_arg,
    eligible_days_arg,
    reference_col: str,
    directory_path_arg: str,
    pd_arg,
    prefix: str,
) -> None:
    """
    Shared logic for failure-to-detect analysis.

    Parameters
    ----------
    reference_col : str
        Either 'sbt_delivery_pass_fail' or 'extubated'.
    prefix : str
        File-name prefix, e.g. 'EHR_VS_SBT' or 'EHR_VS_EXTUBATED'.
    """
    _condition_cols = [
        'flip_skip_reason',
        'cond_device_imv',
        'cond_location_icu',
        'cond_peep_set_le8',
        'cond_ps_set_le8',
        'cond_mode_ps_cpap',
    ]
    for _hosp in mat_df_arg['hospital_id'].unique():
        _mat_hosp = mat_df_arg[mat_df_arg['hospital_id'] == _hosp]
        _filtered_keys = _mat_hosp.loc[
            (_mat_hosp['EHR_Delivery_2mins'] == 0)
            & (_mat_hosp[reference_col] == 1),
            'hosp_id_day_key',
        ].unique()
        _fdf = final_df_3_arg.loc[
            (final_df_3_arg[reference_col] == 1)
            & final_df_3_arg['hosp_id_day_key'].isin(_filtered_keys)
        ]
        _fdf = (
            _fdf.sort_values('event_time')
            .drop_duplicates(subset='hosp_id_day_key', keep='first')
        )
        print(f'Hospital: {_hosp}, filtered shape: {_fdf.shape}')

        # Dependent (waterfall) summary
        _dep_results = []
        _df_rem = _fdf.copy()
        for _col in _condition_cols:
            _step = _df_rem[~_df_rem[_col].isna()]
            _dep_results.append({
                'Step': f'Step {_condition_cols.index(_col) + 1}',
                'FilterColumn': _col,
                'UniqueKeys': _step['hosp_id_day_key'].nunique(),
                'RowCount': _step.shape[0],
                'ValueCounts': _step[_col].value_counts(dropna=False).to_dict(),
            })
            _df_rem = _df_rem[~_df_rem['hosp_id_day_key'].isin(_step['hosp_id_day_key'])]
        _dep_results.append({
            'Step': 'Step 7 (Unmatched)',
            'FilterColumn': None,
            'UniqueKeys': _df_rem['hosp_id_day_key'].nunique(),
            'RowCount': _df_rem.shape[0],
            'ValueCounts': None,
        })
        _dep_df = pd_arg.DataFrame(_dep_results)
        _total_dep = _dep_df['UniqueKeys'].sum()
        _dep_df['% by eligible_days'] = _dep_df['UniqueKeys'].apply(
            lambda x: round(x / eligible_days_arg * 100, 2)
        )
        _dep_df['% of Total'] = _dep_df['UniqueKeys'].apply(
            lambda x: round(x / _total_dep * 100, 2) if _total_dep != 0 else 0
        )
        _out = f'{directory_path_arg}/{prefix}_failure_dependent_summary_{_hosp}.csv'
        _dep_df.to_csv(_out, index=False)
        print(f'Saved dependent summary: {_out}')
        print(_hosp, _dep_df, '\n')

        # Independent (non-exclusive) summary
        _step_dfs = [_fdf[~_fdf[_col].isna()] for _col in _condition_cols]
        _matched_keys = set().union(*[s['hosp_id_day_key'] for s in _step_dfs])
        _unmatched = _fdf[~_fdf['hosp_id_day_key'].isin(_matched_keys)]
        _failure_counts = {
            col: _step_dfs[i]['hosp_id_day_key'].nunique()
            for i, col in enumerate(_condition_cols)
        }
        _failure_counts['unmatched'] = _unmatched['hosp_id_day_key'].nunique()
        _vcounts_map = {
            col: _step_dfs[i][col].value_counts(dropna=False).to_dict()
            for i, col in enumerate(_condition_cols)
        }
        _vcounts_map['unmatched'] = None
        _total_ind = sum(_failure_counts.values())
        _ind_rows = []
        for _reason, _count in _failure_counts.items():
            _ind_rows.append({
                'Failure Reason': _reason,
                'Count': _count,
                '% by eligible_days': round(_count / eligible_days_arg * 100, 2),
                '% of Total (out of total failed cases)': (
                    round(_count / _total_ind * 100, 2) if _total_ind else 0
                ),
                'Value Counts': _vcounts_map[_reason],
            })
        _ind_df = (
            pd_arg.DataFrame(_ind_rows)
            .sort_values(by='Count', ascending=False)
            .reset_index(drop=True)
        )
        _ind_out = f'{directory_path_arg}/{prefix}_failure_independent_summary_hospital_{_hosp}.csv'
        _ind_df.to_csv(_ind_out, index=False)
        print(f'Saved independent summary: {_ind_out}')
        print(_hosp, _ind_df, '\n')


@app.cell
def _(directory_path, eligible_days, final_df_3, mat_df, pd):
    _build_failure_summary(
        final_df_3, mat_df, eligible_days,
        reference_col='sbt_delivery_pass_fail',
        directory_path_arg=directory_path,
        pd_arg=pd,
        prefix='EHR_VS_SBT',
    )
    return


@app.cell
def _(directory_path, eligible_days, final_df_3, mat_df, pd):
    _build_failure_summary(
        final_df_3, mat_df, eligible_days,
        reference_col='extubated',
        directory_path_arg=directory_path,
        pd_arg=pd,
        prefix='EHR_VS_EXTUBATED',
    )
    return


# ---------------------------------------------------------------------------
# Cell 15 — Time-of-day distribution plots
# ---------------------------------------------------------------------------
@app.cell
def _(directory_path, final_df_3, pl, plt):
    _hospital_ids = final_df_3['hospital_id'].dropna().unique()
    _summary_list = []
    for _hosp in _hospital_ids:
        _fh = final_df_3[final_df_3['hospital_id'] == _hosp]
        _sbt_d_time = (
            _fh[(_fh['sbt_delivery_pass_fail'] == 1) & (_fh['eligible_day'] == 1)]
            .sort_values(['hosp_id_day_key', 'event_time'])
            .groupby('hosp_id_day_key', as_index=False)
            .first()[['hosp_id_day_key', 'event_time']]
        )
        _ehr_d_time = (
            _fh[(_fh['EHR_Delivery_2mins'] == 1) & (_fh['eligible_day'] == 1)]
            [['hosp_id_day_key', 'event_time']]
            .drop_duplicates()
        )
        _sbt_hours = _sbt_d_time['event_time'].dt.hour
        _ehr_hours = _ehr_d_time['event_time'].dt.hour
        plt.figure(figsize=(10, 6))
        plt.hist(_sbt_hours, bins=range(0, 25), alpha=0.5,
                 label='SBT Delivery Time', edgecolor='black')
        plt.hist(_ehr_hours, bins=range(0, 25), alpha=0.5,
                 label='EHR Delivery Time', edgecolor='black')
        plt.xlabel('Hour of Day')
        plt.ylabel('Frequency')
        plt.title(f'Event Time Distribution (Hourly) - Hospital {_hosp}')
        plt.legend()
        _ax = plt.gca()
        _ax.spines['top'].set_visible(False)
        _ax.spines['right'].set_visible(False)
        _ax.tick_params(direction='out')
        plt.savefig(f'{directory_path}/event_time_distribution_hospital_{_hosp}.png')
        plt.close()
        _sbt_counts = _sbt_hours.value_counts().sort_index()
        _ehr_counts = _ehr_hours.value_counts().sort_index()
        _hours_pl = (
            pl.DataFrame({'hour': list(range(24))})
            .join(
                pl.DataFrame({'hour': _sbt_counts.index.tolist(), 'SBT_Delivery': _sbt_counts.values.tolist()}),
                on='hour', how='left',
            )
            .join(
                pl.DataFrame({'hour': _ehr_counts.index.tolist(), 'EHR_Delivery': _ehr_counts.values.tolist()}),
                on='hour', how='left',
            )
            .with_columns(
                pl.col('SBT_Delivery').fill_null(0).cast(pl.Int64),
                pl.col('EHR_Delivery').fill_null(0).cast(pl.Int64),
                pl.lit(_hosp).alias('hospital_id'),
            )
        )
        _summary_list.append(_hours_pl)
    pl.concat(_summary_list).write_csv(
        f'{directory_path}/event_time_distribution_summary.csv',
    )
    print('SBT vs EHR overlay plots created.')
    return


@app.cell
def _(directory_path, final_df_3, pl, plt):
    _hospital_ids = final_df_3['hospital_id'].dropna().unique()
    _summary_list = []
    for _hosp in _hospital_ids:
        _fh = final_df_3[final_df_3['hospital_id'] == _hosp]
        _ext_d_time = (
            _fh[(_fh['extubated'] == 1) & (_fh['eligible_day'] == 1)]
            .sort_values(['hosp_id_day_key', 'event_time'])
            .groupby('hosp_id_day_key', as_index=False)
            .first()[['hosp_id_day_key', 'event_time']]
        )
        _ehr_d_time = (
            _fh[(_fh['EHR_Delivery_2mins'] == 1) & (_fh['eligible_day'] == 1)]
            [['hosp_id_day_key', 'event_time']]
            .drop_duplicates()
        )
        _ext_hours = _ext_d_time['event_time'].dt.hour
        _ehr_hours = _ehr_d_time['event_time'].dt.hour
        plt.figure(figsize=(10, 6))
        plt.hist(_ext_hours, bins=range(0, 25), alpha=0.5,
                 label='Extubated Time', edgecolor='black')
        plt.hist(_ehr_hours, bins=range(0, 25), alpha=0.5,
                 label='EHR Delivery Time', edgecolor='black')
        plt.xlabel('Hour of Day')
        plt.ylabel('Frequency')
        plt.title(f'Event Time Distribution (Hourly) - Hospital {_hosp}')
        plt.legend()
        _ax = plt.gca()
        _ax.spines['top'].set_visible(False)
        _ax.spines['right'].set_visible(False)
        _ax.tick_params(direction='out')
        plt.savefig(
            f'{directory_path}/event_time_distribution_hospital_{_hosp}_by_ex.png'
        )
        plt.close()
        _ext_counts = _ext_hours.value_counts().sort_index()
        _ehr_counts = _ehr_hours.value_counts().sort_index()
        _hours_pl = (
            pl.DataFrame({'hour': list(range(24))})
            .join(
                pl.DataFrame({'hour': _ext_counts.index.tolist(), 'SBT_Delivery': _ext_counts.values.tolist()}),
                on='hour', how='left',
            )
            .join(
                pl.DataFrame({'hour': _ehr_counts.index.tolist(), 'EHR_Delivery': _ehr_counts.values.tolist()}),
                on='hour', how='left',
            )
            .with_columns(
                pl.col('SBT_Delivery').fill_null(0).cast(pl.Int64),
                pl.col('EHR_Delivery').fill_null(0).cast(pl.Int64),
                pl.lit(_hosp).alias('hospital_id'),
            )
        )
        _summary_list.append(_hours_pl)
    pl.concat(_summary_list).write_csv(
        f'{directory_path}/event_time_distribution_summary_by_ex.csv',
    )
    print('Extubated vs EHR overlay plots created.')
    return


# ---------------------------------------------------------------------------
# Cell 16 — Final summary stats
# ---------------------------------------------------------------------------
@app.cell
def _(directory_path, final_df_3, mat_df, pl):
    _fpl = pl.from_pandas(final_df_3[
        ['hosp_id_day_key', 'hospitalization_id', 'eligible_day',
         'vent_day_without_paralytics', 'vent_day', 'device_category',
         'location_category']
    ])
    _total_days = _fpl.select(pl.col('hosp_id_day_key').n_unique()).item()
    _eligible_days_1 = (
        _fpl.filter(pl.col('eligible_day') == 1)
        .select(pl.col('hosp_id_day_key').n_unique()).item()
    )
    _imv_days = (
        _fpl.filter(pl.col('vent_day_without_paralytics') == 1)
        .select(pl.col('hosp_id_day_key').n_unique()).item()
    )
    _imv_days_no_filter = (
        _fpl.filter(pl.col('vent_day') == 1)
        .select(pl.col('hosp_id_day_key').n_unique()).item()
    )
    _pct = _eligible_days_1 / _imv_days * 100 if _imv_days > 0 else 0
    _imv_icu_days = (
        _fpl.filter(
            (pl.col('device_category') == 'imv')
            & (pl.col('location_category') == 'icu')
        ).select(pl.col('hosp_id_day_key').n_unique()).item()
    )
    _h_total = _fpl.select(pl.col('hospitalization_id').n_unique()).item()
    _h_eligible = (
        _fpl.filter(pl.col('eligible_day') == 1)
        .select(pl.col('hospitalization_id').n_unique()).item()
    )
    _h_pct = _h_eligible / _h_total * 100 if _h_total > 0 else 0
    _h_imv = (
        _fpl.filter(pl.col('device_category') == 'imv')
        .select(pl.col('hospitalization_id').n_unique()).item()
    )
    _h_imv_icu = (
        _fpl.filter(
            (pl.col('device_category') == 'imv')
            & (pl.col('location_category') == 'icu')
        ).select(pl.col('hospitalization_id').n_unique()).item()
    )

    # Distribution stats from mat_df (already small)
    _mpl = pl.from_pandas(mat_df[
        ['extubated', 'EHR_Delivery_2mins', 'sbt_delivery_pass_fail']
    ])
    _ext = _mpl.filter(pl.col('extubated') == 1)
    _n_ext = _ext.height
    _ehr_vc = (
        _ext.group_by('EHR_Delivery_2mins').len()
        .with_columns((pl.col('len') / _n_ext * 100).alias('pct'))
    )
    _sbt_vc = (
        _ext.group_by('sbt_delivery_pass_fail').len()
        .with_columns((pl.col('len') / _n_ext * 100).alias('pct'))
    )

    print('By n = Days')
    print('Total days:', _total_days)
    print(f'Eligible days: {_eligible_days_1} / {_imv_days} ({_pct:.2f}%)')
    print('IMV days (without paralytics):', _imv_days)
    print('IMV + ICU days:', _imv_icu_days)
    print('\nBy n = Encounter')
    print('Total encounters:', _h_total)
    print(f'Eligible encounters: {_h_eligible} / {_h_total} ({_h_pct:.2f}%)')
    print('IMV encounters:', _h_imv)
    print('IMV + ICU encounters:', _h_imv_icu)
    print('\nEHR_Delivery_2mins distribution (extubated == 1):')
    print(_ehr_vc)
    print('\nsbt_delivery_pass_fail distribution (extubated == 1):')
    print(_sbt_vc)

    # Build stats output with polars
    _stats_pl = pl.DataFrame({
        'Metric': [
            'total_days', 'eligible_days', 'eligible_percentage',
            'imv_days_without_paralytics', 'imv_icu_days', 'imv_days_no_filter',
            'enc_total_days', 'enc_eligible_days', 'enc_eligible_percentage',
            'enc_imv_days', 'enc_imv_icu_days',
        ],
        'Value': [
            float(_total_days), float(_eligible_days_1), _pct,
            float(_imv_days), float(_imv_icu_days), float(_imv_days_no_filter),
            float(_h_total), float(_h_eligible), _h_pct,
            float(_h_imv), float(_h_imv_icu),
        ],
    })
    _ehr_stats = _ehr_vc.select(
        (pl.lit('EHR_Delivery_2mins_')
         + pl.col('EHR_Delivery_2mins').cast(pl.Utf8)
         + pl.lit('_extubated=1')).alias('Metric'),
        pl.col('pct').alias('Value'),
    )
    _sbt_stats = _sbt_vc.select(
        (pl.lit('sbt_delivery_pass_fail_')
         + pl.col('sbt_delivery_pass_fail').cast(pl.Utf8)
         + pl.lit('_extubated=1')).alias('Metric'),
        pl.col('pct').alias('Value'),
    )
    pl.concat([_stats_pl, _ehr_stats, _sbt_stats]).write_csv(
        f'{directory_path}/stats_df.csv',
    )
    print('\nFinal stats saved.')
    return


# ---------------------------------------------------------------------------
# Cell 17 — Table 1 generation
# ---------------------------------------------------------------------------
@app.cell
def _(np, pd, re, t1_cohort):
    def documented(series: "pd.Series") -> str:
        """Return 'Documented' if any non-null value exists, else 'Not Documented'."""
        return 'Documented' if series.notna().any() else 'Not Documented'

    def age_bucket(mean_age: float) -> str | None:
        """Bin a numeric age into decade-span categories."""
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

    def categorize_language(lang: str) -> str:
        """Collapse language to English / Spanish / Other."""
        if re.search('english', str(lang), re.IGNORECASE):
            return 'English'
        elif re.search('spanish', str(lang), re.IGNORECASE):
            return 'Spanish'
        else:
            return 'Other'

    medication_columns = [
        'rass', 'gcs_total', 'cisatracurium', 'vecuronium', 'rocuronium',
        'dobutamine', 'dopamine', 'epinephrine', 'fentanyl', 'hydromorphone',
        'isoproterenol', 'lorazepam', 'midazolam', 'milrinone', 'morphine',
        'norepinephrine', 'phenylephrine', 'propofol', 'vasopressin', 'angiotensin',
    ]
    demographic_columns = [
        'sex_category', 'race_category', 'ethnicity_category', 'language_name',
    ]
    continuous_cols = medication_columns + ['bmi']

    _drugs = [
        'cisatracurium', 'vecuronium', 'rocuronium', 'dobutamine', 'dopamine',
        'epinephrine', 'fentanyl', 'hydromorphone', 'isoproterenol', 'lorazepam',
        'midazolam', 'milrinone', 'morphine', 'norepinephrine', 'phenylephrine',
        'propofol', 'vasopressin', 'angiotensin',
    ]
    # Zero-dose entries are not "documented" as administered
    t1_cohort[_drugs] = t1_cohort[_drugs].applymap(
        lambda x: x if (pd.notna(x) and x > 0) else np.nan
    )
    t1_cohort['bmi'] = t1_cohort['weight_kg'] / (t1_cohort['height_cm'] / 100) ** 2
    t1_cohort['language_name'] = t1_cohort['language_name'].apply(categorize_language)
    t1_cohort[continuous_cols] = t1_cohort[continuous_cols].astype(float)

    return age_bucket, continuous_cols, demographic_columns, documented, medication_columns


# --- Table 1: categorical by hospitalization_id / patient_id ---
@app.cell
def _(
    age_bucket,
    demographic_columns,
    directory_path,
    documented,
    medication_columns,
    sbt,
    t1_cohort,
):
    for _x in ['hospitalization_id', 'patient_id']:
        _t1_summary = t1_cohort.groupby(_x).agg(
            age_at_admission=('age_at_admission', 'mean'),
            **{col: (col, documented) for col in medication_columns},
            **{col: (col, 'first') for col in demographic_columns},
        )
        _t1_summary['age_bucket'] = _t1_summary['age_at_admission'].apply(age_bucket)
        _t1_summary = _t1_summary.drop(columns=['age_at_admission']).reset_index()
        _summary_df = sbt.manual_categorical_tableone(
            _t1_summary, medication_columns + demographic_columns + ['age_bucket']
        )
        _fname = (
            f'{directory_path}/table1_hospitalization_id_categorical.csv'
            if _x == 'hospitalization_id'
            else f'{directory_path}/table1_patient_id_categorical.csv'
        )
        _summary_df.to_csv(_fname, index=False)
    return


# --- Table 1: continuous by hospitalization_id / patient_id ---
@app.cell
def _(continuous_cols, directory_path, sbt, t1_cohort):
    _hosp_agg = (
        t1_cohort.groupby('hospitalization_id')
        .agg({c: 'median' for c in continuous_cols})
        .reset_index()
    )
    _patient_agg = (
        t1_cohort.groupby('patient_id')
        .agg({c: 'median' for c in continuous_cols})
        .reset_index()
    )
    sbt.manual_tableone(_hosp_agg, continuous_cols).to_csv(
        f'{directory_path}/table1_hospitalization_id_continuous.csv', index=False
    )
    sbt.manual_tableone(_patient_agg, continuous_cols).to_csv(
        f'{directory_path}/table1_patient_id_continuous.csv', index=False
    )
    return


# --- Table 1: categorical by flag/day strata ---
@app.cell
def _(
    age_bucket,
    demographic_columns,
    directory_path,
    documented,
    final_df_3,
    medication_columns,
    sbt,
    t1_cohort,
    tqdm,
):
    for _x in tqdm(
        ['vent_day', 'vent_day_without_paralytics', 'eligible_day',
         'EHR_Delivery_2mins', 'EHR_Delivery_30mins'],
        desc='Table 1 categorical (day strata)',
    ):
        _ids = final_df_3[final_df_3[_x] == 1]['hosp_id_day_key'].unique()
        _sub = t1_cohort[t1_cohort['hosp_id_day_key'].isin(_ids)]
        _t1_summary = _sub.groupby('hosp_id_day_key').agg(
            age_at_admission=('age_at_admission', 'mean'),
            **{col: (col, documented) for col in medication_columns},
            **{col: (col, 'first') for col in demographic_columns},
        )
        _t1_summary['age_bucket'] = _t1_summary['age_at_admission'].apply(age_bucket)
        _t1_summary = _t1_summary.drop(columns=['age_at_admission']).reset_index()
        _summary_df = sbt.manual_categorical_tableone(
            _t1_summary, medication_columns + demographic_columns + ['age_bucket']
        )
        _summary_df.to_csv(f'{directory_path}/table1_{_x}_categorical.csv', index=False)
    return


# --- Table 1: continuous by flag/day strata ---
@app.cell
def _(continuous_cols, directory_path, final_df_3, sbt, t1_cohort, tqdm):
    for _x in tqdm(
        ['vent_day', 'vent_day_without_paralytics', 'eligible_day',
         'EHR_Delivery_2mins', 'EHR_Delivery_30mins'],
        desc='Table 1 continuous (day strata)',
    ):
        _ids = final_df_3.loc[final_df_3[_x] == 1, 'hosp_id_day_key'].unique()
        _sub = t1_cohort[t1_cohort['hosp_id_day_key'].isin(_ids)]
        _day_agg = (
            _sub.groupby('hosp_id_day_key')
            .agg({c: 'median' for c in continuous_cols})
            .reset_index()
        )
        sbt.manual_tableone(_day_agg, continuous_cols).to_csv(
            f'{directory_path}/table1_{_x}_continuous.csv', index=False
        )
    return


# ---------------------------------------------------------------------------
# Cell 18 — SOFA computation
# ---------------------------------------------------------------------------
@app.cell
def _():
    import pySofa as sofa

    continuous_cols_sofa = [
        'sofa_cv_97', 'sofa_coag', 'sofa_liver',
        'sofa_resp_pf', 'sofa_resp_pf_imp', 'sofa_resp',
        'sofa_cns', 'sofa_renal', 'sofa_total',
    ]
    return continuous_cols_sofa, sofa


@app.cell
def _(pl):
    _mapping_pl = pl.read_csv(
        '../output/intermediate/hospitalization_to_block_df.csv',
    ).with_columns(
        pl.col('hospitalization_id').cast(pl.Utf8),
        pl.col('encounter_block').cast(pl.Utf8),
    )
    # Convert to pandas for downstream sofa/pySBT compatibility
    mapping_ids = _mapping_pl.to_pandas()
    mapping_ids.head()
    return (mapping_ids,)


@app.cell
def _(cohort_1, pc):
    encounter_level_sofa = (
        cohort_1[['hospitalization_id', 'admission_dttm', 'discharge_dttm']]
        .drop_duplicates()
        .rename(columns={'admission_dttm': 'start_dttm', 'discharge_dttm': 'stop_dttm'})
    )
    encounter_level_sofa = pc.convert_datetime_columns_to_site_tz(
        encounter_level_sofa, pc.helper['your_site_timezone']
    )
    encounter_level_sofa.head()
    return (encounter_level_sofa,)


@app.cell
def _(
    continuous_cols_sofa,
    directory_path,
    encounter_level_sofa,
    mapping_ids,
    sbt,
    sofa,
):
    _sout = sofa.compute_sofa(
        encounter_level_sofa,
        tables_path=None,
        use_hospitalization_id=False,
        id_mapping=mapping_ids,
        group_by_id='encounter_block',
    )
    sbt.manual_tableone(_sout, continuous_cols_sofa).to_csv(
        f'{directory_path}/encounter_level_sofa_t1.csv', index=False
    )
    return


@app.cell
def _(
    continuous_cols_sofa,
    directory_path,
    final_df_3,
    mapping_ids,
    pc,
    pd,
    sbt,
    sofa,
    tqdm,
):
    for _x in tqdm(
        ['vent_day', 'vent_day_without_paralytics', 'eligible_day',
         'EHR_Delivery_2mins', 'EHR_Delivery_30mins', 'sbt_delivery_pass_fail'],
        desc='Generating SOFA Table 1 per flag',
    ):
        _day_df = (
            final_df_3[final_df_3[_x] == 1]
            [['hospitalization_id', 'hosp_id_day_key', 'current_day']]
            .drop_duplicates()
        )
        if _day_df.empty:
            continue
        _day_df['start_dttm'] = pd.to_datetime(_day_df['current_day']).dt.normalize()
        _day_df['stop_dttm'] = _day_df['start_dttm'] + pd.Timedelta(
            hours=23, minutes=59, seconds=59
        )
        _day_df = pc.convert_datetime_columns_to_site_tz(
            _day_df, pc.helper['your_site_timezone']
        )
        _day_sofa = sofa.compute_sofa(
            _day_df,
            tables_path=None,
            use_hospitalization_id=False,
            id_mapping=mapping_ids,
            group_by_id='hosp_id_day_key',
        )
        sbt.manual_tableone(_day_sofa, continuous_cols_sofa).to_csv(
            f'{directory_path}/{_x}_sofa_t1.csv', index=False
        )
    return


# ---------------------------------------------------------------------------
# Entry point — supports `python 02_SBT.py` (defaults to Standard variant)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run()
