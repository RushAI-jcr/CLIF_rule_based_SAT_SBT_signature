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
    import numpy as np
    import pandas as pd
    import re
    import pyCLIF as pc
    from tqdm import tqdm
    from datetime import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    import warnings
    warnings.filterwarnings('ignore')
    from tableone import TableOne
    import pySBT as sbt
    cohort = pd.read_parquet('../output/intermediate/study_cohort.parquet')
    # cohort = pd.read_csv("../output/intermediate/study_cohort.csv")
    cohort['hospital_id'] = cohort['hospital_id'].str.replace('[^a-zA-Z]', '', regex=True)
    return (
        accuracy_score,
        cohort,
        confusion_matrix,
        f1_score,
        np,
        os,
        pc,
        pd,
        plt,
        precision_score,
        re,
        recall_score,
        sbt,
        sns,
        tqdm,
    )


@app.cell
def _(os, pc):
    ## Analysis by
    by = 'Standard'

    # Construct the full directory path
    directory_path = os.path.join("../output/final/", pc.helper["site_name"], f"SBT_{by}")

    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    return by, directory_path


@app.cell
def _(cohort, pc):
    # RUSH-specific SBT timepoint handling (defensive: only runs if column exists)
    if pc.helper.get("site_name") == "RUSH" and "sbt_timepoint" in cohort.columns:
        cohort.loc[
            cohort["sbt_timepoint"] == "3-5 minute evaluation", "pressure_support_set"
        ] = 6.1
        cohort.loc[cohort["sbt_timepoint"] == "3-5 minute evaluation", "mode_category"] = (
            "pressure support/cpap"
        )
        print("RUSH-specific SBT timepoint adjustment applied")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Eligibility Flag making
    """)
    return


@app.cell
def _(cohort, pc, pd):
    t1_cohort = cohort.copy()
    cohort['event_time'] = pd.to_datetime(cohort['event_time'])
    # Ensure all time columns are in datetime format
    cohort['admission_dttm'] = pc.getdttm(cohort['admission_dttm'])
    cohort['discharge_dttm'] = pc.getdttm(cohort['discharge_dttm'])
    cohort_1 = cohort.sort_values(by=['hospitalization_id', 'event_time']).reset_index(drop=True)
    cohort_1['device_category'] = cohort_1['device_category'].str.lower()
    # Ensure the data is sorted by 'hosp_id_day_key' and 'event_time'
    cohort_1['mode_category'] = cohort_1['mode_category'].str.lower()
    cohort_1[['device_category', 'mode_category', 'mode_name', 'location_category', 'hospital_id']] = cohort_1.groupby('hospitalization_id')[['device_category', 'mode_category', 'mode_name', 'location_category', 'hospital_id']].ffill()
    cohort_1[['weight_kg', 'height_cm']] = cohort_1.groupby('hospitalization_id')[['weight_kg', 'height_cm']].ffill().bfill()
    cohort_1[['norepinephrine', 'epinephrine', 'phenylephrine', 'angiotensin', 'vasopressin', 'dopamine', 'dobutamine', 'milrinone', 'isoproterenol']] = cohort_1.groupby('hospitalization_id')[['norepinephrine', 'epinephrine', 'phenylephrine', 'angiotensin', 'vasopressin', 'dopamine', 'dobutamine', 'milrinone', 'isoproterenol']].ffill()
    cohort_1[['fio2_set', 'peep_set', 'spo2', 'pressure_support_set']] = cohort_1.groupby('hospitalization_id')[['fio2_set', 'peep_set', 'spo2', 'pressure_support_set']].ffill()
    cohort_1[['norepinephrine', 'epinephrine', 'phenylephrine', 'dopamine', 'angiotensin', 'vasopressin']] = cohort_1[['norepinephrine', 'epinephrine', 'phenylephrine', 'dopamine', 'angiotensin', 'vasopressin']].fillna(0)
    cohort_1['NEE'] = cohort_1['norepinephrine'] + cohort_1['epinephrine'] + cohort_1['phenylephrine'] / 10 + cohort_1['vasopressin'] * 2.5 + cohort_1['dopamine'] / 100 + cohort_1['angiotensin'] * 10
    # Fill forward's
    cohort_1['Hemodynamic_Stability_by_NEE'] = (cohort_1['NEE'] <= 0.2).astype(int)
    cohort_1['Respiratory_Stability'] = ((cohort_1['fio2_set'] <= 0.5) & (cohort_1['peep_set'] <= 8) & (cohort_1['spo2'] >= 88)).astype(int)
    cohort_1[['cisatracurium', 'vecuronium', 'rocuronium']] = cohort_1.groupby('hospitalization_id')[['cisatracurium', 'vecuronium', 'rocuronium']].ffill()
    # Define Respiratory Stability Flag
    # Fill forward the paralytic by hospitalization columns by 'hosp_id'
    # paralytic max to remove from consideration
    cohort_1['max_paralytics'] = cohort_1[['cisatracurium', 'vecuronium', 'rocuronium']].max(axis=1, skipna=True).fillna(0)
    return cohort_1, t1_cohort


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SBT Eligibility Criteria
    """)
    return


@app.cell
def _(by, cohort_1, sbt):
    final_df = sbt.process_cohort_conditions(cohort_1, by)
    return (final_df,)


@app.cell
def _(final_df):
    # Print statistics
    print('By n = Days')
    _total_days = final_df['hosp_id_day_key'].nunique()
    print('Total number of days for eval in cohort:', _total_days)
    total_vent_days = final_df[final_df['vent_day'] == 1]['hosp_id_day_key'].nunique()
    print('Total number of vent days for eval in cohort: (atleast one IMV event)', total_vent_days)
    total_vent_days_wo_paralytics = final_df[final_df['vent_day_without_paralytics'] == 1]['hosp_id_day_key'].nunique()
    print('Total number of vent days for eval in cohort: (atleast one IMV event & no paralytics given)', total_vent_days_wo_paralytics)
    eligible_days = final_df[final_df['eligible_day'] == 1]['hosp_id_day_key'].nunique()
    _percentage = eligible_days / total_vent_days_wo_paralytics * 100 if _total_days > 0 else 0
    print(f'Eligible days: {eligible_days} / {total_vent_days_wo_paralytics} ({_percentage:.2f}%)')
    print('Hospital days with atleast one IMV event: ', final_df[final_df['device_category'] == 'imv']['hosp_id_day_key'].nunique())
    print('Hospital days with atleast one IMV & ICU event: ', final_df[(final_df['device_category'] == 'imv') & (final_df['location_category'] == 'icu')]['hosp_id_day_key'].nunique())
    print('By n = Encounter')
    _h_total_days = final_df['hospitalization_id'].nunique()
    print('Total number of days for eval in cohort:', _h_total_days)
    _h_eligible_days = final_df[final_df['eligible_day'] == 1]['hospitalization_id'].nunique()
    _h_percentage = _h_eligible_days / _h_total_days * 100 if _h_total_days > 0 else 0
    print(f'Eligible days: {_h_eligible_days} / {_h_total_days} ({_h_percentage:.2f}%)')
    print('Hospital days with atleast one IMV event: ', final_df[final_df['device_category'] == 'imv']['hospitalization_id'].nunique())
    print('Hospital days with atleast one IMV & ICU event: ', final_df[(final_df['device_category'] == 'imv') & (final_df['location_category'] == 'icu')]['hospitalization_id'].nunique())
    return (eligible_days,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FLIP Check
    """)
    return


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


@app.cell
def _(directory_path, final_df_3, pd):
    _hospital_ids = final_df_3['hospital_id'].dropna().unique()
    _bins = list(range(0, 24 * 60 + 1, 60))
    _labels = [f'{i}-{i + 1}hr' for i in range(24)]
    rush_summary = []
    for _hosp in _hospital_ids:
        _delta_series = final_df_3[final_df_3['hospital_id'] == _hosp]['delta_to_extubation_mins'].dropna()
        _delta_binned = pd.cut(_delta_series, bins=_bins, labels=_labels, right=False)
        _rush_counts = _delta_binned.value_counts().sort_index()
        rush_counts_dict = _rush_counts.to_dict()
        rush_counts_dict['hospital_id'] = _hosp
        rush_summary.append(rush_counts_dict)
        pd.DataFrame(final_df_3[final_df_3['hospital_id'] == _hosp]['delta_to_extubation_mins'].describe()).to_csv(f'{directory_path}/delta_stats_between_EHR30Min_Extubated_{_hosp}.csv')
    rush_df = pd.DataFrame(rush_summary)
    rush_df = rush_df.fillna(0).astype({col: 'int' for col in rush_df.columns if col != 'hospital_id'})
    rush_df.to_csv(f'{directory_path}/rush_counts_by_hour_per_hospital.csv', index=False)
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
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results Section
    """)
    return


@app.cell
def _(final_df_3, np):
    final_df_3['sbt_bkp'] = final_df_3['sbt_delivery_pass_fail']
    final_df_3['sbt_delivery_pass_fail_documented'] = final_df_3['sbt_delivery_pass_fail'].notna().astype(float)
    final_df_3['sbt_delivery_pass_fail_documented'] = final_df_3['sbt_delivery_pass_fail_documented'].replace(0, np.nan)
    final_df_3['sbt_screen_pass_fail_documented'] = final_df_3['sbt_screen_pass_fail'].notna().astype(float)
    final_df_3['sbt_screen_pass_fail_documented'] = final_df_3['sbt_screen_pass_fail_documented'].replace(0, np.nan)
    final_df_3['flip_skip_reason'] = final_df_3.groupby('hosp_id_day_key')['flip_skip_reason'].transform(lambda x: _x.ffill().bfill())
    return


@app.cell
def _(final_df_3):
    # Ensure the specified columns are treated as datetime before calculating percentages
    datetime_columns = ['EHR_Delivery_2mins', 'EHR_Delivery_30mins']
    for col in datetime_columns:
        if col in final_df_3.columns:
            final_df_3[col] = final_df_3[col].notna().astype(int)
    return


@app.cell
def _(final_df_3, np):
    grouped_df = final_df_3.groupby('hosp_id_day_key').agg({'hospitalization_id': 'first', 'hospital_id': lambda x: _x.dropna().iloc[-1] if _x.dropna().size > 0 else np.nan, 'eligible_day': 'max', 'EHR_Delivery_2mins': 'max', 'EHR_Delivery_30mins': 'max', 'sbt_screen_pass_fail': 'max', 'sbt_delivery_pass_fail': 'max', 'flag_2_45_extubated': 'max', 'flip_skip_reason': lambda x: _x.dropna().iloc[-1] if _x.dropna().size > 0 else np.nan, 'extubated': 'max'}).reset_index()
    mat_df = grouped_df[grouped_df['eligible_day'] == 1]
    return grouped_df, mat_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Basic counts
    """)
    return


@app.cell
def _(directory_path, grouped_df, pd):
    # Drop NA hospital_ids and get unique ones
    _hospital_ids = grouped_df['hospital_id'].dropna().unique()
    _summary_data = []
    # Container for summary rows
    for _hosp in _hospital_ids:
        _df_hosp = grouped_df[grouped_df['hospital_id'] == _hosp]
    # Loop over hospitals and compute stats
        df_eligible = _df_hosp[_df_hosp['eligible_day'] == 1]
        sbt_S = set(df_eligible[df_eligible['sbt_screen_pass_fail'] == 1]['hosp_id_day_key'].unique())  # Filter data for the hospital and eligible days
        sbt_D = set(df_eligible[df_eligible['sbt_delivery_pass_fail'] == 1]['hosp_id_day_key'].unique())
        ehr_2min = set(df_eligible[df_eligible['EHR_Delivery_2mins'] == 1]['hosp_id_day_key'].unique())
        ehr_30min = set(df_eligible[df_eligible['EHR_Delivery_30mins'] == 1]['hosp_id_day_key'].unique())
        ehr_extubated = set(df_eligible[df_eligible['extubated'] == 1]['hosp_id_day_key'].unique())  # Calculate condition-specific sets (unique hosp_id_day_keys)
        ehr_2min_45min_extubated = set(df_eligible[df_eligible['flag_2_45_extubated'] == 1]['hosp_id_day_key'].unique())
        _summary_data.append({'hospital_id': _hosp, 'sbt_screen_pass': len(sbt_S), 'sbt_delivery_pass': len(sbt_D), 'ehr_2min': len(ehr_2min), 'ehr_30min': len(ehr_30min), 'extubated': len(ehr_extubated), 'ehr_2min_45min_extubated': len(ehr_2min_45min_extubated), 'df_eligible': len(df_eligible)})
        print(f'\nHospital ID: {_hosp}')
        print(f'  SBT Screen Pass: {len(sbt_S)}')
        print(f'  SBT Delivery Pass: {len(sbt_D)}')
        print(f'  EHR 2-min Delivery: {len(ehr_2min)}')
        print(f'  EHR 30-min Delivery: {len(ehr_30min)}')
        print(f'  Extubated: {len(ehr_extubated)}')
        print(f'  ehr_2min_45min_extubated: {len(ehr_2min_45min_extubated)}')
        print(f'  df_eligible: {len(df_eligible)}')
    _summary_df = pd.DataFrame(_summary_data)
    _summary_df
    # Convert summary list to DataFrame
    _summary_df.to_csv(f'{directory_path}/hospital_sbt_ehr_summary_within_eligible_day.csv', index=False)  # Append aggregated counts to summary list  # Optionally print the stats
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### EHR 2 Min vs SBT Flag
    """)
    return


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
    from sklearn.metrics import cohen_kappa_score
    _hospital_ids = mat_df['hospital_id'].unique()
    mat_df['sbt_delivery_pass_fail'] = mat_df['sbt_delivery_pass_fail'].fillna(0)
    metrics_list_2min = []
    for _hosp in _hospital_ids:
        _df_hosp = mat_df[mat_df['hospital_id'] == _hosp]
        if _df_hosp['sbt_delivery_pass_fail'].nunique() <= 1:
            continue
        _conf_matrix = pd.crosstab(_df_hosp['EHR_Delivery_2mins'], _df_hosp['sbt_delivery_pass_fail'])  # Filter the DataFrame for the current hospital
        _conf_matrix_percent = _conf_matrix / _conf_matrix.values.sum() * 100
        _annot = _conf_matrix.astype(str) + '\n' + _conf_matrix_percent.round(1).astype(str) + '%'
        plt.figure(figsize=(6, 4))
        sns.heatmap(_conf_matrix, annot=_annot, fmt='', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
        plt.xlabel('SBT Delivery in Flowsheet')  # Create the confusion matrix using pd.crosstab
        plt.ylabel('EHR Delivery in 2 minutes')
        plt.title(f'Confusion Matrix for Hospital {_hosp}')
        plt.savefig(f'{directory_path}/confusion_matrix_{_hosp}_by_SBT.png')
        plt.close()
        _y_true = _df_hosp['EHR_Delivery_2mins']  # Calculate percentages for each cell
        _y_pred = _df_hosp['sbt_delivery_pass_fail']
        _tn, _fp, _fn, _tp = confusion_matrix(_y_true, _y_pred).ravel()
        _accuracy = accuracy_score(_y_true, _y_pred)  # Create annotation labels that combine count and percentage
        _precision = precision_score(_y_true, _y_pred, zero_division=0)
        _recall = recall_score(_y_true, _y_pred, zero_division=0)
        _f1 = f1_score(_y_true, _y_pred, zero_division=0)
        _specificity = _tn / (_tn + _fp) if _tn + _fp != 0 else 0
        _kappa = cohen_kappa_score(_y_true, _y_pred)  # Plot the confusion matrix
        print(f'Hospital ID: {_hosp}')
        print(f'Accuracy    : {_accuracy:.3f}')
        print(f'Precision   : {_precision:.3f}')
        print(f'Recall      : {_recall:.3f}')
        print(f'F1 Score    : {_f1:.3f}')
        print(f'Specificity : {_specificity:.3f}')
        print(f'Cohen Kappa : {_kappa:.3f}\n')
        _metrics_dict = {'True Positives (TP)': _tp, 'False Positives (FP)': _fp, 'False Negatives (FN)': _fn, 'True Negatives (TN)': _tn, 'Accuracy': _accuracy, 'Precision': _precision, 'Recall': _recall, 'F1 Score': _f1, 'Specificity': _specificity, 'Cohen_Kappa': _kappa}
        _df_metrics = pd.DataFrame(list(_metrics_dict.items()), columns=['Metric', 'Value'])
        _df_metrics.to_csv(f'{directory_path}/EHR_2min_vs_SBT_metrics_{_hosp}.csv', index=False)
        print(_hosp, _df_metrics)
        metrics_list_2min.append({'Column': 'EHR_Delivery_2mins', 'hospital_id': _hosp, 'TP': _tp, 'FP': _fp, 'FN': _fn, 'TN': _tn, 'Accuracy': _accuracy, 'Precision': _precision, 'Recall': _recall, 'F1 Score': _f1, 'Specificity': _specificity, 'Cohen_Kappa': _kappa})
    if metrics_list_2min:  # Save the plot as a PNG file
    # Save pooled concordance summary
        pd.DataFrame(metrics_list_2min).to_csv(f'{directory_path}/delivery_concordance_summary_2min.csv', index=False)  # Close the plot to free memory  # Extract ground truth and predictions for the current hospital  # Compute the confusion matrix and extract TP, FP, FN, TN  # Calculate individual metrics  # Print metrics for current hospital (optional)  # Create a dictionary with the computed metrics  # Build a DataFrame to store the metrics  # Save the metrics DataFrame as a CSV file
    return (cohen_kappa_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### EHR 2 Min vs Extubated Flag
    """)
    return


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
        _df_hosp = mat_df[mat_df['hospital_id'] == _hosp]  # Filter the DataFrame for the current hospital
        _conf_matrix = pd.crosstab(_df_hosp['EHR_Delivery_2mins'], _df_hosp['extubated'])
        _conf_matrix_percent = _conf_matrix / _conf_matrix.values.sum() * 100
        _annot = _conf_matrix.astype(str) + '\n' + _conf_matrix_percent.round(1).astype(str) + '%'  # Create the confusion matrix using pd.crosstab
        plt.figure(figsize=(6, 4))
        sns.heatmap(_conf_matrix, annot=_annot, fmt='', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
        plt.xlabel('extubated')  # Calculate percentages for each cell
        plt.ylabel('EHR Delivery in 2 minutes')
        plt.title(f'Confusion Matrix for Hospital {_hosp}')
        plt.savefig(f'{directory_path}/confusion_matrix_{_hosp}_by_extubated.png')  # Create annotation labels that combine count and percentage
        plt.close()
        _y_true = _df_hosp['EHR_Delivery_2mins']
        _y_pred = _df_hosp['extubated']
        _tn, _fp, _fn, _tp = confusion_matrix(_y_true, _y_pred).ravel()
        _accuracy = accuracy_score(_y_true, _y_pred)  # Plot the confusion matrix
        _precision = precision_score(_y_true, _y_pred, zero_division=0)
        _recall = recall_score(_y_true, _y_pred, zero_division=0)
        _f1 = f1_score(_y_true, _y_pred, zero_division=0)
        _specificity = _tn / (_tn + _fp) if _tn + _fp != 0 else 0
        print(f'Hospital ID: {_hosp}')
        print(f'Accuracy    : {_accuracy:.3f}')
        print(f'Precision   : {_precision:.3f}')
        print(f'Recall      : {_recall:.3f}')
        print(f'F1 Score    : {_f1:.3f}')
        print(f'Specificity : {_specificity:.3f}\n')
        _metrics_dict = {'True Positives (TP)': _tp, 'False Positives (FP)': _fp, 'False Negatives (FN)': _fn, 'True Negatives (TN)': _tn, 'Accuracy': _accuracy, 'Precision': _precision, 'Recall': _recall, 'F1 Score': _f1, 'Specificity': _specificity}
        _df_metrics = pd.DataFrame(list(_metrics_dict.items()), columns=['Metric', 'Value'])
        _df_metrics.to_csv(f'{directory_path}/EHR_2min_VS_EXTUBATED_metrics_{_hosp}.csv', index=False)  # Save the plot as a PNG file
        print(_hosp, _df_metrics)  # Close the plot to free memory  # Extract ground truth and predictions for the current hospital  # Compute the confusion matrix and extract TP, FP, FN, TN  # Calculate individual metrics  # Print metrics for current hospital (optional)  # Create a dictionary with the computed metrics  # Build a DataFrame to store the metrics  # Save the metrics DataFrame as a CSV file
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### EHR 30 Min vs Extubated Flag
    """)
    return


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
        _df_hosp = mat_df[mat_df['hospital_id'] == _hosp]  # Filter the DataFrame for the current hospital
        _conf_matrix = pd.crosstab(_df_hosp['EHR_Delivery_30mins'], _df_hosp['extubated'])
        _conf_matrix_percent = _conf_matrix / _conf_matrix.values.sum() * 100
        _annot = _conf_matrix.astype(str) + '\n' + _conf_matrix_percent.round(1).astype(str) + '%'  # Create the confusion matrix using pd.crosstab
        plt.figure(figsize=(6, 4))
        sns.heatmap(_conf_matrix, annot=_annot, fmt='', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
        plt.xlabel('extubated')  # Calculate percentages for each cell
        plt.ylabel('EHR Delivery in 30 minutes')
        plt.title(f'Confusion Matrix for Hospital {_hosp}')
        plt.savefig(f'{directory_path}/ehr_30_confusion_matrix_{_hosp}_by_extubated.png')  # Create annotation labels that combine count and percentage
        plt.close()
        _y_true = _df_hosp['EHR_Delivery_30mins']
        _y_pred = _df_hosp['extubated']
        _tn, _fp, _fn, _tp = confusion_matrix(_y_true, _y_pred).ravel()
        _accuracy = accuracy_score(_y_true, _y_pred)  # Plot the confusion matrix
        _precision = precision_score(_y_true, _y_pred, zero_division=0)
        _recall = recall_score(_y_true, _y_pred, zero_division=0)
        _f1 = f1_score(_y_true, _y_pred, zero_division=0)
        _specificity = _tn / (_tn + _fp) if _tn + _fp != 0 else 0
        print(f'Hospital ID: {_hosp}')
        print(f'Accuracy    : {_accuracy:.3f}')
        print(f'Precision   : {_precision:.3f}')
        print(f'Recall      : {_recall:.3f}')
        print(f'F1 Score    : {_f1:.3f}')
        print(f'Specificity : {_specificity:.3f}\n')
        _metrics_dict = {'True Positives (TP)': _tp, 'False Positives (FP)': _fp, 'False Negatives (FN)': _fn, 'True Negatives (TN)': _tn, 'Accuracy': _accuracy, 'Precision': _precision, 'Recall': _recall, 'F1 Score': _f1, 'Specificity': _specificity}
        _df_metrics = pd.DataFrame(list(_metrics_dict.items()), columns=['Metric', 'Value'])
        _df_metrics.to_csv(f'{directory_path}/EHR_30_VS_EXTUBATED_metrics_{_hosp}.csv', index=False)  # Save the plot as a PNG file
        print(_hosp, _df_metrics)  # Close the plot to free memory  # Extract ground truth and predictions for the current hospital  # Compute the confusion matrix and extract TP, FP, FN, TN  # Calculate individual metrics  # Print metrics for current hospital (optional)  # Create a dictionary with the computed metrics  # Build a DataFrame to store the metrics  # Save the metrics DataFrame as a CSV file
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### EHR 30 Min VS SBT Flag
    """)
    return


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
            continue  # Filter the DataFrame for the current hospital
        _conf_matrix = pd.crosstab(_df_hosp['EHR_Delivery_30mins'], _df_hosp['sbt_delivery_pass_fail'])
        _conf_matrix_percent = _conf_matrix / _conf_matrix.values.sum() * 100
        _annot = _conf_matrix.astype(str) + '\n' + _conf_matrix_percent.round(1).astype(str) + '%'
        plt.figure(figsize=(6, 4))
        sns.heatmap(_conf_matrix, annot=_annot, fmt='', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])  # Create the confusion matrix using pd.crosstab
        plt.xlabel('SBT Delivery in Flowsheet')
        plt.ylabel('EHR Delivery in 30 minutes')
        plt.title(f'Confusion Matrix for Hospital {_hosp}')
        plt.savefig(f'{directory_path}/ehr_30_confusion_matrix_{_hosp}_by_SBT.png')
        plt.close()  # Calculate percentages for each cell
        _y_true = _df_hosp['EHR_Delivery_30mins']
        _y_pred = _df_hosp['sbt_delivery_pass_fail']
        _tn, _fp, _fn, _tp = confusion_matrix(_y_true, _y_pred).ravel()  # Create annotation labels that combine count and percentage
        _accuracy = accuracy_score(_y_true, _y_pred)
        _precision = precision_score(_y_true, _y_pred, zero_division=0)
        _recall = recall_score(_y_true, _y_pred, zero_division=0)
        _f1 = f1_score(_y_true, _y_pred, zero_division=0)
        _specificity = _tn / (_tn + _fp) if _tn + _fp != 0 else 0  # Plot the confusion matrix
        _kappa = cohen_kappa_score(_y_true, _y_pred)
        print(f'Hospital ID: {_hosp}')
        print(f'Accuracy    : {_accuracy:.3f}')
        print(f'Precision   : {_precision:.3f}')
        print(f'Recall      : {_recall:.3f}')
        print(f'F1 Score    : {_f1:.3f}')
        print(f'Specificity : {_specificity:.3f}')
        print(f'Cohen Kappa : {_kappa:.3f}\n')
        _metrics_dict = {'True Positives (TP)': _tp, 'False Positives (FP)': _fp, 'False Negatives (FN)': _fn, 'True Negatives (TN)': _tn, 'Accuracy': _accuracy, 'Precision': _precision, 'Recall': _recall, 'F1 Score': _f1, 'Specificity': _specificity, 'Cohen_Kappa': _kappa}
        _df_metrics = pd.DataFrame(list(_metrics_dict.items()), columns=['Metric', 'Value'])
        _df_metrics.to_csv(f'{directory_path}/EHR_30_vs_SBT_metrics_{_hosp}.csv', index=False)
        print(_hosp, _df_metrics)
        metrics_list_30min.append({'Column': 'EHR_Delivery_30mins', 'hospital_id': _hosp, 'TP': _tp, 'FP': _fp, 'FN': _fn, 'TN': _tn, 'Accuracy': _accuracy, 'Precision': _precision, 'Recall': _recall, 'F1 Score': _f1, 'Specificity': _specificity, 'Cohen_Kappa': _kappa})  # Save the plot as a PNG file
    if metrics_list_30min:
    # Save pooled concordance summary
        pd.DataFrame(metrics_list_30min).to_csv(f'{directory_path}/delivery_concordance_summary_30min.csv', index=False)  # Close the plot to free memory  # Extract ground truth and predictions for the current hospital  # Compute the confusion matrix and extract TP, FP, FN, TN  # Calculate individual metrics  # Print metrics for current hospital (optional)  # Create a dictionary with the computed metrics  # Build a DataFrame to store the metrics  # Save the metrics DataFrame as a CSV file
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Failure To Detect FLIP vs SBT Flag
    """)
    return


@app.cell
def _(directory_path, eligible_days, final_df_3, mat_df, pd):
    _hospital_ids = mat_df['hospital_id'].unique()
    for _hosp in _hospital_ids:
        _mat_hosp = mat_df[mat_df['hospital_id'] == _hosp]
        _filtered_keys = _mat_hosp.loc[(_mat_hosp['EHR_Delivery_2mins'] == 0) & (_mat_hosp['sbt_delivery_pass_fail'] == 1), 'hosp_id_day_key'].unique()
        _final_filtered_df = final_df_3.loc[(final_df_3['sbt_delivery_pass_fail'] == 1) & final_df_3['hosp_id_day_key'].isin(_filtered_keys)]
        _final_filtered_df = _final_filtered_df.sort_values('event_time')
        _final_filtered_df = _final_filtered_df.drop_duplicates(subset='hosp_id_day_key', keep='first')
        print(f'Hospital: {_hosp}, final_filtered_df shape: {_final_filtered_df.shape}')
        _df = _final_filtered_df.copy()
        _results = []
        _step1 = _df[~_df['flip_skip_reason'].isna()]
        _results.append({'Step': 'Step 1', 'FilterColumn': 'flip_skip_reason', 'UniqueKeys': _step1['hosp_id_day_key'].nunique(), 'RowCount': _step1.shape[0], 'ValueCounts': _step1['flip_skip_reason'].value_counts(dropna=False).to_dict()})
        _df = _df[~_df['hosp_id_day_key'].isin(_step1['hosp_id_day_key'])]
        _step2 = _df[~_df['cond_device_imv'].isna()]
        _results.append({'Step': 'Step 2', 'FilterColumn': 'cond_device_imv', 'UniqueKeys': _step2['hosp_id_day_key'].nunique(), 'RowCount': _step2.shape[0], 'ValueCounts': _step2['cond_device_imv'].value_counts(dropna=False).to_dict()})
        _df = _df[~_df['hosp_id_day_key'].isin(_step2['hosp_id_day_key'])]
        _step3 = _df[~_df['cond_location_icu'].isna()]
        _results.append({'Step': 'Step 3', 'FilterColumn': 'cond_location_icu', 'UniqueKeys': _step3['hosp_id_day_key'].nunique(), 'RowCount': _step3.shape[0], 'ValueCounts': _step3['cond_location_icu'].value_counts(dropna=False).to_dict()})
        _df = _df[~_df['hosp_id_day_key'].isin(_step3['hosp_id_day_key'])]
        _step4 = _df[~_df['cond_peep_set_le8'].isna()]
        _results.append({'Step': 'Step 4', 'FilterColumn': 'cond_peep_set_le8', 'UniqueKeys': _step4['hosp_id_day_key'].nunique(), 'RowCount': _step4.shape[0], 'ValueCounts': _step4['cond_peep_set_le8'].value_counts(dropna=False).to_dict()})
        _df = _df[~_df['hosp_id_day_key'].isin(_step4['hosp_id_day_key'])]
        _step5 = _df[~_df['cond_ps_set_le8'].isna()]
        _results.append({'Step': 'Step 5', 'FilterColumn': 'cond_ps_set_le8', 'UniqueKeys': _step5['hosp_id_day_key'].nunique(), 'RowCount': _step5.shape[0], 'ValueCounts': _step5['cond_ps_set_le8'].value_counts(dropna=False).to_dict()})
        _df = _df[~_df['hosp_id_day_key'].isin(_step5['hosp_id_day_key'])]
        _step6 = _df[~_df['cond_mode_ps_cpap'].isna()]
        _results.append({'Step': 'Step 6', 'FilterColumn': 'cond_mode_ps_cpap', 'UniqueKeys': _step6['hosp_id_day_key'].nunique(), 'RowCount': _step6.shape[0], 'ValueCounts': _step6['cond_mode_ps_cpap'].value_counts(dropna=False).to_dict()})
        _df = _df[~_df['hosp_id_day_key'].isin(_step6['hosp_id_day_key'])]
        _step7 = _df.copy()
        _results.append({'Step': 'Step 7 (Unmatched)', 'FilterColumn': None, 'UniqueKeys': _step7['hosp_id_day_key'].nunique(), 'RowCount': _step7.shape[0], 'ValueCounts': None})
        _detailed_summary_df = pd.DataFrame(_results)
        _total_failures = _detailed_summary_df['UniqueKeys'].sum()
        _detailed_summary_df['% by eligible_days'] = _detailed_summary_df['UniqueKeys'].apply(lambda x: round(_x / eligible_days * 100, 2))
        _detailed_summary_df['% of Total'] = _detailed_summary_df['UniqueKeys'].apply(lambda x: round(_x / _total_failures * 100, 2) if _total_failures != 0 else 0)
        _output_filename = f'{directory_path}/EHR_VS_SBT_failure_dependent_summary_{_hosp}.csv'
        _detailed_summary_df.to_csv(_output_filename, index=False)
        print(f'Saved detailed summary for hospital {_hosp} to {_output_filename}\n')
        print(_hosp, _detailed_summary_df)
        print()
        _ind_step1 = _final_filtered_df[~_final_filtered_df['flip_skip_reason'].isna()]
        _ind_step2 = _final_filtered_df[~_final_filtered_df['cond_device_imv'].isna()]
        _ind_step3 = _final_filtered_df[~_final_filtered_df['cond_location_icu'].isna()]
        _ind_step4 = _final_filtered_df[~_final_filtered_df['cond_peep_set_le8'].isna()]
        _ind_step5 = _final_filtered_df[~_final_filtered_df['cond_ps_set_le8'].isna()]
        _ind_step6 = _final_filtered_df[~_final_filtered_df['cond_mode_ps_cpap'].isna()]
        _matched_keys = set().union(_ind_step1['hosp_id_day_key'], _ind_step2['hosp_id_day_key'], _ind_step3['hosp_id_day_key'], _ind_step4['hosp_id_day_key'], _ind_step5['hosp_id_day_key'], _ind_step6['hosp_id_day_key'])
        _ind_step7 = _final_filtered_df[~_final_filtered_df['hosp_id_day_key'].isin(_matched_keys)]
        _failure_counts = {'flip_skip_reason': _ind_step1['hosp_id_day_key'].nunique(), 'cond_device_imv': _ind_step2['hosp_id_day_key'].nunique(), 'cond_location_icu': _ind_step3['hosp_id_day_key'].nunique(), 'cond_peep_set_le8': _ind_step4['hosp_id_day_key'].nunique(), 'cond_ps_set_le8': _ind_step5['hosp_id_day_key'].nunique(), 'cond_mode_ps_cpap': _ind_step6['hosp_id_day_key'].nunique(), 'unmatched': _ind_step7['hosp_id_day_key'].nunique()}
        _value_counts_map = {'flip_skip_reason': _ind_step1['flip_skip_reason'].value_counts(dropna=False).to_dict(), 'cond_device_imv': _ind_step2['cond_device_imv'].value_counts(dropna=False).to_dict(), 'cond_location_icu': _ind_step3['cond_location_icu'].value_counts(dropna=False).to_dict(), 'cond_peep_set_le8': _ind_step4['cond_peep_set_le8'].value_counts(dropna=False).to_dict(), 'cond_ps_set_le8': _ind_step5['cond_ps_set_le8'].value_counts(dropna=False).to_dict(), 'cond_mode_ps_cpap': _ind_step6['cond_mode_ps_cpap'].value_counts(dropna=False).to_dict(), 'unmatched': None}
        _total_failures_ind = sum(_failure_counts.values())
        _summary_data = []
        for _reason, _count in _failure_counts.items():
            _summary_data.append({'Failure Reason': _reason, 'Count': _count, '% by eligible_days': round(_count / eligible_days * 100, 2), '% of Total (out of total failed cases)': round(_count / _total_failures_ind * 100, 2) if _total_failures_ind else 0, 'Value Counts': _value_counts_map[_reason]})
        _independent_summary_df = pd.DataFrame(_summary_data)
        _independent_summary_df = _independent_summary_df.sort_values(by='Count', ascending=False).reset_index(drop=True)
        _ind_output_filename = f'{directory_path}/EHR_VS_SBT_failure_independent_summary_hospital_{_hosp}.csv'
        _independent_summary_df.to_csv(_ind_output_filename, index=False)
        print(f'Saved independent summary for hospital {_hosp} to {_ind_output_filename}\n')
        print(_hosp, _independent_summary_df)
        print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Failure To Detect FLIP vs Extubated Flag
    """)
    return


@app.cell
def _(directory_path, eligible_days, final_df_3, mat_df, pd):
    _hospital_ids = mat_df['hospital_id'].unique()
    for _hosp in _hospital_ids:
        _mat_hosp = mat_df[mat_df['hospital_id'] == _hosp]
        _filtered_keys = _mat_hosp.loc[(_mat_hosp['EHR_Delivery_2mins'] == 0) & (_mat_hosp['extubated'] == 1), 'hosp_id_day_key'].unique()
        _final_filtered_df = final_df_3.loc[(final_df_3['extubated'] == 1) & final_df_3['hosp_id_day_key'].isin(_filtered_keys)]
        _final_filtered_df = _final_filtered_df.sort_values('event_time')
        _final_filtered_df = _final_filtered_df.drop_duplicates(subset='hosp_id_day_key', keep='first')
        print(f'Hospital: {_hosp}, final_filtered_df shape: {_final_filtered_df.shape}')
        _df = _final_filtered_df.copy()
        _results = []
        _step1 = _df[~_df['flip_skip_reason'].isna()]
        _results.append({'Step': 'Step 1', 'FilterColumn': 'flip_skip_reason', 'UniqueKeys': _step1['hosp_id_day_key'].nunique(), 'RowCount': _step1.shape[0], 'ValueCounts': _step1['flip_skip_reason'].value_counts(dropna=False).to_dict()})
        _df = _df[~_df['hosp_id_day_key'].isin(_step1['hosp_id_day_key'])]
        _step2 = _df[~_df['cond_device_imv'].isna()]
        _results.append({'Step': 'Step 2', 'FilterColumn': 'cond_device_imv', 'UniqueKeys': _step2['hosp_id_day_key'].nunique(), 'RowCount': _step2.shape[0], 'ValueCounts': _step2['cond_device_imv'].value_counts(dropna=False).to_dict()})
        _df = _df[~_df['hosp_id_day_key'].isin(_step2['hosp_id_day_key'])]
        _step3 = _df[~_df['cond_location_icu'].isna()]
        _results.append({'Step': 'Step 3', 'FilterColumn': 'cond_location_icu', 'UniqueKeys': _step3['hosp_id_day_key'].nunique(), 'RowCount': _step3.shape[0], 'ValueCounts': _step3['cond_location_icu'].value_counts(dropna=False).to_dict()})
        _df = _df[~_df['hosp_id_day_key'].isin(_step3['hosp_id_day_key'])]
        _step4 = _df[~_df['cond_peep_set_le8'].isna()]
        _results.append({'Step': 'Step 4', 'FilterColumn': 'cond_peep_set_le8', 'UniqueKeys': _step4['hosp_id_day_key'].nunique(), 'RowCount': _step4.shape[0], 'ValueCounts': _step4['cond_peep_set_le8'].value_counts(dropna=False).to_dict()})
        _df = _df[~_df['hosp_id_day_key'].isin(_step4['hosp_id_day_key'])]
        _step5 = _df[~_df['cond_ps_set_le8'].isna()]
        _results.append({'Step': 'Step 5', 'FilterColumn': 'cond_ps_set_le8', 'UniqueKeys': _step5['hosp_id_day_key'].nunique(), 'RowCount': _step5.shape[0], 'ValueCounts': _step5['cond_ps_set_le8'].value_counts(dropna=False).to_dict()})
        _df = _df[~_df['hosp_id_day_key'].isin(_step5['hosp_id_day_key'])]
        _step6 = _df[~_df['cond_mode_ps_cpap'].isna()]
        _results.append({'Step': 'Step 6', 'FilterColumn': 'cond_mode_ps_cpap', 'UniqueKeys': _step6['hosp_id_day_key'].nunique(), 'RowCount': _step6.shape[0], 'ValueCounts': _step6['cond_mode_ps_cpap'].value_counts(dropna=False).to_dict()})
        _df = _df[~_df['hosp_id_day_key'].isin(_step6['hosp_id_day_key'])]
        _step7 = _df.copy()
        _results.append({'Step': 'Step 7 (No Value)', 'FilterColumn': None, 'UniqueKeys': _step7['hosp_id_day_key'].nunique(), 'RowCount': _step7.shape[0], 'ValueCounts': None})
        _detailed_summary_df = pd.DataFrame(_results)
        _total_failures = _detailed_summary_df['UniqueKeys'].sum()
        _detailed_summary_df['% by eligible_days'] = _detailed_summary_df['UniqueKeys'].apply(lambda x: round(_x / eligible_days * 100, 2))
        _detailed_summary_df['% of Total'] = _detailed_summary_df['UniqueKeys'].apply(lambda x: round(_x / _total_failures * 100, 2) if _total_failures != 0 else 0)
        _output_filename = f'{directory_path}/EHR_VS_EXTUBATED_failure_dependent_summary_{_hosp}.csv'
        _detailed_summary_df.to_csv(_output_filename, index=False)
        print(f'Saved detailed summary for hospital {_hosp} to {_output_filename}\n')
        print(_hosp, _detailed_summary_df)
        print()
        _ind_step1 = _final_filtered_df[~_final_filtered_df['flip_skip_reason'].isna()]
        _ind_step2 = _final_filtered_df[~_final_filtered_df['cond_device_imv'].isna()]
        _ind_step3 = _final_filtered_df[~_final_filtered_df['cond_location_icu'].isna()]
        _ind_step4 = _final_filtered_df[~_final_filtered_df['cond_peep_set_le8'].isna()]
        _ind_step5 = _final_filtered_df[~_final_filtered_df['cond_ps_set_le8'].isna()]
        _ind_step6 = _final_filtered_df[~_final_filtered_df['cond_mode_ps_cpap'].isna()]
        _matched_keys = set().union(_ind_step1['hosp_id_day_key'], _ind_step2['hosp_id_day_key'], _ind_step3['hosp_id_day_key'], _ind_step4['hosp_id_day_key'], _ind_step5['hosp_id_day_key'], _ind_step6['hosp_id_day_key'])
        _ind_step7 = _final_filtered_df[~_final_filtered_df['hosp_id_day_key'].isin(_matched_keys)]
        _failure_counts = {'flip_skip_reason': _ind_step1['hosp_id_day_key'].nunique(), 'cond_device_imv': _ind_step2['hosp_id_day_key'].nunique(), 'cond_location_icu': _ind_step3['hosp_id_day_key'].nunique(), 'cond_peep_set_le8': _ind_step4['hosp_id_day_key'].nunique(), 'cond_ps_set_le8': _ind_step5['hosp_id_day_key'].nunique(), 'cond_mode_ps_cpap': _ind_step6['hosp_id_day_key'].nunique(), 'No Value': _ind_step7['hosp_id_day_key'].nunique()}
        _value_counts_map = {'flip_skip_reason': _ind_step1['flip_skip_reason'].value_counts(dropna=False).to_dict(), 'cond_device_imv': _ind_step2['cond_device_imv'].value_counts(dropna=False).to_dict(), 'cond_location_icu': _ind_step3['cond_location_icu'].value_counts(dropna=False).to_dict(), 'cond_peep_set_le8': _ind_step4['cond_peep_set_le8'].value_counts(dropna=False).to_dict(), 'cond_ps_set_le8': _ind_step5['cond_ps_set_le8'].value_counts(dropna=False).to_dict(), 'cond_mode_ps_cpap': _ind_step6['cond_mode_ps_cpap'].value_counts(dropna=False).to_dict(), 'No Value': None}
        _total_failures_ind = sum(_failure_counts.values())
        _summary_data = []
        for _reason, _count in _failure_counts.items():
            _summary_data.append({'Failure Reason': _reason, 'Count': _count, '% by eligible_days': round(_count / eligible_days * 100, 2), '% of Total (out of total failed cases)': round(_count / _total_failures_ind * 100, 2) if _total_failures_ind else 0, 'Value Counts': _value_counts_map[_reason]})
        _independent_summary_df = pd.DataFrame(_summary_data)
        _independent_summary_df = _independent_summary_df.sort_values(by='Count', ascending=False).reset_index(drop=True)
        _ind_output_filename = f'{directory_path}/EHR_VS_EXTUBATED_failure_independent_summary_{_hosp}.csv'
        _independent_summary_df.to_csv(_ind_output_filename, index=False)
        print(f'Saved independent summary for hospital {_hosp} to {_ind_output_filename}\n')
        print(_hosp, _independent_summary_df)
        print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plots
    """)
    return


@app.cell
def _(directory_path, final_df_3, pd, plt):
    _hospital_ids = final_df_3['hospital_id'].dropna().unique()
    _hospital_summary_list = []
    for _hosp in _hospital_ids:
        _final_hosp = final_df_3[final_df_3['hospital_id'] == _hosp]
        _sbt_d_time = _final_hosp[(_final_hosp['sbt_delivery_pass_fail'] == 1) & (_final_hosp['eligible_day'] == 1)].sort_values(['hosp_id_day_key', 'event_time']).groupby('hosp_id_day_key', as_index=False).first()[['hosp_id_day_key', 'event_time']]
        _ehr_d_time = _final_hosp[(_final_hosp['EHR_Delivery_2mins'] == 1) & (_final_hosp['eligible_day'] == 1)][['hosp_id_day_key', 'event_time']].drop_duplicates()
        _sbt_hours = _sbt_d_time['event_time'].dt.hour
        _ehr_hours = _ehr_d_time['event_time'].dt.hour
        plt.figure(figsize=(10, 6))
        plt.hist(_sbt_hours, bins=range(0, 25), alpha=0.5, label='SBT Delivery Time', edgecolor='black')
        plt.hist(_ehr_hours, bins=range(0, 25), alpha=0.5, label='EHR Delivery Time', edgecolor='black')
        plt.xlabel('Hour of Day')
        plt.ylabel('Frequency')
        plt.title(f'Event Time Distribution (Hourly) - Hospital {_hosp}')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f'{directory_path}/event_time_distribution_hospital_{_hosp}.png')
        plt.close()
        _sbt_counts = _sbt_hours.value_counts().sort_index()
        _ehr_counts = _ehr_hours.value_counts().sort_index()
        _hours_df = pd.DataFrame({'hour': range(24)})
        _hours_df['SBT_Delivery'] = _hours_df['hour'].map(_sbt_counts).fillna(0).astype(int)
        _hours_df['EHR_Delivery'] = _hours_df['hour'].map(_ehr_counts).fillna(0).astype(int)
        _hours_df['hospital_id'] = _hosp
        _hospital_summary_list.append(_hours_df)
    _combined_summary_df = pd.concat(_hospital_summary_list, ignore_index=True)
    _combined_summary_df.to_csv(f'{directory_path}/event_time_distribution_summary.csv', index=False)
    print('Overlay plots created and summary CSV saved.')
    return


@app.cell
def _(directory_path, final_df_3, pd, plt):
    _hospital_ids = final_df_3['hospital_id'].dropna().unique()
    _hospital_summary_list = []
    for _hosp in _hospital_ids:
        _final_hosp = final_df_3[final_df_3['hospital_id'] == _hosp]
        _sbt_d_time = _final_hosp[(_final_hosp['extubated'] == 1) & (_final_hosp['eligible_day'] == 1)].sort_values(['hosp_id_day_key', 'event_time']).groupby('hosp_id_day_key', as_index=False).first()[['hosp_id_day_key', 'event_time']]
        _ehr_d_time = _final_hosp[(_final_hosp['EHR_Delivery_2mins'] == 1) & (_final_hosp['eligible_day'] == 1)][['hosp_id_day_key', 'event_time']].drop_duplicates()
        _sbt_hours = _sbt_d_time['event_time'].dt.hour
        _ehr_hours = _ehr_d_time['event_time'].dt.hour
        plt.figure(figsize=(10, 6))
        plt.hist(_sbt_hours, bins=range(0, 25), alpha=0.5, label='Extubated Time', edgecolor='black')
        plt.hist(_ehr_hours, bins=range(0, 25), alpha=0.5, label='EHR Delivery Time', edgecolor='black')
        plt.xlabel('Hour of Day')
        plt.ylabel('Frequency')
        plt.title(f'Event Time Distribution (Hourly) - Hospital {_hosp}')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f'{directory_path}/event_time_distribution_hospital_{_hosp}_by_ex.png')
        plt.close()
        _sbt_counts = _sbt_hours.value_counts().sort_index()
        _ehr_counts = _ehr_hours.value_counts().sort_index()
        _hours_df = pd.DataFrame({'hour': range(24)})
        _hours_df['SBT_Delivery'] = _hours_df['hour'].map(_sbt_counts).fillna(0).astype(int)
        _hours_df['EHR_Delivery'] = _hours_df['hour'].map(_ehr_counts).fillna(0).astype(int)
        _hours_df['hospital_id'] = _hosp
        _hospital_summary_list.append(_hours_df)
    _combined_summary_df = pd.concat(_hospital_summary_list, ignore_index=True)
    _combined_summary_df.to_csv(f'{directory_path}/event_time_distribution_summary_by_ex.csv', index=False)
    print('Overlay plots created and summary CSV saved.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Final Summary
    """)
    return


@app.cell
def _(directory_path, final_df_3, mat_df, pd):
    _total_days = final_df_3['hosp_id_day_key'].nunique()
    eligible_days_1 = final_df_3[final_df_3['eligible_day'] == 1]['hosp_id_day_key'].nunique()
    imv_days = final_df_3[final_df_3['vent_day_without_paralytics'] == 1]['hosp_id_day_key'].nunique()
    imv_days_with_no_filter = final_df_3[final_df_3['vent_day'] == 1]['hosp_id_day_key'].nunique()
    _percentage = eligible_days_1 / imv_days * 100 if _total_days > 0 else 0
    imv_icu_days = final_df_3[(final_df_3['device_category'] == 'imv') & (final_df_3['location_category'] == 'icu')]['hosp_id_day_key'].nunique()
    _h_total_days = final_df_3['hospitalization_id'].nunique()
    _h_eligible_days = final_df_3[final_df_3['eligible_day'] == 1]['hospitalization_id'].nunique()
    _h_percentage = _h_eligible_days / _h_total_days * 100 if _h_total_days > 0 else 0
    h_imv_days = final_df_3[final_df_3['device_category'] == 'imv']['hospitalization_id'].nunique()
    h_imv_icu_days = final_df_3[(final_df_3['device_category'] == 'imv') & (final_df_3['location_category'] == 'icu')]['hospitalization_id'].nunique()
    ehr_delivery_counts = mat_df[mat_df['extubated'] == 1]['EHR_Delivery_2mins'].value_counts(normalize=True) * 100
    sbt_delivery_counts = mat_df[mat_df['extubated'] == 1]['sbt_delivery_pass_fail'].value_counts(normalize=True) * 100
    print('By n = Days')
    print('Total number of days for eval in cohort:', _total_days)
    print(f'Eligible days: {eligible_days_1} / {imv_days} ({_percentage:.2f}%)')
    print('Hospital days with at least one IMV event:', imv_days)
    print('Hospital days with at least one IMV & ICU event:', imv_icu_days)
    print('\nBy n = Encounter')
    print('Total number of encounters for eval in cohort:', _h_total_days)
    print(f'Eligible encounters: {_h_eligible_days} / {_h_total_days} ({_h_percentage:.2f}%)')
    print('Encounters with at least one IMV event:', h_imv_days)
    print('Encounters with at least one IMV & ICU event:', h_imv_icu_days)
    print('\nEHR_Delivery_2mins distribution (for extubated == 1):')
    print(ehr_delivery_counts)
    print('\nsbt_delivery_pass_fail distribution (for extubated == 1):')
    print(sbt_delivery_counts)
    stats_data = {'Metric': ['total_days', 'eligible_days', 'eligible_percentage', 'imv_days_with_out_paralytics', 'imv_icu_days', 'imv_days_with_no_filter', 'enc_total_days', 'enc_eligible_days', 'enc_eligible_percentage', 'enc_imv_days', 'enc_imv_icu_days'], 'Value': [_total_days, eligible_days_1, _percentage, imv_days, imv_icu_days, imv_days_with_no_filter, _h_total_days, _h_eligible_days, _h_percentage, h_imv_days, h_imv_icu_days]}
    stats_df = pd.DataFrame(stats_data)
    ehr_counts_df = ehr_delivery_counts.reset_index()
    ehr_counts_df.columns = ['Metric', 'Value']
    ehr_counts_df['Metric'] = 'EHR_Delivery_2mins_' + ehr_counts_df['Metric'].astype(str) + '_extubated=1'
    sbt_counts_df = sbt_delivery_counts.reset_index()
    sbt_counts_df.columns = ['Metric', 'Value']
    sbt_counts_df['Metric'] = 'sbt_delivery_pass_fail_' + sbt_counts_df['Metric'].astype(str) + '_extubated=1'
    stats_df = pd.concat([stats_df, ehr_counts_df, sbt_counts_df], ignore_index=True)
    print('\nExtended statistics DataFrame with value counts:')
    print(stats_df)
    stats_df.to_csv(f'{directory_path}/stats_df.csv', index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Table 1 Code
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
    t1_cohort[drugs] = t1_cohort[drugs].applymap(lambda x: _x if _x > 0 else np.nan)
    t1_cohort['bmi'] = t1_cohort['weight_kg'] / (t1_cohort['height_cm'] / 100) ** 2
    t1_cohort['language_name'] = t1_cohort['language_name'].apply(categorize_language)
    # Apply the transformation
    # Apply the function to 'language_name'
    t1_cohort[continuous_cols] = t1_cohort[continuous_cols].astype(float)
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
    sbt,
    t1_cohort,
):
    for _x in ['hospitalization_id', 'patient_id']:
        _t1_summary = t1_cohort.groupby(_x).agg({'age_at_admission': 'mean', **{col: documented for col in medication_columns}, **{col: 'first' for col in demographic_columns}})
        _t1_summary['age_bucket'] = _t1_summary['age_at_admission'].apply(age_bucket)
        _t1_summary = _t1_summary.drop(columns=['age_at_admission'])
        _t1_summary = _t1_summary.reset_index()
        _summary_df = sbt.manual_categorical_tableone(_t1_summary, medication_columns + demographic_columns + ['age_bucket'])
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
def _(continuous_cols, directory_path, sbt, t1_cohort):
    hospitalization_summary = None
    patient_summary = None
    _hosp = t1_cohort.groupby('hospitalization_id').agg({**{c: 'median' for c in continuous_cols}}).reset_index()
    patient = t1_cohort.groupby('patient_id').agg({**{c: 'median' for c in continuous_cols}}).reset_index()
    hospitalization_summary = sbt.manual_tableone(_hosp, continuous_cols)
    patient_summary = sbt.manual_tableone(patient, continuous_cols)
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
    final_df_3,
    medication_columns,
    sbt,
    t1_cohort,
    tqdm,
):
    for _x in tqdm(['vent_day', 'vent_day_without_paralytics', 'eligible_day', 'EHR_Delivery_2mins', 'EHR_Delivery_30mins']):
        ids_to_use = final_df_3[final_df_3[_x] == 1].hosp_id_day_key.unique()
        _t1_summary = t1_cohort[t1_cohort['hosp_id_day_key'].isin(ids_to_use)].groupby('hosp_id_day_key').agg({'age_at_admission': 'mean', **{col: documented for col in medication_columns}, **{col: 'first' for col in demographic_columns}})
        _t1_summary['age_bucket'] = _t1_summary['age_at_admission'].apply(age_bucket)
        _t1_summary = _t1_summary.drop(columns=['age_at_admission'])
        _t1_summary = _t1_summary.reset_index()
        _summary_df = sbt.manual_categorical_tableone(_t1_summary, medication_columns + demographic_columns + ['age_bucket'])
        _summary_df.to_csv(f'{directory_path}/table1_{_x}_categorical.csv', index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Table 1 By Days for Continuous
    """)
    return


@app.cell
def _(continuous_cols, directory_path, final_df_3, sbt, t1_cohort, tqdm):
    for _x in tqdm(['vent_day', 'vent_day_without_paralytics', 'eligible_day', 'EHR_Delivery_2mins', 'EHR_Delivery_30mins']):
        ids = final_df_3.loc[final_df_3[_x] == 1, 'hosp_id_day_key'].unique()
        sub = t1_cohort[t1_cohort['hosp_id_day_key'].isin(ids)]
        day_summary = sub.groupby('hosp_id_day_key').agg({**{c: 'median' for c in continuous_cols}}).reset_index()
        _summary_df = sbt.manual_tableone(day_summary, continuous_cols)
        _summary_df.to_csv(f'{directory_path}/table1_{_x}_continuous.csv', index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sofa T1's
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
def _(cohort_1, pc):
    encounter_level_sofa = cohort_1[['hospitalization_id', 'admission_dttm', 'discharge_dttm']].drop_duplicates().rename(columns={'admission_dttm': 'start_dttm', 'discharge_dttm': 'stop_dttm'})
    encounter_level_sofa = pc.convert_datetime_columns_to_site_tz(encounter_level_sofa, pc.helper['your_site_timezone'])
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
    sout = sofa.compute_sofa(
        encounter_level_sofa,
        tables_path=None,
        use_hospitalization_id = False,
        id_mapping = mapping_ids,
        group_by_id = "encounter_block"
    )
    encounter_level_sofa_t1 = sbt.manual_tableone(sout, continuous_cols_sofa)
    encounter_level_sofa_t1.to_csv(f'{directory_path}/encounter_level_sofa_t1.csv',index=False)
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
    for _x in tqdm(['vent_day', 'vent_day_without_paralytics', 'eligible_day', 'EHR_Delivery_2mins', 'EHR_Delivery_30mins', 'sbt_delivery_pass_fail'], desc='Generating Sofa table 1 for each Flags'):
        day_df = final_df_3[final_df_3[_x] == 1][['hospitalization_id', 'hosp_id_day_key', 'current_day']].drop_duplicates()
        if day_df.empty:
            continue
        day_df['start_dttm'] = pd.to_datetime(day_df['current_day']).dt.normalize()
        day_df['stop_dttm'] = day_df['start_dttm'] + pd.Timedelta(hours=23, minutes=59, seconds=59)
        day_df = pc.convert_datetime_columns_to_site_tz(day_df, pc.helper['your_site_timezone'])
        day_sofa = sofa.compute_sofa(day_df, tables_path=None, use_hospitalization_id=False, id_mapping=mapping_ids, group_by_id='hosp_id_day_key')
        day_sofa_t1 = sbt.manual_tableone(day_sofa, continuous_cols_sofa)
        day_sofa_t1.to_csv(f'{directory_path}/{_x}_sofa_t1.csv', index=False)
    return


if __name__ == "__main__":
    app.run()
