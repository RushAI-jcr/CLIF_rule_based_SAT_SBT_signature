import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import sys
    _CODE_DIR = os.path.dirname(os.path.abspath(__file__))
    _UTILS_DIR = os.path.join(_CODE_DIR, '..', 'utils')
    if _UTILS_DIR not in sys.path:
        sys.path.insert(0, _UTILS_DIR)
    # Change CWD to code/ so relative paths (../output, ../config) work
    os.chdir(_CODE_DIR)
    import pandas as pd
    from tqdm import tqdm
    import numpy as np
    import pytz
    import duckdb
    import pyCLIF as pc
    import warnings
    warnings.filterwarnings('ignore')
    return duckdb, np, os, pc, pd, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Base Population
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### ADT
    """)
    return


@app.cell
def _(pc):
    adt = pc.load_data("clif_adt")
    adt["hospitalization_id"] = adt["hospitalization_id"].astype(str)
    adt = pc.convert_datetime_columns_to_site_tz(adt, pc.helper["your_site_timezone"])
    adt["in_dttm"] = pc.getdttm(adt["in_dttm"])
    pc.deftime(adt["in_dttm"])
    adt.head()
    return (adt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Hospitalization
    """)
    return


@app.cell
def _(adt, pc):
    hosp = pc.load_data("clif_hospitalization")
    hosp["hospitalization_id"] = hosp["hospitalization_id"].astype(str)
    if "hospitalization_joined_id" not in hosp.columns:
        hosp["hospitalization_joined_id"] = hosp["hospitalization_id"]

    hosp["hospitalization_joined_id"] = hosp["hospitalization_joined_id"].astype(str)

    hosp = pc.convert_datetime_columns_to_site_tz(hosp, pc.helper["your_site_timezone"])
    hosp["admission_dttm"] = pc.getdttm(hosp["admission_dttm"])
    hosp["discharge_dttm"] = pc.getdttm(hosp["discharge_dttm"])

    adt["Hosp_key_bkp"] = adt["hospitalization_id"]
    hosp["Hosp_key_bkp"] = hosp["hospitalization_id"]

    hosp.head()
    return (hosp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Hospitalization Stitching
    """)
    return


@app.cell
def _(adt, hosp, pc):
    eblock = pc.stitch_encounters(hosp, adt)

    # Create mapping dictionary
    hospitalization_to_block = {
        hospital_id: block
        for block, hospital_list in zip(
            eblock["encounter_block"].astype(str), eblock["list_hospitalization_id"]
        )
        for hospital_id in hospital_list
    }
    return (hospitalization_to_block,)


@app.cell
def _(hospitalization_to_block, pd):
    # Convert to DataFrame
    hospitalization_to_block_df = pd.DataFrame(
        list(hospitalization_to_block.items()),
        columns=["hospitalization_id", "encounter_block"],
    )
    hospitalization_to_block_df.to_csv(
        "../output/intermediate/hospitalization_to_block_df.csv", index=False
    )
    return


@app.cell
def _(hosp, hospitalization_to_block):
    agg_rules_hosp = {'patient_id': 'first', 'admission_dttm': 'min', 'discharge_dttm': 'max', 'age_at_admission': 'mean', 'discharge_category': 'last', 'hospitalization_joined_id': lambda x: ', '.join(x.unique()), 'Hosp_key_bkp': lambda x: ', '.join(x.unique())}
    if 'discharge_name' in hosp.columns:  # Assuming patient_id is consistent across duplicates
        agg_rules_hosp['discharge_name'] = 'last'  # Earliest admission date
    hosp['hospitalization_id'] = hosp['hospitalization_id'].map(hospitalization_to_block)  # Latest discharge date
    hosp['hospitalization_id'] = hosp['hospitalization_id'].astype(str)  # Take average if different
    hosp_1 = hosp.sort_values(by=['hospitalization_id', 'admission_dttm'])  # Keep the first occurrence
    # Defensive: only include columns that actually exist (discharge_name may not exist at all sites)
    hosp_1 = hosp_1.groupby('hospitalization_id').agg(agg_rules_hosp).reset_index()  # Retain first occurrence  # Backup key, take first occurrence
    return (hosp_1,)


@app.cell
def _(adt, hospitalization_to_block):
    adt_1 = adt[['hospitalization_id', 'in_dttm', 'location_category', 'hospital_id']]
    adt_1['hospitalization_id'] = adt_1['hospitalization_id'].map(hospitalization_to_block).astype(str)
    adt_1 = adt_1.sort_values(by=['hospitalization_id', 'in_dttm'])
    return (adt_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Respiratory Support
    """)
    return


@app.cell
def _(hospitalization_to_block, pc):
    rst = pc.load_data("clif_respiratory_support")
    rst["hospitalization_id"] = rst["hospitalization_id"].astype(str)
    _unmapped_rst = rst["hospitalization_id"].map(hospitalization_to_block).isna().sum()
    if _unmapped_rst > 0:
        print(f"WARNING: {_unmapped_rst} respiratory_support rows have unmapped hospitalization_id (dropped)")
    rst["hospitalization_id"] = (
        rst["hospitalization_id"]
        .map(hospitalization_to_block)
    )
    rst = rst[rst["hospitalization_id"].notna()].copy()
    rst["hospitalization_id"] = rst["hospitalization_id"].astype(str)
    rst = rst[
        ~rst["hospitalization_id"].isin(
            rst[rst["tracheostomy"] == 1].hospitalization_id.unique()
        )
    ]  # exclude trach pats
    return (rst,)


@app.cell
def _(pc, rst):
    rst_1 = pc.convert_datetime_columns_to_site_tz(rst, pc.helper['your_site_timezone'])
    return (rst_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Patient
    """)
    return


@app.cell
def _(pc):
    pat = pc.load_data("clif_patient")
    pat = pat[["patient_id", "sex_category",
            "race_category",
            "ethnicity_category",
            "language_name"]]
    pat = pc.convert_datetime_columns_to_site_tz(pat, pc.helper["your_site_timezone"])
    return (pat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Cohort Filtering
    """)
    return


@app.cell
def _(adt_1, hosp_1, np, pat, pc, pd, rst_1):
    imv_hosp_ids = rst_1[rst_1['device_category'].str.lower() == 'imv'].hospitalization_id.unique()
    icu_hosp_ids = adt_1[adt_1['location_category'].str.lower() == 'icu'].hospitalization_id.unique()
    icu_hosp_ids = [x for x in icu_hosp_ids if x is not None]
    imv_hosp_ids = [x for x in imv_hosp_ids if x is not None]
    hosp_2 = hosp_1[(hosp_1['admission_dttm'].dt.year >= 2022) & (hosp_1['admission_dttm'].dt.year <= 2024) & (hosp_1['discharge_dttm'].dt.year <= 2024) & hosp_1['hospitalization_id'].isin(np.intersect1d(imv_hosp_ids, icu_hosp_ids)) & (hosp_1['age_at_admission'] >= 18) & (hosp_1['age_at_admission'] <= 119)].reset_index(drop=True)
    required_id = hosp_2['hospitalization_id'].unique()
    print(len(required_id), ' : potential cohort count')
    base = pd.merge(hosp_2, pat, on='patient_id', how='inner')[['patient_id', 'hospitalization_id', 'admission_dttm', 'discharge_dttm', 'age_at_admission', 'discharge_category', 'sex_category', 'race_category', 'ethnicity_category', 'language_name']]
    base['admission_dttm'] = pc.getdttm(base['admission_dttm'])
    base.columns
    adt_2 = adt_1[adt_1['hospitalization_id'].isin(required_id)].reset_index(drop=True)
    rst_2 = rst_1[rst_1['hospitalization_id'].isin(required_id)].reset_index(drop=True)
    return adt_2, base, required_id, rst_2


@app.cell
def _(base):
    base.head()
    return


@app.cell
def _(rst_2):
    rst_2.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Water Fall Start
    """)
    return


@app.cell
def _(pc, rst_2):
    ### aswaterfallneedinutc
    rst_2['recorded_dttm'] = rst_2['recorded_dttm'].dt.tz_convert('UTC')
    new_rst = pc.process_resp_support_waterfall(rst_2)
    new_rst.head()
    return (new_rst,)


@app.cell
def _(new_rst, pc):
    new_rst['recorded_dttm'] = new_rst['recorded_dttm'].dt.tz_convert(pc.helper['your_site_timezone'])
    rst_3 = new_rst.copy()
    rst_3['recorded_dttm'] = pc.getdttm(rst_3['recorded_dttm'])
    pc.deftime(rst_3['recorded_dttm'])
    return (rst_3,)


@app.cell
def _(rst_3):
    # V7 Fix: Re-apply tracheostomy exclusion AFTER waterfall forward-fill
    # The waterfall process forward-fills the tracheostomy flag, so patients who
    # become trach'd later in their stay are now properly identified.
    _trach_hosp_ids = rst_3[rst_3['tracheostomy'] == 1]['hospitalization_id'].unique()
    _before = len(rst_3)
    rst_4 = rst_3[~rst_3['hospitalization_id'].isin(_trach_hosp_ids)].reset_index(drop=True)
    print(f'Post-waterfall tracheostomy exclusion: {_before} -> {len(rst_4)} rows (excluded {len(_trach_hosp_ids)} hospitalizations)')
    return (rst_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    water fall end
    """)
    return


@app.cell
def _(np, pc, rst_4):
    if pc.helper['site_name'] == 'RUSH':
        rst_col = ['hospitalization_id', 'recorded_dttm', 'device_category', 'mode_category', 'fio2_set', 'peep_set', 'resp_rate_set', 'pressure_support_set', 'mode_name']
        for extra_col in ['tube_comp_%', 'sbt_timepoint', 'vent_brand_name']:
            if extra_col in rst_4.columns:
                rst_col.append(extra_col)
        rst_4['device_category'] = rst_4['device_category'].replace('nan', np.nan)
    else:
        rst_col = ['hospitalization_id', 'recorded_dttm', 'device_category', 'mode_category', 'fio2_set', 'peep_set', 'resp_rate_set', 'pressure_support_set', 'mode_name']
    rst_col = [c for c in rst_col if c in rst_4.columns]
    rst_5 = rst_4[rst_col]
    rst_5['device_category'] = rst_5['device_category'].str.lower()
    # Defensive: only select columns that exist
    rst_5['mode_category'] = rst_5['mode_category'].str.lower()  # Include RUSH-specific columns only if they exist
    return (rst_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Extubated Flag
    """)
    return


@app.cell
def _(np, rst_5, tqdm):
    # Step 1: Sort and forward-fill device_category by hospitalization_id
    rst_6 = rst_5.sort_values(['hospitalization_id', 'recorded_dttm'])
    rst_6['device_category'] = rst_6.groupby('hospitalization_id')['device_category'].ffill()
    rst_6['device_change'] = (rst_6['device_category'] != rst_6.groupby('hospitalization_id')['device_category'].shift()).astype(int)
    # Step 2: Create a device_segment_id for each change in device_category within a hospitalization
    rst_6['device_segment_id'] = rst_6.groupby('hospitalization_id')['device_change'].cumsum()
    rst_6['mode_category'] = rst_6.groupby(['hospitalization_id', 'device_segment_id'])['mode_category'].ffill()
    rst_6['Device_IMV'] = (rst_6['device_category'] == 'imv').astype(int)
    MAX_EXTUBATION_GAP_HOURS = 24
    rst_6['extubated'] = 0
    for hosp_id, group in tqdm(rst_6.groupby('hospitalization_id'), desc='Generating Extubated Flags'):
    # Step 3: Forward-fill mode_category within each device_segment_id and hospitalization_id
        group = group.sort_values('recorded_dttm')
        idx = group.index
        times = group['recorded_dttm'].values
        imv_vals = group['Device_IMV'].values
    # Step 4: Create Device_IMV column
        for i in range(len(group) - 2):
            if imv_vals[i] == 1 and imv_vals[i + 1] == 0 and (imv_vals[i + 2] == 0):
    # Step 5: Flag extubation when there's a switch from IMV to two consecutive non-IMV entries
    # Fix C4: Add 24-hour time constraint — the 2 consecutive non-IMV observations must
    # occur within 24 hours of each other to count as a true extubation event.
    # Without this, observations days apart could falsely trigger extubation.
                time_gap = (times[i + 2] - times[i]) / np.timedelta64(1, 'h')
                if time_gap <= MAX_EXTUBATION_GAP_HOURS:
                    rst_6.at[idx[i], 'extubated'] = 1  # Check time constraint: both non-IMV observations within MAX_EXTUBATION_GAP_HOURS
    return (rst_6,)


@app.cell
def _(rst_6):
    from definitions_source_of_truth import classify_imv_episodes
    imv_records = rst_6[rst_6['device_category'] == 'imv'][['hospitalization_id', 'recorded_dttm']].copy()
    # Classify IMV episodes using 72-hour gap rule
    imv_records = classify_imv_episodes(imv_records)
    rst_7 = rst_6.merge(imv_records[['hospitalization_id', 'recorded_dttm', 'imv_episode_id']], on=['hospitalization_id', 'recorded_dttm'], how='left')
    rst_7['imv_episode_id'] = rst_7.groupby('hospitalization_id')['imv_episode_id'].ffill()
    # Map episode IDs back to rst
    # Forward-fill episode IDs within hospitalization
    print(f'IMV episodes identified: {rst_7['imv_episode_id'].nunique()}')
    return (rst_7,)


@app.cell
def _(rst_7):
    rst_7.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### MAC
    """)
    return


@app.cell
def _(hospitalization_to_block, pc, required_id):
    mac = pc.load_data("clif_medication_admin_continuous")
    mac["hospitalization_id"] = mac["hospitalization_id"].astype(str)
    mac["hospitalization_id"] = (
        mac["hospitalization_id"].map(hospitalization_to_block).astype(str)
    )
    mac_col = [
        "hospitalization_id",
        "admin_dttm",
        "med_dose",
        "med_category",
        "med_dose_unit",
        "mar_action_group",
    ]

    # Filter to required hospitalizations and med categories
    mac = mac[
        (mac["hospitalization_id"].isin(required_id))
        & (
            mac["med_category"].isin(
                [
                    "norepinephrine",
                    "epinephrine",
                    "phenylephrine",
                    "angiotensin",
                    "vasopressin",
                    "dopamine",
                    "dobutamine",
                    "milrinone",
                    "isoproterenol",
                    "cisatracurium",
                    "vecuronium",
                    "rocuronium",
                    "fentanyl",
                    "propofol",
                    "lorazepam",
                    "midazolam",
                    "hydromorphone",
                    "morphine",
                ]
            )
        )
    ].reset_index(drop=True)

    # Fix 5: Filter to administered records only (CLIF 2.1 requirement)
    if "mar_action_group" in mac.columns:
        _before = len(mac)
        mac = mac[mac["mar_action_group"].str.lower() == "administered"]
        print(f"mar_action_group filter: {_before} -> {len(mac)} rows (removed {_before - len(mac)} non-administered)")
    else:
        print("WARNING: mar_action_group column not found. Cannot filter to administered-only records.")

    mac = mac[mac_col].reset_index(drop=True)

    mac = pc.convert_datetime_columns_to_site_tz(mac, pc.helper["your_site_timezone"])
    mac["admin_dttm"] = pc.getdttm(mac["admin_dttm"])

    mac["med_dose_unit"] = mac["med_dose_unit"].str.lower()
    mac = mac[
        (mac["med_dose_unit"].str.contains(r"/", na=False))
        & (mac["med_dose_unit"] != "units/hr")
    ].reset_index(drop=True)
    return (mac,)


@app.cell
def _(mac, pc):
    pc.deftime(mac['admin_dttm'])
    return


@app.cell
def _(mac):
    mac.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Patient Assessment
    """)
    return


@app.cell
def _(np):
    cat_values_mapping_dict = {
        "negative": 0,
        "fail": 0,
        "pass": 1,
        "positive": 1,
        None: np.nan,
        np.nan: np.nan,
        "yes": 1,
        "no": 0,
    }

    pat_assess_cats_rquired = [
        "sbt_delivery_pass_fail",
        "sbt_screen_pass_fail",
        "sat_delivery_pass_fail",
        "sat_screen_pass_fail",
        "gcs_total",
        "rass",
    ]
    return cat_values_mapping_dict, pat_assess_cats_rquired


@app.cell
def _(pat_assess_cats_rquired, pc):
    pat_at = pc.load_data("clif_patient_assessments", -1)
    pat_at_col = [
        "hospitalization_id",
        "recorded_dttm",
        "numerical_value",
        "categorical_value",
        "assessment_category",
    ]
    pat_at["assessment_category"] = pat_at["assessment_category"].str.lower()
    pat_at = pat_at[(pat_at["assessment_category"].isin(pat_assess_cats_rquired))][
        pat_at_col
    ].reset_index(drop=True)

    pat_at = pc.convert_datetime_columns_to_site_tz(pat_at, pc.helper["your_site_timezone"])
    pat_at["recorded_dttm"] = pc.getdttm(pat_at["recorded_dttm"])
    pc.deftime(pat_at["recorded_dttm"])
    return pat_at, pat_at_col


@app.cell
def _(
    cat_values_mapping_dict,
    hospitalization_to_block,
    pat_at,
    pat_at_col,
    pd,
    required_id,
):
    pat_at['hospitalization_id'] = pat_at['hospitalization_id'].astype(str)
    _unmapped_pat = pat_at['hospitalization_id'].map(hospitalization_to_block).isna().sum()
    if _unmapped_pat > 0:
        print(f'WARNING: {_unmapped_pat} patient_assessment rows have unmapped hospitalization_id (dropped)')
    pat_at['hospitalization_id'] = pat_at['hospitalization_id'].map(hospitalization_to_block)
    pat_at_1 = pat_at[pat_at['hospitalization_id'].notna()].copy()
    pat_at_1['hospitalization_id'] = pat_at_1['hospitalization_id'].astype(str)
    pat_at_1 = pat_at_1[pat_at_1['hospitalization_id'].isin(required_id)][pat_at_col].reset_index(drop=True)
    pat_at_1['categorical_value'] = pat_at_1['categorical_value'].str.lower().map(cat_values_mapping_dict)

    def compute_assessment_value(row):
        if row['assessment_category'].endswith('_pass_fail'):
            return row['categorical_value'] if pd.notnull(row['categorical_value']) else row['numerical_value']
        else:
            return row['numerical_value'] if pd.notnull(row['numerical_value']) else row['categorical_value']
    pat_at_1['assessment_value'] = pat_at_1.apply(compute_assessment_value, axis=1)
    # pat_at["assessment_value"] = pat_at["numerical_value"].combine_first(
    #     pat_at["categorical_value"]
    # )
    #new patch to get category 1st
    # Apply row-wise
    pat_at_1.drop(columns=['numerical_value', 'categorical_value'], inplace=True)
    return (pat_at_1,)


@app.cell
def _(pat_at_1):
    pat_at_1[pat_at_1['assessment_category'] == 'sbt_delivery_pass_fail']['assessment_value'].value_counts(dropna=False)
    return


@app.cell
def _(pat_at_1):
    pat_at_1.assessment_category.unique()
    return


@app.cell
def _(pat_at_1):
    pat_at_1['assessment_value'].value_counts()
    return


@app.cell
def _(pat_at_1):
    pat_at_1.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Vitals
    """)
    return


@app.cell
def _(hospitalization_to_block, pc):
    vit = pc.load_data("clif_vitals", -1)
    vit["hospitalization_id"] = vit["hospitalization_id"].astype(str)
    vit["hospitalization_id"] = (
        vit["hospitalization_id"].map(hospitalization_to_block).astype(str)
    )
    vit_col = ["hospitalization_id", "recorded_dttm_min", "vital_category", "vital_value"]
    vit["vital_category"] = vit["vital_category"].str.lower()


    vit = pc.convert_datetime_columns_to_site_tz(vit, pc.helper["your_site_timezone"])
    return vit, vit_col


@app.cell
def _(pc, required_id, vit, vit_col):
    vit['recorded_dttm_min'] = pc.getdttm(vit['recorded_dttm'])
    pc.deftime(vit['recorded_dttm_min'])
    vit_1 = vit[vit['hospitalization_id'].isin(required_id) & vit['vital_category'].isin(['map', 'heart_rate', 'sbp', 'dbp', 'spo2', 'respiratory_rate', 'weight_kg', 'height_cm'])][vit_col].reset_index(drop=True)
    vit_1 = vit_1.sort_values(by=['hospitalization_id', 'recorded_dttm_min'])
    vit_1 = vit_1.groupby(['hospitalization_id', 'vital_category', 'recorded_dttm_min'], as_index=False).agg({'vital_value': 'first'})
    vit_1['vital_value'] = vit_1['vital_value'].astype(float)
    # Sort by hospitalization_id and recorded_dttm
    # Group by hospitalization_id, vital_category, and recorded_dttm_min, then take the first occurrence of vital_value
    # make sure float
    # for meds
    vit_weight = vit_1[vit_1['vital_category'] == 'weight_kg'].reset_index(drop=True)
    return (vit_1,)


@app.cell
def _(np, vit_1):
    # H10 Fix: Apply outlier thresholds to vitals from config
    import json as _json
    from pathlib import Path as _Path
    _outlier_path = _Path('../config/outlier_config.json')
    try:
        with open(_outlier_path, 'r') as _f:
            outlier_cfg = _json.load(_f)
    except (FileNotFoundError, _json.JSONDecodeError):
        outlier_cfg = {}
    if outlier_cfg:
        _vitals_before = len(vit_1)
        for _vcat, _cfg_key in [('spo2', 'spo2'), ('map', 'map'), ('heart_rate', 'heart_rate'), ('sbp', 'sbp'), ('dbp', 'dbp'), ('respiratory_rate', 'respiratory_rate'), ('weight_kg', 'weight_kg'), ('height_cm', 'height_cm')]:
            if _cfg_key in outlier_cfg:
                _lo, _hi = outlier_cfg[_cfg_key]
                _mask = (vit_1['vital_category'] == _vcat) & ~vit_1['vital_value'].between(_lo, _hi)
                vit_1.loc[_mask, 'vital_value'] = np.nan
        _nulled = vit_1['vital_value'].isna().sum()
        print(f'Vitals outlier thresholds applied. NaN values after: {_nulled}')
    vit_weight_1 = vit_1[vit_1['vital_category'] == 'weight_kg'].reset_index(drop=True)  # Also update vit_weight with cleaned weights
    return outlier_cfg, vit_weight_1


@app.cell
def _(vit_1):
    vit_1.head()
    return


@app.cell
def _(vit_1):
    # Count duplicates
    duplicates = vit_1.duplicated(subset=['hospitalization_id', 'vital_category', 'recorded_dttm_min'], keep=False)
    # Show any duplicates (should be empty if grouping worked correctly)
    vit_1[duplicates]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Med Unit Conversion
    """)
    return


@app.cell
def _(mac, pd, vit_weight_1):
    vit_weight_1.rename({'vital_category': 'med_category', 'recorded_dttm_min': 'admin_dttm'}, axis='columns', inplace=True)
    new_mac = pd.concat([mac, vit_weight_1], ignore_index=True)
    new_mac = new_mac.sort_values(by=['hospitalization_id', 'admin_dttm'])
    new_mac['vital_value'] = new_mac.groupby('hospitalization_id')['vital_value'].ffill().bfill()
    new_mac = new_mac[~(new_mac['med_category'] == 'weight_kg')].reset_index(drop=True)
    # del vit_weight
    print('mac rows:', mac.shape, 'New mac rows:', new_mac.shape)
    return (new_mac,)


@app.cell
def _(new_mac):
    new_mac_1 = new_mac[new_mac['vital_value'] > 0]
    new_mac_1.head(5)
    return (new_mac_1,)


@app.cell
def _(new_mac_1, tqdm):
    # The med_unit_info dictionary
    med_unit_info = {'norepinephrine': {'required_unit': 'mcg/kg/min', 'acceptable_units': ['mcg/kg/min', 'mcg/kg/hr', 'mg/kg/hr', 'mcg/min', 'mg/hr']}, 'epinephrine': {'required_unit': 'mcg/kg/min', 'acceptable_units': ['mcg/kg/min', 'mcg/kg/hr', 'mg/kg/hr', 'mcg/min', 'mg/hr']}, 'phenylephrine': {'required_unit': 'mcg/kg/min', 'acceptable_units': ['mcg/kg/min', 'mcg/kg/hr', 'mg/kg/hr', 'mcg/min', 'mg/hr']}, 'angiotensin': {'required_unit': 'ng/kg/min', 'acceptable_units': ['ng/kg/min', 'ng/kg/hr']}, 'vasopressin': {'required_unit': 'units/min', 'acceptable_units': ['units/min', 'units/hr', 'milliunits/min', 'milliunits/hr']}, 'dopamine': {'required_unit': 'mcg/kg/min', 'acceptable_units': ['mcg/kg/min', 'mcg/kg/hr', 'mg/kg/hr', 'mcg/min', 'mg/hr']}, 'dobutamine': {'required_unit': 'mcg/kg/min', 'acceptable_units': ['mcg/kg/min', 'mcg/kg/hr', 'mg/kg/hr', 'mcg/min', 'mg/hr']}, 'milrinone': {'required_unit': 'mcg/kg/min', 'acceptable_units': ['mcg/kg/min', 'mcg/kg/hr', 'mg/kg/hr', 'mcg/min', 'mg/hr']}, 'isoproterenol': {'required_unit': 'mcg/kg/min', 'acceptable_units': ['mcg/kg/min', 'mcg/kg/hr', 'mg/kg/hr', 'mcg/min', 'mg/hr']}}

    def convert_med_dose(row):
        category = row['med_category']
        if category not in med_unit_info:
            return row
        info = med_unit_info[category]
        required_unit = info['required_unit']
        acceptable_units = info['acceptable_units']
        current_unit = row['med_dose_unit']
        dose = row['med_dose']
        weight = row['vital_value']
        if current_unit == required_unit:
            return row
        if current_unit not in acceptable_units:
            return row
        conversion_factor = 1.0
        if 'kg' in current_unit and 'kg' not in required_unit:
            conversion_factor = conversion_factor * weight
        elif 'kg' not in current_unit and 'kg' in required_unit:
            conversion_factor = conversion_factor / weight
        if 'hr' in current_unit and 'min' in required_unit:
            conversion_factor = conversion_factor / 60.0
        elif 'min' in current_unit and 'hr' in required_unit:
            conversion_factor = conversion_factor * 60.0
        current_med_unit = current_unit.split('/')[0]
        required_med_unit = required_unit.split('/')[0]
        med_conversion = {('mg', 'mcg'): 1000, ('mcg', 'mg'): 0.001, ('milliunits', 'units'): 0.001, ('units', 'milliunits'): 1000}
        if current_med_unit != required_med_unit:
            factor = med_conversion.get((current_med_unit, required_med_unit))
            if factor is not None:
                conversion_factor = conversion_factor * factor
            else:
                return row
        new_dose = dose * conversion_factor
        row['med_dose'] = new_dose
        row['med_dose_unit'] = required_unit
        return row
    tqdm.pandas(desc='Converting medication doses')
    # Apply the conversion function with tqdm for progress tracking
    new_mac_2 = new_mac_1.progress_apply(convert_med_dose, axis=1)  # If the category is not in our dictionary, skip conversion.  # patient's weight in kg  # If the current unit already matches the required unit, nothing to do.  # If the current unit is not in the acceptable list, skip conversion.  # Start with a conversion factor of 1.  # --------------------------------------------------  # 1. Weight conversion: if the current unit is per kg but the required is not,  # then multiply by the patient’s weight.  # --------------------------------------------------  # 2. Time conversion: convert from per hour to per minute or vice versa.  # --------------------------------------------------  # 3. Medication unit conversion (e.g., mg to mcg, milliunits to units)  # We assume the first part (before the first '/') is the measurement unit.  # If no conversion factor is defined, skip conversion.  # --------------------------------------------------  # Apply the conversion  # Update the row with the converted dose and unit.
    return (new_mac_2,)


@app.cell
def _(new_mac_2, np, outlier_cfg):
    # H10 Fix: Apply outlier thresholds to medication doses after unit conversion
    if outlier_cfg:
        _med_before = new_mac_2['med_dose'].notna().sum()
        for _mcat in new_mac_2['med_category'].unique():
            if _mcat in outlier_cfg:
                _lo, _hi = outlier_cfg[_mcat]
                _mask = (new_mac_2['med_category'] == _mcat) & ~new_mac_2['med_dose'].between(_lo, _hi)
                new_mac_2.loc[_mask, 'med_dose'] = np.nan
        _med_after = new_mac_2['med_dose'].notna().sum()
        print(f'Med outlier thresholds applied: {_med_before} -> {_med_after} non-null doses')
    return


@app.cell
def _(new_mac_2):
    # Create a summary table for each med_category
    summary_table = new_mac_2.groupby(['med_category', 'med_dose_unit']).agg(total_N=('med_category', 'size'), min=('med_dose', 'min'), max=('med_dose', 'max'), first_quantile=('med_dose', lambda x: x.quantile(0.25)), second_quantile=('med_dose', lambda x: x.quantile(0.5)), third_quantile=('med_dose', lambda x: x.quantile(0.75)), missing_values=('med_dose', lambda x: x.isna().sum())).reset_index()
    ## check the distrbituon of required continuous meds
    summary_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wide Dataset
    """)
    return


@app.cell
def _(adt_2, base, duckdb, new_mac_2, pat_at_1, pc, rst_7, vit_1):
    duckdb.register('base', base)
    duckdb.register('pat_at', pat_at_1)
    duckdb.register('rst', rst_7)
    duckdb.register('mac', new_mac_2)
    duckdb.register('adt', adt_2)
    duckdb.register('vit', vit_1)
    _q = '\nWITH\n    uni_event_dttm as (\n        select distinct\n            hospitalization_id,\n            event_time\n        from\n            (\n                SELECT\n                    hospitalization_id,\n                    in_dttm AS event_time\n                FROM\n                    adt\n                where\n                    in_dttm is not null\n                UNION\n                SELECT\n                    hospitalization_id,\n                    recorded_dttm AS event_time\n                FROM\n                    rst\n                where\n                    recorded_dttm is not null\n                UNION\n                SELECT\n                    hospitalization_id,\n                    recorded_dttm AS event_time\n                FROM\n                    pat_at\n                where\n                    recorded_dttm is not null\n                UNION\n                SELECT\n                    hospitalization_id,\n                    admin_dttm AS event_time\n                FROM\n                    mac\n                where\n                    admin_dttm is not null\n                UNION\n                SELECT\n                    hospitalization_id,\n                    recorded_dttm_min AS event_time\n                FROM\n                    vit\n                where\n                    recorded_dttm_min is not null\n            ) uni_time\n    )\nselect distinct\n    patient_id,\n    a.hospitalization_id,\n    admission_dttm,\n    discharge_dttm,\n    age_at_admission,\n    discharge_category,\n    sex_category,\n    race_category,\n    ethnicity_category,\n    language_name,\n    event_time\nfrom\n    base a\n    left join uni_event_dttm b on a.hospitalization_id = b.hospitalization_id\n'
    wide_cohort_df = duckdb.sql(_q).df()
    pc.deftime(wide_cohort_df['event_time'])
    return (wide_cohort_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### pivots for assessment and mac table
    """)
    return


@app.cell
def _(duckdb):
    _query = "\nWITH pas_data AS (\n    SELECT  distinct assessment_value ,\tassessment_category\t,\n    hospitalization_id || '_' || strftime(recorded_dttm, '%Y%m%d%H%M') AS combo_id\n    FROM pat_at where recorded_dttm is not null \n) \nPIVOT pas_data\nON assessment_category\nUSING first(assessment_value)\nGROUP BY combo_id\n"
    p_pas = duckdb.sql(_query).df()
    _query = "\nWITH mac_data AS (\n    SELECT  distinct med_dose ,\tmed_category\t,\n    hospitalization_id || '_' || strftime(admin_dttm, '%Y%m%d%H%M') AS combo_id\n    FROM mac where admin_dttm is not null \n) \nPIVOT mac_data\nON med_category\nUSING min(med_dose)\nGROUP BY combo_id\n"
    p_mac = duckdb.sql(_query).df()
    return p_mac, p_pas


@app.cell
def _(duckdb):
    _query = "\nWITH vital_data AS (\n    SELECT  distinct vital_category,\tvital_value\t,\n    hospitalization_id || '_' || strftime(recorded_dttm_min, '%Y%m%d%H%M') AS combo_id\n    FROM vit where recorded_dttm_min is not null \n)\nPIVOT vital_data\nON vital_category\nUSING first(vital_value)\nGROUP BY combo_id\n"
    p_vitals = duckdb.sql(_query).df()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### id-ing all unique timestamps
    """)
    return


@app.cell
def _(duckdb, p_mac, p_pas, wide_cohort_df):
    duckdb.register('expanded_df', wide_cohort_df)
    duckdb.register('p_pas', p_pas)
    duckdb.register('p_mac', p_mac)
    _q = "\n  WITH\n    u_rst as (\n        select\n            *,\n            hospitalization_id || '_' || strftime (recorded_dttm, '%Y%m%d%H%M') AS combo_id\n        from\n            rst\n    ),\n    u_adt as (\n        select\n            *,\n            hospitalization_id || '_' || strftime (in_dttm, '%Y%m%d%H%M') AS combo_id\n        from\n            adt\n    ),\n    u_expanded_df as (\n        select\n            *,\n            hospitalization_id || '_' || strftime (event_time, '%Y%m%d%H%M') AS combo_id\n        from\n            expanded_df\n    )\nselect\n    *\nfrom\n    u_expanded_df a\n    left join u_adt d on a.combo_id = d.combo_id\n    left join u_rst e on a.combo_id = e.combo_id\n    left join p_mac g on a.combo_id = g.combo_id\n    left join p_pas h on a.combo_id = h.combo_id\n    left join p_vitals i on a.combo_id=i.combo_id \n\n                    \n"
    all_join_df = duckdb.sql(_q).df().drop_duplicates()
    return (all_join_df,)


@app.cell
def _(all_join_df):
    all_join_df.shape
    return


@app.cell
def _(wide_cohort_df):
    wide_cohort_df.shape
    return


@app.cell
def _(all_join_df, pd):
    # all_join_df.drop(columns= ['hospitalization_id_2','hospitalization_id_3','combo_id', 'combo_id_2' ,'combo_id_3','combo_id_4','combo_id_5','recorded_dttm','combo_id_6','in_dttm'], axis = 1,inplace=True)
    all_join_df['event_time'] = pd.to_datetime(all_join_df['event_time'])
    all_join_df['date'] = (all_join_df['event_time'] - pd.Timedelta(hours=6)).dt.date
    all_join_df_1 = all_join_df.sort_values(['hospitalization_id', 'event_time']).reset_index(drop=True)
    # Fix 3: Ventilator-day anchor at 06:00 (shift back 6h before flooring to date)
    if 'imv_episode_id' in all_join_df_1.columns:
        all_join_df_1['day_number'] = all_join_df_1.groupby(['hospitalization_id', 'imv_episode_id'])['date'].rank(method='dense').astype('Int64')
        all_join_df_1['hosp_id_day_key'] = all_join_df_1['imv_episode_id'].astype(str) + '_day_' + all_join_df_1['day_number'].astype(str)
    else:
        all_join_df_1['day_number'] = all_join_df_1.groupby('hospitalization_id')['date'].rank(method='dense').astype('Int64')
        all_join_df_1['hosp_id_day_key'] = all_join_df_1['hospitalization_id'].astype(str) + '_day_' + all_join_df_1['day_number'].astype(str)
    # Fix 4: Day numbering within IMV episode (not just hospitalization)
        print('WARNING: imv_episode_id not found. Using hospitalization-level day numbering.')  # Fallback if imv_episode_id not present
    return (all_join_df_1,)


@app.cell
def _(all_join_df_1, np):
    ##SAT SBT columns check
    columns_to_check = ['sbt_delivery_pass_fail', 'sbt_screen_pass_fail', 'sat_delivery_pass_fail', 'sat_screen_pass_fail', 'rass', 'gcs_total']
    for col in columns_to_check:
        if col not in all_join_df_1.columns:
            all_join_df_1[col] = np.nan
            print(f"Column '{col}' is missing for your site. Filling with NaN for now. If this is unintended, please verify your data element.")
    meds_check = ['norepinephrine', 'epinephrine', 'phenylephrine', 'angiotensin', 'vasopressin', 'dopamine', 'dobutamine', 'milrinone', 'isoproterenol', 'cisatracurium', 'vecuronium', 'rocuronium', 'fentanyl', 'propofol', 'lorazepam', 'midazolam', 'hydromorphone', 'morphine']
    for col in meds_check:
        if col not in all_join_df_1.columns:
            all_join_df_1[col] = np.nan
    ## meds check
            print(f"mCide: '{col}' is missing. Please check your CLIF Meds table, it's can be missing if your site doesn't use it.")
    return


@app.cell
def _(all_join_df_1):
    all_join_df_1.to_csv('../output/intermediate/study_cohort.csv', index=False)
    all_join_df_1.to_parquet('../output/intermediate/study_cohort.parquet', index=False)
    return


@app.cell
def _(os, pc):
    directory_path = os.path.join("../output/final/", pc.helper["site_name"])
    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    return


@app.cell
def _():
    print('cohort creation completed!')
    return


if __name__ == "__main__":
    app.run()
