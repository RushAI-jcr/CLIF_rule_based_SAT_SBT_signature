import json
import os
import duckdb
import pandas as pd
import numpy as np
import pytz


def load_config():
    json_path = '../config/config.json'
    
    with open(json_path, 'r') as file:
        config = json.load(file)
    print("Loaded configuration from config.json")
    
    return config

def load_data(table,sample_size=None):
    """
    Load the patient data from a file in the specified directory.

    Returns:
        pd.DataFrame: DataFrame containing patient data.
    """
    # Determine the file path based on the directory and filetype
    file_path = helper['tables_path'] + table + '.' + helper['file_type']
    
    # Load the data based on filetype
    if os.path.exists(file_path):
        if helper['file_type'] == 'csv':
            df = duckdb.read_csv(file_path,sample_size=sample_size).df()
        elif helper['file_type'] == 'parquet':
            df = duckdb.read_parquet(file_path).df()
        else:
            raise ValueError("Unsupported filetype. Only 'csv' and 'parquet' are supported.")
        print(f"Data loaded successfully from {file_path}")
        return df
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist in the specified directory.")
    
def deftime(df):
    
    # Count entries with both hours and minutes
    has_hr_min = df.notna() & (df.dt.hour.notna() & df.dt.minute.notna())
    count_with_hr_min = has_hr_min.sum()

    # Count entries without hours and minutes
    count_without_hr_min = (~has_hr_min).sum()

    # Print the results
    print(f"Count with hours and minutes: {count_with_hr_min}")
    print(f"Count without hours and minutes: {count_without_hr_min}")


def getdttm(df, cutby='min'):
    """
    Convert datetime to the required format, remove timezone if present,
    ensure format is always '%Y-%m-%d %H:%M:%S', and ceil to minute if needed.
    """
    # Convert column to datetime, handling unknown formats
    dt_series = pd.to_datetime(df, errors='coerce')

    # Remove timezone if present
    if dt_series.dt.tz is not None:
        dt_series = dt_series.dt.tz_localize(None)
    # If already naive, no conversion needed

    # Ceil to the nearest minute if cutby='min'
    if cutby == 'min':
        dt_series = dt_series.dt.ceil('min')

    # Format the output to ensure consistent '%Y-%m-%d %H:%M:%S'
    return dt_series

def process_resp_support(df):
    """
    Process the respiratory support data using waterfall logic.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing respiratory support data.
        
    Returns:
        pd.DataFrame: Processed DataFrame with filled values.
    """
    print("Initiating waterfall processing...")
    
    # Ensure 'recorded_dttm' is in datetime format
    df['recorded_dttm'] = pd.to_datetime(df['recorded_dttm'])
    
    # Convert categories to lowercase to standardize
    df['device_category'] = df['device_category'].str.lower()
    df['mode_category'] = df['mode_category'].str.lower()
    df['device_name'] = df['device_name'].str.lower()
    df['mode_name'] = df['mode_name'].str.lower()

    #dtype cast
    float_columns = ['fio2_set', 'flow_rate_set', 'inspiratory_time_set', 'lpm_set',
           'mean_airway_pressure_obs', 'minute_vent_obs', 'peak_inspiratory_pressure_obs', 'peak_inspiratory_pressure_set',
           'peep_obs', 'peep_set', 'plateau_pressure_obs', 'pressure_support_set',
           'resp_rate_obs', 'resp_rate_set', 'tidal_volume_obs',
           'tidal_volume_set', 'pressure_control_set']

    for col in float_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Converts to float and handles non-numeric values
    
    # # Fix out-of-range values
    # print("Fixing out-of-range values for 'fio2_set', 'peep_set', and 'resp_rate_set'...")
    # df['fio2_set'] = df['fio2_set'].where(df['fio2_set'].between(0.21, 1), np.nan)
    # df['peep_set'] = df['peep_set'].where(df['peep_set'].between(0, 50), np.nan)
    # df['resp_rate_set'] = df['resp_rate_set'].where(df['resp_rate_set'].between(0, 60), np.nan)
    
    # Create 'recorded_date' and 'recorded_hour'
    print('Creating recorded_date and recorded_hour...')
    df['recorded_date'] = df['recorded_dttm'].dt.date
    df['recorded_hour'] = df['recorded_dttm'].dt.hour
    
    # Sort data
    print("Sorting data by 'hospitalization_id' and 'recorded_dttm'...")
    df.sort_values(by=['hospitalization_id', 'recorded_dttm'], inplace=True)
    
    # Fix missing 'device_category' and 'device_name' based on 'mode_category'
    print("Fixing missing 'device_category' and 'device_name' based on 'mode_category'...")
    mask = (
        df['device_category'].isna() &
        df['device_name'].isna() &
        df['mode_category'].str.contains('assist control-volume control|simv|pressure control', case=False, na=False)
    )
    df.loc[mask, 'device_category'] = 'imv'
    df.loc[mask, 'device_name'] = 'mechanical ventilator'
    
    # Fix 'device_category' and 'device_name' based on neighboring records
    print("Fixing 'device_category' and 'device_name' based on neighboring records...")
    # Create shifted columns once to avoid multiple shifts
    df['device_category_shifted'] = df['device_category'].shift()
    df['device_category_shifted_neg'] = df['device_category'].shift(-1)
    
    condition_prev = (
        df['device_category'].isna() &
        (df['device_category_shifted'] == 'imv') &
        df['resp_rate_set'].gt(1) &
        df['peep_set'].gt(1)
    )
    condition_next = (
        df['device_category'].isna() &
        (df['device_category_shifted_neg'] == 'imv') &
        df['resp_rate_set'].gt(1) &
        df['peep_set'].gt(1)
    )
    
    condition = condition_prev | condition_next
    df.loc[condition, 'device_category'] = 'imv'
    df.loc[condition, 'device_name'] = 'mechanical ventilator'
    
    # Drop the temporary shifted columns
    df.drop(['device_category_shifted', 'device_category_shifted_neg'], axis=1, inplace=True)
    
    # Handle duplicates and missing data
    print("Handling duplicates and removing rows with all key variables missing...")
    df['n'] = df.groupby(['hospitalization_id', 'recorded_dttm'])['recorded_dttm'].transform('size')
    df = df[~((df['n'] > 1) & (df['device_category'] == 'nippv'))]
    df = df[~((df['n'] > 1) & (df['device_category'].isna()))]
    subset_vars = ['device_category', 'device_name', 'mode_category', 'mode_name', 'fio2_set']
    df.dropna(subset=subset_vars, how='all', inplace=True)
    df.drop_duplicates(subset=['hospitalization_id', 'recorded_dttm'], keep='first', inplace=True)
    df.drop('n', axis=1, inplace=True)  # Drop 'n' as it's no longer needed
    
    # Fill forward 'device_category' within each hospitalization
    print("Filling forward 'device_category' within each hospitalization...")
    df['device_category'] = df.groupby('hospitalization_id')['device_category'].ffill()
    
    # Create 'device_cat_id' based on changes in 'device_category'
    print("Creating 'device_cat_id' to track changes in 'device_category'...")
    df['device_cat_f'] = df['device_category'].fillna('missing').astype('category').cat.codes
    df['device_cat_change'] = df['device_cat_f'] != df.groupby('hospitalization_id')['device_cat_f'].shift()
    df['device_cat_change'] = df['device_cat_change'].astype(int)
    df['device_cat_id'] = df.groupby('hospitalization_id')['device_cat_change'].cumsum()
    df.drop('device_cat_change', axis=1, inplace=True)
    
    # Fill 'device_name' within 'device_cat_id'
    print("Filling 'device_name' within each 'device_cat_id'...")
    df['device_name'] = df.groupby(['hospitalization_id', 'device_cat_id'])['device_name'].ffill().bfill()
    
    # Create 'device_id' based on changes in 'device_name'
    print("Creating 'device_id' to track changes in 'device_name'...")
    df['device_name_f'] = df['device_name'].fillna('missing').astype('category').cat.codes
    df['device_name_change'] = df['device_name_f'] != df.groupby('hospitalization_id')['device_name_f'].shift()
    df['device_name_change'] = df['device_name_change'].astype(int)
    df['device_id'] = df.groupby('hospitalization_id')['device_name_change'].cumsum()
    df.drop('device_name_change', axis=1, inplace=True)
    
    # Fill 'mode_category' within 'device_id'
    print("Filling 'mode_category' within each 'device_id'...")
    df['mode_category'] = df.groupby(['hospitalization_id', 'device_id'])['mode_category'].ffill().bfill()
    
    # Create 'mode_cat_id' based on changes in 'mode_category'
    print("Creating 'mode_cat_id' to track changes in 'mode_category'...")
    df['mode_cat_f'] = df['mode_category'].fillna('missing').astype('category').cat.codes
    df['mode_cat_change'] = df['mode_cat_f'] != df.groupby(['hospitalization_id', 'device_id'])['mode_cat_f'].shift()
    df['mode_cat_change'] = df['mode_cat_change'].astype(int)
    df['mode_cat_id'] = df.groupby(['hospitalization_id', 'device_id'])['mode_cat_change'].cumsum()
    df.drop('mode_cat_change', axis=1, inplace=True)
    
    # Fill 'mode_name' within 'mode_cat_id'
    print("Filling 'mode_name' within each 'mode_cat_id'...")
    df['mode_name'] = df.groupby(['hospitalization_id', 'mode_cat_id'])['mode_name'].ffill().bfill()
    
    # Create 'mode_name_id' based on changes in 'mode_name'
    print("Creating 'mode_name_id' to track changes in 'mode_name'...")
    df['mode_name_f'] = df['mode_name'].fillna('missing').astype('category').cat.codes
    df['mode_name_change'] = df['mode_name_f'] != df.groupby(['hospitalization_id', 'mode_cat_id'])['mode_name_f'].shift()
    df['mode_name_change'] = df['mode_name_change'].astype(int)
    df['mode_name_id'] = df.groupby(['hospitalization_id', 'mode_cat_id'])['mode_name_change'].cumsum()
    df.drop('mode_name_change', axis=1, inplace=True)
    
    # Adjust 'fio2_set' for 'room air' device_category
    print("Adjusting 'fio2_set' for 'room air' device_category...")
    df['fio2_set'] = np.where(df['fio2_set'].isna() & (df['device_category'] == 'room air'), 0.21, df['fio2_set'])
    
    # Adjust 'mode_category' for 't-piece' devices
    print("Adjusting 'mode_category' for 't-piece' devices...")
    mask_tpiece = (
        df['mode_category'].isna() &
        df['device_name'].str.contains('t-piece', case=False, na=False)
    )
    df.loc[mask_tpiece, 'mode_category'] = 'blow by'
    
    # Fill remaining variables within 'mode_name_id'
    print("Filling remaining variables within each 'mode_name_id'...")
    fill_vars = [
        'fio2_set', 'lpm_set', 'peep_set', 'resp_rate_set',
        'resp_rate_obs'
    ]
    df[fill_vars] = df.groupby(['hospitalization_id', 'mode_name_id'])[fill_vars].transform(lambda x: x.ffill().bfill())
    
    # Fill 'tracheostomy' forward within each hospitalization
    print("Filling 'tracheostomy' forward within each hospitalization...")
    df['tracheostomy'] = df.groupby('hospitalization_id')['tracheostomy'].ffill()
    
    # Remove duplicates
    print("Removing duplicates...")
    df.drop_duplicates(inplace=True)
    
    # # Select relevant columns
    # columns_to_keep = [
    #     'hospitalization_id', 'recorded_dttm', 'recorded_date', 'recorded_hour',
    #     'device_category', 'device_name', 'mode_category', 'mode_name',
    #     'device_cat_id', 'device_id', 'mode_cat_id', 'mode_name_id',
    #     'fio2_set', 'lpm_set', 'peep_set', 'resp_rate_set',
    #     'tracheostomy', 'resp_rate_obs','pressure_support_set'
    # ]
    # # Ensure columns exist before selecting
    # existing_columns = [col for col in columns_to_keep if col in df.columns]
    # df = df[existing_columns]
    
    print("Waterfall processing completed.")
    return df

def stitch_encounters(hospitalization, adt, time_interval=6):
    """
    Stitches together related hospital encounters that occur within a specified time interval of each other.
    
    Args:
        hospitalization (pd.DataFrame): Hospitalization table with required columns
        adt (pd.DataFrame): ADT table with required columns
        time_interval (int, optional): Number of hours between encounters to consider them linked. Defaults to 6.
        
    Returns:
        pd.DataFrame: Stitched encounters with encounter blocks
    """
    hospitalization_filtered = hospitalization[["patient_id","hospitalization_id","admission_dttm","discharge_dttm","age_at_admission"]].copy()
    hospitalization_filtered['admission_dttm'] = pd.to_datetime(hospitalization_filtered['admission_dttm'])
    hospitalization_filtered['discharge_dttm'] = pd.to_datetime(hospitalization_filtered['discharge_dttm'])

    hosp_adt_join = pd.merge(hospitalization_filtered[["patient_id","hospitalization_id","admission_dttm","discharge_dttm"]],
                      adt[["hospitalization_id","in_dttm","out_dttm","location_category","hospital_id"]],
                 on="hospitalization_id",how="left")

    hospital_cat = hosp_adt_join[["hospitalization_id","in_dttm","out_dttm","hospital_id"]]

    # Step 1: Sort by patient_id and admission_dttm
    hospital_block = hosp_adt_join[["patient_id","hospitalization_id","admission_dttm","discharge_dttm"]]
    hospital_block = hospital_block.drop_duplicates()
    hospital_block = hospital_block.sort_values(by=["patient_id", "admission_dttm"]).reset_index(drop=True)
    hospital_block = hospital_block[["patient_id","hospitalization_id","admission_dttm","discharge_dttm"]]

    # Step 2: Calculate time between discharge and next admission
    hospital_block["next_admission_dttm"] = hospital_block.groupby("patient_id")["admission_dttm"].shift(-1)
    hospital_block["discharge_to_next_admission_hrs"] = (
        (hospital_block["next_admission_dttm"] - hospital_block["discharge_dttm"]).dt.total_seconds() / 3600
    )

    # Step 3: Create linked column based on time_interval
    hospital_block["linked6hrs"] = hospital_block["discharge_to_next_admission_hrs"] < time_interval

    # Sort values to ensure correct order
    hospital_block = hospital_block.sort_values(by=["patient_id", "admission_dttm"]).reset_index(drop=True)

    # Initialize encounter_block with row indices + 1
    hospital_block['encounter_block'] = hospital_block.index + 1

    # Iteratively propagate the encounter_block values
    while True:
        shifted = hospital_block['encounter_block'].shift(-1)
        mask = hospital_block['linked6hrs'] & (hospital_block['patient_id'] == hospital_block['patient_id'].shift(-1))
        hospital_block.loc[mask, 'encounter_block'] = shifted[mask]
        if hospital_block['encounter_block'].equals(hospital_block['encounter_block'].bfill()):
            break

    hospital_block['encounter_block'] = hospital_block['encounter_block'].bfill().astype(int)
    hospital_block = pd.merge(hospital_block,hospital_cat,how="left",on="hospitalization_id")
    hospital_block = hospital_block.sort_values(by=["patient_id", "admission_dttm","in_dttm","out_dttm"]).reset_index(drop=True)
    hospital_block = hospital_block.drop_duplicates()

    hospital_block2 = hospital_block.groupby(['patient_id','encounter_block']).agg(
        admission_dttm=pd.NamedAgg(column='admission_dttm', aggfunc='min'),
        discharge_dttm=pd.NamedAgg(column='discharge_dttm', aggfunc='max'),
        hospital_id = pd.NamedAgg(column='hospital_id', aggfunc='last'),
        list_hospitalization_id=pd.NamedAgg(column='hospitalization_id', aggfunc=lambda x: sorted(x.unique()))
    ).reset_index()

    df = pd.merge(hospital_block[["patient_id",
                                  "hospitalization_id",
                                  "encounter_block"]].drop_duplicates(),
             hosp_adt_join[["hospitalization_id","location_category","in_dttm","out_dttm"]], on="hospitalization_id",how="left")

    df = pd.merge(df,hospital_block2[["encounter_block",
                                      "admission_dttm",
                                      "discharge_dttm",
                                      "hospital_id",
                                     "list_hospitalization_id"]],on="encounter_block",how="left")
    df = df.drop_duplicates(subset=["patient_id","encounter_block","in_dttm","out_dttm","location_category"])
    
    return df

def standardize_datetime_tz(df, dttm_columns, your_timezone):
   
    if isinstance(dttm_columns, str):
        dttm_columns = [dttm_columns]  # Convert single column name to list
    
    if your_timezone==None:
        return df

    for col in dttm_columns:
        if col in df.columns:
            # Convert to datetime if not already
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # Check if timezone-aware
            if df[col].dt.tz is None:
                print(f"Column {col} is timezone aware: {df[col].dt.tz}")
                df[col] = df[col].dt.tz_localize(your_timezone, ambiguous='NaT', nonexistent='shift_forward')
            else:
                df[col] = df[col].dt.tz_convert(your_timezone)
            # Remove timezone and convert to standard format
            df[col] = df[col].dt.tz_convert(None)
        else:
            print(f"Couldn't find {col} column in this df")
    return df


def convert_datetime_columns_to_site_tz(df, site_tz_str, verbose=True):
    """
    Convert all datetime columns in the DataFrame to the specified site timezone.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - site_tz_str (str): Timezone string, e.g., "America/New_York".
    - verbose (bool): Whether to print detailed output (default: True).

    Returns:
    - pd.DataFrame: Modified DataFrame with datetime columns converted.
    """
    site_tz = pytz.timezone(site_tz_str)

    # Identify datetime-related columns
    dttm_columns = [col for col in df.columns if 'dttm' in col]

    for col in dttm_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

        if pd.api.types.is_datetime64tz_dtype(df[col]):
            current_tz = df[col].dt.tz
            if current_tz == site_tz:
                if verbose:
                    print(f"{col}: Already in your timezone ({current_tz}), no conversion needed.")
            elif current_tz == pytz.UTC:
                df[col] = df[col].dt.tz_convert(site_tz)
                if verbose:
                    print(f"{col}: Converted from UTC to your timezone ({site_tz}).")
            else:
                df[col] = df[col].dt.tz_convert(site_tz)
                if verbose:
                    print(f"{col}: Your timezone is {current_tz}, Converting to your site timezone ({site_tz}).")
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            if verbose:
                df[col] = df[col].dt.tz_localize(site_tz, ambiguous=True, nonexistent='shift_forward')
                print(f"WARNING: {col}: Naive datetime, NOT converting. Assuming it's in your LOCAL ZONE. Please check ETL!")
        else:
            if verbose:
                print(f"WARNING: {col}: Not a datetime column. Please check ETL and run again!")

        if verbose:
            print(f"{col}: null count = {df[col].isna().sum()}")

    return df


def process_resp_support_waterfall(
    resp_support: pd.DataFrame,
    *,
    id_col: str = "hospitalization_id",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Clean + water-fall-fill the CLIF `resp_support` table **exactly** like the
    Nick's reference R pipeline.

    Parameters
    ----------
    resp_support : pd.DataFrame
        Raw `clif_respiratory_support` table **already in UTC**.
        All datetime columns must be timezone-aware as UTC.
    id_col : str, default ``"hospitalization_id"``
        Encounter-level identifier.
    verbose : bool, default ``True``
        If *True* prints progress banners for every phase.

    Returns
    -------
    pd.DataFrame
        Fully processed respiratory-support DataFrame with

        *   hourly scaffold rows inserted (at ``HH:59:59``),
        *   device / mode heuristics applied,
        *   hierarchical IDs: ``device_cat_id ➜ device_id ➜
            mode_cat_id ➜ mode_name_id``,
        *   bidirectional numeric waterfall fill inside the most granular ID,
        *   tracheostomy forward-filled,
        *   one row per timestamp (duplicate-safe), ordered
            ``id_col, recorded_dttm``.

    Notes
    -----
    * This function **does not** change timezones; convert before calling
      if needed.
    * No column re-ordering trick is used (avoids duplicated column error).
    """

    p = print if verbose else (lambda *a, **k: None)

    # ------------------------------------------------------------------ #
    # Phase 0 – set-up & hourly scaffold                                 #
    # ------------------------------------------------------------------ #
    p("✦ Phase 0 – initialise & create hourly scaffold")

    rs = resp_support.copy()

    # Normalise categorical strings
    for c in ["device_category", "device_name", "mode_category", "mode_name"]:
        if c in rs.columns:
            rs[c] = rs[c].str.lower()

    # Numeric coercion
    num_cols = [
        "tracheostomy",
        "fio2_set",
        "lpm_set",
        "peep_set",
        "tidal_volume_set",
        "resp_rate_set",
        "resp_rate_obs",
        "pressure_support_set",
        "peak_inspiratory_pressure_set",
    ]
    num_cols = [c for c in num_cols if c in rs.columns]
    rs[num_cols] = rs[num_cols].apply(pd.to_numeric, errors="coerce")

    rs['fio2_set'] = pd.to_numeric(rs['fio2_set'], errors='coerce')
    # (Optional) If FiO2 is >1 on average => scale by /100
    fio2_mean = rs['fio2_set'].mean(skipna=True)
    # If the mean is greater than 1, divide 'fio2_set' by 100
    if fio2_mean and fio2_mean > 1.0:
        # Only divide values greater than 1 to avoid re-dividing already correct values
        rs.loc[rs['fio2_set'] > 1, 'fio2_set'] = \
            rs.loc[rs['fio2_set'] > 1, 'fio2_set'] / 100
        print("Updated fio2_set to be between 0.21 and 1")
    else:
        print("FIO2_SET mean=", fio2_mean, "is within the required range")

    # Helpers
    rs["recorded_date"] = rs["recorded_dttm"].dt.date
    rs["recorded_hour"] = rs["recorded_dttm"].dt.hour

    # Hourly scaffold rows (HH:59:59)
    min_max = (
        rs.groupby(id_col)["recorded_dttm"]
        .agg(["min", "max"])
        .reset_index()
    )
    scaffold = (
        min_max.apply(
            lambda r: pd.date_range(
                r["min"].floor("h"),
                r["max"].floor("h"),
                freq="1h",
                tz="UTC",
            ),
            axis=1,
        )
        .explode()
        .rename("recorded_dttm")
    )
    scaffold = (
        min_max[[id_col]]
        .join(scaffold)
        .assign(
            recorded_dttm=lambda d: d["recorded_dttm"].dt.floor("h")
            + pd.Timedelta(minutes=59, seconds=59)
        )
    )
    scaffold["recorded_date"] = scaffold["recorded_dttm"].dt.date
    scaffold["recorded_hour"] = scaffold["recorded_dttm"].dt.hour
    scaffold["is_scaffold"] = True
    # rs = pd.concat([rs, scaffold], ignore_index=True).sort_values(
    #     [id_col, "recorded_dttm", "recorded_date", "recorded_hour"]
    # )

    # ------------------------------------------------------------------ #
    # Phase 1 – heuristics to infer / clean device & mode                #
    # ------------------------------------------------------------------ #
    p("✦ Phase 1 – heuristic inference of device / mode")

    # Most-common names (used as fall-back labels)
    device_counts = (
        rs[["device_name", "device_category"]]
        .value_counts()
        .to_frame("count")
        .reset_index()
    )
    _imv_names = device_counts.loc[device_counts["device_category"] == "imv", "device_name"]
    most_common_imv_name = _imv_names.iloc[0] if len(_imv_names) > 0 else "mechanical ventilator"
    _nippv_names = device_counts.loc[device_counts["device_category"] == "nippv", "device_name"]
    most_common_nippv_name = _nippv_names.iloc[0] if len(_nippv_names) > 0 else "bipap"

    mode_counts = (
        rs[["mode_name", "mode_category"]]
        .value_counts()
        .to_frame("count")
        .reset_index()
    )
    _cmv_names = mode_counts.loc[
        mode_counts["mode_category"] == "assist control-volume control", "mode_name"
    ]
    most_common_cmv_name = _cmv_names.iloc[0] if len(_cmv_names) > 0 else "volume control"

    # 1-a  fill IMV from mode_category
    mask = (
        rs["device_category"].isna()
        & rs["device_name"].isna()
        & rs["mode_category"].str.contains(
            r"(?:assist control-volume control|simv|pressure control)", na=False
        )
    )
    rs.loc[mask, ["device_category", "device_name"]] = ["imv", most_common_imv_name]

    # missing IMV name
    rs.loc[
        (rs["device_category"] == "imv") & rs["device_name"].isna(),
        "device_name",
    ] = most_common_imv_name

    # 1-b  IMV heuristics (look-behind / look-ahead)
    rs = rs.sort_values([id_col, "recorded_dttm"])
    prev_cat = rs.groupby(id_col)["device_category"].shift()
    next_cat = rs.groupby(id_col)["device_category"].shift(-1)

    imv_like = (
        rs["device_category"].isna()
        & ((prev_cat == "imv") | (next_cat == "imv"))
        & rs["peep_set"].gt(1)
        & rs["resp_rate_set"].gt(1)
        & rs["tidal_volume_set"].gt(1)
    )
    rs.loc[imv_like, ["device_category", "device_name"]] = ["imv", most_common_imv_name]

    # 1-c  NIPPV heuristics
    prev_cat = rs.groupby(id_col)["device_category"].shift()
    next_cat = rs.groupby(id_col)["device_category"].shift(-1)
    nippv_like = (
        rs["device_category"].isna()
        & ((prev_cat == "nippv") | (next_cat == "nippv"))
        & rs["peak_inspiratory_pressure_set"].gt(1)
        & rs["pressure_support_set"].gt(1)
    )
    rs.loc[nippv_like, "device_category"] = "nippv"
    rs.loc[nippv_like & rs["device_name"].isna(), "device_name"] = most_common_nippv_name

    # 1-d  clearly IMV again
    prev_cat = rs.groupby(id_col)["device_category"].shift()
    next_cat = rs.groupby(id_col)["device_category"].shift(-1)

    back_to_imv = (
        rs["device_category"].isna()
        & ~rs["device_name"].str.contains("trach", na=False)
        & ((prev_cat == "imv") | (next_cat == "imv"))
        & rs["tidal_volume_set"].gt(0)
        & rs["resp_rate_set"].gt(0)
    )
    rs.loc[back_to_imv, ["device_category", "device_name"]] = [
        "imv",
        most_common_imv_name,
    ]

    fill_mode_mask = back_to_imv & rs["mode_category"].isna()
    rs.loc[
        fill_mode_mask, ["mode_category", "mode_name"]
    ] = ["assist control-volume control", most_common_cmv_name]

    # 1-e  duplicate timestamp handling
    rs["dup_count"] = rs.groupby([id_col, "recorded_dttm"])["recorded_dttm"].transform(
        "size"
    )
    rs = rs[~((rs["dup_count"] > 1) & (rs["device_category"] == "nippv"))]
    rs["dup_count"] = rs.groupby([id_col, "recorded_dttm"])["recorded_dttm"].transform(
        "size"
    )
    rs = rs[~((rs["dup_count"] > 1) & rs["device_category"].isna())].drop(
        columns="dup_count"
    )

    # 1-f  random carried-over BiPAP before trach-collar
    lead_dev = rs.groupby(id_col)["device_category"].shift(-1)
    lag_dev = rs.groupby(id_col)["device_category"].shift()
    drop_bipap = (
        (rs["device_category"] == "nippv")
        & (lead_dev == "trach collar")
        & (lag_dev != "nippv")
    )
    rs = rs[~drop_bipap]

    # 1-g  rows with nothing useful
    all_na_cols = [
        "device_category",
        "device_name",
        "mode_category",
        "mode_name",
        "tracheostomy",
        "fio2_set",
        "lpm_set",
        "peep_set",
        "tidal_volume_set",
        "resp_rate_set",
        "resp_rate_obs",
        "pressure_support_set",
        "peak_inspiratory_pressure_set",
    ]
    rs = rs.dropna(subset=all_na_cols, how="all")

    # unique per timestamp
    rs = rs.drop_duplicates(subset=[id_col, "recorded_dttm"], keep="first")

    rs["is_scaffold"] = False 
    rs = pd.concat([rs, scaffold], ignore_index=True).sort_values(
        [id_col, "recorded_dttm", "recorded_date", "recorded_hour"]
    )

    # ------------------------------------------------------------------ #
    # Phase 2 – hierarchical IDs                                         #
    # ------------------------------------------------------------------ #
    p("✦ Phase 2 – build device / mode hierarchical IDs")

    def change_id(col: pd.Series, by: pd.Series) -> pd.Series:
        return (
            col.fillna("missing")
            .groupby(by)
            .transform(lambda s: s.ne(s.shift()).cumsum())
            .astype("int32")
        )

    # 2-A  device_cat_id
    rs["device_category"] = rs.groupby(id_col)["device_category"].ffill()
    rs["device_cat_id"] = change_id(rs["device_category"], rs[id_col])

    # 2-B  device_id
    rs["device_name"] = (
        rs.sort_values("recorded_dttm")
        .groupby([id_col, "device_cat_id"])["device_name"]
        .transform(lambda s: s.ffill().bfill()).infer_objects(copy=False)
    )
    rs["device_id"] = change_id(rs["device_name"], rs[id_col])

    # 2-C  mode_cat_id
    rs = rs.sort_values([id_col, "recorded_dttm"])
    rs["mode_category"] = (
        rs.groupby([id_col, "device_id"])["mode_category"]
        .transform(lambda s: s.ffill().bfill()).infer_objects(copy=False)
    )
    dev_curr = rs["device_id"]
    dev_prev = rs.groupby(id_col)["device_id"].shift()
    mode_curr = rs["mode_category"].fillna("missing")
    mode_prev = rs.groupby(id_col)["mode_category"].shift().fillna("missing")
    mode_cat_bump = ((dev_curr != dev_prev) | (mode_curr != mode_prev)).astype(int)
    rs["mode_cat_id"] = mode_cat_bump.groupby(rs[id_col]).cumsum().astype("int32")

    # 2-D  mode_name_id
    rs["mode_name"] = (
        rs.groupby([id_col, "mode_cat_id"])["mode_name"]
        .transform(lambda s: s.ffill().bfill()).infer_objects(copy=False)
    )
    cat_curr = rs["mode_cat_id"]
    cat_prev = rs.groupby(id_col)["mode_cat_id"].shift()
    name_curr = rs["mode_name"].fillna("missing")
    name_prev = rs.groupby(id_col)["mode_name"].shift().fillna("missing")
    mode_name_bump = ((cat_curr != cat_prev) | (name_curr != name_prev)).astype(int)
    rs["mode_name_id"] = mode_name_bump.groupby(rs[id_col]).cumsum().astype("int32")

    # ------------------------------------------------------------------ #
    # Phase 3 – numeric waterfall inside mode_name_id                    #
    # ------------------------------------------------------------------ #
    p("✦ Phase 3 – numeric down/up-fill inside mode_name_id blocks")

    # FiO2 default for room-air
    ra_mask = (rs["device_category"] == "room air") & rs["fio2_set"].isna()
    rs.loc[ra_mask, "fio2_set"] = 0.21

    # Bad tidal-volume rows → NA
    tv_bad = (
        ((rs["mode_category"] == "pressure support/cpap") & rs["pressure_support_set"].notna())
        | (rs["mode_category"].isna() & rs["device_name"].str.contains("trach", na=False))
        | (
            (rs["mode_category"] == "pressure support/cpap")
            & rs["device_name"].str.contains("trach", na=False)
        )
    )
    rs.loc[tv_bad, "tidal_volume_set"] = np.nan

    num_cols_fill = [
        c
        for c in [
            "fio2_set",
            "lpm_set",
            "peep_set",
            "tidal_volume_set",
            "pressure_support_set",
            "resp_rate_set",
            "resp_rate_obs",
            "peak_inspiratory_pressure_set",
        ]
        if c in rs.columns
    ]

    def fill_block(g: pd.DataFrame) -> pd.DataFrame:
        if (g["device_category"] == "trach collar").any():
            breaker = (g["device_category"] == "trach collar").cumsum()
            return (
                g.groupby(breaker)[num_cols_fill]
                .apply(lambda x: x.ffill().bfill()).infer_objects(copy=False)
                # .apply(lambda x: x.ffill()).infer_objects(copy=False)
            )
        return g[num_cols_fill].ffill().bfill()
        # return g[num_cols_fill].ffill()

    rs[num_cols_fill] = (
        rs.groupby([id_col, "mode_name_id"], group_keys=False, sort=False)
        .apply(fill_block)
    )

    # “t-piece” rows with blank mode_category → classify as blow-by
    tpiece_mask = rs['mode_category'].isna() & rs['device_name'].str.contains('t-piece', na=False)
    rs.loc[tpiece_mask, 'mode_category'] = 'blow by'

    # Tracheostomy forward-fill only
    rs["tracheostomy"] = rs.groupby(id_col)["tracheostomy"].ffill()

    # ------------------------------------------------------------------ #
    # Phase 4 – final tidy-up                                            #
    # ------------------------------------------------------------------ #
    p("✦ Phase 4 – final deduplication & ordering")
    rs = (
        rs.drop_duplicates()
        .sort_values([id_col, "recorded_dttm"])
        .reset_index(drop=True)
    )
    # rs = rs[~rs["is_scaffold"]] 
    # rs = rs.drop(columns="is_scaffold")

    # Final columns to drop
    drop_cols = [
        "recorded_date",
        "recorded_hour"
        # "device_cat_id",
        # "device_id",
        # "mode_cat_id",
        # "mode_name_id",
    ]
    drop_cols = [c for c in drop_cols if c in rs.columns]
    rs = rs.drop(columns=drop_cols)

    p("✅ Respiratory-support waterfall complete.")
    return rs


try:
    helper = load_config()
    print(helper)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"WARNING: Could not load config.json: {e}")
    print("Set pyCLIF.helper manually or ensure ../config/config.json exists.")
    helper = {}