#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys, os
sys.path.insert(0, os.path.join(os.pardir, "utils"))
import numpy as np
import pandas as pd
import re
import pyCLIF as pc
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import warnings

warnings.filterwarnings("ignore")
from tableone import TableOne

import pySBT as sbt

# cohort = pd.read_csv("../output/intermediate/study_cohort.csv")
cohort = pd.read_parquet('../output/intermediate/study_cohort.parquet')

cohort['hospital_id'] = cohort['hospital_id'].str.replace(r'[^a-zA-Z]', '', regex=True)


# In[ ]:


## Analysis by
by = 'Respiratory_Stability'

# Construct the full directory path
directory_path = os.path.join("../output/final/", pc.helper["site_name"], f"SBT_{by}")

# Create the directory if it doesn't exist
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"Directory '{directory_path}' created.")
else:
    print(f"Directory '{directory_path}' already exists.")


# In[ ]:


if pc.helper["site_name"] == "RUSH":
    cohort.loc[
        cohort["sbt_timepoint"] == "3-5 minute evaluation", "pressure_support_set"
    ] = 6.1
    cohort.loc[cohort["sbt_timepoint"] == "3-5 minute evaluation", "mode_category"] = (
        "Pressure Support/CPAP"
    )
    print("its a rush thing")


# # Eligibility Flag making

# In[ ]:


t1_cohort = cohort.copy()

# Ensure all time columns are in datetime format
cohort["event_time"] = pd.to_datetime(cohort["event_time"])
cohort["admission_dttm"] = pc.getdttm(cohort["admission_dttm"])
cohort["discharge_dttm"] = pc.getdttm(cohort["discharge_dttm"])

# Ensure the data is sorted by 'hosp_id_day_key' and 'event_time'
cohort = cohort.sort_values(by=["hospitalization_id", "event_time"]).reset_index(
    drop=True
)

cohort["device_category"] = cohort["device_category"].str.lower()
cohort["mode_category"] = cohort["mode_category"].str.lower()

# Fill forward's
cohort[
    [
        "device_category",
        "mode_category",
        "mode_name",
        "location_category",
        "hospital_id",
    ]
] = cohort.groupby("hospitalization_id")[
    [
        "device_category",
        "mode_category",
        "mode_name",
        "location_category",
        "hospital_id",
    ]
].ffill()


cohort[["weight_kg", "height_cm"]] = (
    cohort.groupby("hospitalization_id")[["weight_kg", "height_cm"]].ffill().bfill()
)

cohort[
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
    ]
] = cohort.groupby("hospitalization_id")[
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
    ]
].ffill()

cohort[["fio2_set", "peep_set", "spo2", "pressure_support_set"]] = cohort.groupby(
    "hospitalization_id"
)[["fio2_set", "peep_set", "spo2", "pressure_support_set"]].ffill()

cohort[
    [
        "norepinephrine",
        "epinephrine",
        "phenylephrine",
        "dopamine",
        "angiotensin",
        "vasopressin",
    ]
] = cohort[
    [
        "norepinephrine",
        "epinephrine",
        "phenylephrine",
        "dopamine",
        "angiotensin",
        "vasopressin",
    ]
].fillna(
    0
)

cohort["NEE"] = (
    cohort["norepinephrine"]
    + cohort["epinephrine"]
    + (cohort["phenylephrine"] / 10)
    + (cohort["vasopressin"] * 2.5)
    + (cohort["dopamine"] / 100)
    + (cohort["angiotensin"] * 10)
)

cohort["Hemodynamic_Stability_by_NEE"] = (((cohort["NEE"] <= 0.2))).astype(int)

# Define Respiratory Stability Flag
cohort["Respiratory_Stability"] = (
    (cohort["fio2_set"] <= 0.5) & (cohort["peep_set"] <= 8) & (cohort["spo2"] >= 88)
).astype(int)

# Fill forward the paralytic by hospitalization columns by 'hosp_id'
cohort[["cisatracurium", "vecuronium", "rocuronium"]] = cohort.groupby(
    "hospitalization_id"
)[["cisatracurium", "vecuronium", "rocuronium"]].ffill()

# paralytic max to remove from consideration
cohort["max_paralytics"] = (
    cohort[["cisatracurium", "vecuronium", "rocuronium"]]
    .max(axis=1, skipna=True)
    .fillna(0)
)


# ## SBT Eligibility Criteria

# In[ ]:


final_df = sbt.process_cohort_conditions(cohort,by)


# In[ ]:


# Print statistics
print("By n = Days")
total_days = final_df["hosp_id_day_key"].nunique()
print("Total number of days for eval in cohort:", total_days)
total_vent_days = final_df[final_df["vent_day"] == 1]["hosp_id_day_key"].nunique()
print(
    "Total number of vent days for eval in cohort: (atleast one IMV event)",
    total_vent_days,
)

total_vent_days_wo_paralytics = final_df[final_df["vent_day_without_paralytics"] == 1][
    "hosp_id_day_key"
].nunique()
print(
    "Total number of vent days for eval in cohort: (atleast one IMV event & no paralytics given)",
    total_vent_days_wo_paralytics,
)

eligible_days = final_df[final_df["eligible_day"] == 1]["hosp_id_day_key"].nunique()

percentage = (
    (eligible_days / total_vent_days_wo_paralytics) * 100 if total_days > 0 else 0
)
print(
    f"Eligible days: {eligible_days} / {total_vent_days_wo_paralytics} ({percentage:.2f}%)"
)
print(
    "Hospital days with atleast one IMV event: ",
    final_df[final_df["device_category"] == "imv"]["hosp_id_day_key"].nunique(),
)
print(
    "Hospital days with atleast one IMV & ICU event: ",
    final_df[
        (final_df["device_category"] == "imv")
        & (final_df["location_category"] == "icu")
    ]["hosp_id_day_key"].nunique(),
)

print("By n = Encounter")
h_total_days = final_df["hospitalization_id"].nunique()
print("Total number of days for eval in cohort:", h_total_days)
h_eligible_days = final_df[final_df["eligible_day"] == 1][
    "hospitalization_id"
].nunique()
h_percentage = (h_eligible_days / h_total_days) * 100 if h_total_days > 0 else 0
print(f"Eligible days: {h_eligible_days} / {h_total_days} ({h_percentage:.2f}%)")
print(
    "Hospital days with atleast one IMV event: ",
    final_df[final_df["device_category"] == "imv"]["hospitalization_id"].nunique(),
)
print(
    "Hospital days with atleast one IMV & ICU event: ",
    final_df[
        (final_df["device_category"] == "imv")
        & (final_df["location_category"] == "icu")
    ]["hospitalization_id"].nunique(),
)


# ## FLIP Check

# In[ ]:


final_df = sbt.process_diagnostic_flip_sbt_optimized_v2(final_df)


# In[ ]:


final_df = sbt.apply_2_45_extubated_flag(final_df)


# In[ ]:


final_df = sbt.compute_time_to_extubation(final_df)


# In[ ]:


# Drop NA hospital_ids and get unique ones
hospital_ids = final_df['hospital_id'].dropna().unique()

# Define hourly bins (0–1440 mins, i.e., 24 hrs) and labels
bins = list(range(0, 24 * 60 + 1, 60))  # 0 to 1440 mins in 60-min intervals
labels = [f'{i}-{i+1}hr' for i in range(24)]  # '0-1hr', '1-2hr', ..., '23-24hr'

# List to store per-hospital rush count rows
rush_summary = []

# Loop over each hospital
for hosp in hospital_ids:
    # Filter and bin delta times
    delta_series = final_df[final_df['hospital_id'] == hosp]['delta_to_extubation_mins'].dropna()
    delta_binned = pd.cut(delta_series, bins=bins, labels=labels, right=False)

    # Count entries per bin and convert to dictionary
    rush_counts = delta_binned.value_counts().sort_index()
    rush_counts_dict = rush_counts.to_dict()

    # Add hospital_id to the result row
    rush_counts_dict['hospital_id'] = hosp

    # Append to summary list
    rush_summary.append(rush_counts_dict)

    pd.DataFrame(final_df[final_df['hospital_id'] == hosp]['delta_to_extubation_mins'].describe()).to_csv(f"{directory_path}/delta_stats_between_EHR30Min_Extubated_{hosp}.csv")

# Convert all to a DataFrame
rush_df = pd.DataFrame(rush_summary)

# Fill missing bins (if any hospital didn’t have extubations in certain bins)
rush_df = rush_df.fillna(0).astype({col: 'int' for col in rush_df.columns if col != 'hospital_id'})

# Save to CSV
rush_df.to_csv(f"{directory_path}/rush_counts_by_hour_per_hospital.csv", index=False)


# In[ ]:


delta_series = final_df.delta_to_extubation_mins.dropna()
# Create bins for each hour till 24 hours
bins = list(range(0, 24*60 + 1, 60))  # from 0 to 1440 minutes (24 hrs) with 60-min intervals
labels = [f'{i}-{i+1}hr' for i in range(24)]  # Label bins as '0-1hr', '1-2hr', ..., '23-24hr'
delta_binned = pd.cut(delta_series, bins=bins, labels=labels, right=False)

# Count the number of entries in each bin
rush_counts = delta_binned.value_counts().sort_index()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(rush_counts.index, rush_counts.values, marker='o')
plt.title('Overall: Count of Extubation Events per Hour Bin after EHR signature (30 mins)')
plt.xlabel('Hours since event (binned)')
plt.ylabel('Number of Extubations')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.show()


# ## Results Section

# In[ ]:


final_df["sbt_bkp"] = final_df["sbt_delivery_pass_fail"]
final_df["sbt_delivery_pass_fail"] = final_df["sbt_delivery_pass_fail"].astype(float).map(
    {0: 1, 1: 1}
)
final_df["sbt_screen_pass_fail"] = final_df["sbt_screen_pass_fail"].astype(float).map({0: 1, 1: 1})

# fill forward reason of flip fail
final_df["flip_skip_reason"] = final_df.groupby("hosp_id_day_key")[
    "flip_skip_reason"
].transform(lambda x: x.ffill().bfill())


# In[ ]:


# Ensure the specified columns are treated as datetime before calculating percentages
datetime_columns = ["EHR_Delivery_2mins", "EHR_Delivery_30mins"]

for col in datetime_columns:
    if col in final_df.columns:
        final_df[col] = final_df[col].notna().astype(int)


# In[ ]:


# Group and aggregate the DataFrame including the extubation check
grouped_df = (
    final_df.groupby("hosp_id_day_key")
    .agg(
        {
            "hospitalization_id": "first",
            "hospital_id": lambda x: (
                x.dropna().iloc[-1] if x.dropna().size > 0 else np.nan
            ),
            "eligible_day": "max",
            "EHR_Delivery_2mins": "max",
            "EHR_Delivery_30mins": "max",
            "sbt_screen_pass_fail": "max",
            "sbt_delivery_pass_fail": "max",
            "flag_2_45_extubated": "max",  # Uncomment if needed
            "flip_skip_reason": lambda x: (
                x.dropna().iloc[-1] if x.dropna().size > 0 else np.nan
            ),
            "extubated": "max",
        }
    )
    .reset_index()
)

# Rename the aggregated device_category column to extubated and fill NaN values
mat_df = grouped_df[grouped_df["eligible_day"] == 1]


# #### Basic counts

# In[ ]:


# Drop NA hospital_ids and get unique ones
hospital_ids = grouped_df["hospital_id"].dropna().unique()

# Container for summary rows
summary_data = []

# Loop over hospitals and compute stats
for hosp in hospital_ids:
    # Filter data for the hospital and eligible days
    df_hosp = grouped_df[grouped_df["hospital_id"] == hosp]
    df_eligible = df_hosp[df_hosp["eligible_day"] == 1]

    # Calculate condition-specific sets (unique hosp_id_day_keys)
    sbt_S = set(
        df_eligible[df_eligible["sbt_screen_pass_fail"] == 1][
            "hosp_id_day_key"
        ].unique()
    )
    sbt_D = set(
        df_eligible[df_eligible["sbt_delivery_pass_fail"] == 1][
            "hosp_id_day_key"
        ].unique()
    )
    ehr_2min = set(
        df_eligible[df_eligible["EHR_Delivery_2mins"] == 1]["hosp_id_day_key"].unique()
    )
    ehr_30min = set(
        df_eligible[df_eligible["EHR_Delivery_30mins"] == 1]["hosp_id_day_key"].unique()
    )
    ehr_extubated = set(
        df_eligible[df_eligible["extubated"] == 1]["hosp_id_day_key"].unique()
    )
    ehr_2min_45min_extubated = set(
        df_eligible[df_eligible["flag_2_45_extubated"] == 1]["hosp_id_day_key"].unique()
    )

    # Append aggregated counts to summary list
    summary_data.append(
        {
            "hospital_id": hosp,
            "sbt_screen_pass": len(sbt_S),
            "sbt_delivery_pass": len(sbt_D),
            "ehr_2min": len(ehr_2min),
            "ehr_30min": len(ehr_30min),
            "extubated": len(ehr_extubated),
            "ehr_2min_45min_extubated": len(ehr_2min_45min_extubated),
        }
    )

    # Optionally print the stats
    print(f"\nHospital ID: {hosp}")
    print(f"  SBT Screen Pass: {len(sbt_S)}")
    print(f"  SBT Delivery Pass: {len(sbt_D)}")
    print(f"  EHR 2-min Delivery: {len(ehr_2min)}")
    print(f"  EHR 30-min Delivery: {len(ehr_30min)}")
    print(f"  Extubated: {len(ehr_extubated)}")
    print(f"  ehr_2min_45min_extubated: {len(ehr_2min_45min_extubated)}")

# Convert summary list to DataFrame
summary_df = pd.DataFrame(summary_data)
summary_df

summary_df.to_csv(f"{directory_path}/hospital_sbt_ehr_summary_within_eligible_day.csv", index=False)


# ##### EHR 2 Min vs SBT Flag

# In[ ]:


hospital_ids = mat_df["hospital_id"].unique()
mat_df["sbt_delivery_pass_fail"] = mat_df["sbt_delivery_pass_fail"].fillna(0)

for hosp in hospital_ids:
    # Filter the DataFrame for the current hospital
    df_hosp = mat_df[mat_df["hospital_id"] == hosp]
    if df_hosp["sbt_delivery_pass_fail"].nunique() <= 1:
        continue
    # Create the confusion matrix using pd.crosstab
    conf_matrix = pd.crosstab(
        df_hosp["EHR_Delivery_2mins"], df_hosp["sbt_delivery_pass_fail"]
    )

    # Calculate percentages for each cell
    conf_matrix_percent = conf_matrix / conf_matrix.values.sum() * 100

    # Create annotation labels that combine count and percentage
    annot = (
        conf_matrix.astype(str) + "\n" + conf_matrix_percent.round(1).astype(str) + "%"
    )

    # Plot the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        conf_matrix,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=["0", "1"],
        yticklabels=["0", "1"],
    )
    plt.xlabel("SBT Delivery in Flowsheet")
    plt.ylabel("EHR Delivery in 2 minutes")
    plt.title(f"Confusion Matrix for Hospital {hosp}")
    # Save the plot as a PNG file
    plt.savefig(f"{directory_path}/confusion_matrix_{hosp}_by_SBT.png")
    plt.close()  # Close the plot to free memory

    # Extract ground truth and predictions for the current hospital
    y_true = df_hosp["EHR_Delivery_2mins"]
    y_pred = df_hosp["sbt_delivery_pass_fail"]

    # Compute the confusion matrix and extract TP, FP, FN, TN
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate individual metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    # Print metrics for current hospital (optional)
    print(f"Hospital ID: {hosp}")
    print(f"Accuracy    : {accuracy:.3f}")
    print(f"Precision   : {precision:.3f}")
    print(f"Recall      : {recall:.3f}")
    print(f"F1 Score    : {f1:.3f}")
    print(f"Specificity : {specificity:.3f}\n")

    # Create a dictionary with the computed metrics
    metrics_dict = {
        "True Positives (TP)": tp,
        "False Positives (FP)": fp,
        "False Negatives (FN)": fn,
        "True Negatives (TN)": tn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Specificity": specificity,
    }

    # Build a DataFrame to store the metrics
    df_metrics = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])

    # Save the metrics DataFrame as a CSV file
    df_metrics.to_csv(f"{directory_path}/EHR_2min_vs_SBT_metrics_{hosp}.csv", index=False)
    print(hosp, df_metrics)


# ##### EHR 2 Min vs Extubated Flag

# In[ ]:


hospital_ids = mat_df["hospital_id"].unique()
mat_df["extubated"] = mat_df["extubated"].fillna(0)
for hosp in hospital_ids:
    # Filter the DataFrame for the current hospital
    df_hosp = mat_df[mat_df["hospital_id"] == hosp]

    # Create the confusion matrix using pd.crosstab
    conf_matrix = pd.crosstab(df_hosp["EHR_Delivery_2mins"], df_hosp["extubated"])

    # Calculate percentages for each cell
    conf_matrix_percent = conf_matrix / conf_matrix.values.sum() * 100

    # Create annotation labels that combine count and percentage
    annot = (
        conf_matrix.astype(str) + "\n" + conf_matrix_percent.round(1).astype(str) + "%"
    )

    # Plot the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        conf_matrix,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=["0", "1"],
        yticklabels=["0", "1"],
    )
    plt.xlabel("extubated")
    plt.ylabel("EHR Delivery in 2 minutes")
    plt.title(f"Confusion Matrix for Hospital {hosp}")
    # Save the plot as a PNG file
    plt.savefig(f"{directory_path}/confusion_matrix_{hosp}_by_extubated.png")
    plt.close()  # Close the plot to free memory

    # Extract ground truth and predictions for the current hospital
    y_true = df_hosp["EHR_Delivery_2mins"]
    y_pred = df_hosp["extubated"]

    # Compute the confusion matrix and extract TP, FP, FN, TN
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate individual metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    # Print metrics for current hospital (optional)
    print(f"Hospital ID: {hosp}")
    print(f"Accuracy    : {accuracy:.3f}")
    print(f"Precision   : {precision:.3f}")
    print(f"Recall      : {recall:.3f}")
    print(f"F1 Score    : {f1:.3f}")
    print(f"Specificity : {specificity:.3f}\n")

    # Create a dictionary with the computed metrics
    metrics_dict = {
        "True Positives (TP)": tp,
        "False Positives (FP)": fp,
        "False Negatives (FN)": fn,
        "True Negatives (TN)": tn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Specificity": specificity,
    }

    # Build a DataFrame to store the metrics
    df_metrics = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])

    # Save the metrics DataFrame as a CSV file
    df_metrics.to_csv(
        f"{directory_path}/EHR_2min_VS_EXTUBATED_metrics_{hosp}.csv", index=False
    )
    print(hosp, df_metrics)


# ##### EHR 30 Min vs Extubated Flag

# In[ ]:


hospital_ids = mat_df["hospital_id"].unique()
mat_df["extubated"] = mat_df["extubated"].fillna(0)
for hosp in hospital_ids:
    # Filter the DataFrame for the current hospital
    df_hosp = mat_df[mat_df["hospital_id"] == hosp]

    # Create the confusion matrix using pd.crosstab
    conf_matrix = pd.crosstab(df_hosp["EHR_Delivery_30mins"], df_hosp["extubated"])

    # Calculate percentages for each cell
    conf_matrix_percent = conf_matrix / conf_matrix.values.sum() * 100

    # Create annotation labels that combine count and percentage
    annot = (
        conf_matrix.astype(str) + "\n" + conf_matrix_percent.round(1).astype(str) + "%"
    )

    # Plot the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        conf_matrix,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=["0", "1"],
        yticklabels=["0", "1"],
    )
    plt.xlabel("extubated")
    plt.ylabel("EHR Delivery in 30 minutes")
    plt.title(f"Confusion Matrix for Hospital {hosp}")
    # Save the plot as a PNG file
    plt.savefig(f"{directory_path}/ehr_30_confusion_matrix_{hosp}_by_extubated.png")
    plt.close()  # Close the plot to free memory

    # Extract ground truth and predictions for the current hospital
    y_true = df_hosp["EHR_Delivery_30mins"]
    y_pred = df_hosp["extubated"]

    # Compute the confusion matrix and extract TP, FP, FN, TN
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate individual metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    # Print metrics for current hospital (optional)
    print(f"Hospital ID: {hosp}")
    print(f"Accuracy    : {accuracy:.3f}")
    print(f"Precision   : {precision:.3f}")
    print(f"Recall      : {recall:.3f}")
    print(f"F1 Score    : {f1:.3f}")
    print(f"Specificity : {specificity:.3f}\n")

    # Create a dictionary with the computed metrics
    metrics_dict = {
        "True Positives (TP)": tp,
        "False Positives (FP)": fp,
        "False Negatives (FN)": fn,
        "True Negatives (TN)": tn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Specificity": specificity,
    }

    # Build a DataFrame to store the metrics
    df_metrics = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])

    # Save the metrics DataFrame as a CSV file
    df_metrics.to_csv(
        f"{directory_path}/EHR_30_VS_EXTUBATED_metrics_{hosp}.csv", index=False
    )
    print(hosp, df_metrics)


# ##### EHR 30 Min VS SBT Flag

# In[ ]:


hospital_ids = mat_df["hospital_id"].unique()
mat_df["sbt_delivery_pass_fail"] = mat_df["sbt_delivery_pass_fail"].fillna(0)

for hosp in hospital_ids:
    # Filter the DataFrame for the current hospital
    df_hosp = mat_df[mat_df["hospital_id"] == hosp]
    if df_hosp["sbt_delivery_pass_fail"].nunique() <= 1:
        continue
    # Create the confusion matrix using pd.crosstab
    conf_matrix = pd.crosstab(
        df_hosp["EHR_Delivery_30mins"], df_hosp["sbt_delivery_pass_fail"]
    )

    # Calculate percentages for each cell
    conf_matrix_percent = conf_matrix / conf_matrix.values.sum() * 100

    # Create annotation labels that combine count and percentage
    annot = (
        conf_matrix.astype(str) + "\n" + conf_matrix_percent.round(1).astype(str) + "%"
    )

    # Plot the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        conf_matrix,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=["0", "1"],
        yticklabels=["0", "1"],
    )
    plt.xlabel("SBT Delivery in Flowsheet")
    plt.ylabel("EHR Delivery in 30 minutes")
    plt.title(f"Confusion Matrix for Hospital {hosp}")
    # Save the plot as a PNG file
    plt.savefig(f"{directory_path}/ehr_30_confusion_matrix_{hosp}_by_SBT.png")
    plt.close()  # Close the plot to free memory

    # Extract ground truth and predictions for the current hospital
    y_true = df_hosp["EHR_Delivery_30mins"]
    y_pred = df_hosp["sbt_delivery_pass_fail"]

    # Compute the confusion matrix and extract TP, FP, FN, TN
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate individual metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    # Print metrics for current hospital (optional)
    print(f"Hospital ID: {hosp}")
    print(f"Accuracy    : {accuracy:.3f}")
    print(f"Precision   : {precision:.3f}")
    print(f"Recall      : {recall:.3f}")
    print(f"F1 Score    : {f1:.3f}")
    print(f"Specificity : {specificity:.3f}\n")

    # Create a dictionary with the computed metrics
    metrics_dict = {
        "True Positives (TP)": tp,
        "False Positives (FP)": fp,
        "False Negatives (FN)": fn,
        "True Negatives (TN)": tn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Specificity": specificity,
    }

    # Build a DataFrame to store the metrics
    df_metrics = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])

    # Save the metrics DataFrame as a CSV file
    df_metrics.to_csv(f"{directory_path}/EHR_30_vs_SBT_metrics_{hosp}.csv", index=False)
    print(hosp, df_metrics)


# #### Failure To Detect FLIP vs SBT Flag

# In[ ]:


hospital_ids = mat_df['hospital_id'].unique()

for hosp in hospital_ids:
    # -------------------------------
    # Filter the data for the current hospital
    # -------------------------------
    mat_hosp = mat_df[mat_df['hospital_id'] == hosp]
    
    # -------------------------------
    # Step 1: Extract filtered keys from mat_hosp
    # -------------------------------
    filtered_keys = mat_hosp.loc[
        (mat_hosp['EHR_Delivery_2mins'] == 0) & (mat_hosp['sbt_delivery_pass_fail'] == 1),
        'hosp_id_day_key'
    ].unique()
    
    # -------------------------------
    # Step 2: Filter final_hosp using these keys
    # -------------------------------
    final_filtered_df = final_df.loc[
        (final_df['sbt_delivery_pass_fail'] == 1) & 
        (final_df['hosp_id_day_key'].isin(filtered_keys))
    ]
    
    final_filtered_df = final_filtered_df.sort_values('event_time')
    final_filtered_df = final_filtered_df.drop_duplicates(subset='hosp_id_day_key', keep='first')
    
    print(f"Hospital: {hosp}, final_filtered_df shape: {final_filtered_df.shape}")
    
    # -------------------------------
    # Work on a copy for filtering steps
    # -------------------------------
    df = final_filtered_df.copy()
    results = []
    
    # ---------------------------------------
    # Step 1: Filter on 'flip_skip_reason'
    # ---------------------------------------
    step1 = df[~df['flip_skip_reason'].isna()]
    results.append({
        'Step': 'Step 1',
        'FilterColumn': 'flip_skip_reason',
        'UniqueKeys': step1['hosp_id_day_key'].nunique(),
        'RowCount': step1.shape[0],
        'ValueCounts': step1['flip_skip_reason'].value_counts(dropna=False).to_dict()
    })
    df = df[~df['hosp_id_day_key'].isin(step1['hosp_id_day_key'])]
    
    # ---------------------------------------
    # Step 2: Filter on 'cond_device_imv'
    # ---------------------------------------
    step2 = df[~df['cond_device_imv'].isna()]
    results.append({
        'Step': 'Step 2',
        'FilterColumn': 'cond_device_imv',
        'UniqueKeys': step2['hosp_id_day_key'].nunique(),
        'RowCount': step2.shape[0],
        'ValueCounts': step2['cond_device_imv'].value_counts(dropna=False).to_dict()
    })
    df = df[~df['hosp_id_day_key'].isin(step2['hosp_id_day_key'])]
    
    # ---------------------------------------
    # Step 3: Filter on 'cond_location_icu'
    # ---------------------------------------
    step3 = df[~df['cond_location_icu'].isna()]
    results.append({
        'Step': 'Step 3',
        'FilterColumn': 'cond_location_icu',
        'UniqueKeys': step3['hosp_id_day_key'].nunique(),
        'RowCount': step3.shape[0],
        'ValueCounts': step3['cond_location_icu'].value_counts(dropna=False).to_dict()
    })
    df = df[~df['hosp_id_day_key'].isin(step3['hosp_id_day_key'])]
    
    # ---------------------------------------
    # Step 4: Filter on 'cond_peep_set_le8'
    # ---------------------------------------
    step4 = df[~df['cond_peep_set_le8'].isna()]
    results.append({
        'Step': 'Step 4',
        'FilterColumn': 'cond_peep_set_le8',
        'UniqueKeys': step4['hosp_id_day_key'].nunique(),
        'RowCount': step4.shape[0],
        'ValueCounts': step4['cond_peep_set_le8'].value_counts(dropna=False).to_dict()
    })
    df = df[~df['hosp_id_day_key'].isin(step4['hosp_id_day_key'])]
    
    # ---------------------------------------
    # Step 5: Filter on 'cond_ps_set_le8'
    # ---------------------------------------
    step5 = df[~df['cond_ps_set_le8'].isna()]
    results.append({
        'Step': 'Step 5',
        'FilterColumn': 'cond_ps_set_le8',
        'UniqueKeys': step5['hosp_id_day_key'].nunique(),
        'RowCount': step5.shape[0],
        'ValueCounts': step5['cond_ps_set_le8'].value_counts(dropna=False).to_dict()
    })
    df = df[~df['hosp_id_day_key'].isin(step5['hosp_id_day_key'])]
    
    # ---------------------------------------
    # Step 6: Filter on 'cond_mode_ps_cpap'
    # ---------------------------------------
    step6 = df[~df['cond_mode_ps_cpap'].isna()]
    results.append({
        'Step': 'Step 6',
        'FilterColumn': 'cond_mode_ps_cpap',
        'UniqueKeys': step6['hosp_id_day_key'].nunique(),
        'RowCount': step6.shape[0],
        'ValueCounts': step6['cond_mode_ps_cpap'].value_counts(dropna=False).to_dict()
    })
    df = df[~df['hosp_id_day_key'].isin(step6['hosp_id_day_key'])]
    
    # ---------------------------------------
    # Step 7: Remaining (unmatched) rows
    # ---------------------------------------
    step7 = df.copy()
    results.append({
        'Step': 'Step 7 (Unmatched)',
        'FilterColumn': None,
        'UniqueKeys': step7['hosp_id_day_key'].nunique(),
        'RowCount': step7.shape[0],
        'ValueCounts': None
    })
    
    # ---------------------------------------
    # Create Detailed Step-by-Step Summary DataFrame
    # ---------------------------------------
    detailed_summary_df = pd.DataFrame(results)
    
    # Calculate total_failures as the sum of UniqueKeys across all steps for this hospital
    total_failures = detailed_summary_df['UniqueKeys'].sum()
    
   
    
    # Add "% Per 100" and "% of Total" columns
    detailed_summary_df['% by eligible_days'] = detailed_summary_df['UniqueKeys'].apply(
        lambda x: round((x / eligible_days) * 100, 2)
    )
    detailed_summary_df['% of Total'] = detailed_summary_df['UniqueKeys'].apply(
        lambda x: round((x / total_failures) * 100, 2) if total_failures != 0 else 0
    )
    
    # ---------------------------------------
    # Save the detailed summary DataFrame as a CSV file for the current hospital
    # ---------------------------------------
    output_filename = f"{directory_path}/EHR_VS_SBT_failure_dependent_summary_{hosp}.csv"
    detailed_summary_df.to_csv(output_filename, index=False)
    print(f"Saved detailed summary for hospital {hosp} to {output_filename}\n")
    print(hosp,detailed_summary_df)
    print()

    # ============================================================
    # B. Independent Filtering Summary (Apply each filter independently)
    # ============================================================
    ind_step1 = final_filtered_df[~final_filtered_df['flip_skip_reason'].isna()]
    ind_step2 = final_filtered_df[~final_filtered_df['cond_device_imv'].isna()]
    ind_step3 = final_filtered_df[~final_filtered_df['cond_location_icu'].isna()]
    ind_step4 = final_filtered_df[~final_filtered_df['cond_peep_set_le8'].isna()]
    ind_step5 = final_filtered_df[~final_filtered_df['cond_ps_set_le8'].isna()]
    ind_step6 = final_filtered_df[~final_filtered_df['cond_mode_ps_cpap'].isna()]
    
    # Determine the union of keys matched by any filter
    matched_keys = set().union(
        ind_step1['hosp_id_day_key'],
        ind_step2['hosp_id_day_key'],
        ind_step3['hosp_id_day_key'],
        ind_step4['hosp_id_day_key'],
        ind_step5['hosp_id_day_key'],
        ind_step6['hosp_id_day_key']
    )
    # Unmatched keys: those not included in any of the independent filters
    ind_step7 = final_filtered_df[~final_filtered_df['hosp_id_day_key'].isin(matched_keys)]
    
    # Compute unique key counts per filter
    failure_counts = {
        'flip_skip_reason': ind_step1['hosp_id_day_key'].nunique(),
        'cond_device_imv': ind_step2['hosp_id_day_key'].nunique(),
        'cond_location_icu': ind_step3['hosp_id_day_key'].nunique(),
        'cond_peep_set_le8': ind_step4['hosp_id_day_key'].nunique(),
        'cond_ps_set_le8': ind_step5['hosp_id_day_key'].nunique(),
        'cond_mode_ps_cpap': ind_step6['hosp_id_day_key'].nunique(),
        'unmatched': ind_step7['hosp_id_day_key'].nunique()
    }
    
    # Compute value counts for each filter column
    value_counts_map = {
        'flip_skip_reason': ind_step1['flip_skip_reason'].value_counts(dropna=False).to_dict(),
        'cond_device_imv': ind_step2['cond_device_imv'].value_counts(dropna=False).to_dict(),
        'cond_location_icu': ind_step3['cond_location_icu'].value_counts(dropna=False).to_dict(),
        'cond_peep_set_le8': ind_step4['cond_peep_set_le8'].value_counts(dropna=False).to_dict(),
        'cond_ps_set_le8': ind_step5['cond_ps_set_le8'].value_counts(dropna=False).to_dict(),
        'cond_mode_ps_cpap': ind_step6['cond_mode_ps_cpap'].value_counts(dropna=False).to_dict(),
        'unmatched': None
    }
    
    total_failures_ind = sum(failure_counts.values())
    summary_data = []
    for reason, count in failure_counts.items():
        summary_data.append({
            'Failure Reason': reason,
            'Count': count,
            '% by eligible_days': round((count / eligible_days) * 100, 2),
            '% of Total (out of total failed cases)': round((count / total_failures_ind) * 100, 2) if total_failures_ind else 0,
            'Value Counts': value_counts_map[reason]
        })
    
    independent_summary_df = pd.DataFrame(summary_data)
    independent_summary_df = independent_summary_df.sort_values(by='Count', ascending=False).reset_index(drop=True)
    
    ind_output_filename = f"{directory_path}/EHR_VS_SBT_failure_independent_summary_hospital_{hosp}.csv"
    independent_summary_df.to_csv(ind_output_filename, index=False)
    print(f"Saved independent summary for hospital {hosp} to {ind_output_filename}\n")
    print(hosp, independent_summary_df)
    print()


# #### Failure To Detect FLIP vs Extubated Flag

# In[ ]:


hospital_ids = mat_df['hospital_id'].unique()

for hosp in hospital_ids:
    # -------------------------------
    # Filter the data for the current hospital
    # -------------------------------
    mat_hosp = mat_df[mat_df['hospital_id'] == hosp]
    
    # -------------------------------
    # Step 1: Extract filtered keys from mat_hosp
    # -------------------------------
    filtered_keys = mat_hosp.loc[
        (mat_hosp['EHR_Delivery_2mins'] == 0) & (mat_hosp['extubated'] == 1),
        'hosp_id_day_key'
    ].unique()
    
    # -------------------------------
    # Step 2: Filter final_hosp using these keys
    # -------------------------------
    final_filtered_df = final_df.loc[
        (final_df['extubated'] == 1) & 
        (final_df['hosp_id_day_key'].isin(filtered_keys))
    ]
    
    final_filtered_df = final_filtered_df.sort_values('event_time')
    final_filtered_df = final_filtered_df.drop_duplicates(subset='hosp_id_day_key', keep='first')
    
    print(f"Hospital: {hosp}, final_filtered_df shape: {final_filtered_df.shape}")
    
    # -------------------------------
    # Work on a copy for filtering steps
    # -------------------------------
    df = final_filtered_df.copy()
    results = []
    
    # ---------------------------------------
    # Step 1: Filter on 'flip_skip_reason'
    # ---------------------------------------
    step1 = df[~df['flip_skip_reason'].isna()]
    results.append({
        'Step': 'Step 1',
        'FilterColumn': 'flip_skip_reason',
        'UniqueKeys': step1['hosp_id_day_key'].nunique(),
        'RowCount': step1.shape[0],
        'ValueCounts': step1['flip_skip_reason'].value_counts(dropna=False).to_dict()
    })
    df = df[~df['hosp_id_day_key'].isin(step1['hosp_id_day_key'])]
    
    # ---------------------------------------
    # Step 2: Filter on 'cond_device_imv'
    # ---------------------------------------
    step2 = df[~df['cond_device_imv'].isna()]
    results.append({
        'Step': 'Step 2',
        'FilterColumn': 'cond_device_imv',
        'UniqueKeys': step2['hosp_id_day_key'].nunique(),
        'RowCount': step2.shape[0],
        'ValueCounts': step2['cond_device_imv'].value_counts(dropna=False).to_dict()
    })
    df = df[~df['hosp_id_day_key'].isin(step2['hosp_id_day_key'])]
    
    # ---------------------------------------
    # Step 3: Filter on 'cond_location_icu'
    # ---------------------------------------
    step3 = df[~df['cond_location_icu'].isna()]
    results.append({
        'Step': 'Step 3',
        'FilterColumn': 'cond_location_icu',
        'UniqueKeys': step3['hosp_id_day_key'].nunique(),
        'RowCount': step3.shape[0],
        'ValueCounts': step3['cond_location_icu'].value_counts(dropna=False).to_dict()
    })
    df = df[~df['hosp_id_day_key'].isin(step3['hosp_id_day_key'])]
    
    # ---------------------------------------
    # Step 4: Filter on 'cond_peep_set_le8'
    # ---------------------------------------
    step4 = df[~df['cond_peep_set_le8'].isna()]
    results.append({
        'Step': 'Step 4',
        'FilterColumn': 'cond_peep_set_le8',
        'UniqueKeys': step4['hosp_id_day_key'].nunique(),
        'RowCount': step4.shape[0],
        'ValueCounts': step4['cond_peep_set_le8'].value_counts(dropna=False).to_dict()
    })
    df = df[~df['hosp_id_day_key'].isin(step4['hosp_id_day_key'])]
    
    # ---------------------------------------
    # Step 5: Filter on 'cond_ps_set_le8'
    # ---------------------------------------
    step5 = df[~df['cond_ps_set_le8'].isna()]
    results.append({
        'Step': 'Step 5',
        'FilterColumn': 'cond_ps_set_le8',
        'UniqueKeys': step5['hosp_id_day_key'].nunique(),
        'RowCount': step5.shape[0],
        'ValueCounts': step5['cond_ps_set_le8'].value_counts(dropna=False).to_dict()
    })
    df = df[~df['hosp_id_day_key'].isin(step5['hosp_id_day_key'])]
    
    # ---------------------------------------
    # Step 6: Filter on 'cond_mode_ps_cpap'
    # ---------------------------------------
    step6 = df[~df['cond_mode_ps_cpap'].isna()]
    results.append({
        'Step': 'Step 6',
        'FilterColumn': 'cond_mode_ps_cpap',
        'UniqueKeys': step6['hosp_id_day_key'].nunique(),
        'RowCount': step6.shape[0],
        'ValueCounts': step6['cond_mode_ps_cpap'].value_counts(dropna=False).to_dict()
    })
    df = df[~df['hosp_id_day_key'].isin(step6['hosp_id_day_key'])]
    
    # ---------------------------------------
    # Step 7: Remaining (unmatched) rows
    # ---------------------------------------
    step7 = df.copy()
    results.append({
        'Step': 'Step 7 (No Value)',
        'FilterColumn': None,
        'UniqueKeys': step7['hosp_id_day_key'].nunique(),
        'RowCount': step7.shape[0],
        'ValueCounts': None
    })
    
    # ---------------------------------------
    # Create Detailed Step-by-Step Summary DataFrame
    # ---------------------------------------
    detailed_summary_df = pd.DataFrame(results)
    
    # Calculate total_failures as the sum of UniqueKeys across all steps for this hospital
    total_failures = detailed_summary_df['UniqueKeys'].sum()
    
   
    
    # Add "% Per 100" and "% of Total" columns
    detailed_summary_df['% by eligible_days'] = detailed_summary_df['UniqueKeys'].apply(
        lambda x: round((x / eligible_days) * 100, 2)
    )
    detailed_summary_df['% of Total'] = detailed_summary_df['UniqueKeys'].apply(
        lambda x: round((x / total_failures) * 100, 2) if total_failures != 0 else 0
    )
    
    # ---------------------------------------
    # Save the detailed summary DataFrame as a CSV file for the current hospital
    # ---------------------------------------
    output_filename = f"{directory_path}/EHR_VS_EXTUBATED_failure_dependent_summary_{hosp}.csv"
    detailed_summary_df.to_csv(output_filename, index=False)
    print(f"Saved detailed summary for hospital {hosp} to {output_filename}\n")
    print(hosp,detailed_summary_df)
    print()

    # ============================================================
    # B. Independent Filtering Summary (Apply each filter independently)
    # ============================================================
    ind_step1 = final_filtered_df[~final_filtered_df['flip_skip_reason'].isna()]
    ind_step2 = final_filtered_df[~final_filtered_df['cond_device_imv'].isna()]
    ind_step3 = final_filtered_df[~final_filtered_df['cond_location_icu'].isna()]
    ind_step4 = final_filtered_df[~final_filtered_df['cond_peep_set_le8'].isna()]
    ind_step5 = final_filtered_df[~final_filtered_df['cond_ps_set_le8'].isna()]
    ind_step6 = final_filtered_df[~final_filtered_df['cond_mode_ps_cpap'].isna()]
    
    # Determine the union of keys matched by any filter
    matched_keys = set().union(
        ind_step1['hosp_id_day_key'],
        ind_step2['hosp_id_day_key'],
        ind_step3['hosp_id_day_key'],
        ind_step4['hosp_id_day_key'],
        ind_step5['hosp_id_day_key'],
        ind_step6['hosp_id_day_key']
    )
    # Unmatched keys: those not included in any of the independent filters
    ind_step7 = final_filtered_df[~final_filtered_df['hosp_id_day_key'].isin(matched_keys)]
    
    # Compute unique key counts per filter
    failure_counts = {
        'flip_skip_reason': ind_step1['hosp_id_day_key'].nunique(),
        'cond_device_imv': ind_step2['hosp_id_day_key'].nunique(),
        'cond_location_icu': ind_step3['hosp_id_day_key'].nunique(),
        'cond_peep_set_le8': ind_step4['hosp_id_day_key'].nunique(),
        'cond_ps_set_le8': ind_step5['hosp_id_day_key'].nunique(),
        'cond_mode_ps_cpap': ind_step6['hosp_id_day_key'].nunique(),
        'No Value': ind_step7['hosp_id_day_key'].nunique()
    }
    
    # Compute value counts for each filter column
    value_counts_map = {
        'flip_skip_reason': ind_step1['flip_skip_reason'].value_counts(dropna=False).to_dict(),
        'cond_device_imv': ind_step2['cond_device_imv'].value_counts(dropna=False).to_dict(),
        'cond_location_icu': ind_step3['cond_location_icu'].value_counts(dropna=False).to_dict(),
        'cond_peep_set_le8': ind_step4['cond_peep_set_le8'].value_counts(dropna=False).to_dict(),
        'cond_ps_set_le8': ind_step5['cond_ps_set_le8'].value_counts(dropna=False).to_dict(),
        'cond_mode_ps_cpap': ind_step6['cond_mode_ps_cpap'].value_counts(dropna=False).to_dict(),
        'No Value': None
    }
    
    total_failures_ind = sum(failure_counts.values())
    summary_data = []
    for reason, count in failure_counts.items():
        summary_data.append({
            'Failure Reason': reason,
            'Count': count,
            '% by eligible_days': round((count / eligible_days) * 100, 2),
            '% of Total (out of total failed cases)': round((count / total_failures_ind) * 100, 2) if total_failures_ind else 0,
            'Value Counts': value_counts_map[reason]
        })
    
    independent_summary_df = pd.DataFrame(summary_data)
    independent_summary_df = independent_summary_df.sort_values(by='Count', ascending=False).reset_index(drop=True)
    
    ind_output_filename = f"{directory_path}/EHR_VS_EXTUBATED_failure_independent_summary_{hosp}.csv"
    independent_summary_df.to_csv(ind_output_filename, index=False)
    print(f"Saved independent summary for hospital {hosp} to {ind_output_filename}\n")
    print(hosp, independent_summary_df)
    print()


# #### Plots

# In[ ]:


hospital_ids = final_df["hospital_id"].dropna().unique()

# This list will hold the summary data for each hospital
hospital_summary_list = []

for hosp in hospital_ids:
    # Filter final_df for the current hospital
    final_hosp = final_df[final_df["hospital_id"] == hosp]

    # Extract event times for SBT delivery (pass) and EHR delivery (within 2 mins)
    sbt_d_time = (
        final_hosp[
            (final_hosp["sbt_delivery_pass_fail"] == 1)
            & (final_hosp["eligible_day"] == 1)
        ]
        .sort_values(["hosp_id_day_key", "event_time"])  # ensure order
        .groupby("hosp_id_day_key", as_index=False)
        .first()[["hosp_id_day_key", "event_time"]]
    )

    ehr_d_time = final_hosp[
        (final_hosp["EHR_Delivery_2mins"] == 1) & (final_hosp["eligible_day"] == 1)
    ][["hosp_id_day_key", "event_time"]].drop_duplicates()

    # Convert event_time to hour values
    sbt_hours = sbt_d_time["event_time"].dt.hour
    ehr_hours = ehr_d_time["event_time"].dt.hour

    # Create overlay histogram plot for the current hospital
    plt.figure(figsize=(10, 6))
    # Use bins from 0 to 24 (24 bins) to capture each hour of the day
    plt.hist(
        sbt_hours,
        bins=range(0, 25),
        alpha=0.5,
        label="SBT Delivery Time",
        edgecolor="black",
    )
    plt.hist(
        ehr_hours,
        bins=range(0, 25),
        alpha=0.5,
        label="EHR Delivery Time",
        edgecolor="black",
    )
    plt.xlabel("Hour of Day")
    plt.ylabel("Frequency")
    plt.title(f"Event Time Distribution (Hourly) - Hospital {hosp}")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save the plot for the current hospital
    plt.savefig(f"{directory_path}/event_time_distribution_hospital_{hosp}.png")
    plt.close()

    # Build a summary DataFrame for the current hospital:
    # Get counts per hour for each event type
    sbt_counts = sbt_hours.value_counts().sort_index()
    ehr_counts = ehr_hours.value_counts().sort_index()

    # Create a DataFrame with all hours 0-23, merging the counts (fill missing with 0)
    hours_df = pd.DataFrame({"hour": range(24)})
    hours_df["SBT_Delivery"] = hours_df["hour"].map(sbt_counts).fillna(0).astype(int)
    hours_df["EHR_Delivery"] = hours_df["hour"].map(ehr_counts).fillna(0).astype(int)
    hours_df["hospital_id"] = hosp

    hospital_summary_list.append(hours_df)

# Combine the summary data for all hospitals into one DataFrame
combined_summary_df = pd.concat(hospital_summary_list, ignore_index=True)
combined_summary_df.to_csv(
    f"{directory_path}/event_time_distribution_summary.csv", index=False
)

print("Overlay plots created and summary CSV saved.")


# In[ ]:


hospital_ids = final_df["hospital_id"].dropna().unique()

# This list will hold the summary data for each hospital
hospital_summary_list = []

for hosp in hospital_ids:
    # Filter final_df for the current hospital
    final_hosp = final_df[final_df["hospital_id"] == hosp]

    # Extract event times for SBT delivery (pass) and EHR delivery (within 2 mins)
    sbt_d_time = (
        final_hosp[(final_hosp["extubated"] == 1) & (final_hosp["eligible_day"] == 1)]
        .sort_values(["hosp_id_day_key", "event_time"])  # ensure order
        .groupby("hosp_id_day_key", as_index=False)
        .first()[["hosp_id_day_key", "event_time"]]
    )

    ehr_d_time = final_hosp[
        (final_hosp["EHR_Delivery_2mins"] == 1) & (final_hosp["eligible_day"] == 1)
    ][["hosp_id_day_key", "event_time"]].drop_duplicates()

    # Convert event_time to hour values
    sbt_hours = sbt_d_time["event_time"].dt.hour
    ehr_hours = ehr_d_time["event_time"].dt.hour

    # Create overlay histogram plot for the current hospital
    plt.figure(figsize=(10, 6))
    # Use bins from 0 to 24 (24 bins) to capture each hour of the day
    plt.hist(
        sbt_hours,
        bins=range(0, 25),
        alpha=0.5,
        label="Extubated Time",
        edgecolor="black",
    )
    plt.hist(
        ehr_hours,
        bins=range(0, 25),
        alpha=0.5,
        label="EHR Delivery Time",
        edgecolor="black",
    )
    plt.xlabel("Hour of Day")
    plt.ylabel("Frequency")
    plt.title(f"Event Time Distribution (Hourly) - Hospital {hosp}")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save the plot for the current hospital
    plt.savefig(f"{directory_path}/event_time_distribution_hospital_{hosp}_by_ex.png")
    plt.close()

    # Build a summary DataFrame for the current hospital:
    # Get counts per hour for each event type
    sbt_counts = sbt_hours.value_counts().sort_index()
    ehr_counts = ehr_hours.value_counts().sort_index()

    # Create a DataFrame with all hours 0-23, merging the counts (fill missing with 0)
    hours_df = pd.DataFrame({"hour": range(24)})
    hours_df["SBT_Delivery"] = hours_df["hour"].map(sbt_counts).fillna(0).astype(int)
    hours_df["EHR_Delivery"] = hours_df["hour"].map(ehr_counts).fillna(0).astype(int)
    hours_df["hospital_id"] = hosp

    hospital_summary_list.append(hours_df)

# Combine the summary data for all hospitals into one DataFrame
combined_summary_df = pd.concat(hospital_summary_list, ignore_index=True)
combined_summary_df.to_csv(
    f"{directory_path}/event_time_distribution_summary_by_ex.csv", index=False
)

print("Overlay plots created and summary CSV saved.")


# # Final Summary

# In[ ]:


# --- Calculate statistics from final_df ---

# By n = Days
total_days = final_df["hosp_id_day_key"].nunique()
eligible_days = final_df[final_df["eligible_day"] == 1]["hosp_id_day_key"].nunique()
imv_days = final_df[final_df["vent_day_without_paralytics"] == 1][
    "hosp_id_day_key"
].nunique()
imv_days_with_no_filter = final_df[final_df["vent_day"] == 1][
    "hosp_id_day_key"
].nunique()
percentage = (eligible_days / imv_days) * 100 if total_days > 0 else 0
imv_icu_days = final_df[
    (final_df["device_category"] == "imv") & (final_df["location_category"] == "icu")
]["hosp_id_day_key"].nunique()

# By n = Encounter
h_total_days = final_df["hospitalization_id"].nunique()
h_eligible_days = final_df[final_df["eligible_day"] == 1][
    "hospitalization_id"
].nunique()
h_percentage = (h_eligible_days / h_total_days) * 100 if h_total_days > 0 else 0
h_imv_days = final_df[final_df["device_category"] == "imv"][
    "hospitalization_id"
].nunique()
h_imv_icu_days = final_df[
    (final_df["device_category"] == "imv") & (final_df["location_category"] == "icu")
]["hospitalization_id"].nunique()

# --- Calculate statistics from mat_df ---

# Distribution of EHR_Delivery_2mins for extubated == 1 (in percentages)
ehr_delivery_counts = (
    mat_df[mat_df["extubated"] == 1]["EHR_Delivery_2mins"].value_counts(normalize=True)
    * 100
)

# Distribution of sbt_delivery_pass_fail for extubated == 1 (in percentages)
sbt_delivery_counts = (
    mat_df[mat_df["extubated"] == 1]["sbt_delivery_pass_fail"].value_counts(
        normalize=True
    )
    * 100
)

# --- Print the statistics ---

print("By n = Days")
print("Total number of days for eval in cohort:", total_days)
print(f"Eligible days: {eligible_days} / {imv_days} ({percentage:.2f}%)")
print("Hospital days with at least one IMV event:", imv_days)
print("Hospital days with at least one IMV & ICU event:", imv_icu_days)

print("\nBy n = Encounter")
print("Total number of encounters for eval in cohort:", h_total_days)
print(f"Eligible encounters: {h_eligible_days} / {h_total_days} ({h_percentage:.2f}%)")
print("Encounters with at least one IMV event:", h_imv_days)
print("Encounters with at least one IMV & ICU event:", h_imv_icu_days)

print("\nEHR_Delivery_2mins distribution (for extubated == 1):")
print(ehr_delivery_counts)

print("\nsbt_delivery_pass_fail distribution (for extubated == 1):")
print(sbt_delivery_counts)

# --- Create a summary DataFrame for the final_df stats ---

stats_data = {
    "Metric": [
        "total_days",
        "eligible_days",
        "eligible_percentage",
        "imv_days_with_out_paralytics",
        "imv_icu_days",
        "imv_days_with_no_filter",
        "enc_total_days",
        "enc_eligible_days",
        "enc_eligible_percentage",
        "enc_imv_days",
        "enc_imv_icu_days",
    ],
    "Value": [
        total_days,
        eligible_days,
        percentage,
        imv_days,
        imv_icu_days,
        imv_days_with_no_filter,
        h_total_days,
        h_eligible_days,
        h_percentage,
        h_imv_days,
        h_imv_icu_days,
    ],
}

stats_df = pd.DataFrame(stats_data)

# Convert value counts to DataFrames and append to stats_df
ehr_counts_df = ehr_delivery_counts.reset_index()
ehr_counts_df.columns = ["Metric", "Value"]
ehr_counts_df["Metric"] = (
    "EHR_Delivery_2mins_" + ehr_counts_df["Metric"].astype(str) + "_extubated=1"
)

sbt_counts_df = sbt_delivery_counts.reset_index()
sbt_counts_df.columns = ["Metric", "Value"]
sbt_counts_df["Metric"] = (
    "sbt_delivery_pass_fail_" + sbt_counts_df["Metric"].astype(str) + "_extubated=1"
)

# Combine all stats
stats_df = pd.concat([stats_df, ehr_counts_df, sbt_counts_df], ignore_index=True)

print("\nExtended statistics DataFrame with value counts:")
print(stats_df)

# Save to CSV
stats_df.to_csv(f"{directory_path}/stats_df.csv", index=False)


# ## Table 1 Code

# In[ ]:


# Aggregate functions
def documented(series):
    return "Documented" if series.notna().any() else "Not Documented"

def age_bucket(mean_age):
    if pd.isna(mean_age):
        return None
    elif mean_age < 40:
        return "18-39"
    elif mean_age < 60:
        return "40-59"
    elif mean_age < 80:
        return "60-79"
    else:
        return "80+"

# Clean 'language_name' to only "English", "Spanish", or "Other"
def categorize_language(lang):
    if re.search(r'english', str(lang), re.IGNORECASE):
        return 'English'
    elif re.search(r'spanish', str(lang), re.IGNORECASE):
        return 'Spanish'
    else:
        return 'Other'

t1_col = [
    "patient_id",
    "hospitalization_id",
    "hosp_id_day_key",
    "age_at_admission",    "sex_category",    "race_category",    "ethnicity_category",    "language_name",    "weight_kg",
    "height_cm", "cisatracurium",    "vecuronium",    "rocuronium",    "dobutamine",    "dopamine",    "epinephrine",
    "fentanyl",    "hydromorphone",    "isoproterenol",    "lorazepam",    "midazolam",    "milrinone",    "morphine",
    "norepinephrine",    "phenylephrine",    "propofol",    "vasopressin",    "angiotensin",     "rass", "gcs_total"
]

medication_columns = [
    "rass", "gcs_total", "cisatracurium", "vecuronium", "rocuronium",
    "dobutamine", "dopamine", "epinephrine", "fentanyl", "hydromorphone",
    "isoproterenol", "lorazepam", "midazolam", "milrinone", "morphine",
    "norepinephrine", "phenylephrine", "propofol", "vasopressin", "angiotensin"
]

demographic_columns = ["sex_category", "race_category", "ethnicity_category", "language_name"]

continuous_cols = [
    "rass", "gcs_total", "cisatracurium", "vecuronium", "rocuronium",
    "dobutamine", "dopamine", "epinephrine", "fentanyl", "hydromorphone",
    "isoproterenol", "lorazepam", "midazolam", "milrinone", "morphine",
    "norepinephrine", "phenylephrine", "propofol", "vasopressin",
    "angiotensin", "bmi"
]

drugs = [
    "cisatracurium", "vecuronium", "rocuronium",
    "dobutamine", "dopamine", "epinephrine", "fentanyl", "hydromorphone",
    "isoproterenol", "lorazepam", "midazolam", "milrinone", "morphine",
    "norepinephrine", "phenylephrine", "propofol", "vasopressin", "angiotensin"
]

# Apply the transformation
t1_cohort[drugs] = t1_cohort[drugs].applymap(lambda x: x if x > 0 else np.nan)

t1_cohort['bmi'] = t1_cohort['weight_kg'] / ((t1_cohort['height_cm'] / 100) ** 2)

# Apply the function to 'language_name'
t1_cohort['language_name'] = t1_cohort['language_name'].apply(categorize_language)
t1_cohort[continuous_cols] = t1_cohort[continuous_cols].astype(float)


# #### Table 1 By Days for Categorical

# In[ ]:


for x in tqdm(
    [
        "EHR_Delivery_2mins",
        "EHR_Delivery_30mins",
    ]
):
    ids_to_use = final_df[final_df[x]==1].hosp_id_day_key.unique()
    # Groupby aggregation by hospitalization_id
    t1_summary = t1_cohort[t1_cohort['hosp_id_day_key'].isin(ids_to_use)].groupby("hosp_id_day_key").agg(
        {
            "age_at_admission": "mean",
            **{col: documented for col in medication_columns},
            **{col: "first" for col in demographic_columns},
        }
    )

    # Apply age bucketing
    t1_summary["age_bucket"] = t1_summary["age_at_admission"].apply(age_bucket)

    # Drop raw age if you don't need it
    t1_summary = t1_summary.drop(columns=["age_at_admission"])

    # Reset index if needed
    t1_summary = t1_summary.reset_index()

    summary_df = sbt.manual_categorical_tableone(
        t1_summary, 
        medication_columns + demographic_columns + ["age_bucket"]
    )
    summary_df.to_csv(f"{directory_path}/table1_{x}_categorical.csv", index=False)


# #### Table 1 By Days for Continuous

# In[ ]:


for x in tqdm([
    "EHR_Delivery_2mins",
    "EHR_Delivery_30mins",
]):
    # --- filter to only the days in this subcohort
    ids = final_df.loc[final_df[x]==1, "hosp_id_day_key"].unique()
    sub = t1_cohort[t1_cohort["hosp_id_day_key"].isin(ids)]

    # --- 1) Day-level medians + flags + demographics
    day_summary = sub.groupby("hosp_id_day_key").agg(
        {
          **{c: "median" for c in continuous_cols}
        }
      ).reset_index()
    summary_df = sbt.manual_tableone(day_summary, continuous_cols)
    summary_df.to_csv(f"{directory_path}/table1_{x}_continuous.csv", index=False)


# # Sofa T1's

# In[ ]:


import pySofa as sofa
sofa.pyCLIF2.helper

continuous_cols_sofa = [ 'sofa_cv_97', 'sofa_coag',
       'sofa_liver', 'sofa_resp_pf', 'sofa_resp_pf_imp', 'sofa_resp',
       'sofa_cns', 'sofa_renal', 'sofa_total']


# In[ ]:


mapping_ids = pd.read_csv('../output/intermediate/hospitalization_to_block_df.csv')
encounter_dict = dict(zip(mapping_ids['hospitalization_id'].astype(str), mapping_ids['encounter_block'].astype(str)))
mapping_ids[['hospitalization_id','encounter_block']]=mapping_ids[['hospitalization_id','encounter_block']].astype(str)
mapping_ids.head()


# In[ ]:


encounter_level_sofa = cohort[['hospitalization_id', 'admission_dttm',
       'discharge_dttm']].drop_duplicates().rename(columns={'admission_dttm':'start_dttm','discharge_dttm':'stop_dttm'})
encounter_level_sofa = pc.convert_datetime_columns_to_site_tz(encounter_level_sofa, pc.helper['your_site_timezone'])
encounter_level_sofa.head()


# In[ ]:


sout = sofa.compute_sofa(
    encounter_level_sofa,
    tables_path=None,
    use_hospitalization_id = False,
    id_mapping = mapping_ids,
    group_by_id = "encounter_block"
)
encounter_level_sofa_t1 = sbt.manual_tableone(sout, continuous_cols_sofa)
encounter_level_sofa_t1.to_csv(f'{directory_path}/encounter_level_sofa_t1.csv',index=False)


# In[ ]:


for x in tqdm([
        "EHR_Delivery_2mins",
        "EHR_Delivery_30mins"],desc="Generating Sofa table 1 for each Flags"):

        day_df = final_df[final_df[x]==1][['hospitalization_id','hosp_id_day_key','current_day']].drop_duplicates()
        if day_df.empty:
                continue
        # Create start_dttm as current_day at 00:00:00
        day_df['start_dttm'] = pd.to_datetime(day_df['current_day']).dt.normalize()
        # Create end_dttm as current_day at 23:59:59
        day_df['stop_dttm'] = day_df['start_dttm'] + pd.Timedelta(hours=23, minutes=59, seconds=59)
        day_df = pc.convert_datetime_columns_to_site_tz(day_df, pc.helper['your_site_timezone'])

        day_sofa = sofa.compute_sofa(
                                day_df,
                                tables_path=None,
                                use_hospitalization_id = False,
                                id_mapping = mapping_ids,
                                group_by_id = "hosp_id_day_key"
                        )
        
        day_sofa_t1 = sbt.manual_tableone(day_sofa, continuous_cols_sofa)
        day_sofa_t1.to_csv(f'{directory_path}/{x}_sofa_t1.csv',index=False)
        
        

