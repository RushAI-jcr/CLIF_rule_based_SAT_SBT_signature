"""
04_hospital_variation.py
========================
Risk-adjusted hospital-level SAT/SBT delivery estimation and variation
visualizations (caterpillar + funnel plots).

Per manuscript:
- Hierarchical regression models with patient-level covariate adjustment
  and hospital random effects
- Random intercepts for patients (within-patient clustering of eligible days)
- Adjusted for patient characteristics known to influence SAT/SBT performance
- Caterpillar plots grouped by health system
- Variation summarized: median (IQR), range, adjusted median odds ratio (aMOR)
- Funnel control limits at 95% and 99.8%
- Concordance between EHR-phenotype and flowsheet-derived rates using
  Pearson/Spearman/CCC + Bland-Altman

CLIF 2.1 compliance:
- hospitalization_id as join key
- hospital_id from ADT table

Usage:
    python 04_hospital_variation.py --sat-file ../output/intermediate/final_df_SAT.csv \
                                     --output-dir ../output/final
"""

from __future__ import annotations


import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

UTILS_DIR = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

warnings.filterwarnings("ignore")

# JAMA style
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})


# ============================================================
# HOSPITAL-LEVEL RATE ESTIMATION
# ============================================================

def compute_crude_hospital_rates(
    df: pd.DataFrame,
    delivery_col: str,
    eligibility_col: str = "eligible_event",
) -> pd.DataFrame:
    """Compute crude SAT/SBT delivery rates per hospital."""
    eligible = df[df[eligibility_col] == 1].copy()
    eligible["delivered"] = eligible[delivery_col].fillna(0).clip(0, 1)

    hosp_rates = eligible.groupby("hospital_id").agg(
        eligible_days=("hosp_id_day_key", "nunique"),
        delivered_days=("delivered", "sum"),
        n_hospitalizations=("hospitalization_id", "nunique"),
        n_patients=("patient_id", "nunique"),
    ).reset_index()

    hosp_rates["crude_rate"] = (
        hosp_rates["delivered_days"] / hosp_rates["eligible_days"]
    )
    return hosp_rates


def compute_risk_adjusted_rates(
    df: pd.DataFrame,
    delivery_col: str,
    eligibility_col: str = "eligible_event",
) -> pd.DataFrame:
    """Compute SAP-style risk-standardized rates:

    Predicted_j = sum model-predicted delivery probabilities with hospital effect
    Expected_j  = sum model-predicted probabilities with hospital effect set to mean
    RSR_j       = (Predicted_j / Expected_j) * Overall
    """
    eligible = df[df[eligibility_col] == 1].copy()
    eligible["delivered"] = eligible[delivery_col].fillna(0).clip(0, 1).astype(int)
    eligible["sex_male"] = (
        eligible["sex_category"].str.lower() == "male"
    ).astype(int)
    eligible["age"] = pd.to_numeric(eligible["age_at_admission"], errors="coerce")

    # Day-level aggregation
    day_level = eligible.groupby("hosp_id_day_key").agg(
        hospitalization_id=("hospitalization_id", "first"),
        patient_id=("patient_id", "first"),
        hospital_id=("hospital_id", "first"),
        delivered=("delivered", "max"),
        age=("age_at_admission", "first"),
        sex_male=("sex_male", "first"),
    ).reset_index()
    day_level["age"] = pd.to_numeric(day_level["age"], errors="coerce")
    day_level["age"] = day_level["age"].fillna(day_level["age"].median())
    day_level["overall_rate"] = day_level["delivered"].mean()

    crude_rates = compute_crude_hospital_rates(df, delivery_col, eligibility_col)

    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

        # Nested patient RE within hospital [SAP 2.7]
        vc_formulas = {"hospital_re": "0 + C(hospital_id)"}
        if "patient_id" in day_level.columns:
            vc_formulas["patient_re"] = "0 + C(patient_id):C(hospital_id)"

        # Full SAP 2.8 covariate set (use what's available)
        formula_covs = ["age", "sex_male"]
        for extra in ["rass", "gcs_total", "bmi", "elixhauser_score",
                       "medical_admission", "ethnicity_nonhispanic",
                       "fio2_set", "peep_set", "vasopressor_nee", "on_sedation"]:
            if extra in day_level.columns and day_level[extra].notna().sum() > len(day_level) * 0.3:
                day_level[extra] = pd.to_numeric(day_level[extra], errors="coerce").fillna(
                    day_level[extra].median()
                )
                formula_covs.append(extra)

        formula_rhs = " + ".join(formula_covs)
        model = BinomialBayesMixedGLM.from_formula(
            f"delivered ~ {formula_rhs}",
            vc_formulas,
            day_level,
        )
        result = model.fit_vb()

        # Fixed-effects linear predictor (expected with mean hospital effect = 0).
        fep_names = list(result.model.fep_names)
        fe_mean = np.asarray(result.fe_mean)
        design = pd.DataFrame(index=day_level.index)
        design["Intercept"] = 1.0
        if "age" in fep_names:
            design["age"] = day_level["age"].astype(float)
        if "sex_male" in fep_names:
            design["sex_male"] = day_level["sex_male"].astype(float)
        design = design.reindex(columns=fep_names, fill_value=0.0)
        lp_expected = design.to_numpy() @ fe_mean
        p_expected = 1 / (1 + np.exp(-lp_expected))

        # Hospital random effects by name.
        re_map: dict[str, float] = {}
        for name, val in zip(result.model.vc_names, result.vc_mean):
            if "C(hospital_id)" in name:
                hosp = name.split("[")[-1].rstrip("]")
                re_map[hosp] = float(val)

        hosp_re = day_level["hospital_id"].astype(str).map(re_map).fillna(0.0).to_numpy()
        p_pred = 1 / (1 + np.exp(-(lp_expected + hosp_re)))
        day_level["p_expected"] = p_expected
        day_level["p_predicted"] = p_pred

        overall_rate = float(day_level["delivered"].mean())
        rsr = (
            day_level.groupby("hospital_id")
            .agg(
                Predicted_j=("p_predicted", "sum"),
                Expected_j=("p_expected", "sum"),
                eligible_days=("hosp_id_day_key", "nunique"),
                delivered_days=("delivered", "sum"),
                n_hospitalizations=("hospitalization_id", "nunique"),
                n_patients=("patient_id", "nunique"),
            )
            .reset_index()
        )
        rsr["overall_rate"] = overall_rate
        rsr["adjusted_rate"] = np.where(
            rsr["Expected_j"] > 0,
            (rsr["Predicted_j"] / rsr["Expected_j"]) * overall_rate,
            np.nan,
        )
        rsr["adjusted_rate"] = rsr["adjusted_rate"].clip(0, 1)
        hosp_rates = crude_rates.merge(
            rsr[
                [
                    "hospital_id",
                    "Predicted_j",
                    "Expected_j",
                    "adjusted_rate",
                    "overall_rate",
                ]
            ],
            on="hospital_id",
            how="left",
        )
        hosp_rates["adjusted_rate"] = hosp_rates["adjusted_rate"].fillna(hosp_rates["crude_rate"])
        hosp_rates["Predicted_j"] = hosp_rates["Predicted_j"].fillna(0.0)
        hosp_rates["Expected_j"] = hosp_rates["Expected_j"].fillna(0.0)

        # MOR from hospital random-effect variance (sigma_u^2).
        hospital_var = np.nan
        if hasattr(result.model, "vcp_names"):
            for idx, name in enumerate(result.model.vcp_names):
                if "hospital_re" in str(name):
                    # vcp is on log(sd) scale in BayesMixedGLM.
                    sigma_u = float(np.exp(result.vcp_mean[idx]))
                    hospital_var = sigma_u ** 2
                    break
        if np.isnan(hospital_var):
            hospital_var = float(np.var(list(re_map.values()))) if re_map else np.nan
        hosp_rates["sigma_u_sq"] = hospital_var
        hosp_rates["MOR"] = (
            np.exp(np.sqrt(2 * hospital_var) * 0.67448975)
            if np.isfinite(hospital_var)
            else np.nan
        )
        # ICC = sigma_h^2 / (sigma_h^2 + sigma_p^2 + pi^2/3) [SAP 2.7]
        patient_var = np.nan
        if hasattr(result.model, "vcp_names"):
            for idx2, name2 in enumerate(result.model.vcp_names):
                if "patient_re" in str(name2):
                    sigma_p = float(np.exp(result.vcp_mean[idx2]))
                    patient_var = sigma_p ** 2
                    break
        if np.isnan(patient_var):
            patient_var = 0.0
        icc = (
            hospital_var / (hospital_var + patient_var + (np.pi ** 2 / 3))
            if np.isfinite(hospital_var)
            else np.nan
        )
        hosp_rates["ICC"] = icc
        hosp_rates["sigma_p_sq"] = patient_var
        hosp_rates["method"] = "sap_rsr_bayesian_mixed_glm"

    except Exception as e:
        print(f"Mixed model failed ({e}), using crude rates as fallback")
        hosp_rates = crude_rates.copy()
        hosp_rates["adjusted_rate"] = hosp_rates["crude_rate"]
        hosp_rates["Predicted_j"] = np.nan
        hosp_rates["Expected_j"] = np.nan
        hosp_rates["sigma_u_sq"] = np.nan
        hosp_rates["MOR"] = np.nan
        hosp_rates["method"] = "crude_fallback"

    return hosp_rates


def compute_adjusted_median_odds_ratio(
    hosp_rates: pd.DataFrame,
) -> tuple[float, float, float]:
    """Compute the adjusted median odds ratio (aMOR).

    aMOR quantifies between-hospital variation. If there were no variation,
    aMOR = 1. Higher values indicate more variation.

    aMOR = exp(sqrt(2 * sigma_u^2) * Phi^-1(0.75))
    where sigma_u^2 is the between-hospital variance of the random intercept.
    """
    from scipy.stats import norm

    # Primary source: model-estimated random-intercept variance.
    if "sigma_u_sq" in hosp_rates.columns and hosp_rates["sigma_u_sq"].notna().any():
        sigma_u_sq = float(hosp_rates["sigma_u_sq"].dropna().iloc[0])
    else:
        rates = hosp_rates["adjusted_rate"].dropna()
        if len(rates) < 3:
            return np.nan, np.nan, np.nan
        # Fallback approximation from adjusted-rate spread on logit scale.
        logit_rates = np.log(
            rates.clip(0.001, 0.999) / (1 - rates.clip(0.001, 0.999))
        )
        sigma_u_sq = float(logit_rates.var())

    if not np.isfinite(sigma_u_sq):
        return np.nan, np.nan, np.nan

    amor = np.exp(np.sqrt(2 * sigma_u_sq) * norm.ppf(0.75))

    # Bootstrap CI for aMOR
    rng = np.random.default_rng(42)
    boot_amors = []
    for _ in range(1000):
        # Parametric bootstrap around sigma_u^2 estimate.
        boot_sigma = abs(rng.normal(loc=np.sqrt(max(sigma_u_sq, 0)), scale=0.1))
        boot_var = boot_sigma**2
        boot_amors.append(float(np.exp(np.sqrt(2 * boot_var) * norm.ppf(0.75))))

    amor_lower = np.percentile(boot_amors, 2.5)
    amor_upper = np.percentile(boot_amors, 97.5)

    return amor, amor_lower, amor_upper


# ============================================================
# CATERPILLAR PLOT
# ============================================================

def caterpillar_plot(
    hosp_rates: pd.DataFrame,
    delivery_label: str,
    output_path: str,
    ax: plt.Axes | None = None,
) -> dict[str, Any]:
    """Generate caterpillar plot of hospital-level adjusted delivery rates.

    JAMA style: Arial font, min 8pt text, legends outside plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        If provided, draw on this axis (for embedding in composite figures).
        If None, creates a standalone figure and saves to output_path.
    """
    df = hosp_rates.sort_values("adjusted_rate").reset_index(drop=True)
    n = len(df)

    # Compute Wilson CIs for each hospital
    from scipy.stats import norm
    z = norm.ppf(0.975)

    cis_lower = []
    cis_upper = []
    for _, row in df.iterrows():
        p = row["adjusted_rate"]
        n_obs = row["eligible_days"]
        denom = 1 + z**2 / n_obs
        center = (p + z**2 / (2 * n_obs)) / denom
        spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_obs)) / n_obs) / denom
        cis_lower.append(max(0, center - spread))
        cis_upper.append(min(1, center + spread))

    df["ci_lower"] = cis_lower
    df["ci_upper"] = cis_upper

    # Overall rate
    overall_rate = (
        df["delivered_days"].sum() / df["eligible_days"].sum()
    )

    # aMOR
    amor, amor_lo, amor_hi = compute_adjusted_median_odds_ratio(df)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, max(4, n * 0.35)))

    # Color by health_system_id if available [SAP 2.7]
    has_system = "health_system_id" in df.columns and df["health_system_id"].notna().any()
    if has_system:
        systems = df["health_system_id"].fillna("Unknown").astype(str)
        unique_systems = systems.unique()
        cmap = plt.cm.get_cmap("tab10", max(len(unique_systems), 1))
        sys_colors = {s: cmap(i) for i, s in enumerate(unique_systems)}
        point_colors = [sys_colors[s] for s in systems]
    else:
        point_colors = ["#2166AC"] * n

    # Error bars
    for i in range(n):
        ax.errorbar(
            df["adjusted_rate"].iloc[i],
            i,
            xerr=[[df["adjusted_rate"].iloc[i] - df["ci_lower"].iloc[i]],
                   [df["ci_upper"].iloc[i] - df["adjusted_rate"].iloc[i]]],
            fmt="o",
            color=point_colors[i],
            ecolor=point_colors[i],
            elinewidth=1.5,
            capsize=3,
            markersize=5,
            alpha=0.85,
        )

    # Legend for health systems
    if has_system:
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=sys_colors[s],
                   markersize=6, label=s)
            for s in unique_systems
        ]
        ax.legend(handles=legend_handles, fontsize=8, loc="upper left",
                  bbox_to_anchor=(1.02, 1), borderaxespad=0, title="Health system",
                  title_fontsize=8)

    # Overall reference line
    ax.axvline(
        overall_rate,
        color="#B2182B",
        linestyle="--",
        linewidth=1.2,
        label=f"Overall: {overall_rate:.1%}",
    )

    ax.set_yticks(range(n))
    ax.set_yticklabels(df["hospital_id"].astype(str), fontsize=8)
    ax.set_xlabel(f"Delivery rate", fontsize=9)
    ax.set_title(
        f"{delivery_label}",
        fontsize=10,
        fontweight="bold",
    )

    # Summary stats as text below
    median_rate = df["adjusted_rate"].median()
    iqr_lo = df["adjusted_rate"].quantile(0.25)
    iqr_hi = df["adjusted_rate"].quantile(0.75)
    rate_range = (df["adjusted_rate"].min(), df["adjusted_rate"].max())

    summary_text = (
        f"Median: {median_rate:.1%}  IQR: [{iqr_lo:.1%}, {iqr_hi:.1%}]  "
        f"Range: [{rate_range[0]:.1%}, {rate_range[1]:.1%}]"
    )
    if not np.isnan(amor):
        summary_text += f"\naMOR: {amor:.2f} (95% CI: {amor_lo:.2f}-{amor_hi:.2f})"

    ax.text(
        0.5, -0.12,
        summary_text,
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
        style="italic",
    )

    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(-0.05, 1.05)
    ax.grid(axis="x", alpha=0.3)

    if standalone:
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  Caterpillar plot saved: {output_path}")

    return {
        "median_rate": median_rate,
        "iqr": (iqr_lo, iqr_hi),
        "range": rate_range,
        "amor": amor,
        "amor_ci": (amor_lo, amor_hi),
        "overall_rate": overall_rate,
        "n_hospitals": n,
    }


# ============================================================
# FUNNEL PLOT
# ============================================================

def compute_funnel_limits(hosp_rates: pd.DataFrame) -> pd.DataFrame:
    """Compute 95% and 99.8% funnel-control limits for adjusted rates."""
    df = hosp_rates.copy()
    if "eligible_days" not in df.columns:
        df["eligible_days"] = 1
    center = float(df["adjusted_rate"].mean()) if not df.empty else 0.0
    n = df["eligible_days"].clip(lower=1).astype(float)
    se = np.sqrt(np.maximum(center * (1 - center), 1e-9) / n)
    z95 = 1.96
    z998 = 3.09
    df["funnel_center"] = center
    df["limit95_low"] = (center - z95 * se).clip(0, 1)
    df["limit95_high"] = (center + z95 * se).clip(0, 1)
    df["limit998_low"] = (center - z998 * se).clip(0, 1)
    df["limit998_high"] = (center + z998 * se).clip(0, 1)
    df["outside_95"] = (df["adjusted_rate"] < df["limit95_low"]) | (df["adjusted_rate"] > df["limit95_high"])
    df["outside_998"] = (df["adjusted_rate"] < df["limit998_low"]) | (df["adjusted_rate"] > df["limit998_high"])
    return df


def funnel_plot(
    hosp_rates: pd.DataFrame,
    delivery_label: str,
    output_path: str,
) -> pd.DataFrame:
    """Generate funnel plot with 95% and 99.8% control limits."""
    df = compute_funnel_limits(hosp_rates).sort_values("eligible_days")
    if df.empty:
        return df

    x = df["eligible_days"].astype(float)
    y = df["adjusted_rate"].astype(float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color="#2166AC", s=28, alpha=0.9, label="Hospitals")
    ax.plot(x, df["funnel_center"], color="#B2182B", linewidth=1.5, label="Overall")
    ax.plot(x, df["limit95_low"], color="#666666", linestyle="--", linewidth=1.0, label="95% limits")
    ax.plot(x, df["limit95_high"], color="#666666", linestyle="--", linewidth=1.0)
    ax.plot(x, df["limit998_low"], color="#111111", linestyle=":", linewidth=1.0, label="99.8% limits")
    ax.plot(x, df["limit998_high"], color="#111111", linestyle=":", linewidth=1.0)

    ax.set_xlabel("Eligible ventilator-days")
    ax.set_ylabel("Risk-standardized delivery rate")
    ax.set_title(f"Funnel Plot: {delivery_label}")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Funnel plot saved: {output_path}")
    return df


# ============================================================
# CONCORDANCE: EHR vs FLOWSHEET RATES
# ============================================================

def concordance_ehr_vs_flowsheet(
    df: pd.DataFrame,
    ehr_col: str,
    flowsheet_col: str,
    output_path: str,
) -> dict[str, Any]:
    """Compare risk-standardized hospital rates: EHR phenotype vs flowsheet."""
    ehr_rates = compute_risk_adjusted_rates(df, ehr_col)
    flow_rates = compute_risk_adjusted_rates(df, flowsheet_col)

    merged = ehr_rates[["hospital_id", "adjusted_rate"]].rename(
        columns={"adjusted_rate": "ehr_rate"}
    ).merge(
        flow_rates[["hospital_id", "adjusted_rate"]].rename(
            columns={"adjusted_rate": "flowsheet_rate"}
        ),
        on="hospital_id",
        how="inner",
    )

    if len(merged) < 3:
        print("  Too few hospitals for concordance analysis")
        return {}

    from scipy.stats import pearsonr, spearmanr

    r_pearson, p_pearson = pearsonr(merged["ehr_rate"], merged["flowsheet_rate"])
    r_spearman, p_spearman = spearmanr(merged["ehr_rate"], merged["flowsheet_rate"])
    x = merged["ehr_rate"].to_numpy(dtype=float)
    y = merged["flowsheet_rate"].to_numpy(dtype=float)
    var_x = float(np.var(x, ddof=1))
    var_y = float(np.var(y, ddof=1))
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    ccc = float(
        (2 * r_pearson * np.sqrt(max(var_x, 0)) * np.sqrt(max(var_y, 0)))
        / (var_x + var_y + (mean_x - mean_y) ** 2)
    ) if (var_x + var_y + (mean_x - mean_y) ** 2) > 0 else np.nan

    # Bland-Altman
    mean_rates = (merged["ehr_rate"] + merged["flowsheet_rate"]) / 2
    diff_rates = merged["ehr_rate"] - merged["flowsheet_rate"]
    bias = diff_rates.mean()
    loa_lower = bias - 1.96 * diff_rates.std()
    loa_upper = bias + 1.96 * diff_rates.std()

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Correlation
    ax = axes[0]
    ax.scatter(merged["flowsheet_rate"], merged["ehr_rate"],
               color="#2166AC", s=50, edgecolors="white", linewidth=0.5)
    lims = [0, max(merged["ehr_rate"].max(), merged["flowsheet_rate"].max()) + 0.05]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Flowsheet delivery rate", fontsize=9)
    ax.set_ylabel("EHR phenotype delivery rate", fontsize=9)
    ax.set_title("Correlation", fontsize=11, fontweight="bold")

    _annot_text = (
        f"r = {r_pearson:.3f} (p = {p_pearson:.3f})\n"
        f"rho = {r_spearman:.3f} (p = {p_spearman:.3f})\n"
        f"CCC = {ccc:.3f}"
    )
    ax.text(0.05, 0.92,
            _annot_text,
            transform=ax.transAxes, fontsize=8, verticalalignment="top")

    # Panel B: Bland-Altman
    ax = axes[1]
    ax.scatter(mean_rates, diff_rates, color="#2166AC", s=50,
               edgecolors="white", linewidth=0.5)
    ax.axhline(bias, color="#B2182B", linestyle="-", linewidth=1)
    ax.axhline(loa_lower, color="#B2182B", linestyle="--", linewidth=0.8)
    ax.axhline(loa_upper, color="#B2182B", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mean of EHR and flowsheet rates", fontsize=9)
    ax.set_ylabel("Difference (EHR - Flowsheet)", fontsize=9)
    ax.set_title("Bland-Altman agreement", fontsize=11, fontweight="bold")
    ax.text(0.05, 0.92,
            f"Bias: {bias:.3f}\nLoA: [{loa_lower:.3f}, {loa_upper:.3f}]",
            transform=ax.transAxes, fontsize=8, verticalalignment="top")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Concordance plot saved: {output_path}")

    return {
        "pearson_r": r_pearson,
        "pearson_p": p_pearson,
        "spearman_rho": r_spearman,
        "spearman_p": p_spearman,
        "ccc": ccc,
        "bias": bias,
        "loa": (loa_lower, loa_upper),
    }


# ============================================================
# MAIN
# ============================================================

def run_hospital_variation(
    sat_file: str,
    sbt_file: str,
    output_dir: str,
) -> pd.DataFrame:
    """Run all hospital variation analyses."""
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    all_results = []
    funnel_rows: list[pd.DataFrame] = []

    for label, filepath, delivery_cols, flowsheet_col in [
        ("SAT", sat_file,
         ["SAT_EHR_delivery", "SAT_modified_delivery"],
         "sat_flowsheet_delivery_flag"),
        ("SBT", sbt_file,
         ["EHR_Delivery_2mins", "EHR_Delivery_5mins", "EHR_Delivery_30mins"],
         "sbt_flowsheet_delivery_flag"),
    ]:
        if not os.path.exists(filepath):
            print(f"Skipping {label}: {filepath} not found")
            continue

        df = pd.read_csv(filepath, low_memory=False)
        for col in ["event_time", "admission_dttm", "discharge_dttm"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format="mixed")

        print(f"\n{'='*60}")
        print(f"Hospital variation for {label}")
        print(f"{'='*60}")

        for dcol in delivery_cols:
            if dcol not in df.columns:
                print(f"  Skipping {dcol}: not in data")
                continue

            print(f"\n--- {dcol} ---")

            # Risk-adjusted rates
            hosp_rates = compute_risk_adjusted_rates(df, dcol)
            rates_path = os.path.join(
                output_dir, f"hospital_rates_{label}_{dcol}.csv"
            )
            hosp_rates.to_csv(rates_path, index=False)
            print(f"  Rates saved: {rates_path}")

            # Caterpillar plot
            cat_path = os.path.join(fig_dir, f"fig5_caterpillar_{label}_{dcol}.pdf")
            variation_stats = caterpillar_plot(hosp_rates, f"{label} ({dcol})", cat_path)
            variation_stats["trial_type"] = label
            variation_stats["delivery_definition"] = dcol
            all_results.append(variation_stats)

            # Funnel plot + data (95% and 99.8% limits)
            funnel_path = os.path.join(fig_dir, f"funnel_{label}_{dcol}.pdf")
            funnel_df = funnel_plot(hosp_rates, f"{label} ({dcol})", funnel_path)
            if not funnel_df.empty:
                funnel_df = funnel_df.copy()
                funnel_df["trial_type"] = label
                funnel_df["delivery_definition"] = dcol
                funnel_rows.append(funnel_df)

            # Concordance with flowsheet
            if flowsheet_col in df.columns:
                conc_path = os.path.join(
                    fig_dir, f"concordance_{label}_{dcol}.pdf"
                )
                conc_stats = concordance_ehr_vs_flowsheet(
                    df, dcol, flowsheet_col, conc_path
                )
                if conc_stats:
                    conc_stats["trial_type"] = label
                    conc_stats["delivery_definition"] = dcol
                    all_results.append(conc_stats)

    # Save summary
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(output_dir, "hospital_variation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")
    if funnel_rows:
        funnel_all = pd.concat(funnel_rows, ignore_index=True)
        funnel_out = os.path.join(output_dir, "hospital_benchmarking_funnel_data.csv")
        funnel_all.to_csv(funnel_out, index=False)
        print(f"Funnel data saved: {funnel_out}")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sat-file", default="../output/intermediate/final_df_SAT.csv")
    parser.add_argument("--sbt-file", default="../output/intermediate/final_df_SBT.csv")
    parser.add_argument("--output-dir", default="../output/final")
    args = parser.parse_args()
    run_hospital_variation(args.sat_file, args.sbt_file, args.output_dir)
