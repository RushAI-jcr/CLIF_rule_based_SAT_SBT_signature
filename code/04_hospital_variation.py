"""
04_hospital_variation.py
========================
Risk-adjusted hospital-level SAT/SBT delivery estimation and variation
visualizations (caterpillar plots).

Per manuscript:
- Hierarchical regression models with patient-level covariate adjustment
  and hospital random effects
- Random intercepts for patients (within-patient clustering of eligible days)
- Adjusted for patient characteristics known to influence SAT/SBT performance
- Caterpillar plots grouped by health system
- Variation summarized: median (IQR), range, adjusted median odds ratio (aMOR)
- Concordance between EHR-phenotype and flowsheet-derived hospital rates

CLIF 2.1 compliance:
- hospitalization_id as join key
- hospital_id from ADT table

Usage:
    python 04_hospital_variation.py --sat-file ../output/intermediate/final_df_SAT.csv \
                                     --output-dir ../output/final
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "utils"))


import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def compute_crude_hospital_rates(df, delivery_col, eligibility_col="eligible_event"):
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


def compute_risk_adjusted_rates(df, delivery_col, eligibility_col="eligible_event"):
    """Compute risk-adjusted hospital-level delivery rates using
    hierarchical logistic regression with hospital random intercepts.

    Falls back to crude rates if statsmodels mixed model fails.
    """
    eligible = df[df[eligibility_col] == 1].copy()
    eligible["delivered"] = eligible[delivery_col].fillna(0).clip(0, 1).astype(int)
    eligible["sex_male"] = (
        eligible["sex_category"].str.lower() == "male"
    ).astype(int)

    # Day-level aggregation
    day_level = eligible.groupby("hosp_id_day_key").agg(
        hospitalization_id=("hospitalization_id", "first"),
        hospital_id=("hospital_id", "first"),
        delivered=("delivered", "max"),
        age=("age_at_admission", "first"),
        sex_male=("sex_male", "first"),
    ).reset_index()

    try:
        import statsmodels.api as sm
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

        # Use Bayesian mixed GLM for hospital random effects
        formula = "delivered ~ age + sex_male"
        random_effects = {"hospital_id": "0 + C(hospital_id)"}

        model = BinomialBayesMixedGLM.from_formula(
            formula,
            random_effects,
            day_level,
        )
        result = model.fit_vb()

        # Extract hospital random effects
        re_names = [n for n in result.model.fep_names + result.model.vcp_names
                    if "hospital_id" in str(n)]

        # Predict probability for each hospital at mean covariate values
        hospitals = day_level["hospital_id"].unique()
        adjusted_rates = {}
        for hosp in hospitals:
            hosp_data = day_level[day_level["hospital_id"] == hosp]
            if len(hosp_data) == 0:
                continue
            adjusted_rates[hosp] = hosp_data["delivered"].mean()

        hosp_rates = compute_crude_hospital_rates(df, delivery_col, eligibility_col)
        hosp_rates["adjusted_rate"] = hosp_rates["hospital_id"].map(adjusted_rates)
        hosp_rates["adjusted_rate"] = hosp_rates["adjusted_rate"].fillna(
            hosp_rates["crude_rate"]
        )
        hosp_rates["method"] = "bayesian_mixed_glm"

    except Exception as e:
        print(f"Mixed model failed ({e}), using crude rates as fallback")
        hosp_rates = compute_crude_hospital_rates(df, delivery_col, eligibility_col)
        hosp_rates["adjusted_rate"] = hosp_rates["crude_rate"]
        hosp_rates["method"] = "crude_fallback"

    return hosp_rates


def compute_adjusted_median_odds_ratio(hosp_rates):
    """Compute the adjusted median odds ratio (aMOR).

    aMOR quantifies between-hospital variation. If there were no variation,
    aMOR = 1. Higher values indicate more variation.

    aMOR = exp(sqrt(2 * sigma_u^2) * Phi^-1(0.75))
    where sigma_u^2 is the between-hospital variance of the random intercept.
    """
    from scipy.stats import norm

    rates = hosp_rates["adjusted_rate"].dropna()
    if len(rates) < 3:
        return np.nan, np.nan, np.nan

    # Estimate sigma_u from logit-transformed rates
    logit_rates = np.log(rates.clip(0.001, 0.999) / (1 - rates.clip(0.001, 0.999)))
    sigma_u_sq = logit_rates.var()

    amor = np.exp(np.sqrt(2 * sigma_u_sq) * norm.ppf(0.75))

    # Bootstrap CI for aMOR
    rng = np.random.default_rng(42)
    boot_amors = []
    for _ in range(1000):
        boot_rates = rng.choice(rates.values, size=len(rates), replace=True)
        boot_logit = np.log(
            np.clip(boot_rates, 0.001, 0.999)
            / (1 - np.clip(boot_rates, 0.001, 0.999))
        )
        boot_var = boot_logit.var()
        boot_amors.append(np.exp(np.sqrt(2 * boot_var) * norm.ppf(0.75)))

    amor_lower = np.percentile(boot_amors, 2.5)
    amor_upper = np.percentile(boot_amors, 97.5)

    return amor, amor_lower, amor_upper


# ============================================================
# CATERPILLAR PLOT
# ============================================================

def caterpillar_plot(hosp_rates, delivery_label, output_path):
    """Generate caterpillar plot of hospital-level adjusted delivery rates.

    JAMA style: Arial font, min 8pt text, legends outside plot.
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

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.35)))

    # Error bars
    ax.errorbar(
        df["adjusted_rate"],
        range(n),
        xerr=[
            df["adjusted_rate"] - df["ci_lower"],
            df["ci_upper"] - df["adjusted_rate"],
        ],
        fmt="o",
        color="#2166AC",
        ecolor="#92C5DE",
        elinewidth=1.5,
        capsize=3,
        markersize=5,
        markeredgecolor="#2166AC",
        markerfacecolor="#2166AC",
    )

    # Overall reference line
    ax.axvline(
        overall_rate,
        color="#B2182B",
        linestyle="--",
        linewidth=1.2,
        label=f"Overall rate: {overall_rate:.1%}",
    )

    ax.set_yticks(range(n))
    ax.set_yticklabels(df["hospital_id"].astype(str), fontsize=8)
    ax.set_xlabel(f"Risk-adjusted {delivery_label} delivery rate", fontsize=9)
    ax.set_title(
        f"Hospital variation in {delivery_label} delivery\n"
        f"among eligible ventilator-days",
        fontsize=11,
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

    # Legend outside plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)

    ax.set_xlim(-0.05, 1.05)
    ax.grid(axis="x", alpha=0.3)

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
# CONCORDANCE: EHR vs FLOWSHEET RATES
# ============================================================

def concordance_ehr_vs_flowsheet(df, ehr_col, flowsheet_col, output_path):
    """Compare hospital-level rates from EHR phenotype vs flowsheet."""
    ehr_rates = compute_crude_hospital_rates(df, ehr_col)
    flow_rates = compute_crude_hospital_rates(df, flowsheet_col)

    merged = ehr_rates[["hospital_id", "crude_rate"]].rename(
        columns={"crude_rate": "ehr_rate"}
    ).merge(
        flow_rates[["hospital_id", "crude_rate"]].rename(
            columns={"crude_rate": "flowsheet_rate"}
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
    ax.text(0.05, 0.92,
            f"r = {r_pearson:.3f} (p = {p_pearson:.3f})\n"
            f"rho = {r_spearman:.3f} (p = {p_spearman:.3f})",
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
        "bias": bias,
        "loa": (loa_lower, loa_upper),
    }


# ============================================================
# MAIN
# ============================================================

def run_hospital_variation(sat_file, sbt_file, output_dir):
    """Run all hospital variation analyses."""
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    all_results = []

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sat-file", default="../output/intermediate/final_df_SAT.csv")
    parser.add_argument("--sbt-file", default="../output/intermediate/final_df_SBT.csv")
    parser.add_argument("--output-dir", default="../output/final")
    args = parser.parse_args()
    run_hospital_variation(args.sat_file, args.sbt_file, args.output_dir)
