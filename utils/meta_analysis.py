"""
Multi-Site Meta-Analysis for SAT/SBT CLIF Federated Study
==========================================================
Two-stage meta-analysis: site-level summary CSVs -> pooled estimates.

Requires: pip install statsmodels forestplot

Usage:
    from meta_analysis import (
        load_site_data, run_meta_analysis, run_proportion_meta_analysis,
        logit_transform, inv_logit,
        jama_forest_plot, funnel_plot, caterpillar_plot,
        pool_means, pool_proportions, aggregate_table1,
    )
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.stats.meta_analysis import combine_effects

matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
})


# ============================================================
# LOGIT TRANSFORMATION FOR PROPORTIONS
# ============================================================

def logit_transform(p, n):
    """Transform proportion to logit scale with continuity correction.

    For meta-analysis of proportions, pooling on the logit scale is
    preferred over the raw proportion scale because:
    1. The logit scale is unbounded (-inf, inf), avoiding boundary artifacts
    2. Variance stabilization is better
    3. Back-transformed pooled estimates respect (0, 1) bounds

    Parameters
    ----------
    p : float or array
        Proportion(s) (delivery rate)
    n : int or array
        Sample size(s)

    Returns
    -------
    logit_p : logit-transformed proportion
    se_logit : standard error on logit scale
    """
    p = np.asarray(p, dtype=float)
    n = np.asarray(n, dtype=float)
    # Continuity correction for 0 or 1 proportions
    p_adj = np.clip(p, 0.5 / n, 1 - 0.5 / n)
    logit_p = np.log(p_adj / (1 - p_adj))
    se_logit = np.sqrt(1.0 / (n * p_adj * (1 - p_adj)))
    return logit_p, se_logit


def inv_logit(x):
    """Inverse logit (expit) function."""
    return 1.0 / (1.0 + np.exp(-x))


def _hksj_adjustment(eff, var_eff, pooled_eff, tau2, k):
    """Hartung-Knapp-Sidik-Jonkman CI adjustment for random-effects meta-analysis.

    Replaces the standard normal z-based CI with a t-distribution CI using
    a corrected variance estimate. Recommended when k < 20 studies.

    References:
        IntHout J, et al. BMC Med Res Methodol. 2014;14:25. PMID: 24548571
        Hartung J, Knapp G. Stat Med. 2001;20(24):3875-3889.

    Returns (se_hksj, t_crit) for use in CI: pooled ± t_crit * se_hksj
    """
    from scipy import stats as sp_stats
    w = 1.0 / (var_eff + tau2)
    q_hksj = np.sum(w * (eff - pooled_eff) ** 2) / (k - 1)
    # Apply correction factor (ensure >= 1 to avoid anti-conservative CIs)
    q_hksj = max(q_hksj, 1.0)
    se_re = np.sqrt(1.0 / np.sum(w))
    se_hksj = se_re * np.sqrt(q_hksj)
    t_crit = sp_stats.t.ppf(0.975, df=k - 1)
    return se_hksj, t_crit


def cluster_bootstrap_concordance(
    day_level_df: pd.DataFrame,
    ehr_col: str,
    flowsheet_col: str,
    cluster_col: str = "imv_episode_id",
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Cluster-bootstrap concordance metrics with 95% percentile CIs.

    Resamples clusters (ventilator episodes or hospitalizations) with
    replacement to propagate within-cluster correlation into confidence
    intervals.  Falls back to hospitalization_id, then row-level bootstrap
    when the preferred cluster column is absent.

    Parameters
    ----------
    day_level_df : DataFrame
        One row per ventilator-episode-day (or similar grain).  Must contain
        ``ehr_col`` and ``flowsheet_col`` as binary (0/1) columns.
    ehr_col : str
        Binary column for EHR-derived phenotype (reference).
    flowsheet_col : str
        Binary column for flowsheet-derived phenotype (comparator).
    cluster_col : str
        Primary cluster identifier column.  Falls back to
        ``hospitalization_id``, then row-level bootstrap.
    n_boot : int
        Number of bootstrap iterations (default 1 000).
    seed : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    dict
        Keys for each metric (``sensitivity``, ``specificity``, ``ppv``,
        ``npv``, ``f1``, ``accuracy``, ``kappa``) each mapped to a sub-dict
        with ``estimate``, ``ci_low`` (2.5th pct), and ``ci_high``
        (97.5th pct).  Also contains top-level ``n_rows`` and
        ``n_clusters`` for reporting.

    Notes
    -----
    Both columns are coerced to int and rows where either value is NaN are
    dropped before analysis.  Cohen's kappa uses the standard unweighted
    formula.

    Examples
    --------
    >>> result = cluster_bootstrap_concordance(df, "EHR_flag", "flowsheet_flag")
    >>> result["sensitivity"]["estimate"]
    0.872
    >>> result["sensitivity"]["ci_low"], result["sensitivity"]["ci_high"]
    (0.851, 0.891)
    """
    rng = np.random.default_rng(seed)

    # --- Validate and clean ------------------------------------------------
    needed = [ehr_col, flowsheet_col]
    missing = [c for c in needed if c not in day_level_df.columns]
    if missing:
        raise ValueError(f"cluster_bootstrap_concordance: missing columns {missing}")

    df = day_level_df[[c for c in [cluster_col, "imv_episode_id",
                                    "hospitalization_id", ehr_col, flowsheet_col]
                        if c in day_level_df.columns]].copy()
    df = df.dropna(subset=[ehr_col, flowsheet_col])
    df[ehr_col] = df[ehr_col].astype(int)
    df[flowsheet_col] = df[flowsheet_col].astype(int)

    # Determine effective cluster column with fallback chain
    if cluster_col in df.columns:
        effective_cluster = cluster_col
    elif "hospitalization_id" in df.columns:
        effective_cluster = "hospitalization_id"
        print(f"  cluster_bootstrap_concordance: '{cluster_col}' not found, "
              "falling back to 'hospitalization_id'")
    else:
        effective_cluster = None
        print("  cluster_bootstrap_concordance: no cluster column found, "
              "using row-level bootstrap")

    # --- Helper: compute all metrics from flat arrays ----------------------
    def _metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        total = tp + fp + fn + tn

        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        ppv  = tp / (tp + fp) if (tp + fp) else 0.0
        npv  = tn / (tn + fn) if (tn + fn) else 0.0
        f1   = (2 * ppv * sens / (ppv + sens)) if (ppv + sens) else 0.0
        acc  = (tp + tn) / total if total else 0.0

        # Cohen's kappa
        p_obs = acc
        p_yes = ((tp + fp) / total) * ((tp + fn) / total) if total else 0.0
        p_no  = ((tn + fn) / total) * ((tn + fp) / total) if total else 0.0
        p_exp = p_yes + p_no
        kappa = (p_obs - p_exp) / (1 - p_exp) if (1 - p_exp) != 0 else 0.0

        return {
            "sensitivity": sens,
            "specificity": spec,
            "ppv": ppv,
            "npv": npv,
            "f1": f1,
            "accuracy": acc,
            "kappa": kappa,
        }

    # --- Point estimates on full data --------------------------------------
    point = _metrics_from_arrays(df[ehr_col].values, df[flowsheet_col].values)

    # --- Bootstrap ---------------------------------------------------------
    boot_records: list[dict[str, float]] = []

    if effective_cluster is not None:
        clusters = df[effective_cluster].unique()
        n_clusters = len(clusters)

        for _ in range(n_boot):
            sampled = rng.choice(clusters, size=n_clusters, replace=True)
            # Reconstruct bootstrap sample from cluster draws
            frames = [df[df[effective_cluster] == cid] for cid in sampled]
            boot_df = pd.concat(frames, ignore_index=True)
            boot_records.append(
                _metrics_from_arrays(boot_df[ehr_col].values,
                                     boot_df[flowsheet_col].values)
            )
    else:
        # Row-level bootstrap fallback
        n_clusters = len(df)
        idx = np.arange(len(df))
        for _ in range(n_boot):
            sampled_idx = rng.choice(idx, size=len(df), replace=True)
            boot_df = df.iloc[sampled_idx]
            boot_records.append(
                _metrics_from_arrays(boot_df[ehr_col].values,
                                     boot_df[flowsheet_col].values)
            )

    boot_arr = {k: np.array([r[k] for r in boot_records])
                for k in point}

    # --- Assemble output ---------------------------------------------------
    result: dict = {
        "n_rows": len(df),
        "n_clusters": n_clusters,
        "cluster_col_used": effective_cluster or "row-level",
    }
    for metric, est in point.items():
        result[metric] = {
            "estimate": round(est, 4),
            "ci_low":  round(float(np.percentile(boot_arr[metric], 2.5)), 4),
            "ci_high": round(float(np.percentile(boot_arr[metric], 97.5)), 4),
        }

    return result


def run_proportion_meta_analysis(
    sites_df: pd.DataFrame,
    rate_col: str,
    n_col: str,
    label_col: str = "site_name",
    method: str = "reml",
    use_hksj: bool = True,
):
    """Meta-analysis of proportions using logit transformation.

    Transforms proportions to logit scale before pooling, then
    back-transforms pooled estimate to proportion scale.

    Parameters
    ----------
    sites_df : DataFrame with site-level delivery rates and sample sizes
    rate_col : column name for proportion (delivery rate, 0-1)
    n_col : column name for sample sizes (denominators)
    label_col : column name for site labels
    method : tau-squared estimator ('reml' recommended, 'dl' for sensitivity)
    use_hksj : bool
        Apply Hartung-Knapp-Sidik-Jonkman CI adjustment (recommended for k < 20)

    Returns
    -------
    res : CombineResults from statsmodels
    summary : DataFrame with site + pooled rows (on proportion scale)
    """
    p = sites_df[rate_col].values
    n = sites_df[n_col].values

    logit_p, se_logit = logit_transform(p, n)

    res = combine_effects(
        logit_p, se_logit ** 2,
        method_re=method,
        row_names=sites_df[label_col].tolist(),
    )

    # Build summary on proportion scale
    summary = res.summary_frame().reset_index()
    # Robust column naming: use actual columns from summary_frame
    _expected_cols = ["label", "eff", "sd_eff", "ci_low", "ci_upp", "w_fe", "w_re", "z", "pval"]
    if len(summary.columns) == len(_expected_cols):
        summary.columns = _expected_cols
    else:
        # Fallback: rename by position for the columns we need
        summary = summary.rename(columns={summary.columns[0]: "label", summary.columns[1]: "eff",
                                           summary.columns[3]: "ci_low", summary.columns[4]: "ci_upp"})
        if "sd_eff" not in summary.columns and len(summary.columns) > 2:
            summary = summary.rename(columns={summary.columns[2]: "sd_eff"})
    # Back-transform to proportion scale
    summary["eff_prop"] = inv_logit(summary["eff"])
    summary["ci_low_prop"] = inv_logit(summary["ci_low"])
    summary["ci_upp_prop"] = inv_logit(summary["ci_upp"])

    # Add pooled row — eff_re is a scalar float, not an object
    pooled_logit = res.eff_re
    pooled_ci = res.conf_int_re()

    # Apply HKSJ adjustment if requested and k >= 3
    k = len(logit_p)
    ci_method = f"RE ({method})"
    if use_hksj and k >= 3:
        tau2 = getattr(res, "tau2", 0.0)
        se_hksj, t_crit = _hksj_adjustment(logit_p, se_logit ** 2, pooled_logit, tau2, k)
        pooled_ci = (pooled_logit - t_crit * se_hksj, pooled_logit + t_crit * se_hksj)
        ci_method = f"RE ({method}) + HKSJ"

    pooled = {
        "label": f"Pooled ({ci_method})",
        "eff": pooled_logit,
        "sd_eff": res.sd_eff_re if hasattr(res, "sd_eff_re") else np.nan,
        "ci_low": pooled_ci[0],
        "ci_upp": pooled_ci[1],
        "w_fe": np.nan,
        "w_re": np.nan,
        "z": np.nan,
        "pval": np.nan,
        "eff_prop": inv_logit(pooled_logit),
        "ci_low_prop": inv_logit(pooled_ci[0]),
        "ci_upp_prop": inv_logit(pooled_ci[1]),
    }
    summary = pd.concat([summary, pd.DataFrame([pooled])], ignore_index=True)

    return res, summary

# ============================================================
# STAGE 1: LOAD AND VALIDATE SITE CSVs
# ============================================================

def load_site_data(csv_dir: str, pattern: str = "site_*.csv") -> pd.DataFrame:
    """Load all site-level CSV exports into a single DataFrame."""
    frames = []
    for f in sorted(Path(csv_dir).glob(pattern)):
        df = pd.read_csv(f)
        if not {"site_id", "site_name"}.issubset(df.columns):
            raise ValueError(f"Missing required columns in {f.name}")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No files matching '{pattern}' in {csv_dir}")
    return pd.concat(frames, ignore_index=True)


# ============================================================
# STAGE 2: META-ANALYSIS
# ============================================================

def run_meta_analysis(
    sites_df: pd.DataFrame,
    estimate_col: str,
    se_col: str,
    label_col: str = "site_name",
    method: str = "reml",
    use_hksj: bool = True,
):
    """Run fixed + random-effects meta-analysis on site-level estimates.

    Parameters
    ----------
    sites_df : DataFrame with site-level estimates
    estimate_col : column name for effect estimates
    se_col : column name for standard errors
    label_col : column name for site labels
    method : tau-squared estimator ('reml' recommended, 'dl' for sensitivity)
    use_hksj : bool
        Apply HKSJ CI adjustment (recommended for k < 20; IntHout et al. 2014)

    Returns
    -------
    res : CombineResults from statsmodels
    summary : DataFrame with site + pooled rows
    """
    eff = sites_df[estimate_col].values
    se = sites_df[se_col].values
    var = se ** 2

    res = combine_effects(
        eff, var,
        method_re=method,
        row_names=sites_df[label_col].tolist(),
    )

    summary = res.summary_frame().reset_index()
    _expected_cols = ["label", "eff", "sd_eff", "ci_low", "ci_upp", "w_fe", "w_re", "z", "pval"]
    if len(summary.columns) == len(_expected_cols):
        summary.columns = _expected_cols
    else:
        summary = summary.rename(columns={summary.columns[0]: "label", summary.columns[1]: "eff",
                                           summary.columns[3]: "ci_low", summary.columns[4]: "ci_upp"})
        if "sd_eff" not in summary.columns and len(summary.columns) > 2:
            summary = summary.rename(columns={summary.columns[2]: "sd_eff"})

    pooled_eff = res.eff_re
    pooled_ci = res.conf_int_re()
    k = len(eff)
    ci_method = f"RE ({method})"

    if use_hksj and k >= 3:
        tau2 = getattr(res, "tau2", 0.0)
        se_hksj, t_crit = _hksj_adjustment(eff, var, pooled_eff, tau2, k)
        pooled_ci = (pooled_eff - t_crit * se_hksj, pooled_eff + t_crit * se_hksj)
        ci_method = f"RE ({method}) + HKSJ"

    pooled = {
        "label": f"Pooled ({ci_method})",
        "eff": pooled_eff,
        "sd_eff": res.sd_eff_re if hasattr(res, "sd_eff_re") else np.nan,
        "ci_low": pooled_ci[0],
        "ci_upp": pooled_ci[1],
        "w_fe": np.nan,
        "w_re": np.nan,
        "z": np.nan,
        "pval": np.nan,
    }
    summary = pd.concat([summary, pd.DataFrame([pooled])], ignore_index=True)

    return res, summary


def sensitivity_leave_one_out(
    sites_df: pd.DataFrame,
    estimate_col: str,
    se_col: str,
    label_col: str = "site_name",
) -> pd.DataFrame:
    """Leave-one-out sensitivity analysis."""
    results = []
    for i, row in sites_df.iterrows():
        subset = sites_df.drop(i)
        res, _ = run_meta_analysis(subset, estimate_col, se_col, label_col)
        results.append({
            "excluded_site": row[label_col],
            "pooled_eff": res.eff_re,
            "ci_low": res.conf_int_re()[0],
            "ci_upp": res.conf_int_re()[1],
            "i2": getattr(res, "i2", np.nan),
        })
    return pd.DataFrame(results)


# ============================================================
# TABLE 1 POOLING
# ============================================================

def pool_means(df: pd.DataFrame, mean_col: str, sd_col: str, n_col: str):
    """Pool means and SDs across sites using weighted formulas.
    Returns (N_total, pooled_mean, pooled_sd)."""
    n = df[n_col].values.astype(float)
    m = df[mean_col].values.astype(float)
    s = df[sd_col].values.astype(float)
    N = n.sum()
    pooled_mean = np.average(m, weights=n)
    pooled_var = (np.sum((n - 1) * s**2 + n * m**2) / N) - pooled_mean**2
    pooled_sd = np.sqrt(max(pooled_var, 0))
    return int(N), pooled_mean, pooled_sd


def pool_proportions(df: pd.DataFrame, num_col: str, n_col: str):
    """Pool proportions across sites.
    Returns (N_total, n_events, proportion)."""
    total_num = df[num_col].sum()
    total_n = df[n_col].sum()
    return int(total_n), int(total_num), total_num / total_n if total_n > 0 else 0.0


def pool_medians_iqr(df: pd.DataFrame, median_col: str,
                     iqr_low_col: str, iqr_high_col: str, n_col: str):
    """Pool medians using sample-size-weighted median (approximation).
    Returns (N_total, weighted_median, weighted_iqr_low, weighted_iqr_high)."""
    n = df[n_col].values.astype(float)
    N = int(n.sum())
    w_median = np.average(df[median_col].values, weights=n)
    w_iqr_low = np.average(df[iqr_low_col].values, weights=n)
    w_iqr_high = np.average(df[iqr_high_col].values, weights=n)
    return N, w_median, w_iqr_low, w_iqr_high


def aggregate_table1(table1_df: pd.DataFrame) -> dict:
    """Aggregate Table 1 from all sites into pooled summary."""
    N, mean_age, sd_age = pool_means(table1_df, "age_mean", "age_sd", "n_total")
    _, n_male, pct_male = pool_proportions(table1_df, "n_male", "n_total")
    _, n_female, pct_female = pool_proportions(table1_df, "n_female", "n_total")
    _, n_alive, pct_alive = pool_proportions(
        table1_df, "n_discharged_alive", "n_total"
    )
    _, n_expired, pct_expired = pool_proportions(
        table1_df, "n_discharged_expired", "n_total"
    )
    N_los, med_icu, iqr_lo, iqr_hi = pool_medians_iqr(
        table1_df, "icu_los_median", "icu_los_iqr_lower",
        "icu_los_iqr_upper", "n_total"
    )

    return {
        "N": N,
        "Age, mean (SD)": f"{mean_age:.1f} ({sd_age:.1f})",
        "Male, n (%)": f"{n_male} ({pct_male:.1%})",
        "Female, n (%)": f"{n_female} ({pct_female:.1%})",
        "ICU LOS, median [IQR]": f"{med_icu:.1f} [{iqr_lo:.1f}, {iqr_hi:.1f}]",
        "Discharged alive, n (%)": f"{n_alive} ({pct_alive:.1%})",
        "Expired, n (%)": f"{n_expired} ({pct_expired:.1%})",
    }


# ============================================================
# FIGURES (JAMA-STYLE)
# ============================================================

def jama_forest_plot(
    summary_df: pd.DataFrame,
    outcome_label: str = "Delivery rate",
    heterogeneity_text: str = "",
    figsize: tuple = (10, None),
    save_path: str | None = None,
):
    """Create a JAMA-style forest plot using matplotlib.

    summary_df must have columns: label, eff, ci_low, ci_upp
    Last row should be the pooled estimate.
    """
    df = summary_df.copy()
    n = len(df)
    if figsize[1] is None:
        figsize = (figsize[0], max(4, n * 0.4 + 1.5))

    fig, ax = plt.subplots(figsize=figsize)
    y_positions = list(range(n - 1, -1, -1))

    # Site-level estimates (all but last row)
    for i in range(n - 1):
        row = df.iloc[i]
        ax.plot(row["eff"], y_positions[i], "s", color="navy",
                markersize=6, zorder=3)
        ax.plot([row["ci_low"], row["ci_upp"]], [y_positions[i]] * 2,
                "-", color="navy", linewidth=1.5, zorder=2)

    # Pooled estimate (diamond)
    pooled = df.iloc[-1]
    diamond_x = [pooled["ci_low"], pooled["eff"], pooled["ci_upp"], pooled["eff"]]
    diamond_y = [y_positions[-1], y_positions[-1] + 0.2,
                 y_positions[-1], y_positions[-1] - 0.2]
    ax.fill(diamond_x, diamond_y, color="firebrick", zorder=3)

    # Reference line at pooled estimate
    ax.axvline(pooled["eff"], color="gray", linestyle=":", linewidth=0.8, zorder=1)

    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["label"].tolist(), fontsize=9, fontfamily="Arial")
    ax.set_xlabel(outcome_label, fontsize=10, fontfamily="Arial")
    ax.set_title("Forest plot", fontsize=11, fontweight="bold", fontfamily="Arial")

    # Right-side annotations (estimate [CI])
    for idx, (_, row) in enumerate(df.iterrows()):
        ci_text = f"{row['eff']:.3f} [{row['ci_low']:.3f}, {row['ci_upp']:.3f}]"
        ax.annotate(ci_text, xy=(ax.get_xlim()[1], y_positions[idx]),
                    fontsize=8, fontfamily="Arial", va="center",
                    xytext=(5, 0), textcoords="offset points")

    # Heterogeneity annotation
    if heterogeneity_text:
        ax.text(0.02, -0.08, heterogeneity_text, transform=ax.transAxes,
                fontsize=8, fontfamily="Arial", style="italic")

    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def funnel_plot(
    sites_df: pd.DataFrame,
    estimate_col: str,
    se_col: str,
    pooled_est: float,
    label_col: str = "site_name",
    save_path: str | None = None,
):
    """Standard funnel plot: effect vs precision (1/SE)."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    se = sites_df[se_col].values
    eff = sites_df[estimate_col].values
    precision = 1.0 / se

    ax.scatter(eff, precision, s=35, color="navy",
               edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(pooled_est, color="firebrick", linestyle="--", linewidth=1)

    # Pseudo-CI funnel lines (95%)
    se_range = np.linspace(se.min() * 0.3, se.max() * 1.5, 100)
    for z in [1.96]:
        ax.plot(pooled_est + z * se_range, 1 / se_range,
                color="gray", linestyle=":", linewidth=0.8)
        ax.plot(pooled_est - z * se_range, 1 / se_range,
                color="gray", linestyle=":", linewidth=0.8)

    ax.set_xlabel("Effect estimate", fontsize=9)
    ax.set_ylabel("Precision (1/SE)", fontsize=9)
    ax.set_title("Funnel plot", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def caterpillar_plot(
    sites_df: pd.DataFrame,
    estimate_col: str,
    ci_low_col: str,
    ci_high_col: str,
    label_col: str = "site_name",
    pooled_est: float | None = None,
    save_path: str | None = None,
):
    """Caterpillar plot showing site-level estimates ranked by effect size."""
    df = sites_df.sort_values(estimate_col).reset_index(drop=True)
    n = len(df)

    fig, ax = plt.subplots(figsize=(6, max(4, n * 0.35 + 1)))
    y = range(n)

    ax.errorbar(
        df[estimate_col], y,
        xerr=[
            df[estimate_col] - df[ci_low_col],
            df[ci_high_col] - df[estimate_col],
        ],
        fmt="o", color="navy", ecolor="gray",
        elinewidth=1, capsize=2, markersize=4,
    )

    if pooled_est is not None:
        ax.axvline(pooled_est, color="firebrick", linestyle="--",
                   linewidth=1, label="Pooled estimate")
        ax.legend(fontsize=8, loc="lower right")

    ax.set_yticks(list(y))
    ax.set_yticklabels(df[label_col].tolist(), fontsize=8)
    ax.set_xlabel("Estimate (95% CI)", fontsize=9)
    ax.set_title("Hospital variation", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax
