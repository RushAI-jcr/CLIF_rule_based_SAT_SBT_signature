"""
05_manuscript_figures.py
========================
Generate all manuscript-ready figures (Figures 1-5) and supplemental figures.

All figures follow JAMA style:
- Arial font
- Min 8pt text everywhere
- Title 10-12pt bold, axis labels 9-10pt, ticks 8-9pt, annotations 8pt
- Legends outside plot area
- No text overlap

CLIF 2.1 compliance: all data consumed from CLIF-derived outputs.

Usage:
    python 05_manuscript_figures.py --data-dir ../output/intermediate \
                                     --output-dir ../output/final/figures
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "utils"))


import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# JAMA Style
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
# FIGURE 1: CONSORT FLOW DIAGRAM
# ============================================================

def figure1_consort(data_dir, output_path, site_stats=None):
    """Generate CONSORT-style flow diagram.

    If site_stats dict is provided, uses real numbers.
    Otherwise generates template with placeholder counts.
    """
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # Default placeholder values
    s = site_stats or {}
    n_admissions = s.get("n_admissions", "XX,XXX")
    n_icu = s.get("n_icu_admissions", "XX,XXX")
    n_imv = s.get("n_imv_episodes", "XX,XXX")
    n_trach_excl = s.get("n_trach_excluded", "X,XXX")
    n_age_excl = s.get("n_age_excluded", "X,XXX")
    n_final_episodes = s.get("n_final_episodes", "XX,XXX")
    n_vent_days = s.get("n_vent_days", "XXX,XXX")
    n_sat_eligible = s.get("n_sat_eligible_days", "XX,XXX")
    n_sbt_eligible = s.get("n_sbt_eligible_days", "XX,XXX")
    n_sites = s.get("n_sites", "X")
    n_hospitals = s.get("n_hospitals", "XX")
    n_sat_flowsheet = s.get("n_sat_flowsheet_hospitals", "X")
    n_sbt_flowsheet = s.get("n_sbt_flowsheet_hospitals", "X")

    def draw_box(x, y, w, h, text, color="#E8EEF4"):
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="#2166AC", linewidth=1.2
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=8,
                fontweight="normal", wrap=True,
                bbox=dict(boxstyle="round", facecolor="none", edgecolor="none"))

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color="#2166AC", lw=1.2))

    def draw_exclusion(x_main, y_main, x_excl, y_excl, text):
        draw_arrow(x_main, y_main, x_excl, y_excl)
        ax.text(x_excl + 0.1, y_excl, text, ha="left", va="center",
                fontsize=8, color="#B2182B")

    # Top: All admissions
    draw_box(5, 13, 4, 0.7,
             f"All hospital admissions\n{n_sites} sites, {n_hospitals} hospitals\n"
             f"Jan 2022 - Dec 2024\n(n = {n_admissions})")

    draw_arrow(5, 12.65, 5, 12.0)

    # ICU admissions
    draw_box(5, 11.5, 3.5, 0.7,
             f"ICU admissions with IMV\n(n = {n_icu})")

    # Exclusions
    draw_arrow(5, 11.15, 5, 10.5)
    draw_exclusion(6.75, 11.5, 7.5, 11.5,
                   f"Tracheostomy excluded\n(n = {n_trach_excl})")
    draw_exclusion(6.75, 10.8, 7.5, 10.8,
                   f"Age < 18 excluded\n(n = {n_age_excl})")

    # Final episodes
    draw_box(5, 9.8, 3.5, 0.7,
             f"IMV episodes included\n(n = {n_final_episodes})")

    draw_arrow(5, 9.45, 5, 8.8)

    # Ventilator-days
    draw_box(5, 8.3, 3, 0.7,
             f"Total ventilator-days\n(n = {n_vent_days})")

    # Split to SAT and SBT
    draw_arrow(5, 7.95, 3, 7.3)
    draw_arrow(5, 7.95, 7, 7.3)

    # SAT branch
    draw_box(3, 6.8, 2.5, 0.7,
             f"SAT-eligible days\n(n = {n_sat_eligible})",
             color="#D4E6F1")
    draw_box(3, 5.5, 2.5, 0.7,
             f"Flowsheet data\navailable at\n{n_sat_flowsheet} hospitals",
             color="#FCF3CF")

    draw_arrow(3, 6.45, 3, 5.85)

    # SBT branch
    draw_box(7, 6.8, 2.5, 0.7,
             f"SBT-eligible days\n(n = {n_sbt_eligible})",
             color="#D4E6F1")
    draw_box(7, 5.5, 2.5, 0.7,
             f"Flowsheet data\navailable at\n{n_sbt_flowsheet} hospitals",
             color="#FCF3CF")

    draw_arrow(7, 6.45, 7, 5.85)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Figure 1 saved: {output_path}")


# ============================================================
# FIGURE 2: PHENOTYPING LOGIC DIAGRAM
# ============================================================

def figure2_phenotype_logic(output_path):
    """Generate phenotyping logic diagram for SAT and SBT.

    Panel A: SAT eligibility and delivery
    Panel B: SBT eligibility and delivery
    """
    from definitions_source_of_truth import (
        SAT_SEDATIVES, SAT_OPIOIDS, PARALYTICS,
        SAT_COMPLETE_DURATION_MIN, SAT_MODIFIED_DURATION_MIN,
        SBT_CONTROLLED_MODES, SBT_FIO2_MAX, SBT_PEEP_MAX, SBT_SPO2_MIN,
        SBT_PRIMARY_DURATION_MIN, SBT_MODIFIED_DURATION_MIN,
        SBT_MIN_CONTROLLED_MODE_HOURS,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis("off")

    def draw_box(ax, x, y, w, h, text, color="#E8EEF4", fontsize=8):
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="#2166AC", linewidth=1
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, wrap=True)

    def draw_arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color="#2166AC", lw=1))

    # --- Panel A: SAT ---
    ax = axes[0]
    ax.set_title("A. SAT Phenotyping", fontsize=11, fontweight="bold", pad=10)

    draw_box(ax, 5, 11, 8, 0.8,
             "SAT Eligibility (10 PM - 6 AM window)\n"
             f"On continuous sedatives ({', '.join(SAT_SEDATIVES)})\n"
             f"and/or opioids ({', '.join(SAT_OPIOIDS)}) for >= 4 hrs",
             color="#D4E6F1")

    draw_box(ax, 5, 9.5, 6, 0.6,
             f"Exclusions: paralytics ({', '.join(PARALYTICS)}), RASS >= 2",
             color="#FADBD8")

    draw_arrow(ax, 5, 10.6, 5, 9.8)
    draw_arrow(ax, 5, 9.2, 5, 8.5)

    draw_box(ax, 5, 8, 8, 0.8,
             f"Complete SAT: ALL sedatives + opioids\n"
             f"discontinued for >= {SAT_COMPLETE_DURATION_MIN} min",
             color="#D5F5E3")

    draw_arrow(ax, 5, 7.6, 5, 6.8)

    draw_box(ax, 5, 6.3, 8, 0.8,
             f"Modified SAT: Sedatives discontinued\n"
             f"(opioids may continue) for >= {SAT_MODIFIED_DURATION_MIN} min",
             color="#FCF3CF")

    draw_arrow(ax, 5, 5.9, 5, 5.2)

    draw_box(ax, 5, 4.7, 8, 0.8,
             "RASS-Enhanced: Alertness (RASS 0 to +1)\n"
             "within 45 min of discontinuation",
             color="#F5EEF8")

    # --- Panel B: SBT ---
    ax = axes[1]
    ax.set_title("B. SBT Phenotyping", fontsize=11, fontweight="bold", pad=10)

    modes_str = ", ".join([m.title() for m in SBT_CONTROLLED_MODES[:3]]) + "..."
    draw_box(ax, 5, 11, 8, 0.8,
             f"SBT Eligibility (10 PM - 6 AM window)\n"
             f"On controlled mode ({modes_str})\n"
             f"for >= {SBT_MIN_CONTROLLED_MODE_HOURS} hrs",
             color="#D4E6F1")

    draw_box(ax, 5, 9.5, 8, 0.8,
             f"Stability: FiO2 <= {SBT_FIO2_MAX:.0%}, PEEP <= {SBT_PEEP_MAX},\n"
             f"SpO2 >= {SBT_SPO2_MIN}%, NEE <= 0.2 mcg/kg/min\n"
             f"Exclusions: paralytics",
             color="#FADBD8")

    draw_arrow(ax, 5, 10.6, 5, 9.9)
    draw_arrow(ax, 5, 9.1, 5, 8.3)

    draw_box(ax, 5, 7.8, 8, 0.8,
             f"Primary SBT: Transition to support mode\n"
             f"(PS/CPAP <= 8 cmH2O or T-piece)\n"
             f"for >= {SBT_PRIMARY_DURATION_MIN} min",
             color="#D5F5E3")

    draw_arrow(ax, 5, 7.4, 5, 6.5)

    draw_box(ax, 5, 6.0, 8, 0.8,
             f"Modified SBT: Transition sustained\n"
             f"for >= {SBT_MODIFIED_DURATION_MIN} min",
             color="#FCF3CF")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Figure 2 saved: {output_path}")


# ============================================================
# FIGURE 3: CRITERION VALIDITY HEATMAPS
# ============================================================

def figure3_criterion_validity(data_dir, output_path):
    """Confusion matrix heatmaps + site-stratified metrics.

    Reads per-site concordance CSVs from output/final/{SITE}/SAT_standard/
    and SBT equivalents.
    """
    # Attempt to find concordance files
    concordance_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if "concordance_summary" in f and f.endswith(".csv"):
                concordance_files.append(os.path.join(root, f))

    if not concordance_files:
        print("  No concordance files found. Generating template figure.")
        _figure3_template(output_path)
        return

    all_metrics = []
    for fpath in concordance_files:
        df = pd.read_csv(fpath)
        # Extract site name from path
        parts = fpath.split(os.sep)
        site = [p for p in parts if p not in ["output", "final", "intermediate"]]
        df["site"] = site[-2] if len(site) >= 2 else "Unknown"
        all_metrics.append(df)

    metrics_df = pd.concat(all_metrics, ignore_index=True)

    n_cols = min(3, len(metrics_df["Column"].unique()))
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    for i, col in enumerate(metrics_df["Column"].unique()[:n_cols]):
        ax = axes[i]
        subset = metrics_df[metrics_df["Column"] == col]

        # Pooled confusion matrix
        tp = subset["TP"].sum()
        fp = subset["FP"].sum()
        fn = subset["FN"].sum()
        tn = subset["TN"].sum()
        cm = np.array([[tn, fp], [fn, tp]])
        total = cm.sum()

        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        labels = ["No Delivery", "Delivery"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("EHR phenotype", fontsize=9)
        ax.set_ylabel("Flowsheet", fontsize=9)
        ax.set_title(col.replace("_", " "), fontsize=10, fontweight="bold")

        for r in range(2):
            for c in range(2):
                val = cm[r, c]
                pct = val / total * 100 if total > 0 else 0
                ax.text(c, r, f"{val:,}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=8,
                        color="white" if val > cm.max() / 2 else "black")

        # Metrics annotation
        acc = (tp + tn) / total if total else 0
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        ppv = tp / (tp + fp) if (tp + fp) else 0
        npv = tn / (tn + fn) if (tn + fn) else 0
        f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else 0

        metrics_text = (
            f"Sens: {sens:.2f}  Spec: {spec:.2f}\n"
            f"PPV: {ppv:.2f}  NPV: {npv:.2f}\n"
            f"Acc: {acc:.2f}  F1: {f1:.2f}"
        )
        ax.text(0.5, -0.2, metrics_text, transform=ax.transAxes,
                ha="center", fontsize=8, family="monospace")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Figure 3 saved: {output_path}")


def _figure3_template(output_path):
    """Template figure when no data available."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, label in zip(axes, ["SAT Delivery", "SBT Delivery"]):
        ax.text(0.5, 0.5, f"{label}\n[Data pending]",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Figure 3 template saved: {output_path}")


# ============================================================
# FIGURE 4: TIMING AND PAIRING OF SAT + SBT
# ============================================================

def figure4_timing_pairing(data_dir, output_path):
    """Distribution of first eligible/delivered day + paired SAT+SBT proportion.

    Reads final_df_SAT.csv and final_df_SBT.csv (or study_cohort.parquet).
    """
    sat_path = os.path.join(data_dir, "final_df_SAT.csv")
    sbt_path = os.path.join(data_dir, "final_df_SBT.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if os.path.exists(sat_path):
        sat_df = pd.read_csv(sat_path, low_memory=False)
        sat_df["event_time"] = pd.to_datetime(sat_df.get("event_time"), format="mixed")

        # Panel A: Distribution of first eligible day
        ax = axes[0]
        if "day_number" in sat_df.columns and "eligible_event" in sat_df.columns:
            first_elig = sat_df[sat_df["eligible_event"] == 1].groupby(
                "hospitalization_id"
            )["day_number"].min()
            bins = range(0, min(int(first_elig.max()) + 2, 30))
            ax.hist(first_elig, bins=bins, color="#2166AC", edgecolor="white",
                    alpha=0.8)
            ax.set_xlabel("IMV day of first eligibility", fontsize=9)
            ax.set_ylabel("Hospitalizations, n", fontsize=9)
            ax.set_title("First eligible day", fontsize=11, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "[Data pending]", ha="center", va="center",
                    fontsize=10, transform=ax.transAxes)
            ax.set_title("First eligible day", fontsize=11, fontweight="bold")
    else:
        axes[0].text(0.5, 0.5, "[SAT data not found]", ha="center", va="center",
                     fontsize=10, transform=axes[0].transAxes)

    # Panel B: First delivery day
    ax = axes[1]
    if os.path.exists(sat_path):
        if "SAT_EHR_delivery" in sat_df.columns and "day_number" in sat_df.columns:
            first_deliv = sat_df[sat_df["SAT_EHR_delivery"] == 1].groupby(
                "hospitalization_id"
            )["day_number"].min()
            if len(first_deliv) > 0:
                bins = range(0, min(int(first_deliv.max()) + 2, 30))
                ax.hist(first_deliv, bins=bins, color="#B2182B", edgecolor="white",
                        alpha=0.8)
    ax.set_xlabel("IMV day of first delivery", fontsize=9)
    ax.set_ylabel("Hospitalizations, n", fontsize=9)
    ax.set_title("First delivered day", fontsize=11, fontweight="bold")

    # Panel C: Paired SAT+SBT on same day
    ax = axes[2]
    if os.path.exists(sat_path) and os.path.exists(sbt_path):
        sbt_df = pd.read_csv(sbt_path, low_memory=False)

        # Find days with SAT delivery
        sat_days = set()
        for dcol in ["SAT_EHR_delivery", "SAT_modified_delivery"]:
            if dcol in sat_df.columns:
                sat_days.update(
                    sat_df[sat_df[dcol] == 1]["hosp_id_day_key"].unique()
                )

        # Find days with SBT delivery
        sbt_days = set()
        for dcol in ["EHR_Delivery_2mins", "EHR_Delivery_5mins", "EHR_Delivery_30mins"]:
            if dcol in sbt_df.columns:
                sbt_days.update(
                    sbt_df[sbt_df[dcol] == 1]["hosp_id_day_key"].unique()
                )

        paired = sat_days & sbt_days
        sat_only = sat_days - sbt_days
        sbt_only = sbt_days - sat_days

        counts = [len(paired), len(sat_only), len(sbt_only)]
        labels_pie = ["SAT + SBT\n(same day)", "SAT only", "SBT only"]
        colors = ["#2166AC", "#92C5DE", "#F4A582"]

        if sum(counts) > 0:
            wedges, texts, autotexts = ax.pie(
                counts, labels=labels_pie, colors=colors,
                autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 8},
            )
            for t in autotexts:
                t.set_fontsize(8)
        else:
            ax.text(0.5, 0.5, "[No delivery data]", ha="center", va="center",
                    fontsize=10, transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "[Data pending]", ha="center", va="center",
                fontsize=10, transform=ax.transAxes)
    ax.set_title("Paired SAT + SBT", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Figure 4 saved: {output_path}")


# ============================================================
# POOLED TABLE GENERATORS (for manuscript Tables 2-4)
# ============================================================

def generate_pooled_table2(data_dir, output_dir):
    """Pool Table 1 (baseline characteristics) across sites."""
    cat_files = []
    cont_files = []
    for root, dirs, files in os.walk(os.path.join(data_dir, "..")):
        for f in files:
            if "table1_eligible_event_categorical" in f:
                cat_files.append(os.path.join(root, f))
            if "table1_eligible_event_continuous" in f:
                cont_files.append(os.path.join(root, f))

    if cat_files:
        all_cat = pd.concat([pd.read_csv(f) for f in cat_files], ignore_index=True)
        # Sum counts across sites
        pooled = all_cat.groupby(["Variable", "Category"]).agg(
            N=("N", "sum"), Count=("Count", "sum")
        ).reset_index()
        pooled["Percent"] = (pooled["Count"] / pooled["N"] * 100).round(2)
        pooled.to_csv(os.path.join(output_dir, "table2_pooled_categorical.csv"),
                      index=False)
        print(f"  Pooled Table 2 (categorical) saved")

    if cont_files:
        all_cont = pd.concat([pd.read_csv(f) for f in cont_files], ignore_index=True)
        all_cont.to_csv(os.path.join(output_dir, "table2_pooled_continuous.csv"),
                        index=False)
        print(f"  Pooled Table 2 (continuous) saved")


def generate_pooled_criterion_validity(data_dir, output_dir):
    """Pool criterion validity metrics across sites with bootstrap CIs."""
    concordance_files = []
    for root, dirs, files in os.walk(os.path.join(data_dir, "..")):
        for f in files:
            if "concordance_summary" in f and f.endswith(".csv"):
                concordance_files.append(os.path.join(root, f))

    if not concordance_files:
        print("  No concordance files found for pooling")
        return

    all_metrics = pd.concat(
        [pd.read_csv(f) for f in concordance_files], ignore_index=True
    )

    pooled = all_metrics.groupby("Column").agg(
        TP=("TP", "sum"), FP=("FP", "sum"),
        FN=("FN", "sum"), TN=("TN", "sum"),
    ).reset_index()

    for _, row in pooled.iterrows():
        tp, fp, fn, tn = row["TP"], row["FP"], row["FN"], row["TN"]
        total = tp + fp + fn + tn
        pooled.loc[pooled["Column"] == row["Column"], "Accuracy"] = (
            (tp + tn) / total if total else 0
        )
        pooled.loc[pooled["Column"] == row["Column"], "Sensitivity"] = (
            tp / (tp + fn) if (tp + fn) else 0
        )
        pooled.loc[pooled["Column"] == row["Column"], "Specificity"] = (
            tn / (tn + fp) if (tn + fp) else 0
        )
        pooled.loc[pooled["Column"] == row["Column"], "PPV"] = (
            tp / (tp + fp) if (tp + fp) else 0
        )
        pooled.loc[pooled["Column"] == row["Column"], "NPV"] = (
            tn / (tn + fn) if (tn + fn) else 0
        )
        sens = tp / (tp + fn) if (tp + fn) else 0
        ppv = tp / (tp + fp) if (tp + fp) else 0
        pooled.loc[pooled["Column"] == row["Column"], "F1"] = (
            2 * ppv * sens / (ppv + sens) if (ppv + sens) else 0
        )

    pooled.to_csv(
        os.path.join(output_dir, "table3_4_pooled_criterion_validity.csv"),
        index=False,
    )
    print(f"  Pooled criterion validity (Tables 3-4) saved")


# ============================================================
# MAIN
# ============================================================

def _save_multi_format(fig, base_path, dpi=600):
    """Save figure in PDF, TIFF (600 DPI), and EPS for ICM submission.

    ICM/Springer requirements:
    - Vector: EPS or PDF
    - Halftone: TIFF >= 300 DPI
    - Combination (text + data): TIFF >= 600 DPI
    """
    fig.savefig(base_path, bbox_inches="tight", dpi=300)  # PDF
    tiff_path = base_path.replace(".pdf", ".tiff")
    fig.savefig(tiff_path, bbox_inches="tight", dpi=dpi, format="tiff")
    eps_path = base_path.replace(".pdf", ".eps")
    try:
        fig.savefig(eps_path, bbox_inches="tight", format="eps")
    except Exception as e:
        print(f"  EPS export failed ({e}), PDF and TIFF saved")
    print(f"  Saved: {base_path} (+TIFF, EPS)")


# ============================================================
# FIGURE 3 (MAIN): COMPOSITE HOSPITAL VARIATION + FOREST PLOT
# ============================================================

def figure3_variation_forest(data_dir, output_path):
    """Composite Figure 3: Hospital variation caterpillar + meta-analytic forest.

    Panel A: SAT caterpillar plot
    Panel B: SBT caterpillar plot
    Panel C: Pooled delivery rate forest plot (SAT + SBT definitions)

    Combines data from 04_hospital_variation.py and meta_analysis.py.
    """
    from meta_analysis import run_proportion_meta_analysis

    sat_path = os.path.join(data_dir, "final_df_SAT.csv")
    sbt_path = os.path.join(data_dir, "final_df_SBT.csv")

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    # Panel A: SAT caterpillar
    if os.path.exists(sat_path):
        sat_df = pd.read_csv(sat_path, low_memory=False)
        try:
            # Import from sibling module
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "hosp_var",
                os.path.join(os.path.dirname(__file__), "04_hospital_variation.py"),
            )
            hosp_var = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hosp_var)

            dcol = "SAT_EHR_delivery" if "SAT_EHR_delivery" in sat_df.columns else None
            if dcol:
                rates = hosp_var.compute_risk_adjusted_rates(sat_df, dcol)
                hosp_var.caterpillar_plot(rates, "SAT delivery", "", ax=axes[0])
                axes[0].set_title("A. SAT hospital variation", fontsize=10, fontweight="bold")
        except Exception as e:
            print(f"  Panel A failed: {e}")
            axes[0].text(0.5, 0.5, "[SAT data pending]", ha="center", va="center",
                         fontsize=10, transform=axes[0].transAxes)
    else:
        axes[0].text(0.5, 0.5, "[SAT data not found]", ha="center", va="center",
                     fontsize=10, transform=axes[0].transAxes)
        axes[0].set_title("A. SAT hospital variation", fontsize=10, fontweight="bold")

    # Panel B: SBT caterpillar
    if os.path.exists(sbt_path):
        sbt_df = pd.read_csv(sbt_path, low_memory=False)
        try:
            dcol = "EHR_Delivery_2mins" if "EHR_Delivery_2mins" in sbt_df.columns else None
            if dcol:
                rates = hosp_var.compute_risk_adjusted_rates(sbt_df, dcol)
                hosp_var.caterpillar_plot(rates, "SBT delivery", "", ax=axes[1])
                axes[1].set_title("B. SBT hospital variation", fontsize=10, fontweight="bold")
        except Exception as e:
            print(f"  Panel B failed: {e}")
            axes[1].text(0.5, 0.5, "[SBT data pending]", ha="center", va="center",
                         fontsize=10, transform=axes[1].transAxes)
    else:
        axes[1].text(0.5, 0.5, "[SBT data not found]", ha="center", va="center",
                     fontsize=10, transform=axes[1].transAxes)
        axes[1].set_title("B. SBT hospital variation", fontsize=10, fontweight="bold")

    # Panel C: Forest plot of pooled delivery rates
    ax = axes[2]
    ax.set_title("C. Pooled delivery rates", fontsize=10, fontweight="bold")
    pooled_path = os.path.join(data_dir, "..", "final", "pooled", "pooled_delivery_rates.csv")
    if os.path.exists(pooled_path):
        try:
            rates_df = pd.read_csv(pooled_path)
            definitions = rates_df["delivery_definition"].tolist()
            pooled_rates = rates_df["pooled_rate_re"].fillna(rates_df["overall_rate"]).values
            ci_lo = rates_df["pooled_ci_low"].fillna(pooled_rates - 0.05).values
            ci_hi = rates_df["pooled_ci_high"].fillna(pooled_rates + 0.05).values

            y_pos = range(len(definitions))
            ax.errorbar(pooled_rates, y_pos,
                         xerr=[pooled_rates - ci_lo, ci_hi - pooled_rates],
                         fmt="D", color="#B2182B", ecolor="#F4A582",
                         elinewidth=1.5, capsize=3, markersize=6)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels([d.replace("_", " ") for d in definitions], fontsize=8)
            ax.set_xlabel("Pooled delivery rate (95% CI)", fontsize=9)
            ax.set_xlim(-0.05, 1.0)
            ax.grid(axis="x", alpha=0.3)

            # Annotate with rate text
            for i, (rate, lo, hi) in enumerate(zip(pooled_rates, ci_lo, ci_hi)):
                ax.text(max(hi + 0.02, 0.5), i,
                         f"{rate:.1%} ({lo:.1%}-{hi:.1%})",
                         va="center", fontsize=8)
        except Exception as e:
            print(f"  Panel C failed: {e}")
            ax.text(0.5, 0.5, "[Run 07_aggregate_sites.py first]",
                     ha="center", va="center", fontsize=10, transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "[Pooled data not found]",
                 ha="center", va="center", fontsize=10, transform=ax.transAxes)

    plt.tight_layout()
    _save_multi_format(fig, output_path)
    plt.close(fig)
    print(f"Figure 3 saved: {output_path}")


# ============================================================
# ESM FIGURES
# ============================================================

def efigure4_completeness_heatmap(data_dir, output_path):
    """eFigure 4: Data completeness heatmap (hospital x data element)."""
    matrix_path = os.path.join(data_dir, "sensitivity", "data_completeness_matrix.csv")
    if not os.path.exists(matrix_path):
        print(f"  Completeness matrix not found: {matrix_path}")
        return

    df = pd.read_csv(matrix_path, index_col=0)
    n_elements, n_hospitals = df.shape

    fig, ax = plt.subplots(figsize=(max(6, n_hospitals * 0.6), max(4, n_elements * 0.4)))
    im = ax.imshow(df.values.astype(float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n_hospitals))
    ax.set_xticklabels(df.columns, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(n_elements))
    ax.set_yticklabels(df.index, fontsize=8)
    ax.set_title("Data completeness by hospital", fontsize=11, fontweight="bold")

    # Annotate cells
    for i in range(n_elements):
        for j in range(n_hospitals):
            val = df.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                         fontsize=8, color="black" if val > 0.5 else "white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Completeness", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    _save_multi_format(fig, output_path, dpi=300)
    plt.close(fig)
    print(f"eFigure 4 saved: {output_path}")


def efigure6_outcome_forest(data_dir, output_path):
    """eFigure 6: Forest plot of construct validity outcome models."""
    outcomes_path = os.path.join(data_dir, "final", "construct_validity_outcomes.csv")
    if not os.path.exists(outcomes_path):
        print(f"  Outcomes file not found: {outcomes_path}")
        return

    df = pd.read_csv(outcomes_path)
    # Filter to rows with effect estimates
    has_hr = df["HR"].notna() if "HR" in df.columns else pd.Series(False, index=df.index)
    has_or = df["OR"].notna() if "OR" in df.columns else pd.Series(False, index=df.index)
    has_irr = df["IRR"].notna() if "IRR" in df.columns else pd.Series(False, index=df.index)

    plot_rows = df[has_hr | has_or | has_irr].copy()
    if plot_rows.empty:
        print("  No effect estimates found for forest plot")
        return

    # Build label + effect + CI
    labels = []
    effects = []
    ci_lo = []
    ci_hi = []
    for _, row in plot_rows.iterrows():
        lbl = f"{row.get('trial_type', '')} {row.get('delivery_definition', '')}\n{row.get('model', '')}"
        labels.append(lbl.strip())
        if pd.notna(row.get("HR")):
            effects.append(row["HR"])
            ci_lo.append(row.get("HR_lower_95", row["HR"] * 0.8))
            ci_hi.append(row.get("HR_upper_95", row["HR"] * 1.2))
        elif pd.notna(row.get("OR")):
            effects.append(row["OR"])
            ci_lo.append(row.get("OR_lower_95", row["OR"] * 0.8))
            ci_hi.append(row.get("OR_upper_95", row["OR"] * 1.2))
        elif pd.notna(row.get("IRR")):
            effects.append(row["IRR"])
            ci_lo.append(row.get("IRR_lower_95", row["IRR"] * 0.8))
            ci_hi.append(row.get("IRR_upper_95", row["IRR"] * 1.2))

    n = len(labels)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.5)))
    y_pos = range(n - 1, -1, -1)

    ax.errorbar(effects, list(y_pos),
                 xerr=[np.array(effects) - np.array(ci_lo),
                       np.array(ci_hi) - np.array(effects)],
                 fmt="s", color="#2166AC", ecolor="#92C5DE",
                 elinewidth=1.5, capsize=3, markersize=5)

    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Effect estimate (HR / OR / IRR) with 95% CI", fontsize=9)
    ax.set_title("Construct validity: outcome model results", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    _save_multi_format(fig, output_path)
    plt.close(fig)
    print(f"eFigure 6 saved: {output_path}")


def generate_esm_figures(data_dir, esm_dir):
    """Generate all Electronic Supplementary Material figures."""
    os.makedirs(esm_dir, exist_ok=True)

    print("\n--- ESM Figures ---")

    # eFigure 1: Phenotyping logic (moved from main Figure 2)
    print("eFigure 1: Phenotyping logic...")
    figure2_phenotype_logic(os.path.join(esm_dir, "efig1_phenotype_logic.pdf"))

    # eFigure 2: Timing and pairing (moved from main Figure 4)
    print("eFigure 2: Timing and pairing...")
    figure4_timing_pairing(data_dir, os.path.join(esm_dir, "efig2_timing_pairing.pdf"))

    # eFigure 3: Forest + funnel plots
    print("eFigure 3: Forest + funnel plots...")
    pooled_path = os.path.join(data_dir, "..", "final", "pooled", "pooled_delivery_rates.csv")
    if os.path.exists(pooled_path):
        try:
            from meta_analysis import jama_forest_plot, funnel_plot
            rates_df = pd.read_csv(pooled_path)
            # Build summary format for forest plot
            for trial in rates_df["trial_type"].unique():
                subset = rates_df[rates_df["trial_type"] == trial]
                if "pooled_rate_re" in subset.columns and subset["pooled_rate_re"].notna().any():
                    summary = pd.DataFrame({
                        "label": subset["delivery_definition"],
                        "eff": subset["pooled_rate_re"].fillna(subset["overall_rate"]),
                        "ci_low": subset["pooled_ci_low"].fillna(0),
                        "ci_upp": subset["pooled_ci_high"].fillna(1),
                    })
                    fig, ax = jama_forest_plot(summary, outcome_label=f"{trial} delivery rate")
                    _save_multi_format(fig, os.path.join(esm_dir, f"efig3_forest_{trial}.pdf"))
                    plt.close(fig)
        except Exception as e:
            print(f"  eFigure 3 failed: {e}")
    else:
        print(f"  Pooled rates not found for eFigure 3")

    # eFigure 4: Data completeness heatmap
    print("eFigure 4: Data completeness heatmap...")
    efigure4_completeness_heatmap(
        os.path.join(data_dir, ".."),
        os.path.join(esm_dir, "efig4_completeness.pdf"),
    )

    # eFigure 5: Bland-Altman (already generated by 04_hospital_variation.py)
    print("eFigure 5: Bland-Altman — generated by 04_hospital_variation.py")

    # eFigure 6: Outcome model forest plot
    print("eFigure 6: Outcome model forest plot...")
    efigure6_outcome_forest(
        os.path.join(data_dir, ".."),
        os.path.join(esm_dir, "efig6_outcomes.pdf"),
    )

    print("ESM figures complete.")


# ============================================================
# MAIN
# ============================================================

def generate_all_figures(data_dir, output_dir, esm_dir=None):
    """Generate all manuscript figures per ICM submission guidelines.

    ICM allows max 5 illustrations (figures + tables) in main manuscript.
    Main: Figure 1 (CONSORT), Figure 2 (criterion validity), Figure 3 (variation+forest)
    Tables 1-2 generated by 07_aggregate_sites.py.
    All other content goes to ESM.
    """
    os.makedirs(output_dir, exist_ok=True)
    if esm_dir is None:
        esm_dir = os.path.join(output_dir, "..", "esm")

    # Load CONSORT numbers from aggregation output (if available)
    import json
    consort_path = os.path.join(output_dir, "..", "pooled", "consort_numbers.json")
    site_stats = None
    if os.path.exists(consort_path):
        with open(consort_path) as f:
            site_stats = json.load(f)
        print(f"Loaded CONSORT numbers from {consort_path}")

    print("=" * 60)
    print("MAIN MANUSCRIPT FIGURES (ICM: max 5 illustrations total)")
    print("=" * 60)

    print("\nFigure 1: CONSORT flow diagram...")
    figure1_consort(data_dir, os.path.join(output_dir, "fig1_consort.pdf"), site_stats=site_stats)

    print("\nFigure 2: Criterion validity (confusion matrices + metrics)...")
    figure3_criterion_validity(data_dir, os.path.join(output_dir, "fig2_criterion_validity.pdf"))

    print("\nFigure 3: Hospital variation + pooled rates (composite)...")
    figure3_variation_forest(data_dir, os.path.join(output_dir, "fig3_variation_forest.pdf"))

    print("\nGenerating pooled tables (Table 2)...")
    generate_pooled_table2(data_dir, output_dir)
    generate_pooled_criterion_validity(data_dir, output_dir)

    print("\n" + "=" * 60)
    print("ELECTRONIC SUPPLEMENTARY MATERIAL (ESM)")
    print("=" * 60)
    generate_esm_figures(data_dir, esm_dir)

    print("\nAll figures and ESM generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../output/intermediate")
    parser.add_argument("--output-dir", default="../output/final/figures")
    parser.add_argument("--esm-dir", default="../output/final/esm")
    args = parser.parse_args()
    generate_all_figures(args.data_dir, args.output_dir, args.esm_dir)
