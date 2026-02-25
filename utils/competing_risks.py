"""Competing-risk utilities for SAP-aligned VFD analyses.

This module provides a Python-native, method-equivalent implementation of
Fine-Gray style subdistribution modeling via discrete-time cloglog regression,
plus Aalen-Johansen cumulative incidence summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CompetingRiskSummary:
    model: str
    estimator: str
    exposure_col: str
    shr: float | None
    shr_lower_95: float | None
    shr_upper_95: float | None
    p_value: float | None
    n_hospitalizations: int
    note: str | None = None


def _build_hosp_dataset(day_level_df: pd.DataFrame, exposure_col: str) -> pd.DataFrame:
    """Build hospitalization-level competing-risk dataset from ventilator-day data."""
    if "vent_day_index" not in day_level_df.columns:
        day_level_df = day_level_df.copy()
        day_level_df["vent_day_index"] = (
            day_level_df.groupby("hospitalization_id").cumcount() + 1
        )

    baseline = (
        day_level_df.sort_values(["hospitalization_id", "vent_day_index", "hosp_id_day_key"])
        .groupby("hospitalization_id", as_index=False)
        .first()
    )
    ever = (
        day_level_df.groupby("hospitalization_id")["delivery"]
        .max()
        .rename("ever_delivered")
        .reset_index()
    )
    hosp = baseline.merge(ever, on="hospitalization_id", how="left")
    hosp["ever_delivered"] = hosp["ever_delivered"].fillna(0).astype(int)

    if exposure_col not in hosp.columns:
        if exposure_col == "ever_delivered":
            hosp[exposure_col] = hosp["ever_delivered"]
        elif exposure_col == "landmark_delivered":
            hosp[exposure_col] = hosp["delivery"].fillna(0).astype(int)
        else:
            raise KeyError(f"Exposure column '{exposure_col}' missing")

    if "total_vent_days" not in hosp.columns:
        hosp["total_vent_days"] = (
            day_level_df.groupby("hospitalization_id")["hosp_id_day_key"].nunique().reindex(hosp["hospitalization_id"]).values
        )

    if "died" not in hosp.columns:
        if "discharge_category" in hosp.columns:
            hosp["died"] = hosp["discharge_category"].astype(str).str.lower().str.contains(
                "expired|dead|death|died", na=False
            ).astype(int)
        else:
            hosp["died"] = 0

    hosp["event_type"] = np.where(hosp["died"] == 1, 2, 1)
    hosp["time_to_event"] = hosp["total_vent_days"].fillna(1).clip(lower=1).astype(int)
    hosp["sex_male"] = (hosp.get("sex_category", "").astype(str).str.lower() == "male").astype(int)
    hosp["race_white"] = hosp.get("race_category", "").astype(str).str.lower().str.contains("white", na=False).astype(int)
    if "hospital_id" not in hosp.columns:
        hosp["hospital_id"] = "UNKNOWN"
    return hosp


def _build_subdistribution_person_period(
    hosp: pd.DataFrame,
    exposure_col: str,
    covariates: list[str],
    horizon: int,
) -> pd.DataFrame:
    """Create Fine-Gray-equivalent person-period data.

    For competing events (death), subjects remain in the subdistribution risk set
    through horizon with zero event indicator rows.
    """
    rows: list[dict[str, Any]] = []
    covariates = [c for c in covariates if c in hosp.columns]

    for row in hosp.itertuples(index=False):
        t_event = int(min(max(getattr(row, "time_to_event"), 1), horizon))
        event_type = int(getattr(row, "event_type"))
        base_payload = {
            "hospitalization_id": getattr(row, "hospitalization_id"),
            "hospital_id": getattr(row, "hospital_id"),
            exposure_col: int(getattr(row, exposure_col)),
        }
        for cov in covariates:
            base_payload[cov] = getattr(row, cov)

        for t in range(1, horizon + 1):
            in_risk = False
            if t <= t_event:
                in_risk = True
            elif event_type == 2 and t > t_event:
                # Fine-Gray subdistribution keeps competing events in risk set.
                in_risk = True
            if not in_risk:
                break

            y = int(event_type == 1 and t == t_event)
            rec = dict(base_payload)
            rec["time"] = t
            rec["log_time"] = float(np.log(t))
            rec["event_interest"] = y
            rows.append(rec)

    return pd.DataFrame(rows)


def fit_fine_gray_equivalent(
    day_level_df: pd.DataFrame,
    exposure_col: str = "landmark_delivered",
    covariates: list[str] | None = None,
    horizon: int = 28,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Fit a Fine-Gray-equivalent subdistribution model.

    Returns
    -------
    result_dict, cif_df, vfd_components_df
    """
    covariates = covariates or ["age_at_admission", "sex_male", "race_white"]
    hosp = _build_hosp_dataset(day_level_df, exposure_col=exposure_col)

    if hosp[exposure_col].nunique() < 2:
        result = CompetingRiskSummary(
            model="Fine-Gray equivalent - extubation alive by day 28",
            estimator="discrete_time_subdistribution_cloglog",
            exposure_col=exposure_col,
            shr=None,
            shr_lower_95=None,
            shr_upper_95=None,
            p_value=None,
            n_hospitalizations=int(len(hosp)),
            note="Exposure has <2 levels; model not estimable",
        )
        cif_df = aalen_johansen_cif(hosp, exposure_col=exposure_col, horizon=horizon)
        comp_df = summarize_vfd_components(hosp, exposure_col=exposure_col, horizon=horizon)
        return result.__dict__, cif_df, comp_df

    period_df = _build_subdistribution_person_period(
        hosp,
        exposure_col=exposure_col,
        covariates=covariates,
        horizon=horizon,
    )

    if period_df.empty or period_df["event_interest"].sum() == 0:
        result = CompetingRiskSummary(
            model="Fine-Gray equivalent - extubation alive by day 28",
            estimator="discrete_time_subdistribution_cloglog",
            exposure_col=exposure_col,
            shr=None,
            shr_lower_95=None,
            shr_upper_95=None,
            p_value=None,
            n_hospitalizations=int(len(hosp)),
            note="No events of interest in analysis horizon",
        )
        cif_df = aalen_johansen_cif(hosp, exposure_col=exposure_col, horizon=horizon)
        comp_df = summarize_vfd_components(hosp, exposure_col=exposure_col, horizon=horizon)
        return result.__dict__, cif_df, comp_df

    try:
        import statsmodels.api as sm

        model_covs = [exposure_col] + [c for c in covariates if c in period_df.columns] + ["log_time"]
        X = sm.add_constant(period_df[model_covs], has_constant="add")
        y = period_df["event_interest"].astype(int)

        glm = sm.GLM(
            y,
            X,
            family=sm.families.Binomial(link=sm.families.links.cloglog()),
        )
        fit = glm.fit(
            cov_type="cluster",
            cov_kwds={"groups": period_df["hospitalization_id"]},
        )

        if exposure_col in fit.params.index:
            coef = float(fit.params[exposure_col])
            se = float(fit.bse[exposure_col])
            pval = float(fit.pvalues[exposure_col])
            shr = float(np.exp(coef))
            lo = float(np.exp(coef - 1.96 * se))
            hi = float(np.exp(coef + 1.96 * se))
        else:
            shr, lo, hi, pval = (None, None, None, None)

        result = CompetingRiskSummary(
            model="Fine-Gray equivalent - extubation alive by day 28",
            estimator="discrete_time_subdistribution_cloglog",
            exposure_col=exposure_col,
            shr=round(shr, 4) if shr is not None else None,
            shr_lower_95=round(lo, 4) if lo is not None else None,
            shr_upper_95=round(hi, 4) if hi is not None else None,
            p_value=round(pval, 4) if pval is not None else None,
            n_hospitalizations=int(len(hosp)),
            note=None,
        )
    except Exception as exc:  # pragma: no cover
        result = CompetingRiskSummary(
            model="Fine-Gray equivalent - extubation alive by day 28",
            estimator="discrete_time_subdistribution_cloglog",
            exposure_col=exposure_col,
            shr=None,
            shr_lower_95=None,
            shr_upper_95=None,
            p_value=None,
            n_hospitalizations=int(len(hosp)),
            note=f"Model failed: {exc}",
        )

    cif_df = aalen_johansen_cif(hosp, exposure_col=exposure_col, horizon=horizon)
    comp_df = summarize_vfd_components(hosp, exposure_col=exposure_col, horizon=horizon)
    return result.__dict__, cif_df, comp_df


def fit_fine_gray_rpy2(
    df: pd.DataFrame,
    exposure_col: str = "landmark_delivered",
    covariates: list[str] | None = None,
    horizon: int = 28,
) -> dict[str, Any] | None:
    """Fit a Fine-Gray subdistribution hazard model via rpy2 and cmprsk::crr().

    This is the primary Fine-Gray estimator per SAP 2.3.  It bridges to R's
    ``cmprsk`` package, which implements the original Fine & Gray (1999) weighted
    partial-likelihood estimator.  If rpy2, R, or cmprsk are unavailable the
    function returns ``None`` so callers can fall back to
    ``fit_fine_gray_equivalent`` (discrete-time cloglog).

    Parameters
    ----------
    df:
        Either a hospitalization-level DataFrame (must contain
        ``time_to_event`` and ``event_type`` columns) or a ventilator-day
        DataFrame.  When ``time_to_event`` is absent the function calls
        ``_build_hosp_dataset`` internally.
    exposure_col:
        Binary (0/1) treatment/exposure column name.
    covariates:
        Additional adjustment covariates.  Defaults to
        ``["age_at_admission", "sex_male", "race_white"]``.
    horizon:
        Analysis horizon in days (used only when ``_build_hosp_dataset`` is
        called; does not truncate the crr fit itself).

    Returns
    -------
    dict with keys matching ``CompetingRiskSummary`` field names, or ``None``
    when rpy2/R/cmprsk is unavailable.

    Notes
    -----
    * ``event_type`` coding: 1 = extubation alive (event of interest),
      2 = death (competing event), 0 = censored — matches cmprsk convention.
    * Cluster-robust SEs are not available via ``crr``; Huber-White SEs from
      the crr object's ``var`` matrix are used (square-root of diagonal).
    * The returned dict is structurally identical to the one returned by
      ``fit_fine_gray_equivalent`` so callers can treat both uniformly.
    """
    covariates = covariates or ["age_at_admission", "sex_male", "race_white"]

    # ------------------------------------------------------------------
    # 1. Prepare hospitalization-level dataset
    # ------------------------------------------------------------------
    if "time_to_event" not in df.columns or "event_type" not in df.columns:
        hosp = _build_hosp_dataset(df, exposure_col=exposure_col)
    else:
        hosp = df.copy()

    # Ensure required binary exposure column is present.
    if exposure_col not in hosp.columns:
        raise KeyError(f"Exposure column '{exposure_col}' not found in DataFrame.")

    # Filter to only covariates that are actually present.
    available_covs = [c for c in covariates if c in hosp.columns]

    # ------------------------------------------------------------------
    # 2. Attempt rpy2 import — return None on any ImportError
    # ------------------------------------------------------------------
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri, pandas2ri
        from rpy2.robjects.packages import importr, PackageNotInstalledError
    except ImportError:
        return None  # rpy2 not installed; caller uses cloglog fallback

    # ------------------------------------------------------------------
    # 3. Activate automatic conversion and import cmprsk
    # ------------------------------------------------------------------
    try:
        pandas2ri.activate()
        numpy2ri.activate()
        cmprsk = importr("cmprsk")
    except Exception:
        # R not available or cmprsk not installed.
        try:
            pandas2ri.deactivate()
            numpy2ri.deactivate()
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # 4. Build covariate matrix and convert to R objects
    # ------------------------------------------------------------------
    try:
        cov_cols = [exposure_col] + available_covs

        # Drop rows with any NA in analysis columns.
        analysis_cols = ["time_to_event", "event_type"] + cov_cols
        clean = hosp[analysis_cols].dropna()

        if clean.empty or clean[exposure_col].nunique() < 2:
            return None

        ftime = ro.FloatVector(clean["time_to_event"].astype(float).tolist())
        fstatus = ro.IntVector(clean["event_type"].astype(int).tolist())

        # cmprsk::crr expects a plain numeric matrix for covariates.
        cov_array = clean[cov_cols].astype(float).values  # shape (n, p)
        r_cov = ro.r.matrix(
            ro.FloatVector(cov_array.flatten(order="C").tolist()),
            nrow=int(cov_array.shape[0]),
            ncol=int(cov_array.shape[1]),
            byrow=True,
        )
        # Assign column names so crr output is labelled.
        r_cov.colnames = ro.StrVector(cov_cols)

        # ------------------------------------------------------------------
        # 5. Call cmprsk::crr — failcode=1 (extubation alive is event of interest)
        # ------------------------------------------------------------------
        crr_fit = cmprsk.crr(
            ftime=ftime,
            fstatus=fstatus,
            cov1=r_cov,
            failcode=ro.IntVector([1]),
            cencode=ro.IntVector([0]),
        )

        # ------------------------------------------------------------------
        # 6. Extract coefficients, variance, and p-values from the fit object
        # ------------------------------------------------------------------
        coef_r = dict(zip(list(ro.r["names"](crr_fit.rx2("coef"))),
                          list(crr_fit.rx2("coef"))))
        var_r = np.array(crr_fit.rx2("var"))  # p x p variance matrix

        coef_names = list(ro.r["names"](crr_fit.rx2("coef")))
        if exposure_col not in coef_names:
            return None

        exp_idx = coef_names.index(exposure_col)
        coef_val = float(coef_r[exposure_col])
        se_val = float(np.sqrt(var_r[exp_idx, exp_idx]))

        # Two-sided Wald p-value.
        from scipy import stats as scipy_stats
        z = coef_val / se_val if se_val > 0 else float("nan")
        pval = float(2.0 * scipy_stats.norm.sf(abs(z)))

        shr = float(np.exp(coef_val))
        lo = float(np.exp(coef_val - 1.96 * se_val))
        hi = float(np.exp(coef_val + 1.96 * se_val))

        result: dict[str, Any] = {
            "model": "Fine-Gray - extubation alive by day 28",
            "estimator": "cmprsk_crr_rpy2",
            "exposure_col": exposure_col,
            "shr": round(shr, 4),
            "shr_lower_95": round(lo, 4),
            "shr_upper_95": round(hi, 4),
            "p_value": round(pval, 4),
            "n_hospitalizations": int(len(clean)),
            "note": None,
        }
        return result

    except Exception as exc:
        # crr call or extraction failed — return None so caller falls back.
        return {
            "model": "Fine-Gray - extubation alive by day 28",
            "estimator": "cmprsk_crr_rpy2",
            "exposure_col": exposure_col,
            "shr": None,
            "shr_lower_95": None,
            "shr_upper_95": None,
            "p_value": None,
            "n_hospitalizations": int(len(hosp)),
            "note": f"cmprsk::crr failed: {exc}",
        }
    finally:
        try:
            pandas2ri.deactivate()
            numpy2ri.deactivate()
        except Exception:
            pass


def aalen_johansen_cif(
    hosp_df: pd.DataFrame,
    exposure_col: str = "exposure",
    horizon: int = 28,
) -> pd.DataFrame:
    """Compute discrete-time Aalen-Johansen CIF curves for extubation and death."""
    records: list[dict[str, Any]] = []

    for exposure_val, grp in hosp_df.groupby(exposure_col):
        times = grp["time_to_event"].astype(int).clip(lower=1, upper=horizon)
        events = grp["event_type"].astype(int)
        n_total = len(grp)
        s_prev = 1.0
        cif_ext_prev = 0.0
        cif_death_prev = 0.0

        for t in range(1, horizon + 1):
            at_risk = ((times >= t)).sum()
            d1 = ((times == t) & (events == 1)).sum()
            d2 = ((times == t) & (events == 2)).sum()

            if at_risk > 0:
                h1 = d1 / at_risk
                h2 = d2 / at_risk
            else:
                h1 = 0.0
                h2 = 0.0

            cif_ext = cif_ext_prev + s_prev * h1
            cif_death = cif_death_prev + s_prev * h2
            s_new = s_prev * (1.0 - h1 - h2)

            records.append(
                {
                    "exposure_group": int(exposure_val),
                    "time_day": t,
                    "n_total": int(n_total),
                    "n_at_risk": int(at_risk),
                    "cif_extubation_alive": round(float(cif_ext), 6),
                    "cif_death_before_extubation": round(float(cif_death), 6),
                    "survival_no_event": round(float(s_new), 6),
                    "method": "aalen_johansen_discrete",
                }
            )

            s_prev = s_new
            cif_ext_prev = cif_ext
            cif_death_prev = cif_death

    return pd.DataFrame(records)


def summarize_vfd_components(
    hosp_df: pd.DataFrame,
    exposure_col: str = "exposure",
    horizon: int = 28,
) -> pd.DataFrame:
    """Summarize SAP-mandated VFD component rates by exposure group."""
    records: list[dict[str, Any]] = []
    for exposure_val, grp in hosp_df.groupby(exposure_col):
        n = len(grp)
        if n == 0:
            continue
        time = grp["time_to_event"].astype(int)
        event = grp["event_type"].astype(int)

        ext_alive = ((event == 1) & (time <= horizon)).sum()
        death_before_ext = ((event == 2) & (time <= horizon)).sum()

        if "imv_episode_id" in grp.columns:
            reintubated = grp["imv_episode_id"].astype(str).str.contains("_ep_2|_ep_3|_ep_4", na=False).sum()
        else:
            reintubated = 0

        records.append(
            {
                "exposure_group": int(exposure_val),
                "n_hospitalizations": int(n),
                "extubated_alive_day28_n": int(ext_alive),
                "extubated_alive_day28_prop": round(float(ext_alive / n), 6),
                "death_before_extubation_n": int(death_before_ext),
                "death_before_extubation_prop": round(float(death_before_ext / n), 6),
                "reintubated_n": int(reintubated),
                "reintubated_prop": round(float(reintubated / n), 6),
                "analysis_horizon_days": int(horizon),
            }
        )

    return pd.DataFrame(records)
