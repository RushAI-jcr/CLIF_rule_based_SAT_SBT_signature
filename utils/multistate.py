"""Multistate utilities for SAP-aligned VFD secondary analyses.

Python-native, method-equivalent transition modeling for:
    MV -> Extubated
    MV -> Dead
    Extubated -> Reintubated
    Extubated -> Dead
    Reintubated -> Dead
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TransitionModelResult:
    transition: str
    model: str
    model_estimator: str
    exposure_col: str
    hr: float | None
    hr_lower_95: float | None
    hr_upper_95: float | None
    p_value: float | None
    n_subjects: int
    n_events: int
    note: str | None = None


def _build_hosp_dataset(day_level_df: pd.DataFrame, exposure_col: str) -> pd.DataFrame:
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
            raise KeyError(f"Exposure column '{exposure_col}' missing from day-level data")

    if "total_vent_days" not in hosp.columns:
        total_days = (
            day_level_df.groupby("hospitalization_id")["hosp_id_day_key"]
            .nunique()
            .rename("total_vent_days")
            .reset_index()
        )
        hosp = hosp.merge(total_days, on="hospitalization_id", how="left")

    if "died" not in hosp.columns:
        if "discharge_category" in hosp.columns:
            # mCIDE discharge_category for death is "Expired"; match common variants
            hosp["died"] = (
                hosp["discharge_category"].astype(str).str.strip().str.lower()
                .isin(["expired", "dead", "death", "died", "deceased"])
            ).astype(int)
        else:
            hosp["died"] = 0

    if "hospital_id" not in hosp.columns:
        hosp["hospital_id"] = "UNKNOWN"

    if "imv_episode_id" in day_level_df.columns:
        ep_counts = (
            day_level_df.groupby("hospitalization_id")["imv_episode_id"]
            .nunique()
            .rename("n_imv_episodes")
            .reset_index()
        )
        hosp = hosp.merge(ep_counts, on="hospitalization_id", how="left")
    else:
        hosp["n_imv_episodes"] = 1

    hosp["reintubated"] = hosp["n_imv_episodes"].fillna(1).astype(int).gt(1).astype(int)
    hosp["sex_male"] = (hosp.get("sex_category", "").astype(str).str.lower() == "male").astype(int)
    hosp["race_white"] = (
        hosp.get("race_category", "").astype(str).str.lower().str.contains("white", na=False)
    ).astype(int)
    hosp["time_to_mv_exit"] = hosp["total_vent_days"].fillna(1).clip(lower=1).astype(int)
    return hosp


def _build_person_period_mv(
    hosp_df: pd.DataFrame,
    exposure_col: str,
    event_transition: str,
    horizon: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in hosp_df.itertuples(index=False):
        t_exit = int(min(max(getattr(row, "time_to_mv_exit"), 1), horizon))
        died = int(getattr(row, "died"))
        is_event = int(
            (event_transition == "MV_to_Dead" and died == 1)
            or (event_transition == "MV_to_Extubated" and died == 0)
        )
        for t in range(1, t_exit + 1):
            rows.append(
                {
                    "hospitalization_id": getattr(row, "hospitalization_id"),
                    "hospital_id": getattr(row, "hospital_id"),
                    exposure_col: int(getattr(row, exposure_col)),
                    "age_at_admission": float(getattr(row, "age_at_admission", np.nan)),
                    "sex_male": int(getattr(row, "sex_male")),
                    "race_white": int(getattr(row, "race_white")),
                    "time_day": t,
                    "log_time": float(np.log(t)),
                    "event": int(is_event == 1 and t == t_exit),
                }
            )
    return pd.DataFrame(rows)


def _build_person_period_extubated_to_reintubated(
    hosp_df: pd.DataFrame,
    exposure_col: str,
    horizon: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    extubated = hosp_df[hosp_df["died"] == 0].copy()
    for row in extubated.itertuples(index=False):
        t_exit = int(min(max(getattr(row, "time_to_mv_exit"), 1), horizon))
        start_t = min(t_exit + 1, horizon)
        event_time = min(start_t + 1, horizon)
        is_event = int(getattr(row, "reintubated") == 1)
        for t in range(start_t, horizon + 1):
            rows.append(
                {
                    "hospitalization_id": getattr(row, "hospitalization_id"),
                    "hospital_id": getattr(row, "hospital_id"),
                    exposure_col: int(getattr(row, exposure_col)),
                    "age_at_admission": float(getattr(row, "age_at_admission", np.nan)),
                    "sex_male": int(getattr(row, "sex_male")),
                    "race_white": int(getattr(row, "race_white")),
                    "time_day": t,
                    "log_time": float(np.log(max(t, 1))),
                    "event": int(is_event == 1 and t == event_time),
                }
            )
    return pd.DataFrame(rows)


def _build_person_period_extubated_to_dead(
    hosp_df: pd.DataFrame,
    exposure_col: str,
    horizon: int,
) -> pd.DataFrame:
    """Build person-period records for Extubated -> Dead transition.

    At-risk population: ALL patients who were extubated alive (exited MV alive),
    not just those who subsequently died. This includes patients who survived to
    discharge (censored) and those who died after extubation (events).

    Event: death after extubation (died==1 AND was extubated).
    Censored: extubated and survived to discharge (died==0).
    """
    rows: list[dict[str, Any]] = []
    # At-risk: all patients who exited MV alive (extubated).
    # Approximation: patients with time_to_mv_exit < horizon who were not reintubated,
    # OR all patients who survived MV (regardless of reintubation status).
    # Use: died==0 (survived = definitely extubated) + died==1 & reintubated==0
    # (died after extubation without reintubation).
    extubated_pool = hosp_df[
        (hosp_df["died"] == 0) |
        ((hosp_df["died"] == 1) & (hosp_df["reintubated"] == 0))
    ].copy()
    for row in extubated_pool.itertuples(index=False):
        t_exit = int(min(max(getattr(row, "time_to_mv_exit"), 1), horizon))
        start_t = min(t_exit + 1, horizon)
        is_event = int(getattr(row, "died", 0))
        event_time = min(start_t + 1, horizon) if is_event else horizon + 1
        for t in range(start_t, horizon + 1):
            rows.append(
                {
                    "hospitalization_id": getattr(row, "hospitalization_id"),
                    "hospital_id": getattr(row, "hospital_id"),
                    exposure_col: int(getattr(row, exposure_col)),
                    "age_at_admission": float(getattr(row, "age_at_admission", np.nan)),
                    "sex_male": int(getattr(row, "sex_male")),
                    "race_white": int(getattr(row, "race_white")),
                    "time_day": t,
                    "log_time": float(np.log(max(t, 1))),
                    # Event fires at event_time for deaths; censored patients never fire
                    "event": int(is_event == 1 and t == event_time),
                }
            )
    return pd.DataFrame(rows)


def _build_person_period_reintubated_to_dead(
    hosp_df: pd.DataFrame,
    exposure_col: str,
    horizon: int,
) -> pd.DataFrame:
    """Build person-period records for Reintubated -> Dead transition.

    At-risk population: patients who were reintubated (n_imv_episodes > 1).
    Event: death (died == 1). The event time is approximated as the first period
    beyond the MV exit window, consistent with other post-MV transition builders.
    """
    rows: list[dict[str, Any]] = []
    reintubated = hosp_df[hosp_df["reintubated"] == 1].copy()
    for row in reintubated.itertuples(index=False):
        t_exit = int(min(max(getattr(row, "time_to_mv_exit"), 1), horizon))
        start_t = min(t_exit + 1, horizon)
        event_time = min(start_t + 1, horizon)
        is_event = int(getattr(row, "died") == 1)
        for t in range(start_t, horizon + 1):
            rows.append(
                {
                    "hospitalization_id": getattr(row, "hospitalization_id"),
                    "hospital_id": getattr(row, "hospital_id"),
                    exposure_col: int(getattr(row, exposure_col)),
                    "age_at_admission": float(getattr(row, "age_at_admission", np.nan)),
                    "sex_male": int(getattr(row, "sex_male")),
                    "race_white": int(getattr(row, "race_white")),
                    "time_day": t,
                    "log_time": float(np.log(max(t, 1))),
                    "event": int(is_event == 1 and t == event_time),
                }
            )
    return pd.DataFrame(rows)


def _fit_transition_model(
    period_df: pd.DataFrame,
    transition: str,
    exposure_col: str,
) -> dict[str, Any]:
    if period_df.empty:
        return TransitionModelResult(
            transition=transition,
            model=f"Multistate - {transition}",
            model_estimator="discrete_time_cloglog",
            exposure_col=exposure_col,
            hr=None,
            hr_lower_95=None,
            hr_upper_95=None,
            p_value=None,
            n_subjects=0,
            n_events=0,
            note="No at-risk records",
        ).__dict__

    n_subjects = int(period_df["hospitalization_id"].nunique())
    n_events = int(period_df["event"].sum())
    if n_events == 0:
        return TransitionModelResult(
            transition=transition,
            model=f"Multistate - {transition}",
            model_estimator="discrete_time_cloglog",
            exposure_col=exposure_col,
            hr=None,
            hr_lower_95=None,
            hr_upper_95=None,
            p_value=None,
            n_subjects=n_subjects,
            n_events=n_events,
            note="No transition events observed",
        ).__dict__

    try:
        import statsmodels.api as sm

        covariates = [exposure_col, "age_at_admission", "sex_male", "race_white", "log_time"]
        covariates = [c for c in covariates if c in period_df.columns]
        model_df = period_df[covariates + ["event", "hospitalization_id"]].dropna()
        if model_df.empty:
            raise ValueError("No complete cases for transition model")

        X = sm.add_constant(model_df[covariates], has_constant="add")
        y = model_df["event"].astype(int)

        glm = sm.GLM(
            y,
            X,
            family=sm.families.Binomial(link=sm.families.links.cloglog()),
        )
        fit = glm.fit(cov_type="cluster", cov_kwds={"groups": model_df["hospitalization_id"]})

        if exposure_col in fit.params.index:
            coef = float(fit.params[exposure_col])
            se = float(fit.bse[exposure_col])
            pval = float(fit.pvalues[exposure_col])
            hr = float(np.exp(coef))
            lo = float(np.exp(coef - 1.96 * se))
            hi = float(np.exp(coef + 1.96 * se))
        else:
            hr, lo, hi, pval = (None, None, None, None)

        return TransitionModelResult(
            transition=transition,
            model=f"Multistate - {transition}",
            model_estimator="discrete_time_cloglog",
            exposure_col=exposure_col,
            hr=round(hr, 4) if hr is not None else None,
            hr_lower_95=round(lo, 4) if lo is not None else None,
            hr_upper_95=round(hi, 4) if hi is not None else None,
            p_value=round(pval, 4) if pval is not None else None,
            n_subjects=n_subjects,
            n_events=n_events,
            note=None,
        ).__dict__
    except Exception as exc:  # pragma: no cover
        return TransitionModelResult(
            transition=transition,
            model=f"Multistate - {transition}",
            model_estimator="discrete_time_cloglog",
            exposure_col=exposure_col,
            hr=None,
            hr_lower_95=None,
            hr_upper_95=None,
            p_value=None,
            n_subjects=n_subjects,
            n_events=n_events,
            note=f"Model failed: {exc}",
        ).__dict__


def _transition_probabilities(
    hosp_df: pd.DataFrame,
    exposure_col: str,
    horizon: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for exposure_value, grp in hosp_df.groupby(exposure_col):
        times = grp["time_to_mv_exit"].astype(int).clip(lower=1, upper=horizon)
        died = grp["died"].astype(int)
        n_total = len(grp)

        s_prev = 1.0
        cum_ext_prev = 0.0
        cum_dead_prev = 0.0
        for t in range(1, horizon + 1):
            n_at_risk = int((times >= t).sum())
            d_ext = int(((times == t) & (died == 0)).sum())
            d_dead = int(((times == t) & (died == 1)).sum())
            if n_at_risk > 0:
                h_ext = d_ext / n_at_risk
                h_dead = d_dead / n_at_risk
            else:
                h_ext = 0.0
                h_dead = 0.0

            cum_ext = cum_ext_prev + s_prev * h_ext
            cum_dead = cum_dead_prev + s_prev * h_dead
            s_new = s_prev * (1.0 - h_ext - h_dead)

            rows.append(
                {
                    "exposure_group": int(exposure_value),
                    "transition": "MV_to_Extubated",
                    "time_day": t,
                    "cumulative_probability": round(float(cum_ext), 6),
                    "n_at_risk": n_at_risk,
                    "n_total_group": int(n_total),
                    "method": "aalen_johansen_discrete",
                }
            )
            rows.append(
                {
                    "exposure_group": int(exposure_value),
                    "transition": "MV_to_Dead",
                    "time_day": t,
                    "cumulative_probability": round(float(cum_dead), 6),
                    "n_at_risk": n_at_risk,
                    "n_total_group": int(n_total),
                    "method": "aalen_johansen_discrete",
                }
            )
            s_prev = s_new
            cum_ext_prev = cum_ext
            cum_dead_prev = cum_dead

        extubated = grp[grp["died"] == 0]
        denom_ext_reint = len(extubated)
        num_ext_reint = int(extubated["reintubated"].sum()) if denom_ext_reint else 0
        rows.append(
            {
                "exposure_group": int(exposure_value),
                "transition": "Extubated_to_Reintubated",
                "time_day": horizon,
                "cumulative_probability": (
                    round(float(num_ext_reint / denom_ext_reint), 6) if denom_ext_reint else np.nan
                ),
                "n_at_risk": int(denom_ext_reint),
                "n_total_group": int(n_total),
                "method": "group_proportion",
            }
        )

        # Extubated_to_Dead: patients who died (exited MV alive then died)
        extubated_then_dead = grp[grp["died"] == 1]
        denom_ext_dead = len(grp)  # all patients were on MV (at risk for extubation)
        num_ext_dead = len(extubated_then_dead)
        rows.append(
            {
                "exposure_group": int(exposure_value),
                "transition": "Extubated_to_Dead",
                "time_day": horizon,
                "cumulative_probability": (
                    round(float(num_ext_dead / denom_ext_dead), 6) if denom_ext_dead else np.nan
                ),
                "n_at_risk": int(denom_ext_dead),
                "n_total_group": int(n_total),
                "method": "group_proportion",
            }
        )

        # Reintubated_to_Dead: among reintubated patients, proportion who died
        reintubated = grp[grp["reintubated"] == 1]
        denom_reint_dead = len(reintubated)
        num_reint_dead = int(reintubated["died"].sum()) if denom_reint_dead else 0
        rows.append(
            {
                "exposure_group": int(exposure_value),
                "transition": "Reintubated_to_Dead",
                "time_day": horizon,
                "cumulative_probability": (
                    round(float(num_reint_dead / denom_reint_dead), 6)
                    if denom_reint_dead
                    else np.nan
                ),
                "n_at_risk": int(denom_reint_dead),
                "n_total_group": int(n_total),
                "method": "group_proportion",
            }
        )

    return pd.DataFrame(rows)


def fit_multistate_equivalent(
    day_level_df: pd.DataFrame,
    exposure_col: str = "landmark_delivered",
    horizon: int = 28,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit multistate-equivalent secondary VFD analysis outputs.

    Returns
    -------
    transition_hazard_df, transition_probability_df
    """
    hosp = _build_hosp_dataset(day_level_df, exposure_col=exposure_col)
    mv_ext = _build_person_period_mv(
        hosp_df=hosp,
        exposure_col=exposure_col,
        event_transition="MV_to_Extubated",
        horizon=horizon,
    )
    mv_dead = _build_person_period_mv(
        hosp_df=hosp,
        exposure_col=exposure_col,
        event_transition="MV_to_Dead",
        horizon=horizon,
    )
    ext_reint = _build_person_period_extubated_to_reintubated(
        hosp_df=hosp,
        exposure_col=exposure_col,
        horizon=horizon,
    )
    ext_dead = _build_person_period_extubated_to_dead(
        hosp_df=hosp,
        exposure_col=exposure_col,
        horizon=horizon,
    )
    reint_dead = _build_person_period_reintubated_to_dead(
        hosp_df=hosp,
        exposure_col=exposure_col,
        horizon=horizon,
    )

    transition_rows = [
        _fit_transition_model(mv_ext, "MV_to_Extubated", exposure_col),
        _fit_transition_model(mv_dead, "MV_to_Dead", exposure_col),
        _fit_transition_model(ext_reint, "Extubated_to_Reintubated", exposure_col),
        _fit_transition_model(ext_dead, "Extubated_to_Dead", exposure_col),
        _fit_transition_model(reint_dead, "Reintubated_to_Dead", exposure_col),
    ]
    transition_df = pd.DataFrame(transition_rows)
    transition_probs = _transition_probabilities(hosp, exposure_col, horizon)
    return transition_df, transition_probs
