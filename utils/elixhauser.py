"""
Elixhauser Comorbidity Index with Van Walraven Weights.

Implements the Quan et al. (2005) ICD-10 coding algorithm for 31 Elixhauser
comorbidity categories and computes the Van Walraven composite score per
hospitalization.

References:
    - Elixhauser A, et al. Med Care. 1998;36(1):8-27.
    - Quan H, et al. Med Care. 2005;43(11):1130-9. (ICD-10 coding algorithm)
    - van Walraven C, et al. Med Care. 2009;47(6):626-633. (weights)
"""

from __future__ import annotations

import re

import pandas as pd

# ---------------------------------------------------------------------------
# ICD-10-CM regex patterns — AHRQ Elixhauser v2024.1
# Each pattern anchored at start; trailing .* allows any suffix.
# ---------------------------------------------------------------------------
ELIXHAUSER_ICD10_REGEX: dict[str, str] = {
    "congestive_heart_failure":     r"^(I09\.9|I11\.0|I13\.0|I13\.2|I25\.5|I42\.[0-9]|I43|I50|P29\.0)",
    "cardiac_arrhythmias":          r"^(I44\.[1-3]|I45\.[6-9]|I47|I48|I49|R00\.[0128]|T82\.1|Z45\.0|Z95\.0)",
    "valvular_disease":             r"^(A52\.0|I05|I06|I07|I08|I09\.[1-9]|I09\.89|I34|I35|I36|I37|I38|I39|Q23\.[0-3]|Z95\.[2-4])",
    "pulmonary_circulation":        r"^(I26|I27|I28\.[089])",
    "peripheral_vascular":          r"^(I70|I71|I73\.[189]|I77\.1|I79\.[0-9]|K55\.[189]|Z95\.[8-9])",
    "hypertension_uncomplicated":   r"^(I10)",
    "hypertension_complicated":     r"^(I11\.[^0]|I12|I13\.[^02]|I15|I16)",
    "paralysis":                    r"^(G04\.1|G11\.4|G80\.[1-4]|G83\.[0-4]|G83\.9)",
    "other_neurological":           r"^(G10|G11\.[^4]|G12|G13|G20|G21|G22|G25\.4|G25\.5|G31\.[289]|G32|G35|G36|G37|G40|G41|G93\.[1-9]|G93\.89|R47\.0)",
    "chronic_pulmonary":            r"^(I27\.8|I27\.9|J40|J41|J42|J43|J44|J45|J46|J47|J60|J61|J62|J63|J64|J65|J66|J67|J68\.4|J70\.[1-3])",
    "diabetes_uncomplicated":       r"^(E10\.[019]|E11\.[019]|E13\.[019])",
    "diabetes_complicated":         r"^(E10\.[2-8]|E11\.[2-8]|E13\.[2-8])",
    "hypothyroidism":               r"^(E00|E01|E02|E03|E89\.0)",
    "renal_failure":                r"^(I12\.9|I13\.1[01]|N18|N19|N25\.0|Z49\.[0-2]|Z94\.0|Z99\.2)",
    "liver_disease":                r"^(B18|I85\.[09]|I86\.4|I98\.2|K70|K71\.[3-7]|K72|K73|K74|K76\.[0-9]|K76\.89|Z94\.4)",
    "peptic_ulcer":                 r"^(K25|K26|K27|K28)",
    "aids_hiv":                     r"^(B20|B21|B22|B24)",
    "lymphoma":                     r"^(C81|C82|C83|C84|C85|C88|C96\.[0-9]|C96\.Z)",
    "metastatic_cancer":            r"^(C77|C78|C79|C80)",
    "solid_tumor_no_metastasis":    r"^(C[01][0-9]|C2[0-6]|C3[0-4]|C3[7-9]|C4[01]|C43|C45|C46|C47|C48|C49|C5[0-8]|C6[0-9]|C7[0-6]|C80\.1)",
    "rheumatoid_arthritis":         r"^(L40\.5|M05|M06|M08|M12\.0|M12\.3|M32|M33|M34|M35\.[1-3]|M36\.0)",
    "coagulopathy":                 r"^(D65|D66|D67|D68|D69\.[13-69])",
    "obesity":                      r"^(E66)",
    "weight_loss":                  r"^(E40|E41|E42|E43|E44|E45|E46|R63\.4|R64)",
    "fluid_electrolyte":            r"^(E22\.2|E86|E87)",
    "blood_loss_anemia":            r"^(D50\.0)",
    "deficiency_anemia":            r"^(D50\.[^0]|D51|D52|D53)",
    "alcohol_abuse":                r"^(F10|E52|G62\.1|I42\.6|K29\.2|K70\.0|K70\.3|K70\.9|T51|Z50\.2|Z71\.4|Z72\.1)",
    "drug_abuse":                   r"^(F11|F12|F13|F14|F15|F16|F18|F19|Z71\.5|Z72\.2)",
    "psychoses":                    r"^(F20|F22|F23|F24|F25|F28|F29|F30\.2|F31\.2|F31\.5)",
    "depression":                   r"^(F20\.4|F31\.3|F31\.4|F32|F33|F34\.1|F41\.2|F43\.2)",
}

# ---------------------------------------------------------------------------
# Van Walraven weights (van Walraven C, et al. Med Care. 2009;47(6):626-633)
# Ordered to match ELIXHAUSER_ICD10_REGEX keys.
# ---------------------------------------------------------------------------
VAN_WALRAVEN_WEIGHTS: dict[str, int] = {
    "congestive_heart_failure":     7,
    "cardiac_arrhythmias":          5,
    "valvular_disease":            -1,
    "pulmonary_circulation":        4,
    "peripheral_vascular":          2,
    "hypertension_uncomplicated":   0,  # van Walraven 2009 Table 2
    "hypertension_complicated":     0,
    "paralysis":                    7,
    "other_neurological":           6,
    "chronic_pulmonary":            3,
    "diabetes_uncomplicated":       0,
    "diabetes_complicated":        -1,  # van Walraven 2009 Table 2
    "hypothyroidism":               0,
    "renal_failure":                5,
    "liver_disease":               11,
    "peptic_ulcer":                 0,
    "aids_hiv":                     0,
    "lymphoma":                     9,
    "metastatic_cancer":           12,
    "solid_tumor_no_metastasis":    4,
    "rheumatoid_arthritis":         0,
    "coagulopathy":                 3,
    "obesity":                     -4,
    "weight_loss":                  6,
    "fluid_electrolyte":           -3,  # van Walraven 2009 Table 2 (was incorrectly +5)
    "blood_loss_anemia":           -2,
    "deficiency_anemia":           -2,
    "alcohol_abuse":                0,
    "drug_abuse":                  -7,
    "psychoses":                    0,
    "depression":                  -3,
}

# Pre-compile regex patterns for performance.
_COMPILED: dict[str, re.Pattern[str]] = {
    cat: re.compile(pattern, re.IGNORECASE)
    for cat, pattern in ELIXHAUSER_ICD10_REGEX.items()
}


def _normalize_icd10(code: str) -> str:
    """Strip whitespace and dots for consistent matching."""
    return code.strip().replace(".", "").upper() if isinstance(code, str) else ""


def _code_matches(code: str, pattern: re.Pattern[str]) -> bool:
    """Test a single ICD code against a compiled pattern.

    Matches both dotted (I50.9) and dot-free (I509) codes by testing
    the raw code first, then the normalized (dot-stripped) form.
    """
    c = code.strip() if isinstance(code, str) else ""
    if not c:
        return False
    # Try raw code first (handles dotted codes matching dot-inclusive patterns),
    # then try normalized form (handles dot-free codes)
    return bool(pattern.match(c)) or bool(pattern.match(_normalize_icd10(c)))


def compute_elixhauser_van_walraven(dx_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Van Walraven Elixhauser composite score per hospitalization.

    Parameters
    ----------
    dx_df:
        CLIF ``diagnosis`` table containing at minimum:
        - ``hospitalization_id``: unique encounter identifier
        - ``icd_code``: raw ICD code string (dots optional)
        - ``icd_version``: numeric or string version; rows where this is not
          10 (or "10") are silently excluded.

    Returns
    -------
    pd.DataFrame
        One row per ``hospitalization_id`` with columns:
        - ``hospitalization_id``
        - ``elixhauser_score`` (int): Van Walraven composite score

    Notes
    -----
    - Only ICD-10-CM codes are processed; ICD-9 rows are dropped.
    - Each comorbidity category is flagged at most once per hospitalization
      regardless of how many matching codes appear.
    - The composite score is the sum of Van Walraven weights across all
      flagged categories and can be negative.

    Examples
    --------
    >>> import pandas as pd
    >>> dx = pd.DataFrame({
    ...     "hospitalization_id": ["A", "A", "B"],
    ...     "icd_code":           ["I50.9", "E11.9", "C80.1"],
    ...     "icd_version":        [10, 10, 10],
    ... })
    >>> compute_elixhauser_van_walraven(dx)
      hospitalization_id  elixhauser_score
    0                  A                 7
    1                  B                 4
    """
    required = {"hospitalization_id", "icd_code", "icd_version"}
    missing = required - set(dx_df.columns)
    if missing:
        raise ValueError(f"dx_df is missing required columns: {missing}")

    # Keep only ICD-10 rows.
    icd10_mask = dx_df["icd_version"].astype(str).str.strip().isin({"10", "10.0"})
    df = dx_df.loc[icd10_mask, ["hospitalization_id", "icd_code"]].copy()
    df["icd_code"] = df["icd_code"].astype(str).str.strip()

    # Flag each category per hospitalization (boolean, then any() by group).
    for cat, pattern in _COMPILED.items():
        df[cat] = df["icd_code"].apply(lambda c, p=pattern: _code_matches(c, p))

    category_cols = list(_COMPILED.keys())
    flagged = (
        df.groupby("hospitalization_id")[category_cols]
        .any()
        .astype(int)
    )

    # Apply Van Walraven weights and sum.
    weight_series = pd.Series(VAN_WALRAVEN_WEIGHTS)
    flagged["elixhauser_score"] = flagged[category_cols].mul(weight_series).sum(axis=1).astype(int)

    result = flagged[["elixhauser_score"]].reset_index()

    # Ensure all original hospitalization_ids appear (even those with no ICD-10 codes).
    all_ids = pd.DataFrame({"hospitalization_id": dx_df["hospitalization_id"].unique()})
    result = all_ids.merge(result, on="hospitalization_id", how="left")
    result["elixhauser_score"] = result["elixhauser_score"].fillna(0).astype(int)

    return result
