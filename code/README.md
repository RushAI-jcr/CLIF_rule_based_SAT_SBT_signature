# Code Directory

## Pipeline Overview

This project follows a three-phase CLIF workflow:

1. **Cohort Identification** (`00_*`) -- Build study cohort from CLIF tables
2. **Phenotyping and Validation** (`01_*`, `02_*`) -- SAT/SBT phenotypes with per-site criterion validity
3. **Analysis** (`03_*` through `07_*`) -- Outcome models, hospital variation, figures, sensitivity, aggregation

## Script Execution Order

| Script | Phase | Description | Input | Output |
|---|---|---|---|---|
| `00_cohort_id.ipynb` | Cohort ID | Identifies IMV episodes, builds wide dataset | Raw CLIF tables | `study_cohort.parquet` |
| `01_SAT_standard.ipynb` | Phenotyping | SAT eligibility + delivery + concordance | `study_cohort.parquet` | `final_df_SAT.csv`, concordance CSVs, Table 1 |
| `02_SBT_Standard.ipynb` | Phenotyping | SBT eligibility + delivery + concordance | `study_cohort.parquet` | `final_df_SBT.csv`, concordance CSVs, Table 1 |
| `02_SBT_*_Stability.ipynb` | Phenotyping | SBT variants (hemodynamic/respiratory) | `study_cohort.parquet` | Variant-specific SBT outputs |
| `03_outcome_models.py` | Analysis | Cause-specific Cox, Fine-Gray-equivalent VFD primary, multistate VFD secondary, mixed-effects LOS/mortality | `final_df_SAT.csv`, `final_df_SBT.csv` | `construct_validity_outcomes.csv`, `construct_validity_vfd_components.csv`, `construct_validity_cif_curves.csv`, `construct_validity_multistate_transitions.csv` |
| `04_hospital_variation.py` | Analysis | SAP risk-standardized rates (`Predicted/Expected*Overall`), MOR, caterpillar + funnel + agreement | `final_df_SAT.csv`, `final_df_SBT.csv` | Hospital rates, funnel data, concordance plots |
| `05_manuscript_figures.py` | Analysis | CONSORT, logic diagrams, criterion validity, timing | All upstream | Figures 1-4, pooled Tables 2-4 |
| `06_sensitivity_analyses.py` | Analysis | Data completeness, alternative thresholds | `study_cohort.parquet` | Sensitivity CSVs |
| `07_aggregate_sites.py` | Analysis | Pools per-site outputs, generates manuscript numbers | All site outputs | `manuscript_numbers.json`, pooled CSVs |

## Site-Level vs Central Execution

**At each CLIF site (local):**
```
00_cohort_id.ipynb → 01_SAT_standard.ipynb + 02_SBT_Standard.ipynb
```

**Central analysis (after site outputs collected):**
```
07_aggregate_sites.py → 03_outcome_models.py + 04_hospital_variation.py
                      → 05_manuscript_figures.py + 06_sensitivity_analyses.py
```

## Environment

All scripts require Python 3.9+ with dependencies listed in `../requirements.txt`. Scripts assume:
- Working directory is `code/`
- Config file exists at `../config/config.json`
- CLIF tables exist at the path specified in config
- Utility modules are in `../utils/` (added to sys.path automatically)

## Notebook Parity Policy

This repository follows a dual notebook policy:

- `*.ipynb` files are the authoring source for interactive development.
- Paired `*.py` Marimo notebooks are executable/script artifacts kept in version control.

Rules:

- Keep a one-to-one pair for every numbered notebook (`00_*`, `01_*`, `02_*`).
- After editing an `ipynb`, regenerate and validate its paired `.py`:
  - `uvx marimo convert <notebook>.ipynb -o <notebook>.py`
  - `uvx marimo check <notebook>.py`
- CI/local checks should fail if a notebook pair is missing.

## Definitions

All phenotyping thresholds, windows, and parameters are centralized in `../utils/definitions_source_of_truth.py`. No analysis script should hardcode these values.

## SAP Conformance Interfaces

Pinned SAP version: `Statistical_Analysis_Plan_SAT_SBT.docx` (2026-02-24).

Output contract additions:

- `construct_validity_outcomes.csv` must include:
  - `sap_requirement_id`
  - `analysis_role`
  - `prespec_status`
  - `model_estimator`
  - `ph_assumption_status`
  - `competing_risk_method`
- Concordance pooling should emit CI provenance:
  - `cluster_col_used`
  - `ci_method`
  - `n_boot`
  - `bootstrap_level`
