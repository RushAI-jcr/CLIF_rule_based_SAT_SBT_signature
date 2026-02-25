# SAP Method-Equivalence Notes

SAP source: `Statistical_Analysis_Plan_SAT_SBT.docx` (dated 2026-02-24).

## Principle

Where SAP names methods commonly implemented in R-first workflows, this repository uses mathematically equivalent Python-native estimators.

## Mapping

- Fine-Gray subdistribution hazard (SAP primary VFD):
  - Implemented as discrete-time cloglog subdistribution model in [`utils/competing_risks.py`](/Users/JCR/Desktop/SAT/claude-code-form/utils/competing_risks.py).
  - Equivalent estimand: subdistribution hazard ratio for extubation alive by day 28.

- Multistate VFD secondary:
  - Implemented as transition-specific discrete-time cloglog hazards plus transition probability outputs in [`utils/multistate.py`](/Users/JCR/Desktop/SAT/claude-code-form/utils/multistate.py).

- ICU LOS and mortality mixed-effects models:
  - Implemented with `PoissonBayesMixedGLM` and `BinomialBayesMixedGLM` random-intercept models in [`code/03_outcome_models.py`](/Users/JCR/Desktop/SAT/claude-code-form/code/03_outcome_models.py).
  - These satisfy hierarchical random-intercept intent in SAP.

- Criterion-validity CIs:
  - Implemented with BCa cluster bootstrap in [`utils/meta_analysis.py`](/Users/JCR/Desktop/SAT/claude-code-form/utils/meta_analysis.py).
  - Primary cluster: `imv_episode_id`; fallback: `hospitalization_id`.

## Section 7 Notes

- SAP missing-data windows and missing-source exclusion logging are implemented in the sensitivity pipeline (`code/06_sensitivity_analyses.py`) with explicit audit CSV outputs.
