# Python Code Audit

Date: 2026-02-25
Scope:

- `code/03_outcome_models.py`
- `code/04_hospital_variation.py`
- `code/06_sensitivity_analyses.py`
- `code/07_aggregate_sites.py`
- `utils/meta_analysis.py`
- `utils/competing_risks.py`
- `utils/multistate.py`

## Summary

Severity-ranked review focused on correctness, maintainability, typing, and statistical implementation safety.

## Findings

1. High: Notebook-derived SAT/SBT scripts still contain large multi-purpose cells and inconsistent function-level typing.
   - Files: `code/01_SAT_standard.py`, `code/02_SBT_Standard.py`, `code/02_SBT_*_Stability.py`
   - Risk: harder validation and change control for criterion-validity execution paths.
   - Status: deferred; core SAP outputs enforced in downstream audited scripts.

2. Medium: `03_outcome_models.py` has expanded orchestration complexity.
   - Mitigation: metadata helper and modular utility extraction (`utils/competing_risks.py`, `utils/multistate.py`).
   - Residual risk: future changes should continue splitting orchestration from estimator functions.

3. Medium: Path bootstrapping with `sys.path` remains in numbered scripts.
   - Risk: execution-context fragility.
   - Status: retained for backward compatibility with notebook/CLI contract.

4. Low: `BayesMixedGLM` p-values use normal approximation from posterior mean/sd.
   - Risk: slight inferential calibration differences vs frequentist GLMM outputs.
   - Status: documented as method-equivalent approximation under Python-only constraint.

## Remediations Implemented

- Added typed competing-risk and multistate utility modules.
- Replaced primary LOS/mortality model paths with mixed-effects random-intercept models.
- Added SAP metadata contract fields to construct-validity outputs.
- Added BCa cluster bootstrap implementation and CI provenance fields.
- Added executable SAP conformance gate script.

## Recommended Next Pass

1. Break out concordance logic from notebook-derived SAT/SBT scripts into shared typed utility modules.
2. Add `mypy` baseline for `code/03`, `code/04`, `code/06`, `code/07`, and `utils/`.
3. Add regression tests for new hospital RSR/MOR and multistate outputs on synthetic datasets.
