# SAP Audit Report

Date: 2026-02-25
Authoritative spec: `Statistical_Analysis_Plan_SAT_SBT.docx` (2026-02-24)

## Status Summary

- Total requirements tracked: 22
- `match`: 22
- `partial`: 0
- `mismatch`: 0

Source table: [`docs/sap_code_traceability.csv`](/Users/JCR/Desktop/SAT/claude-code-form/docs/sap_code_traceability.csv)

## Critical/High Findings

No open critical/high mismatches in the traceability table.

## Closed Remediations

- BCa cluster-bootstrap criterion CIs with episode-level clustering.
- Fine-Gray-equivalent VFD primary + CIF and component reporting outputs.
- Multistate transition modeling outputs.
- Mixed-effects random-intercept LOS/mortality primaries.
- Hospital RSR formula (`Predicted/Expected * Overall`), MOR from model variance.
- Funnel control limits (95%, 99.8%).
- Concordance agreement switched to Pearson, Spearman, Bland-Altman, CCC.
- Multiplicity default disabled in primary output path.
- Required SAP metadata fields in construct-validity outputs.

## Decision Log

- Python-native method-equivalent implementations are accepted where SAP names methods with typical R package implementations.
- `--require-sap-full-pass` now enforces a traceability gate via `scripts/audit_sap_alignment.py`.
