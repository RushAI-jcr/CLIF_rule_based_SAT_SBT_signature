# Statistical Methods Audit (SAP v1.0 Conformance)

Date: 2026-02-25
Authoritative SAP: `Statistical_Analysis_Plan_SAT_SBT.docx` (2026-02-24)

## Summary

This document tracks SAP conformance using requirement IDs and code evidence.

Primary traceability artifacts:

- [`docs/sap_requirement_registry.yaml`](/Users/JCR/Desktop/SAT/claude-code-form/docs/sap_requirement_registry.yaml)
- [`docs/sap_code_traceability.csv`](/Users/JCR/Desktop/SAT/claude-code-form/docs/sap_code_traceability.csv)
- [`docs/sap_audit_report.md`](/Users/JCR/Desktop/SAT/claude-code-form/docs/sap_audit_report.md)

## Current Status

- `match`: 22
- `partial`: 0
- `mismatch`: 0

## Closed (Match)

- Criterion-validity BCa cluster bootstrap CIs and provenance fields.
- Pooled random-effects sensitivity/specificity meta-analysis with heterogeneity.
- Cause-specific time-varying Cox extubation path retained.
- VFD primary competing-risk path (Fine-Gray-equivalent) and required component outputs.
- VFD multistate secondary transition outputs.
- ICU LOS and mortality moved to mixed-effects random-intercept primaries.
- Joint SAT+SBT 4-level exposure model output.
- Hospital benchmarking remapped to SAP RSR formula + MOR from model variance.
- Funnel 95% and 99.8% control limits.
- Agreement metrics include CCC and Bland-Altman.
- Multiplicity disabled by default for primary path.
- `analysis_role` and `prespec_status` tags present in construct-validity rows.

## Open (Partial)

No open `partial` requirements in `sap_code_traceability.csv`.

## Gate

Run:

```bash
python3 scripts/audit_sap_alignment.py
```

Strict gate:

```bash
python3 scripts/audit_sap_alignment.py --fail-on-mismatch
```
