# SAT/SBT EHR Phenotyping (CLIF 2.1)

Computable EHR phenotypes for protocolized liberation from sedation and invasive mechanical ventilation:

- SAT (Spontaneous Awakening Trial): eligibility + delivery detection
- SBT (Spontaneous Breathing Trial): eligibility + delivery detection

This repository is organized for federated multi-site CLIF workflows.

## Purpose And Clinical Question

Primary question:

- Among mechanically ventilated ICU patients, can we reliably identify SAT/SBT eligibility and delivery from EHR data, and are these phenotype detections associated with expected outcomes?

Core outputs:

- Site-level phenotype delivery and concordance summaries
- Pooled cross-site rates and manuscript-ready artifacts
- Construct-validity outcome models (`code/03_outcome_models.py`)

Statistical methods status and decisions are maintained in:

- [`docs/statistical_methods_audit.md`](docs/statistical_methods_audit.md)
- SAP traceability registry/report:
  - [`docs/sap_requirement_registry.yaml`](docs/sap_requirement_registry.yaml)
  - [`docs/sap_code_traceability.csv`](docs/sap_code_traceability.csv)
  - [`docs/sap_audit_report.md`](docs/sap_audit_report.md)

## SAP Version Pin

- Authoritative plan: `Statistical_Analysis_Plan_SAT_SBT.docx`
- Version date: February 24, 2026
- Conformance policy: Python-native method-equivalent implementations allowed

## Role Split

Primary path:

- Site implementer (local CLIF site): run `00`-`02`, generate site outputs.

Secondary path:

- Central analyst: aggregate site outputs, run pooled analyses and figures.

## Minimum CLIF Data Contract

Required CLIF tables and key fields:

- `patient`: `patient_id`, `sex_category`, `race_category`, `ethnicity_category`
- `hospitalization`: `patient_id`, `hospitalization_id`, `admission_dttm`, `discharge_dttm`, `age_at_admission`
- `adt`: `hospitalization_id`, `in_dttm`, `location_category`, `hospital_id`
- `medication_admin_continuous`: `hospitalization_id`, `admin_dttm`, `med_category`, `med_dose`, `med_dose_unit`, `mar_action_group`
- `respiratory_support`: `hospitalization_id`, `recorded_dttm`, `device_category`, `mode_category`, `mode_name`, `fio2_set`, `peep_set`, `pressure_support_set`, `tracheostomy`
- `patient_assessments`: `hospitalization_id`, `recorded_dttm`, `assessment_category`, `numerical_value`, `categorical_value`
- `vitals`: `hospitalization_id`, `recorded_dttm`, `vital_category`, `vital_value`

Reference sources:

- CLIF 2.1 dictionary: <https://clif-icu.com/data-dictionary>
- CLIF vocabularies: <https://github.com/clif-consortium/CLIF>

## Environment And Runtime Matrix

| Component | Requirement | Notes |
|---|---|---|
| Python | 3.11+ recommended (3.9+ supported) | use virtual environment |
| Dependencies | `pip install -r requirements.txt` | includes statsmodels/lifelines/marimo deps |
| Authoring notebooks | `*.ipynb` | source-of-truth for interactive editing |
| Executable notebooks | paired `*.py` Marimo files | committed alongside `ipynb` |
| Notebook validation | `uvx marimo check code/<file>.py` | required for parity checks |

## 15-Minute Quickstart (Site Implementer)

1. Create environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure your site:

```bash
cp config/config_template.json config/config.json
# Edit config/config.json
```

3. Run local pipeline in order (from `code/`):

```bash
cd code
# Step 1
python3 00_cohort_id.py
# Step 2 (SAT + SBT)
python3 01_SAT_standard.py
python3 02_SBT_Standard.py
```

4. Confirm expected artifacts:

- `output/intermediate/study_cohort.parquet`
- `output/intermediate/final_df_SAT.csv`
- `output/intermediate/final_df_SBT.csv`

## Site-Local Runbook (Mandatory Order)

1. Cohort build:

- `00_cohort_id.ipynb` (or paired `.py`)
- Output: `study_cohort.*`

2. SAT phenotyping:

- `01_SAT_standard.ipynb` (or paired `.py`)
- Output: `final_df_SAT.csv` + SAT concordance artifacts

3. SBT phenotyping:

- `02_SBT_Standard.ipynb` (or paired `.py`)
- Optional sensitivity variants: `02_SBT_*_Stability.*`
- Output: `final_df_SBT.csv` + SBT concordance artifacts

### Site QA Checks

- Cohort cardinality checks: non-zero `patient_id`, `hospitalization_id`, `hosp_id_day_key`
- Eligibility sanity: non-zero eligible SAT and SBT days
- Delivery sanity: delivery rates in plausible range (0, 1)
- Column availability: required assessment and medication categories present or explicitly missing by site policy

## Central Aggregation Runbook (Brief)

After site outputs are collected:

1. `python3 code/07_aggregate_sites.py`
2. `python3 code/03_outcome_models.py`
3. `python3 code/04_hospital_variation.py`
4. `python3 code/05_manuscript_figures.py`
5. `python3 code/06_sensitivity_analyses.py`

Primary pooled outputs:

- `output/final/pooled/manuscript_numbers.json`
- `output/final/construct_validity_outcomes.csv`
- `output/final/construct_validity_vfd_components.csv`
- `output/final/construct_validity_cif_curves.csv`
- `output/final/construct_validity_multistate_transitions.csv`
- `output/final/hospital_benchmarking_funnel_data.csv`

## Reproducibility And Validation Checklist

- [ ] `config/config.json` created from template and reviewed
- [ ] Notebook parity confirmed (`ipynb` + paired `.py` exists for each numbered notebook)
- [ ] `uvx marimo check` passes for paired notebook `.py` files
- [ ] Python compile checks pass for `code/` and `utils/`
- [ ] Local outputs regenerated from code (no stale committed artifacts)
- [ ] Statistical method/version columns present in `construct_validity_outcomes.csv`

## Troubleshooting Matrix

| Symptom / Signature | Likely Cause | Fix |
|---|---|---|
| `FileNotFoundError: config/config.json` | config not initialized | copy template and fill required fields |
| `column ... not found` in SAT/SBT notebooks | site CLIF mapping incomplete | validate table schema and controlled vocab mappings |
| `lifelines not installed` | missing dependency | `pip install -r requirements.txt` |
| `Model not fit - check dependencies or data` in outcomes CSV | too few events, separation, or missing covariates | inspect cohort size/events and required columns |
| `marimo check ... failed to parse` | notebook `.py` drift or manual syntax issue | regenerate with `uvx marimo convert ...` and re-check |
| all pooled rates empty in aggregation | site outputs missing or not discoverable | verify directory structure and filenames under `output/final/{SITE}/...` |

## Project Layout

```text
claude-code-form/
  code/                # numbered pipeline scripts + paired notebooks
  config/              # local site config templates
  docs/                # audit and method status docs
  outlier-thresholds/  # threshold CSVs
  output/              # generated artifacts (placeholders tracked)
  utils/               # shared core modules
```

## Notebook Source-Of-Truth Policy

- `*.ipynb` = authoring source
- paired `*.py` Marimo notebook = executable artifact in version control

After notebook edits:

```bash
uvx marimo convert code/<name>.ipynb -o code/<name>.py
uvx marimo check code/<name>.py
```
