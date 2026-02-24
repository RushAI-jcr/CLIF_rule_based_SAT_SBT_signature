# Phenotyping Protocolized Liberation from Sedation and Invasive Mechanical Ventilation in the Electronic Health Record

**CLIF Version:** 2.1

## Objective

Develop and validate computable EHR phenotypes for two related objectives:
1. Identifying **eligibility** for Spontaneous Awakening Trials (SATs) and Spontaneous Breathing Trials (SBTs)
2. Detecting **delivery** of SATs and SBTs among eligible ventilator-days

Phenotypes are validated against clinician flowsheet documentation (criterion validity) and patient outcomes (construct validity) across a federated multi-site CLIF consortium.

## Required CLIF Tables

Please refer to the [CLIF 2.1 data dictionary](https://clif-icu.com/data-dictionary) and [controlled vocabularies (mCIDE)](https://github.com/clif-consortium/CLIF) for constructing the required tables.

| Table | Required Fields | Rationale |
|---|---|---|
| `patient` | `patient_id`, `sex_category`, `race_category`, `ethnicity_category` | Demographics for Table 1 and covariate adjustment |
| `hospitalization` | `patient_id`, `hospitalization_id`, `admission_dttm`, `discharge_dttm`, `age_at_admission` | Cohort identification, outcomes (mortality, LOS) |
| `adt` | `hospitalization_id`, `in_dttm`, `location_category`, `hospital_id` | ICU admission identification, hospital-level variation |
| `medication_admin_continuous` | `hospitalization_id`, `admin_dttm`, `med_category`, `med_dose`, `med_dose_unit`, `mar_action_group` | SAT eligibility (sedatives/opioids), SBT eligibility (vasopressors), paralytic exclusions |
| `respiratory_support` | `hospitalization_id`, `recorded_dttm`, `device_category`, `mode_category`, `mode_name`, `fio2_set`, `peep_set`, `pressure_support_set`, `tracheostomy` | IMV identification, SBT eligibility (controlled mode), SBT delivery (mode transition) |
| `patient_assessments` | `hospitalization_id`, `recorded_dttm`, `assessment_category`, `numerical_value`, `categorical_value` | RASS scores, flowsheet SAT/SBT documentation |
| `vitals` | `hospitalization_id`, `recorded_dttm`, `vital_category`, `vital_value` | SpO2 for SBT eligibility |

**Relevant `med_category` values:**
```
norepinephrine, epinephrine, phenylephrine, angiotensin, vasopressin,
dopamine, dobutamine, milrinone, isoproterenol,
cisatracurium, vecuronium, rocuronium,
fentanyl, propofol, lorazepam, midazolam, hydromorphone, morphine
```

**Relevant `assessment_category` values:**
```
sat_screen_pass_fail, sat_delivery_pass_fail,
sbt_screen_pass_fail, sbt_delivery_pass_fail,
rass, gcs_total
```

## Cohort Identification

- **Study period:** January 1, 2022 -- December 31, 2024
- **Inclusion:** Adults (age >= 18) with at least one ICU admission requiring invasive mechanical ventilation (IMV)
- **Exclusion:** Tracheostomy patients, age > 119
- **IMV episodes:** Defined by >= 72-hour gaps in ventilator support
- **Minimum IMV duration:** 6 hours per episode

## Expected Output

Results are written to `output/` following CLIF naming conventions:

| Output | Location | Description |
|---|---|---|
| Study cohort | `output/intermediate/study_cohort.parquet` | Eligible hospitalizations with ventilator-day-level data |
| SAT results | `output/final/{SITE}/SAT_standard/` | Per-site SAT delivery rates, concordance, Table 1 |
| SBT results | `output/final/{SITE}/SBT_standard/` | Per-site SBT delivery rates, concordance, Table 1 |
| Pooled results | `output/final/pooled/` | Cross-site pooled rates, concordance, manuscript numbers |
| Outcome models | `output/final/construct_validity_outcomes.csv` | Construct validity regression results |
| Figures | `output/final/figures/` | Manuscript-ready figures (JAMA style) |
| Sensitivity | `output/final/sensitivity/` | Data completeness, alternative thresholds |

## Running the Project

### 1. Configure

```bash
cp config/config_template.json config/config.json
# Edit config.json with your site_name, tables_path, file_type, timezone
```

See [`config/README.md`](config/README.md) for details.

### 2. Set up Python environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the pipeline

Execute scripts **in order** from the `code/` directory:

```
Step 1:  00_cohort_id.ipynb              Cohort identification (must run first)
Step 2:  01_SAT_standard.ipynb           SAT phenotyping + criterion validity    } run in
         02_SBT_Standard.ipynb           SBT phenotyping + criterion validity    } parallel
Step 3:  07_aggregate_sites.py           Pool cross-site results
Step 4:  03_outcome_models.py            Construct validity models
         04_hospital_variation.py        Hospital-level variation
Step 5:  05_manuscript_figures.py        Manuscript figures (Figs 1-4)
         06_sensitivity_analyses.py      Sensitivity analyses
```

Steps 1-2 are run **at each CLIF site** locally. Steps 3-5 are run centrally on aggregated outputs.

See [`code/README.md`](code/README.md) for detailed script documentation.

### 4. Troubleshooting

If you encounter errors, run notebooks cell-by-cell to identify the failing step. Common issues:
- Missing `config/config.json` -- copy from template
- Missing CLIF tables -- verify `tables_path` in config
- Missing Python packages -- `pip install -r requirements.txt`

## Project Structure

```
├── README.md                    This file
├── LICENSE                      Apache-2.0
├── requirements.txt             Python dependencies
├── .gitignore                   Ignores data, outputs, configs
│
├── code/                        Analysis scripts (numbered execution order)
│   ├── 00_cohort_id.ipynb
│   ├── 01_SAT_standard.ipynb
│   ├── 02_SBT_Standard.ipynb
│   ├── 02_SBT_*.ipynb           SBT sensitivity variants
│   ├── 03_outcome_models.py
│   ├── 04_hospital_variation.py
│   ├── 05_manuscript_figures.py
│   ├── 06_sensitivity_analyses.py
│   └── 07_aggregate_sites.py
│
├── config/                      Site-specific configuration (gitignored)
│   ├── config_template.json
│   └── outlier_config.json
│
├── utils/                       Shared utility modules
│   ├── config.py                Config loader
│   ├── pyCLIF.py                CLIF data loading and processing
│   ├── pyCLIF2.py               Extended CLIF utilities
│   ├── pySBT.py                 SBT/Table 1 helper functions
│   ├── pySofa.py                SOFA score computation
│   ├── definitions_source_of_truth.py   All phenotype definitions
│   ├── site_output_schema.py    Federated output schema
│   └── meta_analysis.py         Meta-analysis pooling functions
│
├── outlier-thresholds/          Outlier threshold CSVs
│   ├── outlier_thresholds_adults_vitals.csv
│   ├── outlier_thresholds_labs.csv
│   └── outlier_thresholds_respiratory_support.csv
│
└── output/
    ├── intermediate/            Per-site intermediate files
    └── final/                   Final results for manuscript
        ├── figures/
        ├── pooled/
        └── sensitivity/
```

## References

- Rojas JC, et al. A common longitudinal intensive care unit data format (CLIF) for critical illness research. *Intensive Care Med*. 2025;51(1).
- CLIF 2.1 Data Dictionary: https://clif-icu.com/data-dictionary
- CLIF Project Template: https://github.com/Common-Longitudinal-ICU-data-Format/CLIF-Project-Template
