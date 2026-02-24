# Deliverable Matrix: Manuscript to Code Outputs

## Tables

| Deliverable | Manuscript Ref | Code Source | Output File(s) | Status |
|---|---|---|---|---|
| **Table 1**: Operational Definitions | Line 122 | `07_aggregate_sites.py` | `output/final/pooled/table1_operational_definitions.csv` | **DONE** |
| **Table 2**: Patient Characteristics | Lines 124-126 | `01_SAT`, `02_SBT`, `05_manuscript_figures.py` | `output/final/{SITE}/*/table1_*.csv` → pooled | **DONE** |
| **Table 3**: Confusion Matrix | Line 130 | `01_SAT` (cell 28), `02_SBT` (cells 21, 27) | `delivery_concordance_summary*.csv` → pooled | **DONE** |
| **Table 4**: Criterion Validity Metrics | Lines 133-134 | `07_aggregate_sites.py` | `output/final/pooled/pooled_concordance.csv` | **DONE** |
| **Table 5**: Construct Validity | Line 135 | `03_outcome_models.py` | `output/final/construct_validity_outcomes.csv` | **DONE** |

## Figures

| Deliverable | Manuscript Ref | Code Source | Output File | Status |
|---|---|---|---|---|
| **Figure 1**: CONSORT Flow | Line 139 | `05_manuscript_figures.py` + `07_aggregate_sites.py` | `output/final/figures/fig1_consort.pdf` | **DONE** |
| **Figure 2**: Phenotyping Logic | Lines 141-142 | `05_manuscript_figures.py` | `output/final/figures/fig2_phenotype_logic.pdf` | **DONE** |
| **Figure 3**: Criterion Validity | Lines 143-144 | `05_manuscript_figures.py` | `output/final/figures/fig3_criterion_validity.pdf` | **DONE** |
| **Figure 4**: Timing/Pairing | Lines 145-146 | `05_manuscript_figures.py` | `output/final/figures/fig4_timing_pairing.pdf` | **DONE** |
| **Figure 5**: Hospital Variation | Lines 147-148 | `04_hospital_variation.py` | `output/final/figures/fig5_caterpillar_*.pdf` | **DONE** |

## Analysis Pipelines

| Pipeline | Script | Status |
|---|---|---|
| Cohort identification | `00_cohort_id.ipynb` | **DONE** |
| SAT phenotyping + criterion validity | `01_SAT_standard.ipynb` | **DONE** |
| SBT phenotyping (4 variants) + criterion validity | `02_SBT_*.ipynb` | **DONE** |
| Construct validity outcome models | `03_outcome_models.py` | **DONE** |
| Hospital-level variation + caterpillar | `04_hospital_variation.py` | **DONE** |
| Manuscript-ready figures | `05_manuscript_figures.py` | **DONE** |
| Sensitivity analyses | `06_sensitivity_analyses.py` | **PARTIAL** (framework ready, re-run with alt params not implemented) |
| Cross-site aggregation | `07_aggregate_sites.py` | **DONE** |

## Manuscript Numbers

All numbers needed to fill manuscript placeholders are compiled by `07_aggregate_sites.py` into `output/final/pooled/manuscript_numbers.json`.
