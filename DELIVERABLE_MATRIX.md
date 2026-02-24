# Deliverable Matrix: Manuscript to Code Outputs

**Target journal:** Intensive Care Medicine (ICM)
**Constraint:** Max 5 illustrations (figures + tables combined) in main manuscript

## Main Manuscript (5 illustrations)

| Deliverable | Code Source | Output File | Status |
|---|---|---|---|
| **Table 1**: Operational Definitions | `07_aggregate_sites.py` | `output/final/pooled/table1_operational_definitions.csv` | **DONE** |
| **Table 2**: Patient Characteristics | `07_aggregate_sites.py`, `05_manuscript_figures.py` | `output/final/figures/table2_pooled_*.csv` | **DONE** |
| **Figure 1**: CONSORT Flow | `05_manuscript_figures.py` | `output/final/figures/fig1_consort.{pdf,tiff,eps}` | **DONE** |
| **Figure 2**: Criterion Validity | `05_manuscript_figures.py` | `output/final/figures/fig2_criterion_validity.{pdf,tiff,eps}` | **DONE** |
| **Figure 3**: Hospital Variation + Pooled Rates | `05_manuscript_figures.py` + `04_hospital_variation.py` | `output/final/figures/fig3_variation_forest.{pdf,tiff,eps}` | **DONE** |

## Electronic Supplementary Material (ESM)

| Deliverable | Code Source | Output File | Status |
|---|---|---|---|
| **eTable 1**: Construct Validity Outcomes | `07_aggregate_sites.py` | `output/final/esm/etable1_construct_validity.csv` | **DONE** |
| **eTable 2**: Site-Stratified Concordance | `07_aggregate_sites.py` | `output/final/esm/etable2_site_concordance.csv` | **DONE** |
| **eTable 3**: Sensitivity Analyses | `07_aggregate_sites.py` | `output/final/esm/etable3_sensitivity_analyses.csv` | **DONE** |
| **eFigure 1**: Phenotyping Logic Diagram | `05_manuscript_figures.py` | `output/final/esm/efig1_phenotype_logic.pdf` | **DONE** |
| **eFigure 2**: SAT+SBT Timing/Pairing | `05_manuscript_figures.py` | `output/final/esm/efig2_timing_pairing.pdf` | **DONE** |
| **eFigure 3**: Forest + Funnel Plots | `05_manuscript_figures.py` | `output/final/esm/efig3_forest_*.pdf` | **DONE** |
| **eFigure 4**: Data Completeness Heatmap | `05_manuscript_figures.py` | `output/final/esm/efig4_completeness.pdf` | **DONE** |
| **eFigure 5**: Bland-Altman EHR vs Flowsheet | `04_hospital_variation.py` | `output/final/figures/concordance_*.pdf` | **DONE** |
| **eFigure 6**: Outcome Model Forest Plot | `05_manuscript_figures.py` | `output/final/esm/efig6_outcomes.pdf` | **DONE** |

## Analysis Pipelines

| Pipeline | Script | Status |
|---|---|---|
| Cohort identification | `00_cohort_id.ipynb` | **DONE** |
| SAT phenotyping + criterion validity | `01_SAT_standard.ipynb` | **DONE** |
| SBT phenotyping (4 variants) + criterion validity | `02_SBT_*.ipynb` | **DONE** |
| Construct validity outcome models | `03_outcome_models.py` | **DONE** |
| Hospital-level variation + caterpillar | `04_hospital_variation.py` | **DONE** |
| Manuscript-ready figures + ESM | `05_manuscript_figures.py` | **DONE** |
| Sensitivity analyses | `06_sensitivity_analyses.py` | **PARTIAL** |
| Cross-site aggregation + ESM tables | `07_aggregate_sites.py` | **DONE** |

## ICM Submission Requirements

- **Word limit:** 3,000 words (original article)
- **Abstract:** 250 words, structured (Purpose, Methods, Results, Conclusions)
- **Keywords:** 3-5
- **Illustrations:** Max 5 (figures + tables) in main manuscript
- **Figure formats:** PDF/EPS (vector), TIFF >= 600 DPI (combination)
- **ESM:** No limit on supplementary figures/tables
- **References:** Follow Vancouver style

## Manuscript Numbers

All numbers needed to fill manuscript placeholders are compiled by `07_aggregate_sites.py` into `output/final/pooled/manuscript_numbers.json`.
