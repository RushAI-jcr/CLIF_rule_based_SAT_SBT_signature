# Statistical Methods Audit — SAT/SBT CLIF Pipeline

**Date:** 2026-02-24
**Scope:** `03_outcome_models.py`, `01_SAT_standard.ipynb`, `utils/meta_analysis.py`
**Target journal:** Intensive Care Medicine (ICM)

---

## 1. Methods Inventory

| # | Method | File | Function |
|---|--------|------|----------|
| 1 | Cox PH with time-varying exposure (counting process) | `03_outcome_models.py` | `fit_cox_extubation()` |
| 2 | Hurdle model (logistic + NB) for VFDs | `03_outcome_models.py` | `fit_vfd_model()` |
| 3 | Negative binomial for ICU LOS | `03_outcome_models.py` | `fit_icu_los_model()` |
| 4 | GEE logistic for mortality (hospital clustering) | `03_outcome_models.py` | `fit_mortality_model()` |
| 5 | DerSimonian-Laird random-effects meta-analysis | `meta_analysis.py` | `run_proportion_meta_analysis()` |
| 6 | Logit transformation for proportion pooling | `meta_analysis.py` | `logit_transform()` |
| 7 | Cohen's kappa for concordance | `01_SAT_standard.ipynb` | inline |
| 8 | Benjamini-Hochberg FDR correction | `03_outcome_models.py` | `apply_multiplicity_correction()` |
| 9 | SOFA-97 scoring | `01_SAT_standard.ipynb` | inline / `pySofa.py` |
| 10 | Cluster bootstrap CIs | `03_outcome_models.py` | `bootstrap_ci()` |

---

## 2. Issue-by-Issue Audit

### CRITICAL-1: VFD Hurdle Model — Inflated Type I Error

**Current implementation:** Two-part hurdle (logistic for death + truncated NB for VFD|alive).

**Problem:** Renard Triché et al. (*Crit Care* 2025; PMID 40537834; DOI: 10.1186/s13054-025-05474-9) conducted a simulation study comparing VFD analysis methods and found that **hurdle and ZINB models have inflated Type I error** rates in ICU settings. The recommended alternatives are:

1. **Multistate model** (preferred) — treats ventilation, extubation, and death as competing states
2. **Proportional odds ordinal regression** — treats VFD as an ordinal outcome
3. **Fine-Gray subdistribution hazard** — if competing-risk framing is preferred

**Recommendation:** Replace the hurdle model with a proportional odds model (R: `polr`; Python: `OrderedModel` from statsmodels) or add it as a sensitivity analysis. At minimum, cite the Renard Triché paper and acknowledge the limitation.

**Citation to add:**
> Renard Triché L, et al. Comparison of statistical methods for analyzing ventilator-free days: a simulation study. *Crit Care*. 2025;29:xxx. doi:10.1186/s13054-025-05474-9

### CRITICAL-2: VFD=0 for All Deaths — Correct but Cite

**Current implementation** (line 83-87): `VFD=0` if patient died, regardless of ventilation duration. This follows the standard convention.

**Required citation:**
> Schoenfeld DA, Bernard GR, for the ARDS Network. Statistical evaluation of ventilator-free days as an efficacy measure in clinical trials of treatments for acute respiratory distress syndrome. *Crit Care Med*. 2002;30(8):1772-1777. PMID: 12163791

### HIGH-1: Cox Model — Death Treated as Censoring, Not Competing Risk

**Current implementation** (line 224): `event = (is_last & died==0)` — extubation is the event; death is censored.

**Problem:** The docstring says "death as competing event" but the implementation treats death as **non-informative censoring**, which is only valid under the strong assumption that death and extubation are independent conditional on covariates. For ICU data, this is unlikely — sicker patients are more likely to both die and remain intubated.

**Recommendation:** Implement a proper cause-specific hazard model (fit two Cox models: one for extubation, one for death) or use Fine-Gray subdistribution hazards via `lifelines` or `cmprsk` equivalent. At minimum, acknowledge the competing risk assumption.

**Citation:**
> Fine JP, Gray RJ. A proportional hazards model for the subdistribution of a competing risk. *J Am Stat Assoc*. 1999;94(446):496-509.

### HIGH-2: Immortal Time Bias — Handled Correctly, Needs Citation

**Current implementation:** Uses `cummax()` on delivery indicator (line 213) so exposure switches 0→1 irreversibly. The counting-process format (start/stop) avoids immortal time bias.

**Recommendation:** Cite the canonical reference for this approach:

> Suissa S. Immortal time bias in pharmacoepidemiology. *Am J Epidemiol*. 2008;167(4):492-499. PMID: 18056625. DOI: 10.1093/aje/kwm324

> Lévesque LE, Hanley JA, Kezouh A, Suissa S. Problem of immortal time bias in cohort studies: example using statins and cancer. *BMJ*. 2010;340:b5087. PMID: 20085988

### HIGH-3: GEE — Exchangeable Correlation Assumed

**Current implementation:** `fit_mortality_model()` uses GEE with exchangeable correlation and hospital as cluster.

**Acceptable** for this design, but should:
1. Report the working correlation estimate
2. Always use robust (sandwich) standard errors (verify this is enabled)
3. Cite:

> Liang KY, Zeger SL. Longitudinal data analysis using generalized linear models. *Biometrika*. 1986;73(1):13-22.

> Zeger SL, Liang KY. Longitudinal data analysis for discrete and continuous outcomes. *Biometrics*. 1986;42(1):121-130. PMID: 3719049

### MEDIUM-1: Meta-Analysis — DerSimonian-Laird Has Known Bias

**Current implementation:** DerSimonian-Laird (DL) random-effects estimator for tau-squared.

**Problem:** DL underestimates tau-squared with few studies (<20) and can produce overly narrow confidence intervals. With a multicenter CLIF consortium, the number of sites may be small.

**Recommendation:** Use the **restricted maximum likelihood (REML)** or **Paule-Mandel** estimator as the primary analysis, with DL as sensitivity. The Hartung-Knapp-Sidik-Jonkman (HKSJ) confidence interval adjustment is also recommended.

**Citations:**
> DerSimonian R, Laird N. Meta-analysis in clinical trials. *Control Clin Trials*. 1986;7(3):177-188. PMID: 3802833

> Veroniki AA, et al. Methods to estimate the between-study variance and its uncertainty in meta-analysis. *Res Synth Methods*. 2016;7(1):55-79. PMID: 26332144

> IntHout J, Ioannidis JP, Borm GF. The Hartung-Knapp-Sidik-Jonkman method for random effects meta-analysis is straightforward and considerably outperforms the standard DerSimonian-Laird method. *BMC Med Res Methodol*. 2014;14:25. PMID: 24548571

### MEDIUM-2: Logit Transformation for Proportions — Correct, Needs Citation

**Current implementation:** Logit-transforms proportions before pooling, back-transforms with delta method.

**Correct approach** — avoids pooled estimates outside [0,1]. Cite:

> Barendregt JJ, et al. Meta-analysis of prevalence. *J Epidemiol Community Health*. 2013;67(11):974-978. PMID: 23963506

### MEDIUM-3: BH FDR — Scope and Reporting

**Current implementation:** Applies BH correction across all models (6 SAT + 3 SBT definitions × 4 models = 36 tests).

**Questions to address in manuscript:**
1. Is the correction applied per-outcome or globally across all 36 tests?
2. Report both raw p-values and BH-adjusted q-values
3. State the FDR threshold (typically q < 0.05)

**Citation:**
> Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *J R Stat Soc Series B*. 1995;57(1):289-300.

### MEDIUM-4: Cohen's Kappa — Threshold Interpretation

**Current implementation:** Computes kappa for EHR phenotype vs. chart review concordance.

**Recommendation:** Report with Landis-Koch interpretation scale and confidence intervals.

**Citations:**
> Cohen J. A coefficient of agreement for nominal scales. *Educ Psychol Meas*. 1960;20(1):37-46.

> Landis JR, Koch GG. The measurement of observer agreement for categorical data. *Biometrics*. 1977;33(1):159-174. PMID: 843571

### LOW-1: SOFA Scoring

**Citation:**
> Vincent JL, et al. The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure. *Intensive Care Med*. 1996;22(7):707-710. PMID: 8844239

> Seymour CW, et al. Assessment of clinical criteria for sepsis: for the Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*. 2016;315(8):762-774. PMID: 26903335. DOI: 10.1001/jama.2016.0288

---

## 3. Essential Manuscript Citations

### Clinical Context
| Citation | Use |
|----------|-----|
| Girard TD, et al. *Lancet*. 2008;371(9607):126-134. PMID: 18191684 | ABC trial — foundational SAT+SBT RCT |
| Rojas JC, et al. *Intensive Care Med*. 2025. PMID: 40080116 | CLIF 2.1 consortium paper |
| Rojas JC, et al. *medRxiv*. 2024. PMID: 39281737 | CLIF preprint (for data model details) |

### Statistical Methods
| Citation | Use |
|----------|-----|
| Schoenfeld & Bernard. *Crit Care Med*. 2002. PMID: 12163791 | VFD definition/convention |
| Renard Triché et al. *Crit Care*. 2025. PMID: 40537834 | VFD analysis methods comparison |
| Suissa S. *Am J Epidemiol*. 2008. PMID: 18056625 | Immortal time bias avoidance |
| Liang & Zeger. *Biometrika*. 1986. | GEE methodology |
| DerSimonian & Laird. *Control Clin Trials*. 1986. PMID: 3802833 | Random-effects meta-analysis |
| Benjamini & Hochberg. *J R Stat Soc B*. 1995. | FDR correction |
| Cohen J. *Educ Psychol Meas*. 1960. | Kappa statistic |
| Fine & Gray. *JASA*. 1999. | Competing risks (if adopted) |
| Vincent et al. *Intensive Care Med*. 1996. PMID: 8844239 | SOFA score |

---

## 4. Priority Action Items

| Priority | Action | Effort |
|----------|--------|--------|
| **CRITICAL** | Add proportional odds model for VFDs as primary or sensitivity analysis | 2-3 hours |
| **CRITICAL** | Fix Cox competing risk: implement cause-specific hazard (two Cox models) or Fine-Gray | 2 hours |
| HIGH | Add all citations above to manuscript Methods section | 1 hour |
| HIGH | Verify GEE uses robust sandwich SEs; report working correlation | 30 min |
| MEDIUM | Switch meta-analysis to REML + HKSJ CI; keep DL as sensitivity | 1-2 hours |
| MEDIUM | Clarify BH correction scope in Methods text | 15 min |
| LOW | Add kappa CIs and Landis-Koch interpretation | 15 min |

---

## 5. Suggested Methods Paragraph (for manuscript)

> **Statistical Analysis.** Associations between EHR-phenotyped SAT/SBT delivery and outcomes were estimated using four models. Time to extubation was modeled using Cox proportional hazards with delivery as a time-varying exposure in counting-process format to avoid immortal time bias [Suissa, 2008]; death was treated as a competing event using cause-specific hazards [Fine & Gray, 1999]. Ventilator-free days (VFDs) to day 28, defined per convention [Schoenfeld & Bernard, 2002], were analyzed using proportional odds ordinal regression given recent evidence that hurdle and zero-inflated models exhibit inflated Type I error for VFD outcomes [Renard Triché et al., 2025]. ICU length of stay was modeled using negative binomial regression. In-hospital mortality was estimated using generalized estimating equations (GEE) with logit link, exchangeable working correlation, and robust standard errors to account for hospital-level clustering [Liang & Zeger, 1986]. All models adjusted for age, sex, race/ethnicity, illness severity (SOFA [Vincent et al., 1996]), hemodynamic support, sedation exposure, FiO2, and PEEP. EHR phenotype concordance was assessed using Cohen's kappa [Cohen, 1960]. Across-site effect estimates were pooled using random-effects meta-analysis with restricted maximum likelihood estimation [DerSimonian & Laird, 1986; Veroniki et al., 2016] on the logit scale [Barendregt et al., 2013]. Multiplicity was addressed using the Benjamini-Hochberg procedure at FDR < 0.05 [Benjamini & Hochberg, 1995].
