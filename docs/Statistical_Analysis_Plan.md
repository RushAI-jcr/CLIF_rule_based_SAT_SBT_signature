# Statistical Analysis Plan
## Phenotyping Protocolized Liberation from Sedation and Invasive Mechanical Ventilation in the Electronic Health Record

**Prepared for:** Biostatistician Audit and Review
**Date:** February 24, 2026
**Version:** 1.0
**Principal Investigators:** Snigdha Jain, MD, MHS; Juan C. Rojas, MD
**Consortium:** CLIF (Common Longitudinal ICU Format) Consortium
**Target Journal:** Intensive Care Medicine

---

## 1. Purpose and Scope

### 1.1 Study Purpose

This study develops and validates **computable phenotypes** for two foundational clinical processes in mechanically ventilated ICU patients:

1. **Spontaneous Awakening Trials (SATs):** protocolized interruption of continuous sedative infusions to assess patient wakefulness
2. **Spontaneous Breathing Trials (SBTs):** protocolized reduction of ventilatory support to assess readiness for extubation

These phenotypes are derived exclusively from structured EHR data (medications, ventilator settings, vital signs, laboratory values) and are designed to be applied uniformly across health systems using the CLIF data model.

### 1.2 Why a Statistical Analysis Plan Is Needed

The statistical complexity of this study is non-trivial for three reasons:

1. **Multi-level data structure:** Observations are ventilator-days nested within ventilation episodes, nested within hospitalizations, nested within patients, nested within hospitals. Ignoring this clustering inflates Type I error and produces artificially narrow confidence intervals.
2. **Dual validation framework:** The study employs both criterion validity (phenotype vs. flowsheet documentation) and construct validity (association with clinical outcomes), each requiring distinct analytic approaches.
3. **Hospital-level benchmarking:** Risk-standardized rate estimation requires careful hierarchical modeling to separate true hospital-level variation from noise, and the choice of summary metrics (e.g., median odds ratio) has direct implications for policy interpretation.

A prespecified, transparent SAP is essential for:
- Ensuring reproducibility across the federated CLIF consortium
- Preventing data-driven analytic decisions (P-hacking)
- Satisfying ICM reporting standards and the RECORD-PE extension for pharmacoepidemiology/EHR studies
- Providing an auditable chain of reasoning for each statistical decision

### 1.3 Scope

This SAP covers:
- Unit of analysis definitions and cohort construction
- Descriptive statistics
- Criterion validity analysis (diagnostic accuracy metrics with clustered data)
- Construct validity analysis (outcome association models)
- Hospital-level variation and benchmarking
- Missing data strategy
- Sensitivity analyses
- Multiplicity considerations

---

## 2. Study Design and Data Architecture

### 2.1 Design
Retrospective, multi-center observational cohort study using harmonized structured EHR data from the CLIF consortium and MIMIC-IV.

### 2.2 Hierarchical Data Structure

```
Level 5: Health System (k systems)
  Level 4: Hospital (j hospitals within systems)
    Level 3: Patient / Hospitalization (i patients)
      Level 2: Ventilation Episode (e episodes per patient)
        Level 1: Ventilator-Day (d days per episode)
```

**Rationale:** Correctly modeling this hierarchy is critical. Ventilator-days within the same episode are correlated (same patient, same illness trajectory). Episodes within the same hospitalization share patient-level confounders. Hospitals share institutional practices and culture. Failure to account for any level produces biased variance estimates.

### 2.3 Unit of Analysis

| Analysis | Primary Unit | Rationale |
|----------|-------------|-----------|
| Phenotype eligibility/delivery | Ventilator-day | SAT/SBT decisions are made daily; anchored at 06:00 to match clinical workflow |
| Criterion validity | Ventilator-day (bootstrapped at episode level) | Flowsheet entries are day-level; clustering at episode level accounts for within-patient correlation |
| Construct validity (time-to-event) | Ventilator-day (time-varying exposure) | SAT/SBT exposure varies day-to-day; Cox models with time-varying covariates appropriately handle this |
| Construct validity (episode-level outcomes) | Ventilation episode | VFDs, mortality, ICU LOS are episode/hospitalization-level outcomes |
| Hospital benchmarking | Ventilator-day (aggregated to hospital) | Risk-standardized rates require patient-day-level modeling with hospital random effects |

### 2.4 Ventilator-Day Definition

A ventilator-day is defined as a 24-hour period from 06:00 to 05:59 the following day. Eligibility assessment windows span 22:00 the prior calendar day to 06:00 on the index day.

**Rationale:** This anchoring aligns with real-world ICU workflow where SAT/SBT screening occurs during morning rounds. The overnight assessment window (22:00-06:00) captures the most recent physiologic state prior to the clinical decision point, minimizing:
- **Immortal time bias:** By assessing eligibility before the decision point, we avoid conditioning on future events
- **Reverse causation:** Physiologic changes occurring *after* a trial (e.g., hemodynamic instability during an SBT) do not contaminate the eligibility assessment

### 2.5 Ventilation Episode Definition

IMV episodes are defined as contiguous periods of ventilator-mode/setting documentation, with a gap of >72 hours of absent ventilator support distinguishing separate episodes. Minimum IMV duration: 6 hours.

**Rationale:** The 72-hour gap threshold is consistent with prior critical care literature (e.g., WEAN SAFE, Pham et al., *Lancet Respir Med* 2023) and prevents brief interruptions (e.g., for transport or procedures) from splitting a single clinical episode. The 6-hour minimum excludes peri-procedural ventilation that is not relevant to the weaning paradigm.

---

## 3. Descriptive Statistics

### 3.1 Cohort Characterization

Patient characteristics will be summarized at the **ventilator-day level** for the overall cohort and stratified by:
- SAT eligibility (eligible vs. not eligible)
- SBT eligibility (eligible vs. not eligible)
- SAT delivery (delivered vs. not delivered, among eligible)
- SBT delivery (delivered vs. not delivered, among eligible)

### 3.2 Reporting Conventions

| Variable Type | Summary Measure |
|--------------|----------------|
| Continuous, normally distributed | Mean (SD) |
| Continuous, skewed | Median [IQR] |
| Categorical | n (%) |

**Rationale:** ICU data are typically right-skewed (e.g., LOS, vasopressor doses), making median/IQR the preferred central tendency measure. We do not perform formal statistical testing of baseline characteristics between groups (e.g., eligible vs. not eligible) because: (a) these are observational, non-randomized groups where differences are expected and clinically meaningful, and (b) P-values for baseline differences have been criticized as uninformative in observational settings (Senn, *Stat Med* 1994; Austin, *Stat Med* 2007).

### 3.3 Data Completeness

We will report completeness of each required data element (ventilator mode, FiO2, PEEP, SpO2, medication administration records, RASS, hemodynamics) by hospital and across the study period. This will be visualized as a heatmap (hospitals x data elements) with completeness thresholds color-coded.

---

## 4. Criterion Validity Analysis

### 4.1 Reference Standard

Structured flowsheet documentation by nurses and/or respiratory therapists serves as the reference standard. We acknowledge this is an **imperfect reference standard** because:
- Flowsheet entries are subject to documentation errors, delays, and omissions
- Documentation practices vary across hospitals (Table S2)
- Not all hospitals have structured SAT/SBT flowsheet fields

This motivates the dual-validation approach (criterion + construct validity) rather than relying on criterion validity alone.

### 4.2 Diagnostic Accuracy Metrics

For each phenotype (SAT eligibility, SAT delivery, SBT eligibility, SBT delivery), we will construct a 2x2 confusion matrix at the ventilator-day level and report:

| Metric | Formula | Interpretation in Context |
|--------|---------|--------------------------|
| Sensitivity (Recall) | TP / (TP + FN) | Proportion of flowsheet-documented SAT/SBT days correctly identified by phenotype |
| Specificity | TN / (TN + FP) | Proportion of flowsheet-negative days correctly identified |
| Positive Predictive Value (Precision) | TP / (TP + FP) | Proportion of phenotype-positive days confirmed by flowsheet |
| Negative Predictive Value | TN / (TN + FN) | Proportion of phenotype-negative days confirmed by flowsheet |
| Accuracy | (TP + TN) / N | Overall concordance |
| F1 Score | 2 * (PPV * Sensitivity) / (PPV + Sensitivity) | Harmonic mean balancing precision and recall |

### 4.3 Confidence Intervals via Cluster Bootstrap

**Method:** 95% confidence intervals for all diagnostic accuracy metrics will be estimated via **BCa (bias-corrected and accelerated) cluster bootstrap resampling at the ventilation-episode level** (1,000 replicates; 2,000 if computationally feasible). In each bootstrap replicate, entire ventilation episodes are resampled with replacement, and metrics are computed from all ventilator-days within the resampled episodes.

**Rationale:** Standard binomial confidence intervals (e.g., Clopper-Pearson, Wilson) assume independent observations. Ventilator-days within the same episode are correlated (same patient, same illness trajectory, autocorrelated medication and ventilator settings). Ignoring this clustering underestimates variance and produces artificially narrow CIs. The cluster bootstrap is a well-established, distribution-free approach for this problem (Obuchowski, *Acad Radiol* 2000; Defined & Defined, *Radiology* 2012). Resampling at the episode level preserves within-episode correlation structure.

**Alternative considered and rejected:** Generalized estimating equations (GEE) with episode-level clustering could also produce valid CIs, but the bootstrap is preferred here because: (1) it naturally accommodates the simultaneous estimation of multiple metrics from the same confusion matrix, (2) it does not require distributional assumptions, and (3) it is straightforward to implement in a federated analysis.

### 4.4 Site-Stratified Performance

We will report diagnostic accuracy metrics stratified by hospital to assess phenotype portability. Heterogeneity across sites will be assessed visually (forest plots) and quantified with I-squared and Cochran's Q statistic from a random-effects meta-analysis of site-specific sensitivities and specificities.

**Rationale:** A phenotype that performs well on average but poorly at specific sites has limited utility for a federated consortium. Site stratification identifies hospitals where local documentation practices or data quality may require phenotype refinement.

---

## 5. Construct Validity Analysis

### 5.1 Rationale for Construct Validity

Because flowsheet documentation is an imperfect gold standard, we augment criterion validity with construct validity: testing whether phenotype-identified SAT/SBT delivery is associated with the clinical outcomes expected from the prior trial literature. Specifically, we hypothesize:

| Outcome | Expected Direction | Supporting Evidence |
|---------|-------------------|---------------------|
| Time to successful extubation | Shorter with SAT/SBT | Girard et al., *Lancet* 2008 (ABC trial); Ely et al., *NEJM* 1996 |
| Ventilator-free days (VFDs) to day 28 | More with SAT/SBT | Kress et al., *NEJM* 2000; Klompas et al., *AJRCCM* 2015 |
| ICU length of stay | Shorter with SAT/SBT | Hsieh et al., *CCM* 2019; Pun et al., *CCM* 2019 |
| In-hospital mortality | Lower with SAT/SBT | Girard et al., *Lancet* 2008 |
| Awakening (RASS 0 to +1) | More frequent with SAT | Kress et al., *NEJM* 2000 |

If phenotype-identified delivery is associated with these outcomes in the expected direction, this provides evidence that the phenotype captures a clinically meaningful construct, even if imperfectly aligned with flowsheet documentation.

### 5.2 Time to Extubation: Cause-Specific Cox Proportional Hazards Model

**Model:**
$$h_k(t | X(t)) = h_{0k}(t) \cdot \exp(\beta_1 \cdot \text{SAT}_{it} + \beta_2 \cdot \text{SBT}_{it} + \boldsymbol{\gamma}' \mathbf{Z}_i + u_j)$$

where $k$ indexes cause (extubation vs. death), $\text{SAT}_{it}$ and $\text{SBT}_{it}$ are time-varying indicators of delivery on day $t$ for patient $i$, $\mathbf{Z}_i$ are baseline and time-varying confounders, and $u_j$ is a hospital random effect.

**Key Design Decisions:**

1. **Time-varying exposure:** SAT/SBT exposure changes day-to-day. Treating it as a baseline (ever/never) variable would introduce immortal time bias because patients must survive and remain ventilated long enough to receive the trial. Time-varying Cox models correctly handle this by updating exposure status at each event time.

2. **Death as competing risk:** Patients who die cannot be extubated. Ignoring death violates the independent censoring assumption of standard Cox models. We use a **cause-specific hazards** approach rather than Fine-Gray subdistribution hazards because:
   - Cause-specific hazards have a direct causal interpretation (the instantaneous rate of extubation among those still at risk)
   - Fine-Gray estimates are useful for prediction but conflate the effect of the exposure on both the event of interest and the competing event
   - The cause-specific approach is recommended for etiologic research (Lau et al., *BMC Med Res Methodol* 2009; Austin & Fine, *Stat Med* 2017)

3. **Robust standard errors:** Clustered at the hospitalization level (sandwich estimator) to account for potential within-patient correlation across ventilator-days.

4. **Hospital random intercepts:** Account for unmeasured between-hospital differences in extubation practices and case-mix.

**Confounder adjustment:**
- Demographics: age, sex, race/ethnicity, BMI
- Comorbidity: Elixhauser or Charlson score (as available in CLIF)
- Illness severity on the prior day: FiO2, PEEP, vasopressor dose, sedation depth (RASS), GCS (if available)
- Admission diagnosis category (medical vs. surgical)

**Proportional hazards assumption:** Will be tested using Schoenfeld residual plots and a global test. If violated for key covariates, we will use time-stratified models or include time-covariate interaction terms.

**Sensitivity analysis:** Fine-Gray subdistribution hazard models will be fit as a sensitivity analysis to estimate the cumulative incidence of extubation accounting for the competing risk of death. Note: this Fine-Gray sensitivity analysis for time-to-extubation (Section 5.2) is distinct from the Fine-Gray competing risk analysis for VFDs (Section 5.3). The former asks "does SAT/SBT change the hazard of extubation?" while the latter asks "what is the cumulative probability of being extubated alive, accounting for death?"

### 5.3 Ventilator-Free Days (VFDs) to Day 28

VFDs are defined as the number of days alive and free from mechanical ventilation in the first 28 days after the start of the ventilation episode. Patients who die receive 0 VFDs regardless of prior ventilator-free time.

**Statistical Challenges with VFDs:**

VFDs are a composite outcome with structural zeros (death) that violate assumptions of ZINB, hurdle, and two-part models. The 2025 simulation study by Renard Triché et al. (*Critical Care* 2025) demonstrated that count-based models (ZINB, hurdle, two-part) fail to adequately control Type I error for VFDs because:
- A **point mass at zero** conflates two distinct clinical states: death (structural zeros) and prolonged ventilation (observed zeros)
- The zeros are **not latent** — they arise from known, observable clinical mechanisms
- Count-based models impose distributional assumptions that are violated by the composite nature of VFDs

**Primary Model: Fine-Gray Competing Risk Regression**

Time to extubation alive is modeled as the event of interest, with death before extubation as the competing event. The Fine-Gray subdistribution hazard model estimates the cumulative incidence of successful extubation while properly accounting for the competing risk of death.

$$\lambda^{sub}_1(t | X) = \lambda^{sub}_{10}(t) \cdot \exp(\beta_1 \cdot \text{SAT}_i + \beta_2 \cdot \text{SBT}_i + \boldsymbol{\gamma}' \mathbf{Z}_i)$$

**Rationale:** Subdistribution hazard ratios are familiar to ICM reviewers and have a direct interpretation in terms of cumulative incidence. This approach avoids the problematic distributional assumptions of count-based models for VFDs. R implementation: `cmprsk` package.

**Secondary Model: Multistate Model**

A multistate model capturing the full clinical trajectory: mechanical ventilation → extubated → reintubated → dead. This model captures reintubation cycles and provides transition-specific hazard ratios.

**Rationale:** Endorsed by Renard Triché et al. (2025) as the optimal approach for VFD-type outcomes. Published precedent in ICM (2010) and AJRCCM (2024: "Improving the Reporting of Trials Evaluating Organ Support Therapies Using Multistate Modeling"). R implementation: `mstate` package.

**Sensitivity Models:**
- Mann-Whitney U test (non-parametric comparison)
- Proportional odds model (ordinal logistic regression)
- Two-part (hurdle) model: Part 1 logistic for P(VFD > 0), Part 2 truncated negative binomial for E(VFD | VFD > 0)
- Zero-inflated negative binomial (ZINB) — included for continuity with manuscript draft

**Mandatory Component Reporting:**

Alongside any VFD analysis, the following must be reported separately:
- Proportion extubated alive by day 28
- Proportion who died before extubation
- Proportion reintubated
- Cumulative incidence function (CIF) plots (NOT Kaplan-Meier, which overestimates event probabilities in the presence of competing risks)
- Transition probability plots from the multistate model

### 5.4 ICU Length of Stay

**Model:** Mixed-effects negative binomial regression with hospital random intercepts.

**Rationale:** ICU LOS is a right-skewed count variable. Negative binomial regression accommodates overdispersion relative to Poisson regression. Mixed effects account for hospital-level clustering. We do not use linear regression on log-transformed LOS because: (1) back-transformation bias (smearing estimator is required), and (2) zero LOS values cannot be log-transformed.

### 5.5 In-Hospital Mortality

**Model:** Mixed-effects logistic regression with hospital random intercepts.

**Rationale:** Standard approach for binary outcomes in clustered data. Random intercepts allow hospital-specific baseline mortality rates while estimating the average exposure-outcome association.

### 5.6 Paired SAT+SBT Analysis

We will separately evaluate the association of **paired SAT and SBT delivery** (both on the same ventilator-day) with outcomes, as the ABC trial (Girard et al., *Lancet* 2008) demonstrated synergistic benefit of the paired approach. This will be modeled as a 4-level categorical time-varying exposure: neither, SAT only, SBT only, both.

---

## 6. Hospital-Level Variation and Benchmarking

### 6.1 Risk-Standardized Delivery Rates

**Model:** Hierarchical logistic regression at the ventilator-day level:

$$\text{logit}(P(\text{SAT delivered}_{ijd})) = \boldsymbol{\beta}' \mathbf{X}_{ij} + u_j$$

where $\mathbf{X}_{ij}$ are patient-level covariates and $u_j \sim N(0, \sigma^2_u)$ is the hospital random intercept. A nested random intercept for patient within hospital accounts for within-patient clustering of eligible days.

**Risk-standardized rate for hospital $j$:**

$$\text{RSR}_j = \frac{\text{Predicted}_j}{\text{Expected}_j} \times \text{Overall Rate}$$

where Predicted$_j$ uses hospital $j$'s random effect and Expected$_j$ uses the average random effect (0), applied to hospital $j$'s actual case-mix.

**Rationale:** This indirect standardization approach is the standard methodology used by CMS for hospital quality measures (Krumholz et al., *JACC* 2006) and allows fair comparison across hospitals with different patient populations.

### 6.2 Quantifying Variation: Median Odds Ratio (MOR)

$$\text{MOR} = \exp\left(\sqrt{2\sigma^2_u} \times \Phi^{-1}(0.75)\right) \approx \exp(0.6745 \times \sqrt{2\sigma^2_u})$$

The MOR represents the median odds ratio of SAT/SBT delivery for two patients with identical characteristics admitted to two randomly chosen hospitals — specifically, the hospital with higher propensity vs. the one with lower propensity.

**Interpretation:** An MOR of 1.0 indicates no between-hospital variation after risk adjustment. An MOR of 2.0 would mean that, for two otherwise identical patients, the one at the higher-performing hospital has twice the odds of receiving an SAT/SBT.

**Rationale:** The MOR is preferred over the intraclass correlation coefficient (ICC) for binary outcomes because the ICC is sensitive to outcome prevalence and has no direct clinical interpretation, whereas the MOR is on the familiar odds ratio scale (Merlo et al., *J Epidemiol Community Health* 2006; Larsen & Merlo, *J Epidemiol Community Health* 2005). The MOR has been widely adopted for quantifying hospital variation in ICU outcomes (e.g., Sjoding et al., *Crit Care Med* 2016).

### 6.3 Visualization

- **Caterpillar plots:** Hospital-specific risk-standardized rates with 95% CIs, ordered by point estimate, grouped by health system
- **Funnel plots:** Risk-standardized rates vs. volume (number of eligible ventilator-days), with 95% and 99.8% control limits to identify outliers

### 6.4 Concordance Between EHR Phenotype and Flowsheet-Derived Rates

For hospitals with flowsheet data, we will assess agreement between phenotype-derived and flowsheet-derived risk-standardized rates using:
- Pearson and Spearman correlation coefficients
- Bland-Altman plots (mean difference and limits of agreement)
- Concordance correlation coefficient (CCC; Lin, *Biometrics* 1989)

**Rationale:** Correlation alone is insufficient — two measures can be highly correlated but systematically biased. The Bland-Altman approach and CCC explicitly assess both correlation and agreement.

---

## 7. Missing Data Strategy

### 7.1 Characterization

Missing data patterns will be reported by:
- Data element (ventilator mode, FiO2, PEEP, SpO2, medication records, RASS, vasopressor dose)
- Hospital
- Time period (to identify temporal trends in documentation)

### 7.2 Approach

| Scenario | Approach | Rationale |
|----------|----------|-----------|
| Ventilator mode missing | Day excluded from SBT eligibility assessment | Cannot determine controlled vs. spontaneous mode; inclusion would introduce misclassification |
| Medication records missing | Day excluded from SAT eligibility assessment | Cannot determine sedative/opioid exposure |
| FiO2 or PEEP missing | Last observation carried forward (LOCF) within a 6-hour window; otherwise excluded | Ventilator settings are typically documented every 1-4 hours; brief gaps likely reflect unchanged settings |
| RASS missing | Primary algorithms run without RASS component; RASS-enhanced algorithms restricted to days with documentation | RASS is not universally documented; primary algorithms should not depend on it |
| SpO2, hemodynamics missing | LOCF within 2-hour window; otherwise excluded | Vital signs may have brief gaps between nursing assessments |

### 7.3 Sensitivity Analysis for Completeness

We will repeat primary analyses restricted to hospitals/time periods meeting prespecified completeness thresholds:
- Ventilator settings: >=90% of ventilator-days with mode documented
- Medication administration records: >=90% of ventilator-days with MAR data

**Rationale:** If results are robust to restriction to high-completeness sites, this supports the generalizability of the phenotype even in settings with imperfect documentation.

---

## 8. Sensitivity Analyses

### 8.1 Alternative Phenotype Thresholds

| Parameter | Primary | Sensitivity | Rationale |
|-----------|---------|-------------|-----------|
| SAT interruption duration | >= 30 min | >= 15 min, >= 60 min | Prior protocols vary; 30 min reflects ABC trial |
| SBT performance duration | >= 2 min | >= 5 min, >= 30 min | 2 min captures initiation; 30 min captures completed trial |
| SBT pressure support threshold | PS <= 8 cmH2O | PS <= 10, PS <= 5 | Guidelines vary; higher thresholds more inclusive |
| SBT CPAP threshold | CPAP <= 8 cmH2O | CPAP <= 5 | Stricter definition of minimal support |

### 8.2 Population Restrictions

| Restriction | Rationale |
|-------------|-----------|
| Exclude day 0 and day 1 of IMV | SAT/SBT typically not indicated on intubation day or immediately after |
| Exclude cardiac arrest admissions | Targeted temperature management precludes SAT/SBT |
| Exclude comfort care days | Palliative goals preclude liberation trials |
| Exclude targeted temperature management days | Neurocritical care protocol supersedes SAT |
| First IMV episode only | Eliminates within-admission correlation and heterogeneity of re-intubated patients |

### 8.3 Alternative VFD Analysis

The VFD analysis hierarchy is structured as follows:

| Role | Method | Rationale |
|------|--------|-----------|
| **Primary** | Fine-Gray competing risk regression | Subdistribution HRs; avoids count-model assumptions |
| **Secondary** | Multistate model (MV → extubated → reintubated → dead) | Full trajectory; endorsed by Renard Triché 2025 |
| **Sensitivity** | Mann-Whitney U, proportional odds, two-part hurdle, ZINB | Robustness; ZINB for manuscript continuity |

**Rationale:** The 2025 simulation study (Renard Triché et al., *Critical Care*) demonstrated that competing risk and multistate approaches have superior operating characteristics compared to count-based models for VFDs. Sensitivity analyses using Mann-Whitney U, proportional odds, two-part hurdle, and ZINB are included to demonstrate robustness across analytic frameworks.

---

## 9. Multiplicity Considerations

This study is primarily descriptive and hypothesis-generating (phenotype development and validation) rather than confirmatory hypothesis-testing. Therefore:

- **We will not apply formal multiplicity corrections** (e.g., Bonferroni) to the construct validity analyses
- We will clearly label all analyses as prespecified vs. exploratory
- We will emphasize effect estimates and confidence intervals over P-values
- Sensitivity analyses are explicitly exploratory and intended to assess robustness, not to generate new hypotheses

**Rationale:** Multiplicity adjustment in the context of a validation study with multiple complementary analyses would be overly conservative and counterproductive. The goal is not to test a single confirmatory hypothesis but to build a comprehensive body of evidence supporting (or refuting) phenotype validity. This approach aligns with recommendations from Rothman (*Epidemiology* 1990) and the ASA Statement on P-values (Wasserstein & Lazar, *Am Stat* 2016).

---

## 10. Software and Reproducibility

- **Statistical software:** [To be specified — likely R or Python given CLIF consortium infrastructure]
- **Key packages:**
  - Hierarchical models: `lme4` (R) or `statsmodels` (Python)
  - Survival analysis: `survival` + `coxme` (R) for frailty models
  - Bootstrap: custom implementation preserving cluster structure
  - Competing risks: `cmprsk` (Fine-Gray), `mstate` (multistate model)
  - Two-part models: `glmmTMB` (R) or custom implementation
  - Visualization: `ggplot2` (R)
- **Code repository:** All analysis code will be version-controlled and shared via the CLIF consortium repository
- **Federated execution:** Phenotype algorithms will be executed locally at each site; only aggregate (de-identified) results will be pooled

---

## 11. Assessment of Current Manuscript Draft

### 11.1 Points of Agreement

The manuscript draft's statistical methods section is largely sound and well-conceived. I agree with:

1. **Ventilator-day as the primary unit** with 06:00 anchoring — appropriate and well-justified
2. **Overnight eligibility window (22:00-06:00)** — correctly minimizes immortal time and reverse causation bias
3. **Cluster bootstrap at ventilation-episode level** for diagnostic accuracy CIs — methodologically appropriate
4. **Cox proportional hazards with time-varying exposure** for time to extubation — correctly addresses immortal time bias
5. **Mixed-effects models with hospital random intercepts** — appropriately handles clustering
6. **Robust standard errors clustered at hospitalization level** — accounts for within-patient correlation
7. **Median odds ratio** for quantifying hospital variation — excellent choice over ICC
8. **Prespecified confounder set** based on prior literature — transparent and defensible
9. **Dual validation framework** (criterion + construct) — essential given imperfect reference standard

### 11.2 Recommendations for Revision

| Issue | Current Draft | Recommendation | Rationale |
|-------|--------------|----------------|-----------|
| **VFD modeling** | "Zero-inflated negative binomial" | Primary: Fine-Gray competing risk; secondary: multistate model; ZINB/hurdle/Mann-Whitney/prop odds as sensitivity | Renard Triché et al. (Crit Care 2025) simulation study shows count-based models fail Type I error control for VFDs. Competing risk and multistate approaches are methodologically superior. See Section 5.3. |
| **Mixed-effects ZINB for VFDs and ICU LOS** | Draft lists "mixed-effects zero-inflated negative binomial for VFDs and ICU LOS" interchangeably | Separate models: Fine-Gray/multistate for VFDs, standard negative binomial for ICU LOS | VFDs and ICU LOS have fundamentally different distributional properties; VFDs have death as a competing risk requiring competing risk/multistate methods; ICU LOS does not |
| **Table 5 VFD effect measure** | "VFD IRR (95% CI)" implies count model | Change to "VFD sHR (95% CI)" for subdistribution hazard ratio from Fine-Gray | IRR is inappropriate when primary model is Fine-Gray competing risk regression |
| **Mortality model inconsistency** | Draft mentions both "logistic regression" and "multivariable mixed-effects regression" for mortality | Clarify as mixed-effects logistic regression | Internal consistency |
| **Competing risk specification** | Draft says "Cox proportional hazards with death as a competing event" | Specify cause-specific hazards approach (not Fine-Gray) and justify | See Section 5.2 for rationale |
| **Proportional hazards testing** | Not mentioned | Add Schoenfeld residual testing plan | Required for Cox model validity |
| **Missing data** | Draft says "** What did we do with missing data**" (placeholder) | Specify the strategy per Section 7 | Critical gap that must be addressed |
| **SBT performance minimum duration** | >= 2 minutes | Acceptable as primary, but justify this liberal threshold | 2 minutes captures trial initiation but not completion; consider >= 5 min as co-primary and >= 30 min as sensitivity |
| **Concordance metrics** | "Correlation and agreement metrics" (unspecified) | Specify Pearson, Spearman, CCC, and Bland-Altman | Vague as written; needs specificity for reproducibility |
| **Bootstrap replicates** | Not specified | Specify 1,000 replicates (minimum; 2,000 preferred) | Standard practice; must be prespecified |

### 11.3 Items Requiring Clarification in the Draft — RESOLVED

All manuscript placeholders have been resolved (February 24, 2026):

1. **IMV episode gap threshold:** Confirmed as 72 hours, consistent with WEAN SAFE (Pham et al., *Lancet Respir Med* 2023)
2. **SBT hemodynamic stability window:** Any 2 consecutive hours of stability within the 10PM–6AM eligibility window
3. **SAT RASS awakening window:** Confirmed as 45 minutes (RASS 0 to +1 within 45 min of sedation discontinuation)
4. **SBT controlled mode list:** AC-VC, AC-PC, PRVC. SIMV excluded (semi-spontaneous component makes it inappropriate as a "controlled" baseline)
5. **SAT sensitivity durations:** ≥15 min and ≥60 min (primary: ≥30 min)
6. **SBT sensitivity thresholds:** PS ≤10, PS ≤5, CPAP ≤5 (primary: PS ≤8, CPAP ≤8)
7. **Missing data:** Resolved per Section 7 (LOCF with defined windows; exclude days with missing mode/meds; restrict to ≥90% completeness sites as sensitivity)

---

## 12. Summary of Statistical Methods by Analysis

| Analysis | Model | Clustering | Key Features |
|----------|-------|------------|--------------|
| Criterion validity | 2x2 confusion matrix | Cluster bootstrap (episode-level) | Sensitivity, specificity, PPV, NPV, F1; 95% CI |
| Time to extubation | Cause-specific Cox PH | Hospital random effect + robust SE (hospitalization) | Time-varying SAT/SBT exposure; death as competing risk |
| VFDs (28-day) | Fine-Gray competing risk (primary); multistate model (secondary) | Hospital frailty / stratification | CIF plots; transition probabilities; mandatory component reporting |
| ICU LOS | Mixed-effects NB regression | Hospital random intercepts | Robust SE at hospitalization level |
| Mortality | Mixed-effects logistic regression | Hospital random intercepts | Robust SE at hospitalization level |
| Hospital variation | Hierarchical logistic regression | Hospital + patient random intercepts | Risk-standardized rates; MOR; caterpillar & funnel plots |
| Phenotype-flowsheet agreement | Correlation + Bland-Altman | At hospital level | Pearson, Spearman, CCC |

---

## 13. References Supporting Statistical Decisions

1. Obuchowski NA. Bootstrap estimation of diagnostic accuracy with patient-clustered data. *Acad Radiol*. 2000;7(10):849-854. — Cluster bootstrap for diagnostic accuracy
2. Defined & Defined. Methods for calculating sensitivity and specificity of clustered data: a tutorial. *Radiology*. 2012;265(2):331-340. — Clustered diagnostic accuracy methodology
3. Austin PC, Fine JP. Practical recommendations for reporting Fine-Gray model analyses for competing risk data. *Stat Med*. 2017;36(27):4391-4400. — Competing risks methodology
4. Merlo J, et al. A brief conceptual tutorial of multilevel analysis in social epidemiology: using measures of clustering in multilevel logistic regression to investigate contextual phenomena. *J Epidemiol Community Health*. 2006;60(4):290-297. — MOR methodology
5. Krumholz HM, et al. An administrative claims measure suitable for profiling hospital performance based on 30-day mortality rates among patients with an acute myocardial infarction. *Circulation*. 2006;113(13):1683-1692. — Risk-standardized rate methodology
6. Renard Triché E, et al. What is the optimal approach to analyse ventilator-free days? A simulation study. *Crit Care*. 2025. — VFD competing risk and multistate methodology
7. *BMC Med Res Methodol*. 2025. Two-part model for ventilator-free days in a cluster randomized cross-over clinical trial. — Two-part model for VFDs
8. Lin LI. A concordance correlation coefficient to evaluate reproducibility. *Biometrics*. 1989;45(1):255-268. — CCC methodology
9. Wasserstein RL, Lazar NA. The ASA statement on P-values: context, process, and purpose. *Am Stat*. 2016;70(2):129-133. — Multiplicity and P-value interpretation
10. Renard Triché E, et al. What is the optimal approach to analyse ventilator-free days? A simulation study comparing competing risk, multistate, and count-based models. *Crit Care*. 2025.
11. Fine JP, Gray RJ. A proportional hazards model for the subdistribution of a competing risk. *J Am Stat Assoc*. 1999;94(446):496-509.
12. AJRCCM. 2024. Improving the Reporting of Trials Evaluating Organ Support Therapies Using Multistate Modeling. *Am J Respir Crit Care Med*. 2024.
13. Richesson RL, et al. PhenoFit: a framework for evaluating the quality and fitness-for-purpose of computable phenotypes. *J Am Med Inform Assoc*. 2025.
14. Lau B, Cole SR, Gange SJ. Competing risk regression models for epidemiologic data. *Am J Epidemiol*. 2009;170(2):244-256.

---

*This statistical analysis plan was prepared to support transparent, reproducible, and methodologically rigorous analysis of the SAT/SBT EHR phenotyping study. All analytic decisions are prespecified and justified with reference to established methodology and the specific data structure of the CLIF consortium.*
