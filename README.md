# AI Model Governance: Algorithmic Bias Audit & Model Drift Detection

[![Domain](https://img.shields.io/badge/Domain-BFSI_FinTech-blue.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](#)
[![AI Ethics](https://img.shields.io/badge/Compliance-80%25_Rule-orange.svg)](#)
[![MLOps](https://img.shields.io/badge/Monitoring-PSI_%26_KS_Tests-purple.svg)](#)

## 📌 Project Overview
This repository documents the methodology and key findings from my project work during the **Zetheta Algorithms WorkBridge Project Work Experience Program**. The work focused on AI governance, algorithmic fairness compliance, and production machine learning monitoring within the Banking, Financial Services, and Insurance (BFSI) sector.

---

## 🎯 Case Studies Completed

### 1. Algorithmic Bias Audit & Mitigation
* **Objective:** Audit automated credit approval models to detect and mitigate historical demographic disparities.
* **Fairness Metrics Audited:** 
  * **Statistical Parity Difference (SPD):** Measured baseline approval gaps across demographic groups.
  * **Disparate Impact Ratio (DIR):** Evaluated compliance against the legal 80% (4/5ths) rule for adverse impact.
* **Mitigation Technique:** Implemented **Sample Reweighting** during model training to balance group-target distributions, improving the Disparate Impact Ratio from **0.57 to 0.91** ($\ge 0.80$ compliant) while preserving predictive accuracy.

### 2. AI Model Drift Detection & Degradation Analysis
* **Objective:** Design a production monitoring pipeline to detect covariate shift and concept drift across streaming data batches.
* **Drift Diagnostics:** 
  * **Population Stability Index (PSI):** Measured distribution shifts between baseline ($T_0$) and production ($T_1$) batches.
  * **Kolmogorov-Smirnov (KS) Test:** Conducted two-sample statistical testing ($p < 0.0001$) to confirm feature shifts in key financial variables (`credit_score`, `dti_ratio`).
* **Remediation:** Quantified downstream ROC-AUC degradation and configured automated retraining alerts ($\text{PSI} \ge 0.25$).

---

## 🛠️ Tech Stack & Methodologies
* **Languages & Libraries:** Python, Pandas, NumPy, Scikit-Learn, SciPy, Matplotlib, Seaborn
* **Governance & MLOps Concepts:** Disparate Impact Analysis, Sample Reweighting, Covariate Shift, Concept Drift, Population Stability Index (PSI), Hypothesis Testing

---

## 🔒 Confidentiality & Intellectual Property Notice
*In strict accordance with the Non-Disclosure Agreement (NDA) and program terms of Zetheta Algorithms Private Limited, proprietary source code, synthetic data schemas, and internal pipelines are maintained in a private repository.*# ai-model-governance-bfsi
