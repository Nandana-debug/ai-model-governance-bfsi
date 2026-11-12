# AI Model Governance in BFSI: Algorithmic Bias Mitigation & Drift Monitoring

[![Domain](https://img.shields.io/badge/Domain-BFSI_FinTech-1a365d.svg)](#)
[![Focus](https://img.shields.io/badge/Focus-AI_Ethics_%26_MLOps-2b6cb0.svg)](#)
[![Compliance](https://img.shields.io/badge/Standard-EEOC_80%25_Rule-22543d.svg)](#)
[![Python](https://img.shields.io/badge/Stack-Python_%7C_Scikit--Learn_%7C_SciPy-4a5568.svg)](#)

## 📌 Executive Summary
In Banking, Financial Services, and Insurance (BFSI), automated credit decisioning models face two critical production risks: **systemic algorithmic bias** inherited from historical lending data, and **silent performance degradation** caused by shifting macroeconomic conditions. 

Completed as part of the **Zetheta Algorithms WorkBridge Program**, this project implements an end-to-end governance framework to audit, mitigate, and continuously monitor machine learning models across a 50,000-record credit evaluation lifecycle.

---

## 🏗️ System Architecture

```mermaid
graph LR
    A[Historical Credit Data<br>N = 50,000] --> B{Phase 1: Fairness Audit}
    B -->|DIR < 0.80| C[Pre-Processing:<br>Sample Reweighting]
    C --> D[Mitigated Classifier<br>DIR = 0.91]
    D --> E[Production Deployment]
    E --> F{Phase 2: Drift Monitor<br>T0 vs T1 Batches}
    F -->|PSI >= 0.25<br>KS p < 0.05| G[Automated Alert &<br>Retraining Trigger]
    F -->|PSI < 0.10| E
