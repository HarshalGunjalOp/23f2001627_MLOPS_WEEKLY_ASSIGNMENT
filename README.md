# 22F3000916 — MLOps Weekly Assignment | Week 9

**Student ID :** 23F2001627  
**Term :** MAY 2026  
**Branch :** `week_9`  
**Assignment :** Explainability, Fairness, and Drift in the IRIS Pipeline

---

## Overview

This assignment integrates essential ML governance practices into the pipeline. It demonstrates how to audit a model for demographic bias, explain black-box predictions using game-theoretic feature attributions, monitor for data drift, and document accountability for production deployment.

| Component | Role |
|---|---|
| **Fairlearn** | Audits the model for performance gaps across a synthetically injected sensitive attribute (`location`). |
| **SHAP** | Generates summary plots to explain which features push the model toward or away from specific classifications. |
| **Drift Monitoring** | Simulates and detects statistical shifts in input features (Data Drift) that could degrade model reliability in production. |
| **Model Card** | Formalizes accountability by documenting the model's intended use, training data, and known limitations. |

---

## Files

| File | Description |
|---|---|
| `week9_analysis.py` | Core script: injects the location attribute, runs Fairlearn metrics, generates SHAP plots, and simulates data drift. |
| `MODEL_CARD.md` | A governance document outlining the model's purpose, fairness audit results, and deployment limitations. |
| `shap_virginica.png` | The generated SHAP summary plot visualizing feature attributions for the Virginica class. |
| `README.md` | Assignment documentation and execution instructions. |

---

## Setup & Execution

**1. Install Dependencies**
Ensure you have the required analytical libraries installed in your environment:
```bash
pip install fairlearn shap scikit-learn pandas matplotlib