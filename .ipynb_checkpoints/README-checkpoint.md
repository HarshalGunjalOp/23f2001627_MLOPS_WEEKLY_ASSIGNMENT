# Week 10: LLMOps — Fine-Tuning Gemini on the IRIS Pipeline

## Student Information
* **Roll Number:** 23F2001627
* **Repository:** `23F2001627_MLOPS_WEEKLY_ASSIGNMENT`
* **Branch:** `week_10`

---

## Overview
This project demonstrates the transition from traditional MLOps to LLMOps by fine-tuning a foundation Gemini model on the IRIS dataset using Vertex AI. It compares two distinct data representations—raw feature values vs. natural language descriptions—to analyze how natural language structuring impacts LLM output accuracy and format compliance.

---

## Repository Structure
```text
.
├── prepare_data.py    # Generates Gemini-compatible JSONL dataset files (v1 & v2)
├── tune_models.py     # Launches supervised fine-tuning jobs on GCP Vertex AI
├── evaluate.py        # Evaluates trained endpoints on accuracy & format compliance
├── README.md          # Project documentation