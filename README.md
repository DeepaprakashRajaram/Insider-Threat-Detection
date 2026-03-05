# Insider Threat Detection Using Behavioral Modeling

Machine Learning based insider threat detection system using behavioral activity modeling and hybrid anomaly detection.

This project combines supervised learning (XGBoost) and unsupervised anomaly detection (Isolation Forest) to identify malicious insider behavior in enterprise activity logs.

## Overview

This project implements a machine learning–based system for detecting insider threats using behavioral activity logs. Insider threats are security risks that originate from individuals within an organization who misuse their authorized access to compromise data, systems, or networks.

The system analyzes user behavior across multiple enterprise activities (logins, file access, web usage, email activity, and device connections) to identify suspicious patterns and assign risk scores to potentially malicious users.

The project combines supervised and unsupervised machine learning techniques to build a hybrid insider threat detection pipeline.

---

## Problem Statement

Traditional security systems focus primarily on external attackers, while insider threats remain difficult to detect because insiders operate with legitimate access.

The goal of this project is to design a system that can:

* Model normal user behavior
* Detect abnormal or suspicious activity
* Identify potential insider threats
* Generate risk scores and alerts for security teams

---

## Dataset

The system uses the **CERT Insider Threat Dataset (r4.2)** developed by the **Carnegie Mellon University Software Engineering Institute**.

Dataset characteristics:

* Synthetic enterprise activity logs
* Multiple activity sources:

  * Logon activity
  * File access
  * Email communication
  * Web browsing
  * USB device connections
* Psychometric employee profiles
* Ground truth insider attack scenarios

Due to its realistic behavioral simulation, this dataset is widely used in insider threat research.

---

## System Architecture

```
Enterprise Activity Logs
        │
        ▼
Feature Engineering
(User behavioral statistics)
        │
        ▼
Supervised Model
(XGBoost Insider Classifier)
        │
        ▼
Anomaly Detection
(Isolation Forest)
        │
        ▼
Hybrid Detection Model
(Weighted Fusion)
        │
        ▼
Risk Scoring Engine
        │
        ▼
Security Alerts
```

---

## Feature Engineering

User activity logs are aggregated into **daily behavioral profiles**. Features include:

* Login frequency
* After-hours login activity
* File access count
* Email activity
* Web browsing activity
* USB device connections
* Behavioral deviation scores (z-score anomalies)

These features capture deviations from normal user behavior.

---

## Machine Learning Models

### Supervised Model

**XGBoost classifier**

* Detects known insider attack patterns
* Trained using labeled CERT attack scenarios

### Anomaly Detection

**Isolation Forest**

* Identifies abnormal behavioral patterns
* Trained using only normal user activity

### Hybrid Model

The final system combines both models:

```
Hybrid Score =
0.85 × Supervised Model
+
0.15 × Anomaly Model
```

This improves detection performance while maintaining low false positives.

---

## Results

Evaluation on the CERT r4.2 dataset:

| Metric    | Value |
| --------- | ----- |
| Precision | 0.885 |
| Recall    | 0.538 |
| F1 Score  | 0.669 |
| ROC-AUC   | 0.992 |

Given the extreme class imbalance (≈0.29% malicious events), the system achieves strong detection capability with a low false positive rate.

---

## Risk Scoring System

Each user-day activity is assigned a **risk score (0–100)** based on the hybrid model output.

Example alert:

```
User: MCF0600
Date: 2010-09-20
Risk Score: 98.9
```

Security analysts can investigate users with the highest risk scores.

---

## Key Behavioral Indicators

The model identified the following as strong insider threat signals:

* USB device connections
* File access spikes
* After-hours logins
* Abnormal email activity
* Web browsing deviations

These behaviors align with real-world insider threat patterns such as data exfiltration.

---

## Project Structure

```
Insider-Threat-Detection
│
├── data
│   ├── raw
│   └── processed
│
├── src
│   ├── extract_labels.py
│   ├── build_user_day_labels.py
│   ├── build_features_r4.py
│   ├── train_supervised.py
│   ├── train_anomaly.py
│   ├── hybrid_model.py
│   ├── generate_risk_scores.py
│
├── models
│
├── results
│
├── notebooks
│
├── README.md
└── requirements.txt
```

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* XGBoost
* Isolation Forest
* Large-scale log data engineering

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Insider-Threat-Detection.git
cd Insider-Threat-Detection
```

Create a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

The full detection pipeline can be executed using the scripts inside the `src` directory.

Extract malicious labels:

```bash
python src/extract_labels.py
```

Build user-day labels:

```bash
python src/build_user_day_labels.py
```

Generate behavioral features:

```bash
python src/build_features_r4.py
```

Train supervised model:

```bash
python src/train_supervised.py
```

Train anomaly detection model:

```bash
python src/train_anomaly.py
```

Generate risk scores:

```bash
python src/generate_risk_scores.py
```

---

## Future Improvements

Possible extensions include:

* Temporal behavioral modeling
* Sequence-based detection (LSTM / Transformer)
* Graph-based insider relationship modeling
* Real-time streaming detection systems

---

## Author

Deepak Prakash Rajaram
