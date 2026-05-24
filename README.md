# Fraud Detection System using Machine Learning & Explainable AI

A production-oriented fraud detection system built using machine learning, explainable AI (SHAP), and interactive Streamlit dashboards. This project uses the IEEE-CIS Fraud Detection dataset to identify fraudulent transactions, analyze risk behavior, and provide explainable predictions for operational fraud monitoring.

---

# Project Overview

This project is designed as a complete fraud analytics ecosystem rather than a basic ML classification notebook.

The system includes:

* Fraud prediction using machine learning
* Feature engineering for transaction behavior analysis
* Risk segmentation
* Explainable AI with SHAP
* Interactive Streamlit dashboards
* Fraud pattern monitoring
* Drift and stability analysis
* Deployment-ready workflow

---

# Features

## Machine Learning Pipeline

* Data preprocessing
* Missing value handling
* Categorical encoding
* Feature engineering
* Model training and evaluation

## Fraud Detection Models

* Random Forest
* XGBoost
* LightGBM
* Logistic Regression

## Explainable AI

* SHAP summary plots
* Waterfall explanations
* Transaction-level prediction reasoning

## Risk Segmentation

Transactions are classified into:

* Low Risk
* Medium Risk
* High Risk

## Interactive Dashboard

The Streamlit dashboard includes:

* System Performance Overview
* Transaction Explorer
* SHAP Explainability
* Fraud Pattern Analysis
* Drift Monitoring

---

# Dataset

Dataset Used:

* IEEE-CIS Fraud Detection Dataset

Files:

* `train_transaction.csv`
* `train_identity.csv`

Merged using:

* `TransactionID`

---

# Tech Stack

## Programming Language

* Python

## Libraries

* pandas
* numpy
* matplotlib
* seaborn
* plotly
* scikit-learn
* XGBoost
* LightGBM
* SHAP
* imbalanced-learn
* Streamlit

---

# Project Structure

```bash
Fraud-Detection-System/
│
├── data/
│   ├── train_transaction.csv
│   ├── train_identity.csv
│
├── artifacts/
│   ├── trained_model.pkl
│   ├── final_predictions.csv
│   ├── drift_metrics.csv
│
├── notebooks/
│   ├── analysis.ipynb
│
├── app/
│   ├── streamlit_app.py
│
├── requirements.txt
├── README.md
```

---

# Workflow

## 1. Data Collection

Load transaction and identity datasets.

## 2. Data Preprocessing

* Handle missing values
* Encode categorical features
* Normalize data
* Remove noisy columns

## 3. Feature Engineering

Create fraud-related behavioral features:

* transaction velocity
* amount anomalies
* device consistency
* transaction frequency

## 4. Imbalance Handling

Use SMOTE to balance fraud and non-fraud samples.

## 5. Model Training

Train multiple ML models and compare:

* Precision
* Recall
* F1-score
* ROC-AUC

## 6. Threshold Optimization

Tune probability thresholds for operational fraud detection.

## 7. Explainability

Use SHAP to explain:

* important features
* fraud reasoning
* transaction-level predictions

## 8. Dashboard Deployment

Deploy interactive Streamlit dashboard for real-time fraud analytics.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/rohanjangra07/Frauddetection
cd Fraud-Detection-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run the Project

## Run Notebook

Open:

```bash
analysis.ipynb
```

## Run Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

---

# Important Deployment Notes

The dashboard requires generated deployment artifacts.

Required columns in prediction datasets:

```python
prediction_probability
Risk_Tier
```

If these columns are missing, dashboard modules such as:

* Transaction Explorer
* SHAP Explainability
* Risk Analysis

will fail during deployment.

---

# Common Deployment Issues

## Missing Columns Error

Error:

```python
Data artifact missing required schema columns:
['prediction_probability', 'Risk_Tier']
```

Fix:

```python
df['prediction_probability'] = model.predict_proba(X)[:,1]

df['Risk_Tier'] = df['prediction_probability'].apply(assign_risk)
```

---

## Drift Monitoring Error

Error:

```python
Drift monitoring data not found.
```

Cause:

* missing drift artifact files
* incomplete training pipeline execution

---

# Strengths of the Project

* Production-oriented architecture
* Explainable AI integration
* Risk segmentation
* Interactive dashboards
* Threshold optimization
* Fraud analytics workflow

---

# Limitations

* Potential temporal leakage risk
* SMOTE-generated synthetic patterns
* Deployment artifact dependency
* Drift monitoring still incomplete

---

# Future Improvements

* Real-time streaming fraud detection
* Kafka integration
* Online learning
* Automated retraining pipelines
* Better drift monitoring
* Cloud deployment optimization
* API-based prediction service

---

# Final Conclusion

This project demonstrates a complete fraud detection ecosystem integrating:

* Machine Learning
* Explainable AI
* Risk Analytics
* Operational Monitoring
* Dashboard Deployment

The system is designed to bridge the gap between predictive modeling and real-world fraud investigation workflows.
