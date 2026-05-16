# Real-Time Fraud Detection System

A fully orchestrated, end-to-end fraud detection pipeline built on the IEEE-CIS dataset.

## Architecture

This project is structured as a production-grade ML pipeline, featuring strict temporal splitting, leak-free feature engineering, fold-safe cross validation, and centralized inference powering a Streamlit dashboard.

## Methodology Disclosure

Cross-validation is performed on the chronologically split training partition using `TimeSeriesSplit`.

To prevent synthetic leakage, SMOTE is applied strictly inside each training fold and never on validation folds.

Preprocessing transformers are fit once on the chronological training partition before cross-validation for computational efficiency on the high-dimensional IEEE-CIS dataset.

This may introduce minor preprocessing-statistics leakage across folds, but avoids the significantly higher computational overhead of fully fold-specific preprocessing.

## Execution

To run the full end-to-end pipeline:
```powershell
python run_pipeline.py
```

To start the Operations Dashboard:
```powershell
streamlit run dashboard/app.py
```
