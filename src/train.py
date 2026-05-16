import os
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score
from sklearn.isotonic import IsotonicRegression
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, Tuple
import json
from src.utils import logger, validate_file_exists
from src.config import (
    TARGET_COL, TIME_COL, PROCESSED_TRAIN_ORIGINAL, BEST_MODEL_PATH, 
    THRESHOLD_PATH, RANDOM_STATE, CALIBRATION_MODEL_PATH, FEATURE_STATS_PATH
)

class ModelTrainer:
    """
    Production-grade training pipeline for fraud detection models.
    Implements TimeSeriesSplit for cross-validation to prevent temporal leakage.
    """
    
    def __init__(self, use_smote: bool = True):
        self.use_smote = use_smote
        
        # Configure models without aggressive imbalance weighting initially
        # Double-dipping (SMOTE + weights) is strictly avoided
        self.models = {
            'lightgbm': lgb.LGBMClassifier(
                objective='binary',
                metric='average_precision',

                n_estimators=2000,
                learning_rate=0.02,

                max_depth=6,
                num_leaves=31,
                min_child_samples=20,

                colsample_bytree=0.8,   # sklearn alias for feature_fraction
                subsample=0.8,          # sklearn alias for bagging_fraction
                subsample_freq=1,       # sklearn alias for bagging_freq

                reg_alpha=0.1,          # sklearn alias for lambda_l1
                reg_lambda=0.1,         # sklearn alias for lambda_l2

                force_col_wise=True,
                verbosity=-1,           # Suppress "no further splits" warning spam
                random_state=RANDOM_STATE,
                is_unbalance=not self.use_smote,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                eval_metric='aucpr',
                n_jobs=-1
            ),
            'isolation_forest': IsolationForest(
                n_estimators=200,
                contamination=0.05, # Estimate of outlier proportion
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        }
        self.trained_models = {}
        self.cv_scores = {}

    def train_cv(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        fe=None,           # FeatureEngineer instance for fold-safe target encoding
        raw_train_df=None, # Raw (pre-preprocessing) training rows aligned with X
    ) -> Any:
        """
        Trains a model using TimeSeriesSplit to prevent look-ahead bias.
        When fe + raw_train_df are provided, target-encoded columns (DeviceRisk,
        EmailDomainRisk) are recomputed per fold using ONLY fold-train rows,
        eliminating target encoding leakage into validation folds.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not supported.")

        # Columns that are target-derived and must be recomputed per fold
        TARGET_ENCODED_COLS = ['DeviceRisk', 'EmailDomainRisk']
        fold_safe_encoding = (
            fe is not None
            and raw_train_df is not None
            and any(c in X.columns for c in TARGET_ENCODED_COLS)
        )
        if fold_safe_encoding:
            logger.info("Fold-safe target encoding ENABLED: DeviceRisk/EmailDomainRisk will be recomputed per fold.")
        else:
            logger.info("Fold-safe target encoding DISABLED (fe/raw_train_df not provided).")

        logger.info(f"Starting Cross-Validation for {model_name}...")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        model = self.models[model_name]

        pr_aucs = []
        oof_preds = []
        oof_targets = []
        feature_importances = []

        # Strict Schema Validation
        if TARGET_COL in X.columns:
            raise ValueError(f"CRITICAL LEAKAGE: {TARGET_COL} found in feature matrix.")

        import hashlib
        # Pre-cast target-encoded columns to float32 so master dtype matches fold assignments.
        # Without this, parquet loads them as float64 but fold loop casts to float32 → hash mismatch.
        TARGET_ENCODED_COLS = ['DeviceRisk', 'EmailDomainRisk']
        for _col in TARGET_ENCODED_COLS:
            if _col in X.columns:
                X[_col] = X[_col].astype('float32')

        # Hash column names AND dtypes — catches any remaining type drift between folds
        master_schema = tuple(zip(X.columns.tolist(), X.dtypes.astype(str).tolist()))
        master_schema_hash = hashlib.md5(str(master_schema).encode()).hexdigest()
        logger.info(f"Master Feature Schema Hash: {master_schema_hash} (Features: {len(X.columns)})")

        # Validate temporal ordering
        if TIME_COL in X.columns and not X[TIME_COL].is_monotonic_increasing:
            logger.warning(f"Temporal validation failed! Data is not sorted by {TIME_COL}. Sorting now...")
            sorted_idx = X.sort_values(TIME_COL).index
            X = X.loc[sorted_idx]
            y = y.loc[sorted_idx]

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, y_tr = X.iloc[train_idx].copy(), y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]

            # ── Fold-safe target encoding ─────────────────────────────────────
            if fold_safe_encoding:
                fold_raw_tr = raw_train_df.iloc[train_idx]
                fe.fit_fold(fold_raw_tr)   # Refit risk maps on fold-train only

                # Overwrite pre-computed (globally-leaked) encoded columns
                # Use .loc[] to avoid pandas block reordering
                # Cast to float32 immediately to guarantee dtype consistency across folds
                if 'DeviceRisk' in X_tr.columns and 'id_31' in fold_raw_tr.columns:
                    X_tr.loc[:, 'DeviceRisk'] = (
                        fold_raw_tr['id_31'].map(fe.device_risk_map)
                        .fillna(fe.global_fraud_rate).values.astype('float32')
                    )
                    X_val.loc[:, 'DeviceRisk'] = (
                        raw_train_df.iloc[val_idx]['id_31'].map(fe.device_risk_map)
                        .fillna(fe.global_fraud_rate).values.astype('float32')
                    )
                if 'EmailDomainRisk' in X_tr.columns and 'P_emaildomain' in fold_raw_tr.columns:
                    grp_tr = fold_raw_tr['P_emaildomain'].map(fe._group_email)
                    grp_val = raw_train_df.iloc[val_idx]['P_emaildomain'].map(fe._group_email)
                    X_tr.loc[:, 'EmailDomainRisk'] = (
                        grp_tr.map(fe.email_risk_map)
                        .fillna(fe.global_fraud_rate).values.astype('float32')
                    )
                    X_val.loc[:, 'EmailDomainRisk'] = (
                        grp_val.map(fe.email_risk_map)
                        .fillna(fe.global_fraud_rate).values.astype('float32')
                    )
            # ─────────────────────────────────────────────────────────────────
            # Per-fold strict schema checks
            assert list(X_tr.columns) == list(X_val.columns), f"Fold {fold+1} train/val schema mismatch"
            assert TARGET_COL not in X_tr.columns, f"Fold {fold+1} TARGET_COL leaked into feature matrix"

            fold_schema = tuple(zip(X_tr.columns.tolist(), X_tr.dtypes.astype(str).tolist()))
            fold_hash = hashlib.md5(str(fold_schema).encode()).hexdigest()
            if fold_hash != master_schema_hash:
                raise ValueError(
                    f"Fold {fold+1} schema hash ({fold_hash}) "
                    f"differs from master schema hash ({master_schema_hash})"
                )
            logger.info(f"Fold {fold+1} schema checks passed. Hash={fold_hash}, Features={len(X_tr.columns)}")

            
            # Dynamic Weighting if SMOTE is disabled
            if not self.use_smote and model_name == 'xgboost':
                neg_count = (y_tr == 0).sum()
                pos_count = (y_tr == 1).sum()
                scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
                model.set_params(scale_pos_weight=scale_pos_weight)
            
            # Apply SMOTE inside the fold-train only
            if self.use_smote and model_name != 'isolation_forest':
                logger.info(f"Fold {fold+1}: Applying SMOTE to train partition (Train Shape: {X_tr.shape})")
                # 0.2 ratio: enough minority signal without distorting base rate
                sm = SMOTE(sampling_strategy=0.2, random_state=RANDOM_STATE)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

            # Overflow sanitation: clip extreme values before model.fit to prevent
            # RuntimeWarning: overflow in cast from large V-column ratios
            X_tr = X_tr.replace([np.inf, -np.inf], np.nan).clip(-1e9, 1e9)
            X_val = X_val.replace([np.inf, -np.inf], np.nan).clip(-1e9, 1e9)
            
            # Isolation forest is unsupervised for training
            if model_name == 'isolation_forest':
                model.fit(X_tr)
                # Predict returns -1 for outliers, 1 for inliers. Map to 1 (fraud), 0 (legit)
                preds = model.predict(X_val)
                y_pred_proba = np.where(preds == -1, 1, 0) 
            else:
                # --- Recency-Aware Sample Weighting (Step 5) ---
                # Since we use TimeSeriesSplit, index order = temporal order.
                # Linear ramp from 0.5 (oldest) to 1.0 (newest) reduces impact of stale patterns.
                sample_weight = np.linspace(0.5, 1.0, len(X_tr))

                if model_name == 'lightgbm':
                    model.fit(
                        X_tr, y_tr,
                        sample_weight=sample_weight,
                        eval_set=[(X_val, y_val)],
                        eval_metric='average_precision',
                        callbacks=[lgb.early_stopping(stopping_rounds=100)]
                    )
                elif model_name == 'xgboost':
                    model.fit(
                        X_tr, y_tr,
                        sample_weight=sample_weight,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            if hasattr(model, 'feature_importances_'):
                feature_importances.append(model.feature_importances_)
                
            oof_preds.extend(y_pred_proba)
            oof_targets.extend(y_val)
            
            fold_pr_auc = average_precision_score(y_val, y_pred_proba)

            # Log best_iteration to diagnose under/overfitting
            if hasattr(model, 'best_iteration_') and model.best_iteration_ is not None:
                logger.info(f"Fold {fold+1} Best Iteration: {model.best_iteration_}")

            # ── Drift diagnostics ────────────────────────────────────────────
            fold_fraud_rate = float(y_val.mean())
            fold_mean_pred  = float(np.mean(y_pred_proba))
            global_fraud_rate = float(y.mean())
            drift_delta = fold_fraud_rate - global_fraud_rate
            
            # Store for persistence (Step 6)
            if not hasattr(self, 'fold_diagnostics'): self.fold_diagnostics = []
            self.fold_diagnostics.append({
                "fold": fold + 1,
                "fraud_rate": fold_fraud_rate,
                "mean_pred": fold_mean_pred,
                "pr_auc": fold_pr_auc
            })

            logger.info(
                f"Fold {fold+1} Drift | fraud_rate={fold_fraud_rate:.4f} "
                f"(drift={drift_delta:+.4f} vs global={global_fraud_rate:.4f}) "
                f"| mean_pred_prob={fold_mean_pred:.4f}"
            )
            # ───────────────────────────────────────────────────────────────

            # Realistic Metric Protection
            if fold_pr_auc > 0.98:
                logger.warning(f"Fold {fold+1}: Unusually high PR-AUC ({fold_pr_auc:.4f}) detected. Review for leakage.")
                
            pr_aucs.append(fold_pr_auc)
            logger.info(f"Fold {fold+1} Realistic PR-AUC: {fold_pr_auc:.4f} (Val Shape: {X_val.shape})")
            
        avg_pr_auc = np.mean(pr_aucs)
        logger.info(f"{model_name} Average Realistic CV PR-AUC: {avg_pr_auc:.4f}")
        self.cv_scores[model_name] = avg_pr_auc
        
        # Train Probability Calibrator on OOF predictions
        logger.info("Fitting Isotonic Regression Calibration Model on OOF predictions...")
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(oof_preds, oof_targets)
        joblib.dump(calibrator, CALIBRATION_MODEL_PATH)
        logger.info(f"Calibration model saved to {CALIBRATION_MODEL_PATH}")
        
        # Final Model Training Pipeline
        logger.info(f"Retraining {model_name} on full training data...")
        if self.use_smote and model_name != 'isolation_forest':
            logger.info("Applying SMOTE before final model fit (ratio=0.2, matching CV distribution)...")
            sm = SMOTE(sampling_strategy=0.2, random_state=RANDOM_STATE)
            X_final, y_final = sm.fit_resample(X, y)
        else:
            X_final, y_final = X, y
            if model_name == 'xgboost':
                neg_count = (y_final == 0).sum()
                pos_count = (y_final == 1).sum()
                model.set_params(scale_pos_weight=neg_count / pos_count if pos_count > 0 else 1)
                
        if model_name == 'isolation_forest':
            model.fit(X_final)
        else:
            model.fit(X_final, y_final)
            
        self.trained_models[model_name] = model
        
        # Persist Drift Monitoring Stats (Step 6)
        # Sanitize X before computing stats: inf and extreme values cause overflow in nanops
        logger.info("Persisting Feature Stats and Drift Monitoring (outputs/drift_monitoring.json)...")
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        _num = X_clean.select_dtypes(include=np.number).columns
        # Clip to 1e6: this matches the safe_divide limit and is safe for mean/std ops
        X_clean[_num] = X_clean[_num].clip(-1e6, 1e6).astype(np.float32)
        
        # Step 4: Identify Weakest Features
        mean_importance = np.mean(feature_importances, axis=0) if feature_importances else None
        if mean_importance is not None:
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': mean_importance
            }).sort_values('importance', ascending=True)
            weak_features = importance_df.head(20)
            logger.info(f"Top 20 Weakest Features (Candidates for Pruning):\n{weak_features.to_string(index=False)}")
        
        feature_stats = {
            "means": X_clean.mean(numeric_only=True).to_dict(),
            "stds": X_clean.std(numeric_only=True).to_dict(),
            "missing_pct": (X_clean.isnull().mean() * 100).to_dict(),
            "feature_importance_std": np.std(feature_importances, axis=0).tolist() if feature_importances else None,
            "fold_fraud_rates": [float(f['fraud_rate']) for f in self.fold_diagnostics],
            "fold_mean_preds": [float(f['mean_pred']) for f in self.fold_diagnostics],
            "fold_pr_aucs": [float(f['pr_auc']) for f in self.fold_diagnostics],
            "weak_features": weak_features['feature'].tolist() if mean_importance is not None else []
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(FEATURE_STATS_PATH), exist_ok=True)
        with open(FEATURE_STATS_PATH, 'w') as f:
            json.dump(feature_stats, f)

        # Also save to dedicated drift monitoring path
        DRIFT_MONITOR_PATH = os.path.join(os.path.dirname(FEATURE_STATS_PATH), 'drift_monitoring.json')
        with open(DRIFT_MONITOR_PATH, 'w') as f:
            json.dump(feature_stats, f)
            
        return model

    def save_model(self, model_name: str):
        """
        Persists the trained model artifact to models/trained_models/best_model.pkl.
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} has not been trained yet.")
            
        joblib.dump(self.trained_models[model_name], BEST_MODEL_PATH)
        logger.info(f"Best model saved to {BEST_MODEL_PATH}")
        
        # Save placeholder optimal threshold (would be tuned in tuning phase)
        threshold_data = {"optimal_threshold": 0.40, "critical_threshold": 0.75}
        with open(THRESHOLD_PATH, 'w') as f:
            json.dump(threshold_data, f)
            
    def run_training(self, fe=None, raw_train_df=None):
        """
        Executes the full training pipeline using exclusively the persisted parquet dataset.
        Optionally accepts fe (FeatureEngineer) and raw_train_df to enable
        fold-safe target encoding inside TimeSeriesSplit CV.
        """
        validate_file_exists(PROCESSED_TRAIN_ORIGINAL, "Processed Training Data")

        logger.info(f"Loading processed training data from {PROCESSED_TRAIN_ORIGINAL}...")
        train_df = pd.read_parquet(PROCESSED_TRAIN_ORIGINAL)

        X = train_df.drop(columns=[TARGET_COL])
        y = train_df[TARGET_COL]

        # Align raw_train_df index with X so iloc-based fold indexing is consistent
        if raw_train_df is not None:
            raw_train_df = raw_train_df.reset_index(drop=True)

        # Defaulting to lightgbm as the champion model
        self.train_cv('lightgbm', X, y, fe=fe, raw_train_df=raw_train_df)
        self.save_model('lightgbm')
