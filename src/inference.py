import pandas as pd
import numpy as np
import joblib
import json
import os
from src.config import (
    BEST_MODEL_PATH, THRESHOLD_PATH, FEATURE_COLS_PATH, 
    PROCESSED_TEST, PREDICTIONS_PATH, TARGET_COL, CALIBRATION_MODEL_PATH
)
from src.utils import logger, validate_file_exists

class InferenceEngine:
    """
    Centralized Inference Engine.
    Ensures that dashboard, SHAP, and evaluation all use the exact same prediction logic.
    """
    def __init__(self):
        validate_file_exists(BEST_MODEL_PATH, "Trained Model")
        validate_file_exists(THRESHOLD_PATH, "Threshold Configuration")
        validate_file_exists(FEATURE_COLS_PATH, "Feature Columns Configuration")
        
        logger.info("Loading Inference Artifacts...")
        self.model = joblib.load(BEST_MODEL_PATH)
        
        with open(THRESHOLD_PATH, 'r') as f:
            self.thresholds = json.load(f)
            
        self.feature_cols = joblib.load(FEATURE_COLS_PATH)
        
        self.calibrator = None
        if os.path.exists(CALIBRATION_MODEL_PATH):
            logger.info("Loading Probability Calibration Model...")
            self.calibrator = joblib.load(CALIBRATION_MODEL_PATH)
        else:
            logger.info("No Calibration Model found. Using raw probabilities.")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a processed dataframe, aligns features, and generates standard predictions.
        """
        logger.info(f"Running inference on {len(df)} records...")
        
        # Align features strictly to what the model was trained on
        missing_cols = set(self.feature_cols) - set(df.columns)
        if missing_cols:
            logger.warning(f"Inference input is missing {len(missing_cols)} features. Imputing with 0.")
            
        # Strict deterministic reindexing to match exact order and columns
        df_reindexed = df.reindex(columns=self.feature_cols, fill_value=0)
        X = df_reindexed[self.feature_cols]

        
        # Generate probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[:, 1]
        else:
            probabilities = self.model.predict(X)
            
        # Apply Isotonic Calibration if available
        if self.calibrator is not None:
            probabilities = self.calibrator.predict(probabilities)
            
        # Create output dataframe
        out_df = df.copy()
        out_df['prediction_probability'] = probabilities
        
        # Assign hard labels based on optimal threshold
        opt_thresh = self.thresholds.get('optimal_threshold', 0.40)
        out_df['prediction_label'] = (probabilities >= opt_thresh).astype(int)
        
        # Assign Business Risk Tiers
        crit_thresh = self.thresholds.get('critical_threshold', 0.75)
        conditions = [
            (probabilities >= crit_thresh),
            (probabilities >= opt_thresh) & (probabilities < crit_thresh),
            (probabilities < opt_thresh)
        ]
        choices = ['Critical Risk', 'Suspicious', 'Clear']
        out_df['Risk_Tier'] = np.select(conditions, choices, default='Unknown')
        
        # Ensure standard schema if columns exist in raw dataset
        # Map specific internal names to required schema names if needed
        if 'id_31' in out_df.columns and 'DeviceType' not in out_df.columns:
            out_df['DeviceType'] = out_df['id_31']
            
        return out_df

    def run_batch_inference(self):
        """
        Loads processed test dataset, runs inference, and persists predictions artifact.
        """
        validate_file_exists(PROCESSED_TEST, "Processed Test Data")

        logger.info(f"Loading {PROCESSED_TEST} for batch inference...")
        test_df = pd.read_parquet(PROCESSED_TEST)

        # Integrity checks — catches double-merge / duplication bugs before they corrupt predictions
        logger.info(f"Inference dataframe shape BEFORE prediction: {test_df.shape}")
        if 'TransactionID' in test_df.columns:
            n_dupes = test_df['TransactionID'].duplicated().sum()
            if n_dupes > 0:
                logger.warning(
                    f"INTEGRITY WARNING: {n_dupes:,} duplicate TransactionIDs detected. "
                    f"Dropping duplicates to prevent doubled predictions."
                )
                test_df = test_df.drop_duplicates(subset=['TransactionID'])
                logger.info(f"Shape after deduplication: {test_df.shape}")
            else:
                logger.info("TransactionID uniqueness check passed (0 duplicates).")
        
        predictions_df = self.predict(test_df)
        
        logger.info(f"Saving predictions artifact to {PREDICTIONS_PATH}")
        predictions_df.to_parquet(PREDICTIONS_PATH, index=False)
        return predictions_df
