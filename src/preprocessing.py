import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
import joblib
from typing import Tuple, List, Dict
from src.config import (
    MISSING_VALUE_THRESHOLD, RANDOM_STATE, TEST_SIZE, 
    TARGET_COL, TIME_COL, ID_COL, 
    ENCODER_PATH, IMPUTER_VALUES_PATH, FEATURE_COLS_PATH,
    PROCESSED_TRAIN_ORIGINAL, PROCESSED_TEST
)
from src.utils import logger

class PreprocessingPipeline:
    """
    Production-grade pipeline for cleaning, encoding, scaling, and splitting data.
    Ensures zero data leakage by fitting transformers only on training data.
    """
    
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.imputer_values: Dict[str, any] = {}
        self.dropped_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.variance_selector = VarianceThreshold(threshold=5e-4)  # Prunes near-zero-variance sparse noise while preserving rare fraud signals
        self.variance_dropped_cols: List[str] = []

    def handle_missing_values(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Drops columns with high missingness and imputes others.
        Inference/Test set uses training imputer values.
        """
        if is_training:
            # 1. Drop columns with > threshold missing values
            nan_stats = df.isnull().mean()
            self.dropped_cols = nan_stats[nan_stats > MISSING_VALUE_THRESHOLD].index.tolist()
            logger.info(f"Dropping {len(self.dropped_cols)} columns with >{MISSING_VALUE_THRESHOLD*100}% missing data.")
            
        df = df.drop(columns=[c for c in self.dropped_cols if c in df.columns])
        
        # 2. Impute missing values
        if is_training:
            num_cols = df.select_dtypes(include=[np.number]).columns
            cat_cols = df.select_dtypes(exclude=[np.number]).columns
            
            for col in num_cols:
                if col == TARGET_COL: continue
                self.imputer_values[col] = df[col].median()
                
            for col in cat_cols:
                if col == TARGET_COL: continue
                self.imputer_values[col] = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'

        # Vectorized fillna to prevent fragmentation
        df = df.fillna(value=self.imputer_values)
        
        # Categorical fallbacks for lingering NaNs not in imputer_values
        if not is_training:
            cat_cols = df.select_dtypes(exclude=[np.number]).columns
            fill_cats = {c: 'Unknown' for c in cat_cols if c not in self.imputer_values and c != TARGET_COL}
            fill_nums = {c: 0 for c in df.select_dtypes(include=[np.number]).columns if c not in self.imputer_values and c != TARGET_COL}
            df = df.fillna(value={**fill_cats, **fill_nums})
            
        return df

    def encode_categorical(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Applies Label Encoding to all categorical features dynamically.
        """
        if is_training:
            self.cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            if TARGET_COL in self.cat_cols:
                self.cat_cols.remove(TARGET_COL)
        
        available_cat = [c for c in self.cat_cols if c in df.columns]
        new_cols = {}
        
        for col in available_cat:
            # Cast to string safely
            s = df[col].astype(str).fillna('Unknown')
            if is_training:
                le = LabelEncoder()
                # Fit includes an explicit 'Unknown' class to safely handle unseen data in test
                le.fit(np.append(s.unique(), 'Unknown'))
                new_cols[col] = le.transform(s)
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    # Replace unseen labels with 'Unknown' BEFORE transforming
                    unseen_mask = ~s.isin(le.classes_)
                    s.loc[unseen_mask] = 'Unknown'
                    new_cols[col] = le.transform(s)
                else:
                    logger.warning(f"No encoder found for {col}. Skipping.")
                    
        if new_cols:
            df = df.drop(columns=list(new_cols.keys()))
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
            
        return df

    # Removed scale_features since tree-based models don't need scaling

    def _validate_numeric(self, X: pd.DataFrame):
        """
        Strictly ensures no object/string/category columns remain before SMOTE.
        """
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            logger.error(f"Non-numeric columns found before SMOTE: {non_numeric}")
            logger.info(f"Dtypes summary:\n{X[non_numeric].dtypes}")
            raise ValueError(f"Could not convert string to float. Remaining non-numeric columns: {non_numeric}. They must be encoded.")
        logger.info("Validation passed: All features are numeric.")

    # Removed apply_smote since it's now dynamically applied in train.py CV loop

    def save_artifacts(self, feature_cols: List[str]):
        """
        Persists encoders, scalers, and feature columns for inference.
        """
        joblib.dump(self.label_encoders, ENCODER_PATH)
        joblib.dump(self.imputer_values, IMPUTER_VALUES_PATH)
        joblib.dump(feature_cols, FEATURE_COLS_PATH)
        logger.info("Preprocessing artifacts saved to models/artifacts/")

    def fit_transform_train(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Fits transformers on training data and returns imputed/encoded output.
        """
        logger.info("Fitting PreprocessingPipeline on Training Data...")
        # --- EARLY SANITIZATION GATE ---
        # Clipping must happen BEFORE imputer/encoder calculations to prevent overflow 
        # in pandas .median() or .mean() during handle_missing_values.
        # We use float32 range safety limits.
        num_cols = train_df.select_dtypes(include=np.number).columns
        train_df[num_cols] = train_df[num_cols].replace([np.inf, -np.inf], np.nan).clip(-1e9, 1e9)
        
        train_df = self.handle_missing_values(train_df, is_training=True)
        train_df = self.encode_categorical(train_df, is_training=True)

        X_train = train_df.drop(columns=[TARGET_COL, TIME_COL], errors='ignore')
        y_train = train_df[TARGET_COL]
        
        
        # Lightweight Feature Cleanup
        logger.info("Applying lightweight feature cleanup (VarianceThreshold)...")
        self.variance_selector.fit(X_train)
        self.variance_dropped_cols = X_train.columns[~self.variance_selector.get_support()].tolist()
        if self.variance_dropped_cols:
            logger.info(f"Dropped {len(self.variance_dropped_cols)} near-constant features.")
            X_train = X_train.drop(columns=self.variance_dropped_cols)
            
        feature_cols = list(X_train.columns)

        # Downcast float64 → float32: halves memory, reduces overflow risk from near-max values
        f64_cols = X_train.select_dtypes(include=['float64']).columns
        if len(f64_cols) > 0:
            X_train[f64_cols] = X_train[f64_cols].astype('float32')
            logger.info(f"Downcast {len(f64_cols)} float64 columns to float32.")

        # Reset indices to prevent alignment issues during reconstruction/concatenation
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

        return X_train, y_train, feature_cols

    def transform_test(self, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Transforms test data using fitted artifacts.
        """
        # --- EARLY SANITIZATION GATE (Test) ---
        num_cols = test_df.select_dtypes(include=np.number).columns
        test_df[num_cols] = test_df[num_cols].replace([np.inf, -np.inf], np.nan).clip(-1e9, 1e9)

        test_df = self.handle_missing_values(test_df, is_training=False)
        test_df = self.encode_categorical(test_df, is_training=False)
        
        X_test = test_df.drop(columns=[TARGET_COL, TIME_COL], errors='ignore')
        if self.variance_dropped_cols:
            X_test = X_test.drop(columns=[c for c in self.variance_dropped_cols if c in X_test.columns], errors='ignore')

        # Mirror float64→float32 downcast from train to keep parquet dtypes consistent
        f64_cols = X_test.select_dtypes(include=['float64']).columns
        if len(f64_cols) > 0:
            X_test[f64_cols] = X_test[f64_cols].astype('float32')

        y_test = test_df[TARGET_COL] if TARGET_COL in test_df.columns else None

        # Reset indices to prevent alignment issues during reconstruction/concatenation
        X_test = X_test.reset_index(drop=True)
        if y_test is not None:
            y_test = y_test.reset_index(drop=True)

        return X_test, y_test

    def run_preprocessing_and_save(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Orchestrates full train/test transform and persists parquets.
        """
        X_train_res, y_train_res, feature_cols = self.fit_transform_train(train_df)
        X_test, y_test = self.transform_test(test_df)
        
        # Reconstruct Training DataFrame for persistence
        # Reset index ensures X and y align perfectly without duplication
        train_processed = X_train_res.copy()
        train_processed[TARGET_COL] = y_train_res.values if isinstance(y_train_res, pd.Series) else y_train_res
        
        # Reconstruct Test DataFrame for persistence
        test_processed = X_test.copy()
        if y_test is not None:
            test_processed[TARGET_COL] = y_test.values if isinstance(y_test, pd.Series) else y_test
            
        # Final Shape Validation before save
        logger.info(f"Processed train shape BEFORE save: {train_processed.shape}")
        logger.info(f"Processed test shape BEFORE save: {test_processed.shape}")

        # Persist Datasets as Parquet
        logger.info(f"Saving original processed train data to {PROCESSED_TRAIN_ORIGINAL}")
        train_processed.to_parquet(PROCESSED_TRAIN_ORIGINAL, index=False)
        
        logger.info(f"Saving processed test data to {PROCESSED_TEST}")
        test_processed.to_parquet(PROCESSED_TEST, index=False)
        
        # Persist transformers
        self.save_artifacts(feature_cols)
        logger.info("Preprocessing complete.")
