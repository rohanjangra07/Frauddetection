import pandas as pd
import numpy as np
from typing import List, Dict
from src.utils import logger
from src.config import TIME_COL, TARGET_COL

class FeatureEngineer:
    """
    Stateful feature engineering pipeline.
    Calculates behavioral and environmental risk features while preventing leakage.
    """
    
    def __init__(self):
        self.card_means: Dict[str, float] = {}
        self.card_counts: Dict[str, int] = {}
        self.device_risk_map: Dict[str, float] = {}
        self.email_risk_map: Dict[str, float] = {}
        self.global_fraud_rate: float = 0.0
        self.selected_features: List[str] = []

    @staticmethod
    def safe_divide(a, b, fill_value=0.0):
        """
        Universal safety helper for ratio features.
        Prevents division by zero, clips extreme outliers, and handles inf/nan.
        Force-casts to float32 to prevent intermediate overflows during calculation.
        """
        # Ensure inputs are float32 to prevent overflow in intermediate ops
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)

        # Guard denominator: values < 1e-6 are treated as 1e-6 to prevent explosion
        b_guarded = np.where(np.abs(b) < 1e-6, 1e-6, b)
        
        # Guarded division
        result = np.divide(a, b_guarded)
        
        # Handle nan/inf results early
        result = np.nan_to_num(
            result,
            nan=fill_value,
            posinf=1e6,
            neginf=-1e6
        )

        # Final clip to safe production range
        result = np.clip(result, -1e6, 1e6)
        
        return result.astype('float32')

    def fit(self, df: pd.DataFrame):
        """
        Learns statistics from the full training set.
        """
        logger.info("Fitting FeatureEngineer on training data...")

        if 'card1' in df.columns and 'TransactionAmt' in df.columns:
            self.card_means = df.groupby('card1')['TransactionAmt'].mean().to_dict()
            self.card_counts = df.groupby('card1').size().to_dict()

        if TARGET_COL in df.columns:
            self._fit_target_encodings(df)

    def fit_fold(self, fold_train_df: pd.DataFrame):
        """
        Re-fits ONLY the target-derived encodings using fold-train rows.
        """
        if TARGET_COL not in fold_train_df.columns:
            logger.warning("fit_fold: TARGET_COL not found — skipping target encoding refit.")
            return
        logger.info(f"fit_fold: refitting target encodings on fold-train ({len(fold_train_df):,} rows)")
        self._fit_target_encodings(fold_train_df)

    def _fit_target_encodings(self, df: pd.DataFrame):
        """Internal helper: compute smoothed fraud rate maps from df."""
        self.global_fraud_rate = df[TARGET_COL].mean()
        smoothing = 10

        if 'id_31' in df.columns:
            agg = df.groupby('id_31', observed=True)[TARGET_COL].agg(['mean', 'count'])
            self.device_risk_map = (
                (agg['mean'] * agg['count'] + self.global_fraud_rate * smoothing) /
                (agg['count'] + smoothing)
            ).to_dict()

        if 'P_emaildomain' in df.columns:
            email_groups = df['P_emaildomain'].map(self._group_email)
            agg = df.groupby(email_groups, observed=True)[TARGET_COL].agg(['mean', 'count'])
            self.email_risk_map = (
                (agg['mean'] * agg['count'] + self.global_fraud_rate * smoothing) /
                (agg['count'] + smoothing)
            ).to_dict()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies transformations to the dataframe.
        """
        logger.info("Transforming dataframe with engineered features...")
        df = df.copy()
        new_cols = {}

        # --- 1. Temporal Features ---
        if TIME_COL in df.columns:
            new_cols['HourOfDay'] = ((df[TIME_COL] // 3600) % 24).astype('int8')
            new_cols['DayOfWeek'] = ((df[TIME_COL] // (3600 * 24)) % 7).astype('int8')

        # --- 2. Amount Features ---
        if 'TransactionAmt' in df.columns:
            new_cols['TransactionAmt_Log'] = np.log1p(df['TransactionAmt']).astype('float32')
            if 'card1' in df.columns:
                card_mean = df['card1'].map(self.card_means).fillna(df['TransactionAmt'])
                new_cols['AmtToMeanRatio'] = self.safe_divide(df['TransactionAmt'], card_mean)

        # --- 3. Environmental Risk ---
        if 'id_31' in df.columns:
            new_cols['DeviceRisk'] = (
                df['id_31'].map(self.device_risk_map)
                .fillna(self.global_fraud_rate)
                .astype('float32')
            )

        # --- 4. Identity & Email Features ---
        if 'P_emaildomain' in df.columns:
            email_groups = df['P_emaildomain'].map(self._group_email)
            new_cols['EmailDomainRisk'] = (
                email_groups.map(self.email_risk_map)
                .fillna(self.global_fraud_rate)
                .astype('float32')
            )

        # --- 5. Velocity & Sequential Behavioral Features ---
        # Requirement: Sort by TransactionDT to ensure rolling features only use PAST data.
        df_sorted = df.sort_values(TIME_COL)
        
        if 'card1' in df_sorted.columns:
            # Global Frequency
            new_cols['Card_TransactionCount'] = df['card1'].map(self.card_counts).fillna(1).astype('int32')
            
            # A. Time Since Last Transaction
            new_cols['TimeSinceLastTransaction'] = (
                df_sorted.groupby('card1')[TIME_COL]
                .diff()
                .fillna(999999)
                .astype('float32')
            )
            
            # B. Rolling Transaction Count (Last 10)
            new_cols['TransactionsLast10'] = (
                df_sorted.groupby('card1')['TransactionID']
                .transform(lambda x: x.rolling(10, min_periods=1).count())
                .astype('float32')
            )
            
            # C. Rolling Mean Transaction Amount (Last 10)
            rolling_mean = (
                df_sorted.groupby('card1')['TransactionAmt']
                .transform(lambda x: x.rolling(10, min_periods=1).mean())
            )
            new_cols['RollingMeanAmt10'] = rolling_mean.astype('float32')

            # D. Amount Deviation Feature
            new_cols['AmtDeviationFromRollingMean'] = (
                df_sorted['TransactionAmt'] - rolling_mean
            ).astype('float32')

            # E. Device Switching Signal
            if 'id_31' in df_sorted.columns:
                new_cols['UniqueDevicesPerCard'] = (
                    df_sorted.groupby('card1')['id_31']
                    .transform('nunique')
                    .astype('int16')
                )

            # F. Email Instability Signal
            if 'P_emaildomain' in df_sorted.columns:
                new_cols['UniqueEmailsPerCard'] = (
                    df_sorted.groupby('card1')['P_emaildomain']
                    .transform('nunique')
                    .astype('int16')
                )
            
        # Avoid fragmentation and ensure numeric stability
        if new_cols:
            new_cols_df = pd.DataFrame(new_cols, index=df.index)
            
            # Sanitization gate: engineered columns should not have inf or nans
            new_cols_df = new_cols_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Final clip to safe float32 range to prevent overflow in training
            float_cols = new_cols_df.select_dtypes(include=['float64', 'float32']).columns
            if len(float_cols) > 0:
                new_cols_df[float_cols] = new_cols_df[float_cols].clip(-1e6, 1e6).astype('float32')
                
            df = pd.concat([df, new_cols_df], axis=1)

        return df

    def _group_email(self, email: str) -> str:
        if pd.isna(email): return 'unknown'
        email = str(email).lower()
        if 'gmail' in email or 'google' in email: return 'gmail'
        if 'yahoo' in email: return 'yahoo'
        if 'hotmail' in email or 'outlook' in email or 'live' in email or 'msn' in email: return 'microsoft'
        if 'icloud' in email or 'me.com' in email or 'apple' in email: return 'apple'
        if 'aol' in email: return 'aol'
        try:
            parts = email.split('.')
            if len(parts) >= 2: return parts[-2]
            return parts[0]
        except: return 'other'

    def run_full_engineering(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        self.fit(train_df)
        train_eng = self.transform(train_df)
        test_eng = self.transform(test_df)
        self.selected_features = [c for c in train_eng.columns if c not in [TARGET_COL, TIME_COL]]
        return train_eng, test_eng
