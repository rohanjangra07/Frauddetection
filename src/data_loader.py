import pandas as pd
import numpy as np
from src.config import TRAIN_TRANSACTION, TRAIN_IDENTITY, ID_COL
from src.utils import logger, log_dataframe_info

class DataLoader:
    """
    Handles loading of transaction and identity datasets and merging them.
    """
    
    def __init__(self, transaction_path: str = TRAIN_TRANSACTION, identity_path: str = TRAIN_IDENTITY):
        self.transaction_path = transaction_path
        self.identity_path = identity_path

    def load_raw_data(self) -> pd.DataFrame:
        """
        Loads raw CSV files and merges them on TransactionID.
        """
        try:
            logger.info(f"Loading transaction data from {self.transaction_path}...")
            train_trans = pd.read_csv(self.transaction_path)
            
            logger.info(f"Loading identity data from {self.identity_path}...")
            train_id = pd.read_csv(self.identity_path)
            
            # Merge datasets
            logger.info("Merging transaction and identity data...")
            df = pd.merge(train_trans, train_id, on=ID_COL, how='left')
            
            log_dataframe_info(df, "Merged Raw Data")
            return df
            
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during data loading: {e}")
            raise

    @staticmethod
    def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterate through all columns of a dataframe and modify the data type
        to reduce memory usage.
        
        CRITICAL FIX: float16 is removed. It overflows at 65,504, which is 
        frequently exceeded in IEEE-CIS features, causing 'overflow in cast' warnings.
        """
        start_mem = df.memory_usage().sum() / 1024**2
        logger.info(f'Memory usage of dataframe is {start_mem:.2f} MB')
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object and not pd.api.types.is_categorical_dtype(df[col]):
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    # NEVER use float16. It is too small for transaction amounts and V-columns.
                    # float32 is the safe production standard for ML pipelines.
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            elif col_type == object:
                df[col] = df[col].astype('category')

        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f'Memory usage after optimization is: {end_mem:.2f} MB')
        logger.info(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
        
        return df
