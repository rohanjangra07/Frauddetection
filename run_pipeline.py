import logging
from src.utils import logger, validate_file_exists
from src.config import TRAIN_TRANSACTION, TRAIN_IDENTITY, TIME_COL, TEST_SIZE
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.preprocessing import PreprocessingPipeline
from src.train import ModelTrainer
from src.inference import InferenceEngine

def main():
    logger.info("Starting Full End-to-End Real Data Pipeline...")

    # Step 1: Validate Raw Datasets
    logger.info("Step 1: Validating raw data presence...")
    validate_file_exists(TRAIN_TRANSACTION, "Raw Transaction Data")
    validate_file_exists(TRAIN_IDENTITY, "Raw Identity Data")

    # Step 2: Load and Merge Data
    logger.info("Step 2: Loading and merging raw datasets...")
    loader = DataLoader(str(TRAIN_TRANSACTION), str(TRAIN_IDENTITY))
    raw_df = loader.load_raw_data()
    raw_df = loader.reduce_mem_usage(raw_df)

    # Step 3: Chronological Temporal Split
    logger.info("Step 3: Performing chronological temporal split...")
    raw_df = raw_df.sort_values(TIME_COL)
    train_len = int(len(raw_df) * (1 - TEST_SIZE))
    train_df = raw_df.iloc[:train_len].copy()
    test_df = raw_df.iloc[train_len:].copy()
    logger.info(f"Temporal Split -> Train: {train_df.shape}, Test: {test_df.shape}")

    # Step 4, 5, 6: Feature Engineering
    logger.info("Steps 4-6: Fitting FeatureEngineer on Train, transforming Train/Test...")
    fe = FeatureEngineer()
    fe.fit(train_df)
    train_eng = fe.transform(train_df)
    test_eng = fe.transform(test_df)

    # Step 7, 8, 9, 10: Preprocessing & SMOTE
    logger.info("Steps 7-10: Fitting Preprocessor on Train, validating, applying SMOTE, persisting Parquet...")
    preprocessor = PreprocessingPipeline()
    preprocessor.run_preprocessing_and_save(train_eng, test_eng)

    # Step 11: Model Training (with fold-safe target encoding)
    logger.info("Step 11: Training Model on processed Parquet with fold-safe target encoding...")
    trainer = ModelTrainer()
    # Pass fe + raw train_df so target-encoded columns are recomputed per CV fold
    # train_df is reset-indexed to stay aligned with the parquet X matrix (both derive from the same temporal slice)
    trainer.run_training(fe=fe, raw_train_df=train_df.reset_index(drop=True))

    # Step 9-11: Inference and Persistence
    logger.info("Steps 9-11: Running centralized inference and persisting predictions...")
    inference_engine = InferenceEngine()
    inference_engine.run_batch_inference()
    # run_batch_inference saves predictions.parquet

    logger.info("Pipeline Execution Complete. The Dashboard is now ready to launch.")

if __name__ == "__main__":
    main()
