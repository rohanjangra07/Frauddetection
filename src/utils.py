import logging
import sys
from pathlib import Path
from typing import Optional

# Centralized Absolute Path Handling
BASE_DIR = Path(__file__).resolve().parent.parent

def ensure_directories():
    """
    Safely creates all required project directories using absolute paths.
    Prevents FileNotFoundError during artifact saving/logging.
    """
    directories = [
        BASE_DIR / "outputs",
        BASE_DIR / "logs",
        BASE_DIR / "charts",
        BASE_DIR / "models",
        BASE_DIR / "models" / "trained_models",
        BASE_DIR / "models" / "artifacts",
        BASE_DIR / "data",
        BASE_DIR / "data" / "processed",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def validate_file_exists(path: Path, description: str):
    """
    Strict validation to ensure required files exist.
    Raises FileNotFoundError if missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}. Ensure the pipeline is executed in order.")

# Run directory creation immediately upon import to guarantee they exist
ensure_directories()

def setup_logging(log_file: Optional[str] = "pipeline.log") -> logging.Logger:
    """
    Sets up professional, Streamlit-safe logging to both console and file.
    """
    logger = logging.getLogger("FraudDetection")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers (crucial for Streamlit reruns)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        # Use absolute path to the logs directory
        log_path = BASE_DIR / "logs" / log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Initialize common logger
logger = setup_logging()

def log_dataframe_info(df, name: str):
    """
    Utility to log basic dataframe stats.
    """
    logger.info(f"--- DataFrame Info: {name} ---")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    logger.info(f"Columns: {list(df.columns)}")
