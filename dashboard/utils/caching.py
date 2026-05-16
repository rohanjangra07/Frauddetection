import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import os

# Resolving paths robustly without circular imports
DASHBOARD_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = DASHBOARD_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_PATH = OUTPUTS_DIR / "predictions.parquet"
BEST_MODEL_PATH = TRAINED_MODELS_DIR / "best_model.pkl"

@st.cache_resource
def load_model():
    """
    Loads the real trained LightGBM model. 
    Searches multiple paths to handle both local and cloud deployment.
    """
    # Try multiple common paths for robustness
    possible_paths = [
        BEST_MODEL_PATH,
        PROJECT_ROOT / "models" / "trained_models" / "best_model.pkl",
        DASHBOARD_DIR.parent / "models" / "trained_models" / "best_model.pkl"
    ]
    
    model_path = None
    for p in possible_paths:
        if p.exists():
            model_path = p
            break
            
    if not model_path:
        st.error(f"❌ Model artifact not found. Please run 'python run_pipeline.py' or check models/trained_models/best_model.pkl")
        st.stop()
        
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()

@st.cache_data
def load_processed_data():
    """
    Loads the REAL predictions dataset. 
    Fallbacks to sample_data.parquet for cloud deployment demonstration.
    """
    SAMPLE_PATH = PROJECT_ROOT / "data" / "sample_data.parquet"
    
    path_to_load = None
    is_sample = False
    
    if PREDICTIONS_PATH.exists():
        path_to_load = PREDICTIONS_PATH
    elif SAMPLE_PATH.exists():
        path_to_load = SAMPLE_PATH
        is_sample = True
    else:
        st.error(f"❌ No data artifacts found (checked outputs/predictions.parquet and data/sample_data.parquet).")
        st.stop()
        
    try:
        df = pd.read_parquet(path_to_load)
        
        if is_sample:
            st.warning("⚠️ Running in **Simulation Mode** using sample deployment data.")
        
        # Strict validation of required inference schema
        required_cols = ['TransactionID', 'TransactionAmt', 'prediction_probability', 'Risk_Tier']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Data artifact missing required schema columns: {missing_cols}")
            st.stop()
            
        return df
    except Exception as e:
        st.error(f"❌ Failed to load data: {str(e)}")
        st.stop()
