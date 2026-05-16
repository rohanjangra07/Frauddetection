import optuna
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score
from typing import Dict, Any
from src.utils import logger

class HyperparameterTuner:
    """
    Production-grade hyperparameter tuning module using Optuna.
    Optimizes models strictly on Precision-Recall AUC (PR-AUC) using 
    TimeSeriesSplit to prevent validation leakage.
    """
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 3):
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.study = None
        self.best_params = {}

    def _lgb_objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function for LightGBM.
        """
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
            'is_unbalance': True,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        pr_aucs = []
        
        for train_idx, val_idx in tscv.split(self.X):
            X_tr, y_tr = self.X.iloc[train_idx], self.y.iloc[train_idx]
            X_val, y_val = self.X.iloc[val_idx], self.y.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**param)
            
            # Using Early Stopping to prevent excessive search complexity and overfitting
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            fold_pr_auc = average_precision_score(y_val, y_pred_proba)
            pr_aucs.append(fold_pr_auc)
            
        return np.mean(pr_aucs)

    def tune(self, model_name: str, n_trials: int = 20) -> Dict[str, Any]:
        """
        Executes the Optuna study to find optimal hyperparameters.
        """
        logger.info(f"Starting Hyperparameter Tuning for {model_name} ({n_trials} trials)...")
        
        # Maximize PR-AUC
        self.study = optuna.create_study(direction="maximize")
        
        if model_name == 'lightgbm':
            self.study.optimize(self._lgb_objective, n_trials=n_trials, n_jobs=1)
        else:
            raise NotImplementedError(f"Tuning for {model_name} is currently not implemented.")
            
        self.best_params = self.study.best_params
        
        logger.info(f"Tuning Complete. Best PR-AUC: {self.study.best_value:.4f}")
        logger.info(f"Best Params: {self.best_params}")
        
        return self.best_params

    def plot_optimization_history(self):
        """
        Visualizes the improvement of the objective value over all trials.
        """
        if self.study:
            fig = optuna.visualization.plot_optimization_history(self.study)
            fig.show()
        else:
            logger.warning("No tuning history to plot.")

    def plot_param_importances(self):
        """
        Visualizes which hyperparameters had the most impact on PR-AUC.
        """
        if self.study:
            fig = optuna.visualization.plot_param_importances(self.study)
            fig.show()
        else:
            logger.warning("No tuning history to plot.")
