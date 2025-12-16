"""
src/forecasting/train.py
Train demand forecasting model
Reads: outputs/features/train.parquet
Creates: outputs/models/model.joblib
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import logging
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.io import load_config, load_parquet, ensure_dir, save_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = load_config()


def load_features():
    """Load feature data created by build_features.py"""
    feat_dir = CONFIG['paths']['features']
    train = load_parquet(f"{feat_dir}/train.parquet")
    test = load_parquet(f"{feat_dir}/test.parquet")
    score = load_parquet(f"{feat_dir}/score.parquet")
    feature_cols = pd.read_csv(f"{feat_dir}/feature_cols.csv", header=None)[0].tolist()
    return train, test, score, feature_cols


def train_model():
    """Train forecasting model with cross-validation"""
    logger.info("=" * 60)
    logger.info("TRAINING DEMAND FORECAST MODEL")
    logger.info("=" * 60)
    
    train_df, test_df, _, feature_cols = load_features()
    
    X_train = train_df[feature_cols]
    y_train = train_df['units']
    X_test = test_df[feature_cols]
    y_test = test_df['units']
    
    # Model parameters from config
    params = CONFIG['forecasting']['params']
    
    # Cross-validation
    logger.info("Running time-series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=CONFIG['forecasting']['cv_folds'])
    cv_rmse = []
    
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
        model = GradientBoostingRegressor(**params)
        model.fit(X_tr, y_tr)
        
        pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        cv_rmse.append(rmse)
        logger.info(f"  Fold {fold+1}: RMSE = {rmse:.2f}")
    
    # Train final model on all training data
    logger.info("\nTraining final model...")
    final_model = GradientBoostingRegressor(**params)
    final_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = final_model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)  # No negative predictions
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100
    
    metrics = {
        'cv_rmse_mean': float(np.mean(cv_rmse)),
        'cv_rmse_std': float(np.std(cv_rmse)),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'test_mape': float(test_mape),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': len(feature_cols)
    }
    
    logger.info(f"\n{'='*40}")
    logger.info("MODEL PERFORMANCE")
    logger.info(f"{'='*40}")
    logger.info(f"CV RMSE: {metrics['cv_rmse_mean']:.2f} (+/- {metrics['cv_rmse_std']:.2f})")
    logger.info(f"Test RMSE: {metrics['test_rmse']:.2f}")
    logger.info(f"Test MAE: {metrics['test_mae']:.2f}")
    logger.info(f"Test MAPE: {metrics['test_mape']:.1f}%")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"\nTop 5 features:")
    for _, row in importance.head(5).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model and metrics
    model_dir = CONFIG['paths']['models']
    ensure_dir(model_dir)
    
    joblib.dump(final_model, f"{model_dir}/model.joblib")
    save_json(metrics, f"{model_dir}/metrics.json")
    importance.to_parquet(f"{model_dir}/feature_importance.parquet", index=False)
    
    logger.info(f"\nModel saved to {model_dir}/model.joblib")
    logger.info("=" * 60)
    
    return final_model, metrics


if __name__ == '__main__':
    train_model()