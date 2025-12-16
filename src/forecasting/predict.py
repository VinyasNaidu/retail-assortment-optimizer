"""
src/forecasting/predict.py
Generate demand forecasts for optimization
Reads: outputs/models/model.joblib, outputs/features/score.parquet
Creates: outputs/forecasts/forecasts.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.io import load_config, load_parquet, save_parquet, ensure_dir

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = load_config()


def load_model():
    """Load trained model"""
    model_dir = CONFIG['paths']['models']
    model = joblib.load(f"{model_dir}/model.joblib")
    logger.info("Model loaded")
    return model


def load_score_data():
    """Load scoring data and feature columns"""
    feat_dir = CONFIG['paths']['features']
    score_df = load_parquet(f"{feat_dir}/score.parquet")
    feature_cols = pd.read_csv(f"{feat_dir}/feature_cols.csv", header=None)[0].tolist()
    return score_df, feature_cols


def generate_forecasts():
    """Generate demand forecasts for all store-SKU pairs"""
    logger.info("=" * 60)
    logger.info("GENERATING FORECASTS")
    logger.info("=" * 60)
    
    # Load model and data
    model = load_model()
    score_df, feature_cols = load_score_data()
    
    # Predict
    X = score_df[feature_cols]
    predictions = model.predict(X)
    predictions = np.maximum(predictions, 0)  # No negative demand
    
    # Scale to forecast horizon
    horizon = CONFIG['forecasting']['forecast_weeks']
    
    forecasts = pd.DataFrame({
        'store_id': score_df['store_id'],
        'sku_id': score_df['sku_id'],
        'forecast_units': (predictions * horizon).round(0).astype(int),
        'forecast_weekly': predictions.round(1)
    })
    
    logger.info(f"Generated forecasts for {len(forecasts)} store-SKU pairs")
    logger.info(f"Forecast horizon: {horizon} weeks")
    logger.info(f"Total units forecasted: {forecasts['forecast_units'].sum():,}")
    
    # Save
    forecast_dir = CONFIG['paths']['forecasts']
    ensure_dir(forecast_dir)
    save_parquet(forecasts, f"{forecast_dir}/forecasts.parquet")
    
    logger.info("=" * 60)
    logger.info("FORECASTS COMPLETE")
    logger.info("=" * 60)
    
    return forecasts


if __name__ == '__main__':
    generate_forecasts()