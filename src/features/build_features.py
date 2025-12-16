"""
src/features/build_features.py
Feature engineering for demand forecasting
Reads: outputs/raw/ (stores, skus, sales)
Creates: outputs/features/ (train.parquet, score.parquet)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.io import load_config, load_parquet, save_parquet, ensure_dir

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = load_config()


def load_raw_data() -> dict:
    """Load raw data files created by generate_data.py"""
    raw_dir = CONFIG['paths']['raw']
    return {
        'stores': load_parquet(f"{raw_dir}/stores.parquet"),
        'skus': load_parquet(f"{raw_dir}/skus.parquet"),
        'sales': load_parquet(f"{raw_dir}/sales.parquet")
    }


def create_lag_features(df: pd.DataFrame, lags: list) -> pd.DataFrame:
    """Create lag features for time series"""
    df = df.sort_values(['store_id', 'sku_id', 'week'])
    
    for lag in lags:
        df[f'units_lag_{lag}'] = df.groupby(['store_id', 'sku_id'])['units'].shift(lag)
    
    return df


def create_rolling_features(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """Create rolling mean/std features"""
    df = df.sort_values(['store_id', 'sku_id', 'week'])
    
    for window in windows:
        df[f'units_rmean_{window}'] = (
            df.groupby(['store_id', 'sku_id'])['units']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f'units_rstd_{window}'] = (
            df.groupby(['store_id', 'sku_id'])['units']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
        )
    
    return df


def build_features() -> tuple:
    """Build train and score feature sets"""
    logger.info("=" * 60)
    logger.info("BUILDING FEATURES")
    logger.info("=" * 60)
    
    # Load raw data (created by generate_data.py)
    data = load_raw_data()
    sales = data['sales'].copy()
    stores = data['stores']
    skus = data['skus']
    
    # Parse dates
    sales['week_date'] = pd.to_datetime(sales['week_start'])
    sales = sales.sort_values(['store_id', 'sku_id', 'week_date'])
    
    # Temporal features
    logger.info("Creating temporal features...")
    sales['week_of_year'] = sales['week_date'].dt.isocalendar().week.astype(int)
    sales['month'] = sales['week_date'].dt.month
    sales['quarter'] = sales['week_date'].dt.quarter
    sales['is_holiday_season'] = sales['month'].isin([11, 12]).astype(int)
    
    # Lag features
    logger.info("Creating lag features...")
    lags = CONFIG['features']['lag_weeks']
    sales = create_lag_features(sales, lags)
    
    # Rolling features
    logger.info("Creating rolling features...")
    windows = CONFIG['features']['rolling_windows']
    sales = create_rolling_features(sales, windows)
    
    # Merge store attributes
    logger.info("Merging store/SKU attributes...")
    store_cols = ['store_id', 'region', 'store_type', 'store_size', 'shelf_capacity']
    sales = sales.merge(stores[store_cols], on='store_id', how='left')
    
    # Merge SKU attributes
    sku_cols = ['sku_id', 'category', 'brand', 'price', 'margin', 'space_units']
    sales = sales.merge(skus[sku_cols], on='sku_id', how='left')
    
    # Encode categoricals
    logger.info("Encoding categorical variables...")
    cat_cols = ['store_id', 'sku_id', 'region', 'store_type', 'store_size', 'category', 'brand']
    for col in cat_cols:
        sales[f'{col}_enc'] = sales[col].astype('category').cat.codes
    
    # Drop NaN rows (from lags at start of time series)
    initial_count = len(sales)
    sales = sales.dropna()
    logger.info(f"Dropped {initial_count - len(sales)} rows with NaN")
    
    # Define feature columns (used by forecasting model)
    feature_cols = (
        ['week_of_year', 'month', 'quarter', 'is_holiday_season', 'promo_flag'] +
        [f'units_lag_{l}' for l in lags] +
        [f'units_rmean_{w}' for w in windows] +
        [f'units_rstd_{w}' for w in windows] +
        ['shelf_capacity', 'price', 'margin', 'space_units'] +
        [f'{c}_enc' for c in cat_cols]
    )
    
    # Train/test split (time-based)
    test_weeks = CONFIG['features']['test_weeks']
    max_date = sales['week_date'].max()
    split_date = max_date - pd.Timedelta(weeks=test_weeks)
    
    train_df = sales[sales['week_date'] <= split_date].copy()
    test_df = sales[sales['week_date'] > split_date].copy()
    
    # Score set: latest week per store-sku (for generating forecasts)
    score_df = sales.sort_values('week_date').groupby(['store_id', 'sku_id']).last().reset_index()
    
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}, Score: {len(score_df)}")
    
    # Save to outputs/features/
    output_dir = CONFIG['paths']['features']
    ensure_dir(output_dir)
    
    # Keep ID columns + features + target
    id_cols = ['store_id', 'sku_id', 'week', 'week_date']
    target = ['units']
    all_cols = id_cols + feature_cols + target
    
    save_parquet(train_df[all_cols], f"{output_dir}/train.parquet")
    save_parquet(test_df[all_cols], f"{output_dir}/test.parquet")
    save_parquet(score_df[all_cols], f"{output_dir}/score.parquet")
    
    # Save feature column names (used by train.py)
    pd.Series(feature_cols).to_csv(f"{output_dir}/feature_cols.csv", index=False, header=False)
    
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 60)
    
    return train_df, test_df, score_df, feature_cols


if __name__ == '__main__':
    build_features()