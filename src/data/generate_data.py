"""
src/data/generate_data.py
Generate synthetic retail data (Parquet output)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.io import load_config, save_parquet, ensure_dir

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = load_config()
np.random.seed(CONFIG['data']['seed'])


def generate_stores() -> pd.DataFrame:
    """Generate store master data"""
    n = CONFIG['data']['num_stores']
    regions = CONFIG['data']['regions']
    types = CONFIG['data']['store_types']
    sizes = CONFIG['data']['store_sizes']
    
    capacity_map = {'Small': 5000, 'Medium': 10000, 'Large': 18000}
    
    stores = []
    for i in range(n):
        size = np.random.choice(sizes)
        stores.append({
            'store_id': f'S{str(i+1).zfill(4)}',
            'region': np.random.choice(regions),
            'store_type': np.random.choice(types),
            'store_size': size,
            'shelf_capacity': capacity_map[size] + np.random.randint(-500, 500),
            'weekly_traffic': np.random.randint(5000, 50000)
        })
    
    logger.info(f"Generated {n} stores")
    return pd.DataFrame(stores)


def generate_skus() -> pd.DataFrame:
    """Generate SKU master data"""
    n = CONFIG['data']['num_skus']
    categories = CONFIG['data']['categories']
    num_suppliers = CONFIG['data']['num_suppliers']
    
    skus = []
    per_category = n // len(categories)
    
    for cat_idx, category in enumerate(categories):
        for i in range(per_category):
            sku_num = cat_idx * per_category + i + 1
            
            # Price varies by category
            if category in ['Meat', 'Personal Care']:
                price = np.random.uniform(6, 22)
            elif category in ['Beverages', 'Snacks']:
                price = np.random.uniform(1.5, 7)
            else:
                price = np.random.uniform(2, 12)
            
            cost = price * np.random.uniform(0.45, 0.65)
            
            skus.append({
                'sku_id': f'SKU{str(sku_num).zfill(5)}',
                'category': category,
                'brand': f'Brand_{np.random.randint(1, 35)}',
                'supplier_id': f'SUP{str(np.random.randint(1, num_suppliers+1)).zfill(3)}',
                'cost': round(cost, 2),
                'price': round(price, 2),
                'margin': round(price - cost, 2),
                'space_units': np.random.randint(1, 4),
                'is_must_carry': np.random.random() < 0.03
            })
    
    logger.info(f"Generated {len(skus)} SKUs across {len(categories)} categories")
    return pd.DataFrame(skus)


def generate_sales(stores_df: pd.DataFrame, skus_df: pd.DataFrame) -> pd.DataFrame:
    """Generate weekly sales history"""
    weeks = CONFIG['data']['weeks_history']
    logger.info(f"Generating {weeks} weeks of sales...")
    
    store_ids = stores_df['store_id'].tolist()
    sku_ids = skus_df['sku_id'].tolist()
    
    # Lookups
    store_data = stores_df.set_index('store_id')
    sku_data = skus_df.set_index('sku_id')
    
    size_mult = {'Small': 0.6, 'Medium': 1.0, 'Large': 1.4}
    cat_mult = {'Beverages': 1.3, 'Snacks': 1.2, 'Dairy': 1.0, 'Meat': 0.7, 'Frozen': 0.8}
    
    start_date = datetime.now() - timedelta(weeks=weeks)
    records = []
    
    for week_num in range(weeks):
        if week_num % 25 == 0:
            logger.info(f"  Week {week_num}/{weeks}")
        
        week_start = start_date + timedelta(weeks=week_num)
        week_str = week_start.strftime('%Y-W%W')
        month = week_start.month
        
        # Seasonality
        seasonal = {12: 1.4, 11: 1.2, 1: 0.85, 7: 1.1}.get(month, 1.0)
        
        for store_id in store_ids:
            # Each store carries ~50% of SKUs
            active_skus = np.random.choice(sku_ids, size=int(len(sku_ids)*0.5), replace=False)
            store_size = store_data.loc[store_id, 'store_size']
            
            for sku_id in active_skus:
                category = sku_data.loc[sku_id, 'category']
                promo = int(np.random.random() < 0.15)
                
                base = 10 * size_mult.get(store_size, 1.0) * cat_mult.get(category, 1.0)
                units = max(1, int(base * seasonal * (1.4 if promo else 1.0) * np.random.exponential(1.2)))
                
                records.append({
                    'store_id': store_id,
                    'sku_id': sku_id,
                    'week': week_str,
                    'week_start': week_start.strftime('%Y-%m-%d'),
                    'units': units,
                    'promo_flag': promo
                })
    
    logger.info(f"Generated {len(records)} sales records")
    return pd.DataFrame(records)


def generate_constraints_category(stores_df: pd.DataFrame) -> pd.DataFrame:
    """Generate category min/max constraints per store"""
    categories = CONFIG['data']['categories']
    
    records = []
    for _, store in stores_df.iterrows():
        size_mult = {'Small': 0.5, 'Medium': 1.0, 'Large': 1.5}[store['store_size']]
        
        for category in categories:
            records.append({
                'store_id': store['store_id'],
                'category': category,
                'min_skus': int(CONFIG['optimization']['defaults']['min_skus_per_category'] * size_mult),
                'max_skus': int(CONFIG['optimization']['defaults']['max_skus_per_category'] * size_mult)
            })
    
    logger.info(f"Generated {len(records)} category constraints")
    return pd.DataFrame(records)


def generate_constraints_supplier(stores_df: pd.DataFrame) -> pd.DataFrame:
    """Generate supplier minimum constraints"""
    num_suppliers = CONFIG['data']['num_suppliers']
    
    records = []
    # Only some suppliers have minimums
    suppliers_with_mins = np.random.choice(
        range(1, num_suppliers + 1), 
        size=num_suppliers // 3, 
        replace=False
    )
    
    for sup_num in suppliers_with_mins:
        supplier_id = f'SUP{str(sup_num).zfill(3)}'
        # Apply to random subset of stores
        selected_stores = np.random.choice(
            stores_df['store_id'].tolist(),
            size=int(len(stores_df) * 0.6),
            replace=False
        )
        for store_id in selected_stores:
            records.append({
                'store_id': store_id,
                'supplier_id': supplier_id,
                'min_skus': np.random.choice([3, 5, 8])
            })
    
    logger.info(f"Generated {len(records)} supplier constraints")
    return pd.DataFrame(records)


def generate_all():
    """Generate all datasets"""
    logger.info("=" * 60)
    logger.info("GENERATING SYNTHETIC RETAIL DATA")
    logger.info("=" * 60)
    
    output_dir = CONFIG['paths']['raw']
    ensure_dir(output_dir)
    
    # Generate in order (dependencies matter)
    stores = generate_stores()
    skus = generate_skus()
    sales = generate_sales(stores, skus)
    cat_constraints = generate_constraints_category(stores)
    sup_constraints = generate_constraints_supplier(stores)
    
    # Save all files
    save_parquet(stores, f"{output_dir}/stores.parquet")
    save_parquet(skus, f"{output_dir}/skus.parquet")
    save_parquet(sales, f"{output_dir}/sales.parquet")
    save_parquet(cat_constraints, f"{output_dir}/constraints_category.parquet")
    save_parquet(sup_constraints, f"{output_dir}/constraints_supplier.parquet")
    
    logger.info("=" * 60)
    logger.info("DATA GENERATION COMPLETE")
    logger.info("=" * 60)
    
    return {'stores': stores, 'skus': skus, 'sales': sales}


if __name__ == '__main__':
    generate_all()