"""
src/data/io.py
Local file read/write helpers - used by all other modules
"""

import pandas as pd
from pathlib import Path
import json
import yaml
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_parquet(df: pd.DataFrame, path: str):
    """Save DataFrame as Parquet"""
    p = Path(path)
    ensure_dir(p.parent)
    df.to_parquet(p, index=False)
    logger.info(f"Saved {len(df)} rows → {p}")


def load_parquet(path: str) -> pd.DataFrame:
    """Load Parquet file"""
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows ← {path}")
    return df


def save_json(data: dict, path: str):
    """Save dictionary as JSON"""
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved JSON → {p}")


def load_json(path: str) -> dict:
    """Load JSON file"""
    with open(path, 'r') as f:
        return json.load(f)