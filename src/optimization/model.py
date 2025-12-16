"""
src/optimization/model.py
MILP Assortment Optimization using Pyomo
Reads: outputs/raw/ (stores, skus, constraints), outputs/forecasts/
Creates: outputs/assortments/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import logging
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.io import load_config, load_parquet, ensure_dir, save_json, save_parquet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = load_config()


def load_optimization_data() -> dict:
    """Load all data needed for optimization"""
    raw_dir = CONFIG['paths']['raw']
    forecast_dir = CONFIG['paths']['forecasts']
    
    data = {
        'stores': load_parquet(f"{raw_dir}/stores.parquet"),
        'skus': load_parquet(f"{raw_dir}/skus.parquet"),
        'forecasts': load_parquet(f"{forecast_dir}/forecasts.parquet"),
        'cat_constraints': load_parquet(f"{raw_dir}/constraints_category.parquet"),
        'sup_constraints': load_parquet(f"{raw_dir}/constraints_supplier.parquet"),
    }
    
    return data


def build_model(data: dict) -> pyo.ConcreteModel:
    """
    Build Pyomo MILP model for assortment optimization
    
    Decision Variable:
        x[s,p] ∈ {0,1} = 1 if SKU p is carried in store s
    
    Objective:
        Maximize total expected profit
    
    Constraints:
        1. Shelf capacity per store
        2. Min/max SKUs per category per store
        3. Supplier minimum requirements
        4. Must-carry items
    """
    logger.info("=" * 60)
    logger.info("BUILDING OPTIMIZATION MODEL")
    logger.info("=" * 60)
    
    stores = data['stores']
    skus = data['skus']
    forecasts = data['forecasts']
    cat_const = data['cat_constraints']
    sup_const = data['sup_constraints']
    
    model = pyo.ConcreteModel("AssortmentOptimization")
    
    # ==================== SETS ====================
    store_ids = stores['store_id'].tolist()
    sku_ids = skus['sku_id'].tolist()
    categories = skus['category'].unique().tolist()
    suppliers = skus['supplier_id'].unique().tolist()
    
    model.S = pyo.Set(initialize=store_ids)
    model.P = pyo.Set(initialize=sku_ids)
    model.C = pyo.Set(initialize=categories)
    model.R = pyo.Set(initialize=suppliers)
    
    # Mappings for constraints
    sku_cat = skus.set_index('sku_id')['category'].to_dict()
    sku_sup = skus.set_index('sku_id')['supplier_id'].to_dict()
    
    model.skus_in_cat = {c: [p for p in sku_ids if sku_cat.get(p) == c] for c in categories}
    model.skus_from_sup = {r: [p for p in sku_ids if sku_sup.get(p) == r] for r in suppliers}
    
    # ==================== PARAMETERS ====================
    
    # Forecasted demand
    fc = forecasts.set_index(['store_id', 'sku_id'])['forecast_units'].to_dict()
    model.forecast = pyo.Param(model.S, model.P, 
                                initialize=lambda m, s, p: fc.get((s, p), 0), 
                                default=0)
    
    # SKU attributes
    sku_data = skus.set_index('sku_id')
    model.margin = pyo.Param(model.P, initialize=lambda m, p: sku_data.loc[p, 'margin'])
    model.space = pyo.Param(model.P, initialize=lambda m, p: sku_data.loc[p, 'space_units'])
    model.must_carry = pyo.Param(model.P, initialize=lambda m, p: 1 if sku_data.loc[p, 'is_must_carry'] else 0)
    
    # Store attributes
    store_data = stores.set_index('store_id')
    model.capacity = pyo.Param(model.S, initialize=lambda m, s: store_data.loc[s, 'shelf_capacity'])
    
    # Category constraints
    cc = cat_const.set_index(['store_id', 'category'])
    model.cat_min = pyo.Param(model.S, model.C,
                               initialize=lambda m, s, c: cc.loc[(s, c), 'min_skus'] if (s, c) in cc.index else 0,
                               default=0)
    model.cat_max = pyo.Param(model.S, model.C,
                               initialize=lambda m, s, c: cc.loc[(s, c), 'max_skus'] if (s, c) in cc.index else 100,
                               default=100)
    
    # Supplier constraints
    if len(sup_const) > 0:
        sc = sup_const.set_index(['store_id', 'supplier_id'])
        model.sup_min = pyo.Param(model.S, model.R,
                                   initialize=lambda m, s, r: sc.loc[(s, r), 'min_skus'] if (s, r) in sc.index else 0,
                                   default=0)
    else:
        model.sup_min = pyo.Param(model.S, model.R, default=0)
    
    # ==================== DECISION VARIABLE ====================
    model.x = pyo.Var(model.S, model.P, domain=pyo.Binary)
    
    # ==================== OBJECTIVE ====================
    # Maximize: sum of (forecast * margin) for selected SKUs
    def obj_rule(m):
        return sum(m.forecast[s, p] * m.margin[p] * m.x[s, p] 
                   for s in m.S for p in m.P)
    
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
    
    # ==================== CONSTRAINTS ====================
    
    # 1. Shelf capacity constraint
    def shelf_rule(m, s):
        return sum(m.space[p] * m.x[s, p] for p in m.P) <= m.capacity[s]
    model.shelf_con = pyo.Constraint(model.S, rule=shelf_rule)
    
    # 2. Category minimum constraint
    def cat_min_rule(m, s, c):
        skus_c = model.skus_in_cat.get(c, [])
        if not skus_c:
            return pyo.Constraint.Skip
        return sum(m.x[s, p] for p in skus_c) >= m.cat_min[s, c]
    model.cat_min_con = pyo.Constraint(model.S, model.C, rule=cat_min_rule)
    
    # 3. Category maximum constraint
    def cat_max_rule(m, s, c):
        skus_c = model.skus_in_cat.get(c, [])
        if not skus_c:
            return pyo.Constraint.Skip
        return sum(m.x[s, p] for p in skus_c) <= m.cat_max[s, c]
    model.cat_max_con = pyo.Constraint(model.S, model.C, rule=cat_max_rule)
    
    # 4. Must-carry constraint
    def must_rule(m, s, p):
        if m.must_carry[p] == 1:
            return m.x[s, p] == 1
        return pyo.Constraint.Skip
    model.must_con = pyo.Constraint(model.S, model.P, rule=must_rule)
    
    # 5. Supplier minimum constraint
    def sup_rule(m, s, r):
        skus_r = model.skus_from_sup.get(r, [])
        min_req = pyo.value(m.sup_min[s, r])
        if not skus_r or min_req == 0:
            return pyo.Constraint.Skip
        return sum(m.x[s, p] for p in skus_r) >= min_req
    model.sup_con = pyo.Constraint(model.S, model.R, rule=sup_rule)
    
    # Log model size
    n_vars = len(store_ids) * len(sku_ids)
    logger.info(f"Stores: {len(store_ids)}, SKUs: {len(sku_ids)}")
    logger.info(f"Binary variables: {n_vars:,}")
    logger.info(f"Categories: {len(categories)}, Suppliers: {len(suppliers)}")
    
    return model


def solve_model(model: pyo.ConcreteModel) -> dict:
    """Solve the optimization model"""
    logger.info("=" * 60)
    logger.info("SOLVING OPTIMIZATION MODEL")
    logger.info("=" * 60)
    
    solver_name = CONFIG['optimization']['solver']
    time_limit = CONFIG['optimization']['time_limit']
    gap = CONFIG['optimization']['mip_gap']
    
    solver = SolverFactory(solver_name)
    
    # Set solver options
    if solver_name == 'cbc':
        solver.options['seconds'] = time_limit
        solver.options['ratioGap'] = gap
    
    logger.info(f"Solver: {solver_name}")
    logger.info(f"Time limit: {time_limit}s, MIP gap: {gap}")
    
    start = time.time()
    result = solver.solve(model, tee=True)
    elapsed = time.time() - start
    
    status = {
        'solver': solver_name,
        'status': str(result.solver.status),
        'termination': str(result.solver.termination_condition),
        'time_seconds': round(elapsed, 1),
        'objective': None,
        'optimal': False
    }
    
    if result.solver.termination_condition == TerminationCondition.optimal:
        status['optimal'] = True
        status['objective'] = round(pyo.value(model.obj), 2)
        logger.info(f"\n✓ OPTIMAL SOLUTION FOUND")
        logger.info(f"  Objective (Total Profit): ${status['objective']:,.2f}")
    elif result.solver.termination_condition == TerminationCondition.feasible:
        status['objective'] = round(pyo.value(model.obj), 2)
        logger.info(f"\nFeasible solution found: ${status['objective']:,.2f}")
    else:
        logger.warning(f"\nNo solution found: {status['termination']}")
    
    logger.info(f"  Solve time: {elapsed:.1f} seconds")
    
    return status


def extract_solution(model: pyo.ConcreteModel, data: dict) -> pd.DataFrame:
    """Extract solution as DataFrame"""
    skus = data['skus'].set_index('sku_id')
    forecasts = data['forecasts'].set_index(['store_id', 'sku_id'])
    
    rows = []
    for s in model.S:
        for p in model.P:
            if pyo.value(model.x[s, p]) > 0.5:  # Selected
                fc = forecasts.loc[(s, p), 'forecast_units'] if (s, p) in forecasts.index else 0
                margin = skus.loc[p, 'margin']
                rows.append({
                    'store_id': s,
                    'sku_id': p,
                    'category': skus.loc[p, 'category'],
                    'forecast_units': fc,
                    'expected_profit': round(fc * margin, 2)
                })
    
    return pd.DataFrame(rows)


def run_optimization():
    """Full optimization pipeline"""
    # Load data
    data = load_optimization_data()
    
    # Build model
    model = build_model(data)
    
    # Solve
    status = solve_model(model)
    
    if status['objective']:
        # Extract solution
        solution = extract_solution(model, data)
        
        # Calculate summary metrics
        metrics = {
            **status,
            'total_selections': len(solution),
            'unique_stores': solution['store_id'].nunique(),
            'unique_skus': solution['sku_id'].nunique(),
            'total_profit': round(solution['expected_profit'].sum(), 2),
            'avg_skus_per_store': round(len(solution) / solution['store_id'].nunique(), 1)
        }
        
        # Save outputs
        out_dir = CONFIG['paths']['assortments']
        ensure_dir(out_dir)
        
        save_parquet(solution, f"{out_dir}/assortment_output.parquet")
        save_json(metrics, f"{out_dir}/metrics.json")
        
        logger.info("=" * 60)
        logger.info("OPTIMIZATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Expected Profit: ${metrics['total_profit']:,}")
        logger.info(f"Store-SKU Selections: {metrics['total_selections']:,}")
        logger.info(f"Avg SKUs per Store: {metrics['avg_skus_per_store']}")
        logger.info(f"\nSaved to {out_dir}/")
        
        return solution, metrics
    
    return None, status


if __name__ == '__main__':
    run_optimization()