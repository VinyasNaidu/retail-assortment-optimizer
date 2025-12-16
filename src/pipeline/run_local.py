"""
src/pipeline/run_local.py
One-command pipeline runner for entire project
"""

import argparse
import time
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_pipeline(from_step: int = 1):
    """
    Run pipeline starting from a specific step
    
    Steps:
        1 = Data Generation (creates new synthetic data)
        2 = Feature Engineering (reads raw data)
        3 = Model Training (reads features)
        4 = Prediction (reads model + features)
        5 = Optimization (reads forecasts + raw data)
    
    If from_step=3, it skips 1 & 2 and starts from training.
    """
    start = time.time()
    
    print("\n" + "=" * 70)
    print("   RETAIL ASSORTMENT OPTIMIZATION PIPELINE")
    print("=" * 70)
    print(f"   Starting from step {from_step}")
    print("=" * 70 + "\n")
    
    # Step 1: Data Generation
    print("\n[1/5] DATA GENERATION")
    print("-" * 40)
    if from_step <= 1:
        from src.data.generate_data import generate_all
        generate_all()
    else:
        print("⏭ Skipped (using existing data)")
    
    # Step 2: Feature Engineering
    print("\n[2/5] FEATURE ENGINEERING")
    print("-" * 40)
    if from_step <= 2:
        from src.features.build_features import build_features
        build_features()
    else:
        print("⏭ Skipped (using existing features)")
    
    # Step 3: Model Training
    print("\n[3/5] DEMAND FORECASTING - Training")
    print("-" * 40)
    if from_step <= 3:
        from src.forecasting.train import train_model
        train_model()
    else:
        print("⏭ Skipped (using existing model)")
    
    # Step 4: Generate Forecasts
    print("\n[4/5] DEMAND FORECASTING - Prediction")
    print("-" * 40)
    if from_step <= 4:
        from src.forecasting.predict import generate_forecasts
        generate_forecasts()
    else:
        print("⏭ Skipped (using existing forecasts)")
    
    # Step 5: Optimization (always runs)
    print("\n[5/5] ASSORTMENT OPTIMIZATION")
    print("-" * 40)
    from src.optimization.model import run_optimization
    solution, metrics = run_optimization()
    
    # Evaluation
    if solution is not None:
        print("\n[EVALUATION]")
        print("-" * 40)
        from src.optimization.evaluate import evaluate_solution
        evaluate_solution()
    
    # Summary
    elapsed = time.time() - start
    
    print("\n" + "=" * 70)
    print(f"   PIPELINE COMPLETE - Total time: {elapsed/60:.1f} minutes")
    print("=" * 70)
    
    print("\n📁 OUTPUT FILES:")
    print("   outputs/raw/           → Synthetic retail data")
    print("   outputs/features/      → ML features")
    print("   outputs/models/        → Trained demand model")
    print("   outputs/forecasts/     → Demand forecasts")
    print("   outputs/assortments/   → Optimal assortment solution")
    print("   outputs/reports/       → Evaluation & final report")


def main():
    parser = argparse.ArgumentParser(
        description='Retail Assortment Optimization Pipeline',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--from-step', 
        type=int, 
        choices=[1, 2, 3, 4, 5],
        default=5,
        help='''Start pipeline from step N (default: 5 = optimization only)
1 = Full pipeline (regenerate everything)
2 = From features (keep existing raw data)
3 = From training (keep existing features)
4 = From prediction (keep existing model)
5 = Optimization only (keep existing forecasts)'''
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full pipeline from scratch (same as --from-step 1)'
    )
    
    args = parser.parse_args()
    
    if args.full:
        run_pipeline(from_step=1)
    else:
        run_pipeline(from_step=args.from_step)


if __name__ == '__main__':
    main()