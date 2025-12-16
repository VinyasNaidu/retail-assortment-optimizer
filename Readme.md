# 🛒 Retail Assortment Optimization System

> **SKU-level assortment optimization using Mixed-Integer Linear Programming (MILP) and Machine Learning demand forecasting**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Pyomo](https://img.shields.io/badge/Pyomo-6.6+-green.svg)](http://www.pyomo.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Business Problem](#-business-problem)
- [Solution Overview](#-solution-overview)
- [Results & Impact](#-results--impact)
- [Technical Architecture](#-technical-architecture)
- [Data Pipeline](#-data-pipeline)
- [Machine Learning Model](#-machine-learning-model)
- [Optimization Model](#-optimization-model)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [AWS Deployment](#-aws-deployment)
- [Future Enhancements](#-future-enhancements)

---

## 🎯 Business Problem

### The Challenge

Retail chains face a critical decision: **Which products should each store carry?**

This decision directly impacts:
- **Revenue**: Carrying the right products maximizes sales
- **Profitability**: Product margins vary significantly
- **Customer satisfaction**: Customers expect to find what they need
- **Operational efficiency**: Shelf space is limited and expensive

### Real-World Constraints

Retailers can't simply stock the highest-margin products. They must balance:

| Constraint | Description | Business Reason |
|------------|-------------|-----------------|
| **Shelf Space** | Each store has limited shelf capacity | Physical limitation |
| **Category Requirements** | Must carry minimum SKUs per category | Customer expectations |
| **Category Limits** | Maximum SKUs per category | Avoid over-assortment |
| **Supplier Agreements** | Minimum purchases from key suppliers | Contractual obligations |
| **Must-Carry Items** | Certain products are mandatory | Brand/corporate requirements |

### Why Optimization Matters

A typical retailer with:
- 75 stores
- 600 potential SKUs
- 45,000 possible store-SKU combinations

**Manual selection is impossible.** Even experienced merchandisers can't evaluate all combinations while respecting constraints.

---

## 💡 Solution Overview

This project implements an **end-to-end assortment optimization system** that:

1. **Forecasts demand** for every store-SKU combination using machine learning
2. **Optimizes selection** using Mixed-Integer Linear Programming (MILP)
3. **Respects all business constraints** while maximizing profit
4. **Quantifies improvement** vs. baseline heuristics

### Solution Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RETAIL ASSORTMENT OPTIMIZATION                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│  HISTORICAL   │          │    STORE      │          │   PRODUCT     │
│    SALES      │          │  ATTRIBUTES   │          │  ATTRIBUTES   │
│  (52 weeks)   │          │ (75 stores)   │          │  (600 SKUs)   │
└───────┬───────┘          └───────┬───────┘          └───────┬───────┘
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   ▼
                    ┌──────────────────────────┐
                    │   FEATURE ENGINEERING    │
                    │  • Lag features (1,2,4,8)│
                    │  • Rolling stats (4,8,12)│
                    │  • Categorical encoding  │
                    └────────────┬─────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │   DEMAND FORECASTING     │
                    │   (Gradient Boosting)    │
                    │   CV RMSE: 13.84         │
                    └────────────┬─────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │   MILP OPTIMIZATION      │
                    │   (Pyomo + CBC Solver)   │
                    │   45,000 binary vars     │
                    └────────────┬─────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │   OPTIMAL ASSORTMENT     │
                    │   26,820 selections      │
                    │   $5.84M expected profit │
                    └──────────────────────────┘
```

---

## 📊 Results & Impact

### Key Metrics

| Metric | Optimized | Baseline | Improvement |
|--------|-----------|----------|-------------|
| **Total Expected Profit** | $5,844,740 | $5,132,298 | **+$712,442** |
| **Profit Improvement** | - | - | **+13.9%** |
| Store-SKU Selections | 26,820 | 26,820 | Same |
| Avg SKUs per Store | 357.6 | 357.6 | Same |
| Solve Time | 0.31 sec | - | - |
| Optimality Gap | 0% | - | Proven Optimal |

### What This Means

- **$712,442 additional profit** from smarter product selection
- **Same number of products** - improvement comes purely from better selection
- **All constraints satisfied** - practical, implementable solution
- **Proven optimal** - mathematically guaranteed best solution

### Performance by Category

The optimization intelligently allocates shelf space across categories based on profit potential:

```
Category Performance (Optimized vs Baseline):
├── Beverages:     +18.2% profit uplift
├── Snacks:        +15.7% profit uplift  
├── Dairy:         +12.3% profit uplift
├── Frozen:        +14.1% profit uplift
├── Meat:          +11.8% profit uplift
└── ... (12 categories total)
```

---

## 🏗 Technical Architecture

### Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Language** | Python 3.9+ | Core development |
| **Data Processing** | Pandas, NumPy, PyArrow | Data manipulation & Parquet I/O |
| **ML Framework** | Scikit-learn | Demand forecasting |
| **Optimization** | Pyomo 6.6+ | MILP modeling |
| **Solver** | CBC (open source) | Solving MILP |
| **Configuration** | PyYAML | Config management |
| **Cloud Storage** | AWS S3 | Data persistence |
| **Cloud Compute** | AWS SageMaker | Model validation |

### Design Principles

1. **Modularity**: Each component is independent and testable
2. **Configuration-driven**: All parameters in `config.yaml`
3. **File-based**: Parquet files for data exchange (no database required)
4. **Reproducibility**: Seeded random generation, version-controlled
5. **Scalability**: Designed to handle larger datasets

---

## 🔄 Data Pipeline

### Data Flow

```
[1] generate_data.py     [2] build_features.py    [3] train.py
         │                        │                      │
         ▼                        ▼                      ▼
   outputs/raw/            outputs/features/       outputs/models/
   ├── stores.parquet      ├── train.parquet      ├── model.joblib
   ├── skus.parquet        ├── test.parquet       └── metrics.json
   ├── sales.parquet       └── score.parquet
   └── constraints.parquet
         │                        │                      │
         └────────────────────────┼──────────────────────┘
                                  ▼
                          [4] predict.py
                                  │
                                  ▼
                        outputs/forecasts/
                        └── forecasts.parquet
                                  │
                                  ▼
                          [5] model.py (optimization)
                                  │
                                  ▼
                        outputs/assortments/
                        ├── assortment_output.parquet
                        └── metrics.json
```

### Data Description

#### Stores (75 records)
```python
{
    'store_id': 'S0001',           # Unique identifier
    'region': 'Northeast',          # Geographic region
    'store_type': 'Urban',          # Urban/Suburban/Rural
    'store_size': 'Large',          # Small/Medium/Large
    'shelf_capacity': 18000,        # Total shelf units
    'weekly_traffic': 35000         # Customer visits/week
}
```

#### SKUs (600 records)
```python
{
    'sku_id': 'SKU00001',           # Unique identifier
    'category': 'Beverages',        # Product category
    'brand': 'Brand_12',            # Brand name
    'supplier_id': 'SUP005',        # Supplier
    'cost': 2.45,                   # Unit cost
    'price': 4.99,                  # Retail price
    'margin': 2.54,                 # Profit margin
    'space_units': 2,               # Shelf space required
    'is_must_carry': False          # Mandatory item flag
}
```

#### Sales History (1.17M records)
```python
{
    'store_id': 'S0001',
    'sku_id': 'SKU00001',
    'week': '2024-W01',
    'week_start': '2024-01-01',
    'units': 47,                    # Units sold
    'promo_flag': 0                 # Promotional indicator
}
```

### Feature Engineering

The ML model uses **31 features** across these categories:

| Category | Features | Description |
|----------|----------|-------------|
| **Temporal** | 5 | week_of_year, month, quarter, is_holiday_season, promo_flag |
| **Lag** | 4 | units_lag_1, units_lag_2, units_lag_4, units_lag_8 |
| **Rolling** | 6 | units_rmean_4/8/12, units_rstd_4/8/12 |
| **Store** | 5 | shelf_capacity, region_enc, store_type_enc, store_size_enc, store_id_enc |
| **Product** | 6 | price, margin, space_units, category_enc, brand_enc, sku_id_enc |
| **Derived** | 5 | Various encoded categorical features |

---

## 🤖 Machine Learning Model

### Model Selection

**Gradient Boosting Regressor** was chosen for:
- Strong performance on tabular data
- Handles mixed feature types
- Feature importance interpretability
- No extensive hyperparameter tuning needed

### Training Process

```python
# Time-series cross-validation (3 folds)
Fold 1: RMSE = 13.30
Fold 2: RMSE = 13.72  
Fold 3: RMSE = 14.49

# Final Model Performance
CV RMSE:    13.84 (+/- 0.49)
Test RMSE:  15.83
Test MAE:   10.21
```

### Feature Importance

```
Top 10 Most Important Features:
─────────────────────────────────────
1. store_size_enc     39.73%  ████████████████████
2. shelf_capacity     22.73%  ███████████
3. sku_id_enc         12.61%  ██████
4. promo_flag         10.47%  █████
5. units_rstd_12       1.34%  █
6. category_enc        1.21%  █
7. units_rmean_12      1.15%  █
8. month               0.98%  █
9. units_lag_1         0.87%  
10. price              0.82%  
```

### Key Insights

- **Store size dominates**: Larger stores have fundamentally different demand patterns
- **Promotions matter**: 10.47% importance shows promo_flag is significant
- **Product identity**: Individual SKU patterns (sku_id_enc) capture brand loyalty
- **Seasonality captured**: Temporal features contribute to predictions

---

## 🔧 Optimization Model

### Mathematical Formulation

#### Sets
- `S` = Set of stores (75 stores)
- `P` = Set of products/SKUs (600 SKUs)
- `C` = Set of categories (12 categories)
- `R` = Set of suppliers (25 suppliers)

#### Parameters
| Parameter | Description |
|-----------|-------------|
| `forecast[s,p]` | Forecasted demand for SKU p at store s |
| `margin[p]` | Profit margin for SKU p |
| `space[p]` | Shelf space required for SKU p |
| `capacity[s]` | Total shelf capacity of store s |
| `cat_min[s,c]` | Minimum SKUs from category c at store s |
| `cat_max[s,c]` | Maximum SKUs from category c at store s |
| `sup_min[s,r]` | Minimum SKUs from supplier r at store s |
| `must_carry[p]` | 1 if SKU p is mandatory, 0 otherwise |

#### Decision Variables
```
x[s,p] ∈ {0, 1}  ∀s ∈ S, p ∈ P

x[s,p] = 1 if SKU p is selected for store s
x[s,p] = 0 otherwise
```

#### Objective Function
```
Maximize Z = Σ(s∈S) Σ(p∈P) forecast[s,p] × margin[p] × x[s,p]
```

#### Constraints

**1. Shelf Space Capacity**
```
Σ(p∈P) space[p] × x[s,p] ≤ capacity[s]    ∀s ∈ S
```
*Each store's total shelf usage cannot exceed its capacity*

**2. Category Minimum**
```
Σ(p∈Pc) x[s,p] ≥ cat_min[s,c]    ∀s ∈ S, c ∈ C
```
*Each store must carry at least the minimum SKUs per category*

**3. Category Maximum**
```
Σ(p∈Pc) x[s,p] ≤ cat_max[s,c]    ∀s ∈ S, c ∈ C
```
*Each store cannot exceed maximum SKUs per category*

**4. Supplier Minimum**
```
Σ(p∈Pr) x[s,p] ≥ sup_min[s,r]    ∀s ∈ S, r ∈ R
```
*Stores must meet minimum supplier purchase requirements*

**5. Must-Carry Items**
```
x[s,p] = 1    ∀s ∈ S, p ∈ P where must_carry[p] = 1
```
*Mandatory items must be carried in all stores*

### Model Statistics

| Metric | Value |
|--------|-------|
| Binary Variables | 45,000 |
| Constraints | ~3,700 |
| Non-zeros | ~145,000 |
| Solve Time | 0.31 seconds |
| Optimality Gap | 0% (proven optimal) |

### Solver Output Interpretation

```
Presolve: 1794 rows, 43500 columns
Solution found by feasibility pump
Result - Optimal solution found
Objective value: 5844740.51
Enumerated nodes: 0
Total iterations: 0
```

- **Presolve**: Solver reduced problem size (removed redundant constraints)
- **Feasibility pump**: Found optimal solution quickly
- **0 nodes enumerated**: Solved at root node (no branching needed)

---

## 📁 Project Structure

```
retail-assortment-optimizer/
│
├── 📁 src/                          # Source code
│   ├── 📁 data/
│   │   ├── __init__.py
│   │   ├── generate_data.py         # Synthetic data generation
│   │   └── io.py                    # File I/O utilities
│   │
│   ├── 📁 features/
│   │   ├── __init__.py
│   │   └── build_features.py        # Feature engineering
│   │
│   ├── 📁 forecasting/
│   │   ├── __init__.py
│   │   ├── train.py                 # Model training
│   │   └── predict.py               # Generate forecasts
│   │
│   ├── 📁 optimization/
│   │   ├── __init__.py
│   │   ├── model.py                 # MILP formulation & solving
│   │   └── evaluate.py              # Baseline comparison
│   │
│   └── 📁 pipeline/
│       ├── __init__.py
│       └── run_local.py             # End-to-end pipeline
│
├── 📁 configs/
│   └── config.yaml                  # All configuration parameters
│
├── 📁 outputs/                      # Generated outputs (gitignored)
│   ├── raw/                         # Synthetic data
│   ├── features/                    # ML features
│   ├── models/                      # Trained models
│   ├── forecasts/                   # Demand forecasts
│   ├── assortments/                 # Optimization results
│   └── reports/                     # Evaluation reports
│
├── 📁 notebooks/
│   └── cloud_smoke_test.ipynb       # AWS validation notebook
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager
- CBC solver (for optimization)

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/retail-assortment-optimizer.git
cd retail-assortment-optimizer

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# On Mac/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Install CBC solver
# On Mac:
brew install cbc

# On Ubuntu/Debian:
sudo apt-get install coinor-cbc

# On Windows:
# Download from https://github.com/coin-or/Cbc/releases
```

### Verify Installation

```bash
# Check Python packages
python -c "import pandas, numpy, sklearn, pyomo; print('All packages installed!')"

# Check CBC solver
which cbc  # Should show path like /opt/homebrew/bin/cbc
```

---

## 🚀 Usage

### Run Full Pipeline

```bash
# Generate all data and run complete pipeline
python -m src.pipeline.run_local --full
```

**Expected output:**
```
======================================================================
   RETAIL ASSORTMENT OPTIMIZATION PIPELINE
======================================================================

[1/5] DATA GENERATION
[2/5] FEATURE ENGINEERING  
[3/5] DEMAND FORECASTING - Training
[4/5] DEMAND FORECASTING - Prediction
[5/5] ASSORTMENT OPTIMIZATION

✓ OPTIMAL SOLUTION FOUND
  Objective (Total Profit): $5,844,740.51

[EVALUATION]
  Optimized Profit:  $5,844,740.51
  Baseline Profit:   $5,132,298.06
  Uplift:            +13.9%

======================================================================
   PIPELINE COMPLETE - Total time: 12.5 minutes
======================================================================
```

### Run Specific Steps

```bash
# Run only optimization (fastest - uses existing data)
python -m src.pipeline.run_local

# Run from feature engineering onwards
python -m src.pipeline.run_local --from-step 2

# Run from model training onwards
python -m src.pipeline.run_local --from-step 3

# Run from prediction onwards  
python -m src.pipeline.run_local --from-step 4

# Run only optimization + evaluation
python -m src.pipeline.run_local --from-step 5
```

### Run Individual Modules

```bash
# Data generation only
python -m src.data.generate_data

# Feature engineering only
python -m src.features.build_features

# Model training only
python -m src.forecasting.train

# Prediction only
python -m src.forecasting.predict

# Optimization only
python -m src.optimization.model

# Evaluation only
python -m src.optimization.evaluate
```

---

## ☁️ AWS Deployment

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         AWS Cloud                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    S3 Bucket                         │    │
│  │  s3://your-bucket/retail-opt/                       │    │
│  │  ├── raw/           (stores, skus, sales)           │    │
│  │  ├── features/      (train, test, score)            │    │
│  │  ├── models/        (model.joblib)                  │    │
│  │  ├── forecasts/     (forecasts.parquet)             │    │
│  │  ├── assortments/   (assortment_output.parquet)     │    │
│  │  └── reports/       (evaluation.json)               │    │
│  └─────────────────────────────────────────────────────┘    │
│                              │                               │
│                              ▼                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              SageMaker Studio                        │    │
│  │         (cloud_smoke_test.ipynb)                    │    │
│  │    • Validates pipeline runs in cloud               │    │
│  │    • Reads/writes to S3                             │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Upload to S3

```bash
# Configure AWS CLI
aws configure

# Create bucket (if needed)
aws s3 mb s3://your-bucket-name

# Sync outputs to S3
aws s3 sync outputs/ s3://your-bucket-name/retail-opt/
```

### Cost Estimate

| Resource | Usage | Estimated Cost |
|----------|-------|----------------|
| S3 Storage | ~100 MB | ~$0.002/month |
| SageMaker Studio | 1 hour | ~$0.05 |
| **Total** | One-time | **< $0.10** |

---

## 🔮 Future Enhancements

### Short-term
- [ ] Add XGBoost/LightGBM model comparison
- [ ] Implement product substitution constraints
- [ ] Add seasonal assortment rotation
- [ ] Create interactive dashboard (Streamlit)

### Medium-term
- [ ] Multi-objective optimization (profit + diversity)
- [ ] Robust optimization for demand uncertainty
- [ ] Store clustering for similar assortments
- [ ] A/B testing framework

### Long-term
- [ ] Real-time demand updating
- [ ] Integration with inventory management
- [ ] Reinforcement learning for dynamic assortment
- [ ] Multi-echelon supply chain optimization

---

## 📚 References

### Operations Research
- Kök, A. G., Fisher, M. L., & Vaidyanathan, R. (2015). Assortment Planning: Review of Literature and Industry Practice. *Retail Supply Chain Management*.
- Honhon, D., Gaur, V., & Seshadri, S. (2010). Assortment Planning and Inventory Decisions Under Stockout-Based Substitution. *Operations Research*.

### Tools & Libraries
- [Pyomo Documentation](http://www.pyomo.org/documentation)
- [CBC Solver](https://github.com/coin-or/Cbc)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**
- GitHub: [@YourUsername](https://github.com/YourUsername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 🙏 Acknowledgments

- Pyomo development team for the optimization framework
- COIN-OR project for the CBC solver
- Scikit-learn contributors

---

<p align="center">
  <i>Built with ❤️ for retail optimization</i>
</p>
