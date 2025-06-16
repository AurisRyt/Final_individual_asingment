# Final_individual_asingment
BIG_DATA_individual_project

# Cross-Feature Fraud Pattern Mining

Large-scale fraud pattern discovery in credit card transactions using Apache Spark and association rule mining.

## Overview

This project discovers hidden fraud patterns by analyzing combinations of transaction features that individually appear normal but collectively indicate fraudulent behavior. Using Apache Spark for distributed processing, we systematically explore 2, 3, and 4-feature combinations across 1.3M+ transactions.

**Key Results:**
- 381 statistically significant fraud patterns discovered
- Best pattern: 48.6% fraud detection rate (83.9x baseline improvement)
- 100% validation success rate on independent test data

## Dataset

Credit card transaction data with fraud labels:
- Training: 1,296,675 transactions (0.58% fraud rate)
- Test: 555,719 transactions (0.39% fraud rate)
- Features: temporal, geographic, demographic, behavioral

## Requirements

```bash
pip install pyspark pandas matplotlib seaborn plotly numpy
```

Requires Java 8/11 for Spark.

## Usage

Run the complete pipeline:

```bash
python EDA.py                    # Exploratory analysis and feature engineering
python Association_rule_mining.py   # Pattern discovery and validation  
python Visualisation_scribd.py      # Generate interactive visualizations
```

## Key Findings

### Pattern Discovery Results
```
Pattern Discovery Summary:
- 2-Feature: 43 patterns
- 3-Feature: 197 patterns  
- 4-Feature: 141 patterns
- Total: 381 patterns

TOP DISCOVERED PATTERNS:
time_period=night AND amt_category=very_high                 48.6% fraud rate (83.9x lift)
category=grocery_pos AND amt_category=high AND gender=M      97.9% fraud rate (169.2x lift)
time_period=night AND amt_category=very_high AND distance_  48.2% fraud rate (83.3x lift)
```

### Feature Analysis
From EDA output:
```
FRAUD RATES BY FEATURE:
amt_category=very_high: 23.34% fraud rate
time_period=night: 1.66% fraud rate  
category=shopping_net: 1.76% fraud rate
baseline fraud rate: 0.58%
```

### Validation Performance
```
Validation Summary:
- Validated patterns: 10/10 (100% success)
- Average confidence drop: ~10-15% (training to test)
- All patterns maintain statistical significance
```

## Technical Implementation

**Big Data Processing:** Apache Spark handles combinatorial explosion of feature combinations across large transaction datasets.

**Association Rule Mining:** Calculates support, confidence, and lift metrics for statistical significance.

**Cross-Validation:** Independent test dataset validation ensures pattern generalizability.

## Output Files

The pipeline generates timestamped CSV files:

1. `fraud_patterns_all_FULL_TRAINSET_[timestamp].csv` - All 381 discovered patterns
2. `fraud_patterns_TOP15_[timestamp].csv` - Top performing patterns
3. `fraud_patterns_VALIDATION_RESULTS_[timestamp].csv` - Cross-validation results
4. `fraud_patterns_SUMMARY_[timestamp].csv` - Executive summary

## Visualizations

Interactive dashboards and analysis charts:

- **Pattern Performance Dashboard** - Comprehensive 6-panel analysis
- **Cross-Feature Heatmap** - Feature relationship matrix
- **Temporal Analysis** - Time-based fraud patterns
- **Validation Charts** - Training vs test performance
- **Feature Importance** - Statistical feature rankings

## Configuration

Key parameters in `Association_rule_mining.py`:

```python
MIN_SUPPORT = 0.0005       # Minimum pattern frequency
MIN_CONFIDENCE = 0.02      # Minimum fraud rate (2%)
MIN_LIFT = 5.0            # Minimum fraud enrichment
```

Update file paths for your environment:
```python
train_path = "/path/to/fraudTrain.csv"
test_path = "/path/to/fraudTest.csv"
```

## Sample Console Output

### EDA.py
```
============================================================
CARD FRAUD CROSS-FEATURE PATTERN ANALYSIS
============================================================
Training dataset: 1,296,675 transactions with 22 features
Test dataset: 555,719 transactions with 22 features

Fraud rate summary:
Training data: 0.58%
Test data: 0.39%

Enhanced features created:
- hour, day_of_week, month, is_weekend, time_period
- amt_category, age, age_group  
- distance_km, distance_category
```

### Association_rule_mining.py
```
============================================================
FRAUD PATTERN DISCOVERY PIPELINE
============================================================
Spark Session initialized: 4.0.0

Pattern Discovery Summary:
- 2-Feature: 43 patterns
- 3-Feature: 197 patterns
- 4-Feature: 141 patterns
- Total: 381 patterns

Best Pattern Found:
- Pattern: time_period=night AND amt_category=very_high
- Fraud Rate: 48.6%
- Lift: 83.9x baseline

✓ Used 100% of training data (1,296,675 transactions)
✓ Discovered 381 total patterns
✓ Validated 10 patterns on test data
```

## Project Structure

```
├── EDA.py                     # Exploratory data analysis
├── Association_rule_mining.py # Pattern discovery pipeline
├── Visualisation_scribd.py   # Interactive visualizations
├── output/
│   ├── visualizations/
│   │   ├── interactive/       # HTML dashboards
│   │   └── static/           # PNG charts
│   └── *.csv                 # Pattern analysis results
└── README.md
```

## Academic Context

**Problem:** Cross-feature fraud pattern mining in large transaction datasets

**Method:** Multi-dimensional association rule mining with Apache Spark

**Innovation:** Systematic exploration of feature combinations vs. single-feature analysis

**Validation:** Cross-dataset stability assessment and statistical significance testing

**Technology:** Distributed computing for handling combinatorial complexity

This implementation demonstrates advanced data engineering, statistical analysis, and big data processing capabilities for fraud detection applications.
