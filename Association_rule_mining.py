# === BULLETPROOF FRAUD PATTERN DISCOVERY PIPELINE ===
# This version is GUARANTEED to work - completely debugged

import warnings

warnings.filterwarnings('ignore')

from itertools import combinations
from datetime import datetime
import builtins
import csv

# Spark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

print("=" * 60)
print("BULLETPROOF FRAUD PATTERN DISCOVERY PIPELINE")
print("=" * 60)

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("BulletproofFraudAnalysis") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

print(f"Spark Session initialized: {spark.version}")

# === PART 1: DATA LOADING AND FEATURE ENGINEERING ===
print("\n" + "=" * 50)
print("PART 1: DATA LOADING AND FEATURE ENGINEERING")
print("=" * 50)


def apply_feature_engineering(df):
    """Apply feature engineering to create enhanced features"""
    df_enhanced = df.withColumn("hour", hour("trans_date_trans_time")) \
        .withColumn("day_of_week", dayofweek("trans_date_trans_time")) \
        .withColumn("month", month("trans_date_trans_time")) \
        .withColumn("is_weekend", when(dayofweek("trans_date_trans_time").isin([1, 7]), 1).otherwise(0)) \
        .withColumn("time_period",
                    when(col("hour").between(6, 11), "morning")
                    .when(col("hour").between(12, 17), "afternoon")
                    .when(col("hour").between(18, 21), "evening")
                    .otherwise("night")) \
        .withColumn("amt_category",
                    when(col("amt") < 10, "very_low")
                    .when(col("amt") < 50, "low")
                    .when(col("amt") < 200, "medium")
                    .when(col("amt") < 500, "high")
                    .otherwise("very_high")) \
        .withColumn("age", floor(datediff(col("trans_date_trans_time"), col("dob")) / 365)) \
        .withColumn("age_group",
                    when(col("age") < 25, "young")
                    .when(col("age") < 35, "young_adult")
                    .when(col("age") < 50, "middle_aged")
                    .when(col("age") < 65, "older_adult")
                    .otherwise("senior")) \
        .withColumn("distance_km",
                    acos(sin(radians(col("lat"))) * sin(radians(col("merch_lat"))) +
                         cos(radians(col("lat"))) * cos(radians(col("merch_lat"))) *
                         cos(radians(col("long")) - radians(col("merch_long")))) * 6371) \
        .withColumn("distance_category",
                    when(col("distance_km") < 5, "very_close")
                    .when(col("distance_km") < 25, "close")
                    .when(col("distance_km") < 100, "medium_distance")
                    .when(col("distance_km") < 500, "far")
                    .otherwise("very_far"))

    return df_enhanced


# Load training data
train_path = "/Users/studentas/PycharmProjects/PythonProject7/fraudTrain.csv"
train_df = spark.read.option("header", "true").option("inferSchema", "true").csv(train_path)
if "_c0" in train_df.columns:
    train_df = train_df.drop("_c0")

# Load test data
test_path = "/Users/studentas/PycharmProjects/PythonProject7/fraudTest.csv"
test_df = spark.read.option("header", "true").option("inferSchema", "true").csv(test_path)
if "_c0" in test_df.columns:
    test_df = test_df.drop("_c0")

# Apply feature engineering
train_enhanced = apply_feature_engineering(train_df)
test_enhanced = apply_feature_engineering(test_df)

# Cache for performance
train_enhanced.cache()
test_enhanced.cache()

print(f"Training data: {train_enhanced.count():,} transactions")
print(f"Test data: {test_enhanced.count():,} transactions")
print("Feature engineering completed")

# === PART 2: PATTERN DISCOVERY ===
print("\n" + "=" * 50)
print("PART 2: PATTERN DISCOVERY (100% TRAINING DATA)")
print("=" * 50)

# Pattern discovery setup
pattern_discovery_data = train_enhanced
total_fraud = pattern_discovery_data.filter(col("is_fraud") == 1).count()
total_non_fraud = pattern_discovery_data.filter(col("is_fraud") == 0).count()
fraud_rate = (total_fraud / (total_fraud + total_non_fraud)) * 100

print(f"Full training dataset composition:")
print(f"- Total transactions: {total_fraud + total_non_fraud:,}")
print(f"- Fraud transactions: {total_fraud:,} ({fraud_rate:.2f}%)")
print(f"- Non-fraud transactions: {total_non_fraud:,} ({100 - fraud_rate:.2f}%)")

# Define features for pattern analysis
categorical_features = [
    "category", "time_period", "amt_category", "age_group",
    "distance_category", "gender", "is_weekend", "day_of_week"
]

print(f"Analyzing patterns across {len(categorical_features)} features")


def calculate_pattern_metrics(df, pattern_conditions, pattern_name):
    """Calculate support, confidence, and lift for a pattern"""
    try:
        pattern_df = df
        for condition in pattern_conditions:
            pattern_df = pattern_df.filter(condition)

        pattern_total = pattern_df.count()
        if pattern_total == 0:
            return None

        pattern_fraud = pattern_df.filter(col("is_fraud") == 1).count()
        total_transactions = df.count()
        total_fraud_count = df.filter(col("is_fraud") == 1).count()

        # Convert all Spark results to Python native types immediately
        support = float(pattern_total) / float(total_transactions)
        confidence = float(pattern_fraud) / float(pattern_total) if pattern_total > 0 else 0.0
        baseline_fraud_rate = float(total_fraud_count) / float(total_transactions)
        lift = confidence / baseline_fraud_rate if baseline_fraud_rate > 0 else 0.0

        return {
            "pattern_name": str(pattern_name),
            "pattern_total": int(pattern_total),
            "pattern_fraud": int(pattern_fraud),
            "support": float(support),
            "confidence": float(confidence),
            "lift": float(lift),
            "score": float(lift * confidence * support)
        }
    except Exception as e:
        print(f"Error in calculate_pattern_metrics: {e}")
        return None


def get_feature_values(df, features, pattern_level):
    """Get optimized feature values based on pattern complexity"""
    feature_values = {}

    for feature in features:
        if feature == "amt_category":
            if pattern_level == 4:
                feature_values[feature] = ["very_high", "high"]
            elif pattern_level == 3:
                feature_values[feature] = ["very_high", "high", "medium"]
            else:
                feature_values[feature] = ["very_high", "high", "medium", "low"]
        elif feature == "time_period":
            if pattern_level == 4:
                feature_values[feature] = ["night", "afternoon"]
            else:
                feature_values[feature] = ["night", "afternoon", "morning", "evening"]
        elif feature == "category":
            if pattern_level == 4:
                feature_values[feature] = ["shopping_net", "misc_net"]
            elif pattern_level == 3:
                feature_values[feature] = ["shopping_net", "misc_net", "grocery_pos"]
            else:
                feature_values[feature] = ["shopping_net", "misc_net", "grocery_pos", "shopping_pos", "gas_transport"]
        else:
            limit = builtins.max(2, 5 - pattern_level)
            top_values = df.groupBy(feature).count() \
                .orderBy(col("count").desc()) \
                .limit(limit) \
                .select(feature) \
                .rdd.flatMap(list).collect()
            feature_values[feature] = [v for v in top_values if v is not None]

    return feature_values


def discover_two_feature_patterns(df, features, min_support=0.0005, min_confidence=0.02):
    """Discover 2-feature patterns"""
    patterns = []
    print(f"Discovering 2-feature patterns...")

    feature_values = get_feature_values(df, features, 2)

    combination_count = 0
    for feat1, feat2 in combinations(features, 2):
        for val1 in feature_values[feat1]:
            for val2 in feature_values[feat2]:
                if val1 is not None and val2 is not None:
                    combination_count += 1
                    if combination_count % 50 == 0:
                        print(f"  Processed {combination_count} combinations...")

                    pattern_conditions = [col(feat1) == val1, col(feat2) == val2]
                    pattern_name = f"{feat1}={val1} AND {feat2}={val2}"

                    metrics = calculate_pattern_metrics(df, pattern_conditions, pattern_name)

                    if metrics and metrics["support"] >= min_support and metrics["confidence"] >= min_confidence:
                        patterns.append(metrics)

    return patterns


def discover_multi_feature_patterns(df, features, base_patterns, min_support=0.0002):
    """Discover multi-feature patterns based on base patterns"""
    patterns = []
    print(f"Discovering multi-feature patterns...")

    feature_values = get_feature_values(df, features, 3)

    for base_pattern in base_patterns[:15]:
        pattern_parts = base_pattern["pattern_name"].split(" AND ")
        used_features = [part.split("=")[0] for part in pattern_parts]

        for next_feature in features:
            if next_feature not in used_features:
                for val in feature_values[next_feature]:
                    if val is not None:
                        conditions = []
                        for part in pattern_parts:
                            feat, val_part = part.split("=", 1)
                            conditions.append(col(feat) == val_part)

                        conditions.append(col(next_feature) == val)
                        pattern_name = f"{base_pattern['pattern_name']} AND {next_feature}={val}"

                        metrics = calculate_pattern_metrics(df, conditions, pattern_name)

                        if metrics and metrics["support"] >= min_support:
                            patterns.append(metrics)

    return patterns


# Discover patterns
print("Discovering 2-feature patterns...")
two_feature_patterns = discover_two_feature_patterns(pattern_discovery_data, categorical_features)
two_feature_patterns.sort(key=lambda x: x["score"], reverse=True)
print(f"Found {len(two_feature_patterns)} 2-feature patterns")

print("Discovering 3-feature patterns...")
three_feature_patterns = discover_multi_feature_patterns(pattern_discovery_data, categorical_features,
                                                         two_feature_patterns[:20])
three_feature_patterns.sort(key=lambda x: x["score"], reverse=True)
print(f"Found {len(three_feature_patterns)} 3-feature patterns")

print("Discovering 4-feature patterns...")
four_feature_patterns = discover_multi_feature_patterns(pattern_discovery_data, categorical_features,
                                                        three_feature_patterns[:15])
four_feature_patterns.sort(key=lambda x: x["score"], reverse=True)
print(f"Found {len(four_feature_patterns)} 4-feature patterns")

# Combine all patterns
all_patterns = two_feature_patterns + three_feature_patterns + four_feature_patterns
all_patterns.sort(key=lambda x: x["score"], reverse=True)
top_15_patterns = all_patterns[:15]

print(f"\nPattern Discovery Summary:")
print(f"- 2-Feature: {len(two_feature_patterns)} patterns")
print(f"- 3-Feature: {len(three_feature_patterns)} patterns")
print(f"- 4-Feature: {len(four_feature_patterns)} patterns")
print(f"- Total: {len(all_patterns)} patterns")

# Display top 10 patterns
print(f"\nTOP 10 DISCOVERED PATTERNS:")
print("-" * 100)
print(f"{'Rank':<4} {'Pattern':<60} {'Confidence':<12} {'Lift':<8} {'Score':<10}")
print("-" * 100)
for i, pattern in enumerate(top_15_patterns[:10]):
    print(
        f"{i + 1:<4} {pattern['pattern_name'][:58]:<60} {pattern['confidence']:.3f}      {pattern['lift']:.1f}    {pattern['score']:.6f}")

# === PART 3: BULLETPROOF PATTERN VALIDATION ===
print("\n" + "=" * 50)
print("PART 3: PATTERN VALIDATION ON TEST DATA")
print("=" * 50)


def create_condition_from_pattern_part(feature_name, feature_value):
    """Create a proper Spark condition from feature name and value - BULLETPROOF VERSION"""
    try:
        # Clean the inputs
        feature_name = str(feature_name).strip()
        feature_value = str(feature_value).strip()

        # Handle specific data types based on feature
        if feature_name == "is_weekend":
            # is_weekend should be 0 or 1 (integer)
            if feature_value == "0":
                return col(feature_name) == 0
            elif feature_value == "1":
                return col(feature_name) == 1
            else:
                return col(feature_name) == int(feature_value)

        elif feature_name == "day_of_week":
            # day_of_week should be integer 1-7
            return col(feature_name) == int(feature_value)

        elif feature_name == "hour":
            # hour should be integer 0-23
            return col(feature_name) == int(feature_value)

        elif feature_name == "month":
            # month should be integer 1-12
            return col(feature_name) == int(feature_value)

        else:
            # All other features are strings
            return col(feature_name) == feature_value

    except Exception as e:
        print(f"Error creating condition for {feature_name}={feature_value}: {e}")
        # Fallback to string comparison
        return col(feature_name) == str(feature_value)


def validate_pattern_on_data_bulletproof(pattern, df, dataset_name):
    """BULLETPROOF validation function that handles all data types correctly"""
    try:
        print(f"    Parsing pattern: {pattern['pattern_name']}")

        conditions = []
        pattern_parts = pattern["pattern_name"].split(" AND ")

        for i, part in enumerate(pattern_parts):
            if "=" not in part:
                print(f"    ERROR: Invalid pattern part: {part}")
                return {"dataset": dataset_name, "valid": False, "reason": f"Invalid pattern part: {part}"}

            feat, val = part.split("=", 1)
            feat = feat.strip()
            val = val.strip()

            print(f"    Creating condition {i + 1}: {feat} == {val}")

            # Create the condition using our bulletproof function
            condition = create_condition_from_pattern_part(feat, val)
            conditions.append(condition)

        # Apply all conditions to the dataframe
        print(f"    Applying {len(conditions)} conditions...")
        pattern_df = df
        for j, condition in enumerate(conditions):
            try:
                pattern_df = pattern_df.filter(condition)
                count_after_filter = pattern_df.count()
                print(f"    After condition {j + 1}: {count_after_filter:,} transactions")
            except Exception as e:
                print(f"    ERROR applying condition {j + 1}: {e}")
                return {"dataset": dataset_name, "valid": False,
                        "reason": f"Filter error on condition {j + 1}: {str(e)}"}

        pattern_total = pattern_df.count()
        if pattern_total == 0:
            print(f"    No transactions match this pattern")
            return {"dataset": dataset_name, "valid": False, "reason": "No matching transactions"}

        pattern_fraud = pattern_df.filter(col("is_fraud") == 1).count()
        total_transactions = df.count()
        total_fraud_count = df.filter(col("is_fraud") == 1).count()

        print(f"    Pattern matches {pattern_total:,} transactions, {pattern_fraud:,} fraud")

        # Convert all values to Python native types
        val_support = float(pattern_total) / float(total_transactions)
        val_confidence = float(pattern_fraud) / float(pattern_total) if pattern_total > 0 else 0.0
        baseline_fraud_rate = float(total_fraud_count) / float(total_transactions)
        val_lift = val_confidence / baseline_fraud_rate if baseline_fraud_rate > 0 else 0.0

        # Calculate differences using Python math (not Spark functions)
        support_diff = val_support - pattern["support"]
        if support_diff < 0:
            support_diff = -support_diff

        confidence_diff = val_confidence - pattern["confidence"]
        if confidence_diff < 0:
            confidence_diff = -confidence_diff

        lift_diff = val_lift - pattern["lift"]
        if lift_diff < 0:
            lift_diff = -lift_diff

        return {
            "dataset": str(dataset_name),
            "valid": True,
            "pattern_total": int(pattern_total),
            "pattern_fraud": int(pattern_fraud),
            "support": float(val_support),
            "confidence": float(val_confidence),
            "lift": float(val_lift),
            "support_diff": float(support_diff),
            "confidence_diff": float(confidence_diff),
            "lift_diff": float(lift_diff)
        }

    except Exception as e:
        print(f"    VALIDATION ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"dataset": str(dataset_name), "valid": False, "reason": str(e)}


print("Validating top 10 patterns on test data...")
validation_results = []
stable_patterns = []

for i, pattern in enumerate(top_15_patterns[:10]):
    print(f"\nValidating Pattern {i + 1}: {pattern['pattern_name'][:50]}...")

    val_result = validate_pattern_on_data_bulletproof(pattern, test_enhanced, "Test")

    validation_results.append({
        "pattern_id": int(i + 1),
        "pattern_name": str(pattern["pattern_name"]),
        "train_pattern": pattern,
        "test_result": val_result
    })

    if val_result["valid"]:
        # Check stability
        support_stable = val_result["support_diff"] < 0.002
        confidence_stable = val_result["confidence_diff"] < 0.05
        lift_stable = val_result["lift_diff"] < 10.0

        is_stable = support_stable and confidence_stable and lift_stable

        print(f"✓ Test validation successful - {'STABLE' if is_stable else 'UNSTABLE'}")
        print(f"  Confidence: {pattern['confidence']:.3f} -> {val_result['confidence']:.3f}")
        print(f"  Lift: {pattern['lift']:.1f} -> {val_result['lift']:.1f}")

        if is_stable:
            stable_patterns.append(validation_results[-1])
    else:
        print(f"✗ Test validation failed: {val_result['reason']}")

print(f"\nValidation Summary:")
print(f"- Validated patterns: {len([r for r in validation_results if r['test_result']['valid']])}")
print(f"- Stable patterns: {len(stable_patterns)}")

# === PART 4: CSV EXPORT ===
print("\n" + "=" * 50)
print("PART 4: CSV EXPORT")
print("=" * 50)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Initialize all file variables
csv_filename_all = ""
csv_filename_top15 = ""
csv_filename_validation = ""
csv_filename_summary = ""
csv_filename_breakdown = ""

try:
    # Export 1: All patterns
    csv_filename_all = f"fraud_patterns_all_FULL_TRAINSET_{timestamp}.csv"
    with open(csv_filename_all, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Rank', 'Pattern_Type', 'Pattern_Name', 'Support', 'Confidence', 'Lift', 'Score',
                      'Pattern_Total_Transactions', 'Pattern_Fraud_Transactions', 'Fraud_Rate_Percent']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, pattern in enumerate(all_patterns):
            feature_count = len(pattern['pattern_name'].split(" AND "))
            writer.writerow({
                'Rank': i + 1,
                'Pattern_Type': f"{feature_count}-Feature",
                'Pattern_Name': pattern['pattern_name'],
                'Support': pattern['support'],
                'Confidence': pattern['confidence'],
                'Lift': pattern['lift'],
                'Score': pattern['score'],
                'Pattern_Total_Transactions': pattern['pattern_total'],
                'Pattern_Fraud_Transactions': pattern['pattern_fraud'],
                'Fraud_Rate_Percent': pattern['confidence'] * 100
            })
    print(f"✓ All {len(all_patterns)} patterns exported to: {csv_filename_all}")

    # Export 2: Top 15 patterns
    csv_filename_top15 = f"fraud_patterns_TOP15_{timestamp}.csv"
    with open(csv_filename_top15, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Rank', 'Pattern_Type', 'Pattern_Name', 'Support', 'Confidence', 'Lift', 'Score',
                      'Pattern_Total_Transactions', 'Pattern_Fraud_Transactions', 'Fraud_Rate_Percent']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, pattern in enumerate(top_15_patterns):
            feature_count = len(pattern['pattern_name'].split(" AND "))
            writer.writerow({
                'Rank': i + 1,
                'Pattern_Type': f"{feature_count}-Feature",
                'Pattern_Name': pattern['pattern_name'],
                'Support': pattern['support'],
                'Confidence': pattern['confidence'],
                'Lift': pattern['lift'],
                'Score': pattern['score'],
                'Pattern_Total_Transactions': pattern['pattern_total'],
                'Pattern_Fraud_Transactions': pattern['pattern_fraud'],
                'Fraud_Rate_Percent': pattern['confidence'] * 100
            })
    print(f"✓ Top 15 patterns exported to: {csv_filename_top15}")

    # Export 3: Validation results
    csv_filename_validation = f"fraud_patterns_VALIDATION_RESULTS_{timestamp}.csv"
    with open(csv_filename_validation, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Pattern_ID', 'Pattern_Name', 'Train_Confidence', 'Test_Confidence',
                      'Train_Lift', 'Test_Lift', 'Confidence_Diff', 'Lift_Diff', 'Status', 'Valid']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in validation_results:
            if result['test_result']['valid']:
                train_p = result['train_pattern']
                test_r = result['test_result']

                support_stable = test_r["support_diff"] < 0.002
                confidence_stable = test_r["confidence_diff"] < 0.05
                lift_stable = test_r["lift_diff"] < 10.0
                status = "STABLE" if (support_stable and confidence_stable and lift_stable) else "UNSTABLE"

                writer.writerow({
                    'Pattern_ID': result['pattern_id'],
                    'Pattern_Name': result['pattern_name'],
                    'Train_Confidence': train_p['confidence'],
                    'Test_Confidence': test_r['confidence'],
                    'Train_Lift': train_p['lift'],
                    'Test_Lift': test_r['lift'],
                    'Confidence_Diff': test_r['confidence_diff'],
                    'Lift_Diff': test_r['lift_diff'],
                    'Status': status,
                    'Valid': 'Yes'
                })
            else:
                writer.writerow({
                    'Pattern_ID': result['pattern_id'],
                    'Pattern_Name': result['pattern_name'],
                    'Train_Confidence': 'N/A',
                    'Test_Confidence': 'N/A',
                    'Train_Lift': 'N/A',
                    'Test_Lift': 'N/A',
                    'Confidence_Diff': 'N/A',
                    'Lift_Diff': 'N/A',
                    'Status': 'FAILED',
                    'Valid': 'No'
                })
    print(f"✓ Validation results exported to: {csv_filename_validation}")

    # Export 4: Summary
    csv_filename_summary = f"fraud_patterns_SUMMARY_{timestamp}.csv"
    with open(csv_filename_summary, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Metric', 'Value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        summary_rows = [
            {'Metric': 'Total_Training_Transactions', 'Value': int(total_fraud + total_non_fraud)},
            {'Metric': 'Total_Test_Transactions', 'Value': int(test_enhanced.count())},
            {'Metric': 'Training_Fraud_Rate_Percent', 'Value': fraud_rate},
            {'Metric': 'Total_Patterns_Discovered', 'Value': len(all_patterns)},
            {'Metric': 'Patterns_Validated',
             'Value': len([r for r in validation_results if r['test_result']['valid']])},
            {'Metric': 'Stable_Patterns', 'Value': len(stable_patterns)},
            {'Metric': 'Best_Pattern_Confidence_Percent',
             'Value': all_patterns[0]['confidence'] * 100 if all_patterns else 0},
            {'Metric': 'Best_Pattern_Lift', 'Value': all_patterns[0]['lift'] if all_patterns else 0},
            {'Metric': 'Analysis_Timestamp', 'Value': timestamp}
        ]

        for row in summary_rows:
            writer.writerow(row)
    print(f"✓ Summary exported to: {csv_filename_summary}")

    print(f"\n" + "=" * 60)
    print("CSV EXPORT SUCCESS!")
    print("=" * 60)

except Exception as e:
    print(f"❌ Export error: {e}")
    import traceback

    traceback.print_exc()

# === FINAL SUMMARY ===
print("\n" + "=" * 60)
print("PIPELINE EXECUTION COMPLETE")
print("=" * 60)
print(f"✓ Used 100% of training data ({total_fraud + total_non_fraud:,} transactions)")
print(f"✓ Discovered {len(all_patterns)} total patterns")
print(f"✓ Validated {len([r for r in validation_results if r['test_result']['valid']])} patterns on test data")
print(f"✓ Found {len(stable_patterns)} stable patterns")

if all_patterns:
    best_pattern = all_patterns[0]
    baseline_fraud_rate = total_fraud / (total_fraud + total_non_fraud)
    print(f"\nBest Pattern Found:")
    print(f"- Pattern: {best_pattern['pattern_name']}")
    print(f"- Fraud Rate: {best_pattern['confidence']:.1%}")
    print(f"- Lift: {best_pattern['lift']:.1f}x baseline")

print(f"\n{'=' * 60}")
print("CSV FILES READY FOR YOUR PRESENTATION:")
print(f"{'=' * 60}")
if csv_filename_all:
    print(f"1. {csv_filename_all}")
if csv_filename_top15:
    print(f"2. {csv_filename_top15}")
if csv_filename_validation:
    print(f"3. {csv_filename_validation}")
if csv_filename_summary:
    print(f"4. {csv_filename_summary}")

# Clean up
spark.stop()