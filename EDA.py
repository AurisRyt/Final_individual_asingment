import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import warnings
import builtins  # This gives us access to Python's built-in functions

warnings.filterwarnings('ignore')

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("CardFraudPatternAnalysis") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

print("=" * 60)
print("CARD FRAUD CROSS-FEATURE PATTERN ANALYSIS")
print("=" * 60)

# Load both datasets
print("Loading datasets...")
train_path = "/Users/studentas/PycharmProjects/PythonProject7/fraudTrain.csv"
test_path = "/Users/studentas/PycharmProjects/PythonProject7/fraudTest.csv"

# Load training dataset
train_df = spark.read.option("header", "true").option("inferSchema", "true").csv(train_path)
if "_c0" in train_df.columns:
    train_df = train_df.drop("_c0")

# Load test dataset
test_df = spark.read.option("header", "true").option("inferSchema", "true").csv(test_path)
if "_c0" in test_df.columns:
    test_df = test_df.drop("_c0")

print(f"Training dataset: {train_df.count():,} transactions with {len(train_df.columns)} features")
print(f"Test dataset: {test_df.count():,} transactions with {len(test_df.columns)} features")

print("\n" + "=" * 50)
print("DATA QUALITY ANALYSIS")
print("=" * 50)

# Check missing values in training data
print("Missing values in training data:")
missing_counts_train = []
for column in train_df.columns:
    null_count = train_df.filter(col(column).isNull()).count()
    missing_counts_train.append((column, null_count))

has_missing = False
for col_name, null_count in missing_counts_train:
    if null_count > 0:
        print(f"{col_name}: {null_count}")
        has_missing = True

if not has_missing:
    print("No missing values found in training data")

# Check missing values in test data
print("\nMissing values in test data:")
missing_counts_test = []
for column in test_df.columns:
    null_count = test_df.filter(col(column).isNull()).count()
    missing_counts_test.append((column, null_count))

has_missing = False
for col_name, null_count in missing_counts_test:
    if null_count > 0:
        print(f"{col_name}: {null_count}")
        has_missing = True

if not has_missing:
    print("No missing values found in test data")

print("\n" + "=" * 50)
print("FRAUD DISTRIBUTION COMPARISON")
print("=" * 50)

# Training data fraud distribution
print("Training data fraud distribution:")
train_fraud_dist = train_df.groupBy("is_fraud").count().orderBy("is_fraud")
train_fraud_dist.show()

train_fraud_pd = train_fraud_dist.toPandas()
train_fraud_pd['percentage'] = (train_fraud_pd['count'] / train_fraud_pd['count'].sum()) * 100

# Test data fraud distribution
print("Test data fraud distribution:")
test_fraud_dist = test_df.groupBy("is_fraud").count().orderBy("is_fraud")
test_fraud_dist.show()

test_fraud_pd = test_fraud_dist.toPandas()
test_fraud_pd['percentage'] = (test_fraud_pd['count'] / test_fraud_pd['count'].sum()) * 100

# Print summary
train_fraud_rate = train_fraud_pd[train_fraud_pd['is_fraud'] == 1]['percentage'].values[0]
test_fraud_rate = test_fraud_pd[test_fraud_pd['is_fraud'] == 1]['percentage'].values[0]

print(f"\nFraud rate summary:")
print(f"Training data: {train_fraud_rate:.2f}%")
print(f"Test data: {test_fraud_rate:.2f}%")
difference = builtins.abs(train_fraud_rate - test_fraud_rate)
print(f"Difference: {difference:.2f}%")

print("\n" + "=" * 50)
print("FEATURE ENGINEERING")
print("=" * 50)


def apply_feature_engineering(df, dataset_name):
    """Apply feature engineering to a dataset"""
    print(f"Applying feature engineering to {dataset_name}...")

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


# Apply feature engineering to both datasets
train_enhanced = apply_feature_engineering(train_df, "training data")
test_enhanced = apply_feature_engineering(test_df, "test data")

# Cache for performance
train_enhanced.cache()
test_enhanced.cache()

print("Enhanced features created:")
new_features = ["hour", "day_of_week", "month", "is_weekend", "time_period",
                "amt_category", "age", "age_group", "distance_km", "distance_category"]
for feature in new_features:
    print(f"- {feature}")

print("\n" + "=" * 50)
print("BASIC FEATURE ANALYSIS")
print("=" * 50)

# Analyze fraud rates by key features on training data
categorical_features = ["category", "gender", "time_period", "amt_category",
                        "age_group", "distance_category", "is_weekend"]

print("Fraud rates by categorical features (training data):")
for feature in categorical_features:
    print(f"\n{feature.upper()}:")
    fraud_by_feature = train_enhanced.groupBy(feature, "is_fraud").count() \
        .groupBy(feature).pivot("is_fraud", [0, 1]).sum("count") \
        .fillna(0) \
        .withColumn("total", col("0") + col("1")) \
        .withColumn("fraud_rate", round(col("1") / col("total") * 100, 2)) \
        .select(feature, "total", "1", "fraud_rate") \
        .orderBy(desc("fraud_rate"))

    fraud_by_feature.show(10)

print("\n" + "=" * 50)
print("VISUALIZATIONS")
print("=" * 50)

# Set up plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)

# Visualization 1: Fraud rate comparison between datasets
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Training data fraud distribution
train_labels = ['Non-Fraud', 'Fraud']
train_counts = train_fraud_pd['count'].values
ax1.pie(train_counts, labels=train_labels, autopct='%1.2f%%', startangle=90)
ax1.set_title('Training Data Fraud Distribution')

# Test data fraud distribution
test_labels = ['Non-Fraud', 'Fraud']
test_counts = test_fraud_pd['count'].values
ax2.pie(test_counts, labels=test_labels, autopct='%1.2f%%', startangle=90)
ax2.set_title('Test Data Fraud Distribution')

plt.tight_layout()
plt.show()

# Visualization 2: Fraud rates by time period
time_fraud_data = train_enhanced.groupBy("time_period", "is_fraud").count() \
    .groupBy("time_period").pivot("is_fraud", [0, 1]).sum("count") \
    .fillna(0) \
    .withColumn("total", col("0") + col("1")) \
    .withColumn("fraud_rate", col("1") / col("total") * 100) \
    .select("time_period", "fraud_rate").toPandas()

plt.figure(figsize=(10, 6))
bars = plt.bar(time_fraud_data['time_period'], time_fraud_data['fraud_rate'])
plt.title('Fraud Rate by Time Period')
plt.ylabel('Fraud Rate (%)')
plt.xlabel('Time Period')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{height:.2f}%', ha='center', va='bottom')

plt.show()

# Visualization 3: Fraud rates by amount category
amt_fraud_data = train_enhanced.groupBy("amt_category", "is_fraud").count() \
    .groupBy("amt_category").pivot("is_fraud", [0, 1]).sum("count") \
    .fillna(0) \
    .withColumn("total", col("0") + col("1")) \
    .withColumn("fraud_rate", col("1") / col("total") * 100) \
    .select("amt_category", "fraud_rate").toPandas()

# Sort by predefined order
amt_order = ['very_low', 'low', 'medium', 'high', 'very_high']
amt_fraud_data['amt_category'] = pd.Categorical(amt_fraud_data['amt_category'], categories=amt_order, ordered=True)
amt_fraud_data = amt_fraud_data.sort_values('amt_category')

plt.figure(figsize=(10, 6))
bars = plt.bar(amt_fraud_data['amt_category'], amt_fraud_data['fraud_rate'], color='red', alpha=0.7)
plt.title('Fraud Rate by Amount Category')
plt.ylabel('Fraud Rate (%)')
plt.xlabel('Amount Category')
plt.xticks(rotation=45)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{height:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Visualization 4: Dataset comparison chart
comparison_data = {
    'Dataset': ['Training', 'Test'],
    'Total Transactions': [train_df.count(), test_df.count()],
    'Fraud Rate (%)': [train_fraud_rate, test_fraud_rate]
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Transaction counts
ax1.bar(comparison_data['Dataset'], comparison_data['Total Transactions'],
        color=['blue', 'orange'], alpha=0.7)
ax1.set_title('Transaction Counts by Dataset')
ax1.set_ylabel('Number of Transactions')

# Add value labels
for i, v in enumerate(comparison_data['Total Transactions']):
    ax1.text(i, v + 10000, f'{v:,}', ha='center', va='bottom')

# Fraud rates
ax2.bar(comparison_data['Dataset'], comparison_data['Fraud Rate (%)'],
        color=['red', 'darkred'], alpha=0.7)
ax2.set_title('Fraud Rates by Dataset')
ax2.set_ylabel('Fraud Rate (%)')

# Add value labels
for i, v in enumerate(comparison_data['Fraud Rate (%)']):
    ax2.text(i, v + 0.01, f'{v:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("DATA PREPARATION COMPLETE")
print("=" * 50)
print("Summary:")
print(f"Training data: {train_enhanced.count():,} transactions with {len(train_enhanced.columns)} features")
print(f"Test data: {test_enhanced.count():,} transactions with {len(test_enhanced.columns)} features")
print("Both datasets ready for pattern discovery and validation")
print("Training data will be used for pattern discovery")
print("Test data will be used for pattern validation")

# Make datasets available for other scripts
train_data = train_enhanced
test_data = test_enhanced

print("\nEDA complete - ready for pattern discovery!")