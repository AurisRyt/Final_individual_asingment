
# === FRAUD PATTERN VISUALIZATION ===

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from datetime import datetime

print("=" * 60)
print("SIMPLE FRAUD PATTERN VISUALIZATION PIPELINE")
print("=" * 60)

# Create output directory
os.makedirs("output", exist_ok=True)
os.makedirs("output/visualizations", exist_ok=True)

# Find CSV files - look for the original format files
print("Looking for pattern CSV files...")

# Look for the specific file patterns that exist in your directory
file_patterns = [
    "fraud_patterns_TOP15_*.csv",
    "fraud_patterns_all_FULL_TRAINSET_*.csv"
]

csv_files = []
for pattern in file_patterns:
    files = glob.glob(pattern)
    csv_files.extend(files)

# Remove summary files
csv_files = [f for f in csv_files if 'SUMMARY' not in f]

print(f"Found pattern files: {csv_files}")

if not csv_files:
    print("ERROR: No pattern CSV files found!")
    print("Looking for files like: fraud_patterns_TOP15_*.csv")
    # Show what files exist
    all_csv = glob.glob("*.csv")
    print(f"Available CSV files: {all_csv}")
    exit(1)

# Prefer TOP15 file if available, otherwise use the largest file
top15_files = [f for f in csv_files if 'TOP15' in f]
if top15_files:
    latest_file = max(top15_files, key=os.path.getctime)
else:
    latest_file = max(csv_files, key=os.path.getctime)

print(f"Using file: {latest_file}")

# Load data
try:
    df = pd.read_csv(latest_file)
    print(f"Loaded {len(df)} patterns from CSV")
    print(f"Columns found: {list(df.columns)}")

    # Check if this looks like pattern data
    expected_cols = ['Pattern_Name', 'Confidence', 'Test_Confidence', 'Train_Confidence', 'Fraud_Rate_Percent']
    has_pattern_data = any(col in df.columns for col in expected_cols)

    if not has_pattern_data:
        print("WARNING: This doesn't look like pattern data!")
        print("Trying to find a better file...")

        # Try to find a better file
        for file in csv_files:
            if file != latest_file:
                try:
                    test_df = pd.read_csv(file)
                    if any(col in test_df.columns for col in expected_cols):
                        print(f"Found better file: {file}")
                        df = test_df
                        latest_file = file
                        break
                except:
                    continue

        print(f"Final file being used: {latest_file}")
        print(f"Final columns: {list(df.columns)}")

except Exception as e:
    print(f"ERROR loading CSV: {e}")
    exit(1)

# === VISUALIZATION 1: SIMPLE BAR CHART ===
print("\nCreating bar chart of top patterns...")

try:
    # For the original Association_rule_mining.py output format
    confidence_col = None

    # Check for original format columns
    if 'Confidence' in df.columns:
        confidence_col = 'Confidence'
    elif 'Fraud_Rate_Percent' in df.columns:
        confidence_col = 'Fraud_Rate_Percent'

    print(f"Using confidence column: {confidence_col}")

    if confidence_col is None:
        print("WARNING: No confidence column found!")
        print(f"Available columns: {list(df.columns)}")

        # Create pattern type distribution if available
        if 'Pattern_Type' in df.columns:
            plt.figure(figsize=(10, 6))
            type_counts = df['Pattern_Type'].value_counts()
            bars = plt.bar(type_counts.index, type_counts.values, color=['blue', 'orange', 'green'], alpha=0.7)

            # Add value labels
            for bar, count in zip(bars, type_counts.values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                         f'{count}', ha='center', va='bottom', fontweight='bold')

            plt.title('Pattern Type Distribution', fontsize=16, fontweight='bold')
            plt.ylabel('Number of Patterns')
            plt.xlabel('Pattern Type')
            plt.tight_layout()

            chart_path = "output/visualizations/pattern_types.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            print(f"Pattern type chart saved: {chart_path}")
            plt.close()
        else:
            print("Creating basic summary chart...")
            plt.figure(figsize=(8, 6))
            plt.bar(['Total Patterns'], [len(df)], color='blue', alpha=0.7)
            plt.title('Fraud Pattern Discovery Results', fontsize=16, fontweight='bold')
            plt.ylabel('Number of Patterns')
            plt.text(0, len(df) + len(df) * 0.05, f'{len(df)}', ha='center', va='bottom', fontweight='bold')
            plt.tight_layout()

            basic_path = "output/visualizations/basic_summary.png"
            plt.savefig(basic_path, dpi=300, bbox_inches='tight')
            print(f"Basic summary saved: {basic_path}")
            plt.close()

    else:
        # Show actual pattern names visualization
        print(f"Creating visualization with {len(df)} patterns...")

        # Get top 10 patterns
        top_10 = df.head(10).copy()

        # Get confidence values
        confidence_values = top_10[confidence_col].values

        # Convert to percentage if needed (original format uses 0-1 scale)
        if confidence_col == 'Confidence' and confidence_values.max() <= 1.0:
            confidence_values = confidence_values * 100

        # Print patterns to console first
        print(f"\nTOP 10 FRAUD PATTERNS:")
        print("=" * 100)
        for i, row in top_10.iterrows():
            conf_val = row[confidence_col]
            if confidence_col == 'Confidence' and conf_val <= 1.0:
                conf_val *= 100
            print(f"{i + 1:2d}. {row['Pattern_Name']}")
            print(f"    Fraud Rate: {conf_val:.1f}%  |  Lift: {row.get('Lift', 'N/A'):.1f}x")
            print()

        # Create horizontal bar chart to show full pattern names
        plt.figure(figsize=(16, 12))

        # Get pattern names and clean them up
        pattern_names = []
        for name in top_10['Pattern_Name']:
            # Clean up pattern names to be more readable
            clean_name = name.replace(' AND ', ' & ').replace('=', ': ')
            # Limit length but keep it readable
            if len(clean_name) > 80:
                clean_name = clean_name[:77] + "..."
            pattern_names.append(clean_name)

        # Create horizontal bar chart (better for long text)
        y_positions = range(len(pattern_names))
        bars = plt.barh(y_positions, confidence_values, color='darkred', alpha=0.8)

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, confidence_values)):
            width = bar.get_width()
            plt.text(width + max(confidence_values) * 0.01, bar.get_y() + bar.get_height() / 2.,
                     f'{value:.1f}%', ha='left', va='center', fontweight='bold', fontsize=11)

        # Set y-axis labels to show actual pattern names
        plt.yticks(y_positions, [f"{i + 1}. {name}" for i, name in enumerate(pattern_names)], fontsize=10)
        plt.gca().invert_yaxis()  # Top pattern at top

        plt.title('Top 10 Fraud Patterns - Actual Pattern Names', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Fraud Detection Rate (%)', fontsize=12, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)

        # Adjust layout to fit long pattern names
        plt.tight_layout()
        plt.subplots_adjust(left=0.5)  # Make room for pattern names

        # Save chart
        chart_path = "output/visualizations/fraud_patterns_with_names.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"Fraud patterns chart saved: {chart_path}")
        plt.close()

        # Create a simple vertical chart with numbers for presentation
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(top_10)), confidence_values, color='darkred', alpha=0.8)

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, confidence_values)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + max(confidence_values) * 0.01,
                     f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

        plt.title('Top 10 Fraud Patterns - Detection Rates', fontsize=16, fontweight='bold')
        plt.ylabel('Fraud Detection Rate (%)', fontsize=12, fontweight='bold')
        plt.xlabel('Pattern Rank (see console output for full pattern names)', fontsize=12)
        plt.xticks(range(len(top_10)), [f"#{i + 1}" for i in range(len(top_10))])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        simple_path = "output/visualizations/fraud_patterns_simple.png"
        plt.savefig(simple_path, dpi=300, bbox_inches='tight')
        print(f"Simple chart saved: {simple_path}")
        plt.close()

except Exception as e:
    print(f"ERROR creating visualizations: {e}")
    print("Creating basic summary...")

    # Create very basic summary
    plt.figure(figsize=(8, 6))
    plt.bar(['Patterns Discovered'], [len(df)], color='green', alpha=0.7)
    plt.title('Fraud Pattern Analysis Results', fontsize=16, fontweight='bold')
    plt.ylabel('Count')
    plt.text(0, len(df) + len(df) * 0.05, f'{len(df)}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()

    fallback_path = "output/visualizations/results_summary.png"
    plt.savefig(fallback_path, dpi=300, bbox_inches='tight')
    print(f"Results summary saved: {fallback_path}")
    plt.close()

# === VISUALIZATION 2: PATTERN SUMMARY ===
print("\nCreating pattern summary...")

try:
    # Create summary statistics
    plt.figure(figsize=(10, 6))

    # Summary data
    total_patterns = len(df)

    # Try to find different metrics
    if confidence_col in df.columns:
        avg_confidence = df[confidence_col].mean()
        max_confidence = df[confidence_col].max()

        # Convert to percentage if needed
        if avg_confidence <= 1.0:
            avg_confidence *= 100
            max_confidence *= 100
    else:
        avg_confidence = 0
        max_confidence = 0

    # Create summary bar chart
    categories = ['Total\nPatterns', 'Average\nConfidence (%)', 'Best\nConfidence (%)']
    values = [total_patterns, avg_confidence, max_confidence]
    colors = ['blue', 'orange', 'red']

    bars = plt.bar(categories, values, color=colors, alpha=0.7)

    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        if 'Pattern' in bar.get_x():
            label = f'{int(value)}'
        else:
            label = f'{value:.1f}%'
        plt.text(bar.get_x() + bar.get_width() / 2., height + max(values) * 0.02,
                 label, ha='center', va='bottom', fontweight='bold')

    plt.title('Fraud Pattern Analysis Summary', fontsize=16, fontweight='bold')
    plt.ylabel('Value', fontsize=12)
    plt.tight_layout()

    # Save summary
    summary_path = "output/visualizations/pattern_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Summary chart saved: {summary_path}")
    plt.close()

except Exception as e:
    print(f"ERROR creating summary: {e}")

# === SIMPLE DATA EXPORT ===
print("\nCreating simple data summary...")

try:
    # Create text summary
    summary_text = f"""
FRAUD PATTERN ANALYSIS SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET INFORMATION:
- Total patterns analyzed: {len(df)}
- Data source: {latest_file}

TOP 5 PATTERNS:
"""

    # Add top 5 patterns to summary
    for i in range(min(5, len(df))):
        pattern_name = df.iloc[i].get('Pattern_Name', f'Pattern {i + 1}')
        confidence = df.iloc[i].get(confidence_col, 0)
        if confidence <= 1.0:
            confidence *= 100

        summary_text += f"{i + 1}. {pattern_name[:60]}...\n"
        summary_text += f"   Fraud Rate: {confidence:.1f}%\n\n"

    # Save text summary
    text_path = "output/visualizations/analysis_summary.txt"
    with open(text_path, 'w') as f:
        f.write(summary_text)
    print(f"Text summary saved: {text_path}")

except Exception as e:
    print(f"ERROR creating text summary: {e}")

# === FINAL REPORT ===
print("\n" + "=" * 60)
print("SIMPLE VISUALIZATION COMPLETE!")
print("=" * 60)
print("Generated files:")
print("1. output/visualizations/top_patterns_simple.png - Bar chart of top patterns")
print("2. output/visualizations/pattern_summary.png - Analysis summary")
print("3. output/visualizations/analysis_summary.txt - Text summary")

print(f"\nData Summary:")
print(f"- Processed {len(df)} patterns from {latest_file}")
if confidence_col:
    best_confidence = df[confidence_col].max()
    if best_confidence <= 1.0:
        best_confidence *= 100
    print(f"- Best pattern confidence: {best_confidence:.1f}%")

print("\nFiles are ready for your presentation!")
print("=" * 60)
