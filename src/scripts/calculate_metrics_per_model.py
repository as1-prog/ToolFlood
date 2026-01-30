#!/usr/bin/env python3
"""
Calculate average B, Poisoning rate, ASR, and TDR per model from results_table.csv.
"""

from pathlib import Path
import pandas as pd


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load results from CSV file."""
    return pd.read_csv(csv_path)


def calculate_metrics_per_model(df: pd.DataFrame):
    """
    Calculate average B, Poisoning rate, ASR, and TDR per model.

    Uses columns from results_table.csv:
    - B = Num_Injected_Tools
    - Poisoning rate = Num_Injected_Tools / Benign_Tools (per row)
    - ASR, TDR from CSV

    Args:
        df: DataFrame from results_table.csv (Model, Num_Injected_Tools, Benign_Tools, ASR, TDR)
    """
    df = df.copy()
    df['B'] = df['Num_Injected_Tools']
    df['Poisoning_Rate'] = df['Num_Injected_Tools'] / df['Benign_Tools']

    metrics_per_model = df.groupby('Model').agg({
        'B': 'mean',
        'Poisoning_Rate': 'mean',
        'ASR': 'mean',
        'TDR': 'mean',
    }).reset_index()

    metrics_per_model.columns = ['Model', 'Avg_B', 'Avg_Poisoning_Rate', 'Avg_ASR', 'Avg_TDR']

    return metrics_per_model, df


if __name__ == '__main__':
    # Load results from CSV
    csv_path = Path('experiment_output_poisonrag_blackbox_toole/results_table.csv')
    df = load_results(csv_path)

    # Calculate metrics per model
    metrics_df, full_df = calculate_metrics_per_model(df)

    # Format poisoning rate as percentage for display
    display_df = metrics_df.copy()
    display_df['Avg_Poisoning_Rate'] = (display_df['Avg_Poisoning_Rate'] * 100).round(4)

    # Display results
    print("\n" + "="*80)
    print("METRICS PER MODEL")
    print("="*80)
    print(display_df.to_string(index=False))

    # Save to CSV (keeping poisoning rate as decimal)
    output_csv = Path('experiment_output_poisonrag_blackbox_toole/metrics_per_model.csv')
    metrics_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total rows: {len(df)}")
    print(f"Number of models: {len(metrics_df)}")
    print(f"Models: {', '.join(metrics_df['Model'].tolist())}")
