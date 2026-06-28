#!/usr/bin/env python3
"""
Comprehensive Visualization of All Albumin Partition Coefficient Methods

Compares all implemented methods:
- Previous: Thermodynamic, Group Contribution
- New: Molecular Property, Advanced Physicochemical
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def load_all_predictions(base_dir):
    """Load all prediction files."""
    predictions = {}

    # Thermodynamic methods
    thermo_files = [
        'thermodynamic_flory-huggins_predictions.csv',
        'thermodynamic_delta-G-transfer_predictions.csv',
        'thermodynamic_regular-solution_predictions.csv',
    ]

    # Molecular property methods
    molprop_files = [
        'molecular_property_hydrophobic-balance_predictions.csv',
        'molecular_property_molar-volume_predictions.csv',
        'molecular_property_hybrid-property_predictions.csv',
    ]

    # Advanced physicochemical methods
    advanced_files = [
        'advanced_physicochem_polarizability_predictions.csv',
        'advanced_physicochem_dielectric_predictions.csv',
        'advanced_physicochem_hybrid-advanced_predictions.csv',
    ]

    # Group contribution methods
    gc_files = [
        'group_contrib_unifac_predictions.csv',
    ]

    all_files = thermo_files + molprop_files + advanced_files + gc_files

    for fname in all_files:
        fpath = base_dir / fname
        if fpath.exists():
            try:
                df = pd.read_csv(fpath)
                # Extract model name from filename
                model_name = fname.replace('_predictions.csv', '').replace('thermodynamic_', '').replace('molecular_property_', '').replace('advanced_physicochem_', '').replace('group_contrib_', '')
                predictions[model_name] = df
                print(f"Loaded: {model_name} ({len(df)} compounds)")
            except Exception as e:
                print(f"Error loading {fname}: {e}")

    return predictions


def calculate_metrics(df):
    """Calculate R², RMSE, Spearman, Pearson."""
    from scipy.stats import spearmanr, pearsonr

    y_true = df['logK_exp'].values
    y_pred = df['logK_pred'].values

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return None

    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    spearman = spearmanr(y_true, y_pred).correlation
    pearson = pearsonr(y_true, y_pred)[0]

    return {'r2': r2, 'rmse': rmse, 'spearman': spearman, 'pearson': pearson, 'n': len(y_true)}


def create_summary_table(predictions):
    """Create summary table of all methods."""
    results = []

    for model_name, df in predictions.items():
        metrics = calculate_metrics(df)
        if metrics:
            results.append({
                'Model': model_name,
                'R²': metrics['r2'],
                'RMSE': metrics['rmse'],
                'Spearman ρ': metrics['spearman'],
                'Pearson r': metrics['pearson'],
                'N': metrics['n']
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('R²', ascending=False)
    return results_df


def plot_comparison_scatter(predictions, output_path):
    """Create scatter plots for top methods."""
    top_models = sorted(predictions.items(),
                       key=lambda x: calculate_metrics(x[1])['r2'] if calculate_metrics(x[1]) else 0,
                       reverse=True)[:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (model_name, df) in enumerate(top_models):
        ax = axes[idx]

        y_true = df['logK_exp'].values
        y_pred = df['logK_pred'].values

        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        metrics = calculate_metrics(df)

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)

        # Identity line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y = x')

        # Trend line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax.plot(y_true, p(y_true), 'b-', linewidth=2, label='Trend')

        ax.set_xlabel('Experimental logK', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted logK', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name}\nR² = {metrics["r2"]:.4f} | RMSE = {metrics["rmse"]:.4f}',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison scatter plot to: {output_path}")
    plt.close()


def plot_bar_comparison(results_df, output_path):
    """Create bar chart comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Color by category
    def get_color(model_name):
        if 'flory' in model_name or 'delta-G' in model_name or 'regular-solution' in model_name:
            return '#1f77b4'  # Blue for Thermodynamic
        elif 'hydrophobic' in model_name or 'molar-volume' in model_name or 'parachor' in model_name or 'surface' in model_name:
            return '#2ca02c'  # Green for Molecular Property
        elif 'polarizability' in model_name or 'dielectric' in model_name or 'refractivity' in model_name or 'fractional' in model_name:
            return '#ff7f0e'  # Orange for Advanced
        elif 'unifac' in model_name or 'hansen' in model_name:
            return '#d62728'  # Red for Group Contribution
        elif 'hybrid' in model_name:
            return '#9467bd'  # Purple for Hybrid
        else:
            return '#7f7f7f'  # Gray for others

    colors = [get_color(model) for model in results_df['Model']]

    # R² plot
    axes[0].barh(results_df['Model'], results_df['R²'], color=colors)
    axes[0].set_xlabel('R²', fontsize=12, fontweight='bold')
    axes[0].set_title('Coefficient of Determination (R²)', fontsize=13, fontweight='bold')
    axes[0].set_xlim(0, max(results_df['R²']) * 1.2)
    axes[0].grid(axis='x', alpha=0.3)

    # RMSE plot
    axes[1].barh(results_df['Model'], results_df['RMSE'], color=colors)
    axes[1].set_xlabel('RMSE', fontsize=12, fontweight='bold')
    axes[1].set_title('Root Mean Square Error', fontsize=13, fontweight='bold')
    axes[1].invert_yaxis()  # Best (lowest) at top
    axes[1].grid(axis='x', alpha=0.3)

    # Spearman plot
    axes[2].barh(results_df['Model'], results_df['Spearman ρ'], color=colors)
    axes[2].set_xlabel('Spearman ρ', fontsize=12, fontweight='bold')
    axes[2].set_title('Spearman Correlation', fontsize=13, fontweight='bold')
    axes[2].set_xlim(0, 1)
    axes[2].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved bar comparison to: {output_path}")
    plt.close()


def plot_category_comparison(results_df, output_path):
    """Compare method categories."""
    # Categorize models
    categories = []
    for model in results_df['Model']:
        if 'flory' in model or 'delta-G' in model or 'regular-solution' in model:
            categories.append('Thermodynamic')
        elif 'hydrophobic' in model or 'molar-volume' in model or 'parachor' in model or 'surface' in model:
            categories.append('Molecular Property')
        elif 'polarizability' in model or 'dielectric' in model or 'refractivity' in model or 'fractional' in model:
            categories.append('Advanced Physicochem')
        elif 'unifac' in model or 'hansen' in model:
            categories.append('Group Contribution')
        elif 'hybrid' in model:
            categories.append('Hybrid')
        else:
            categories.append('Other')

    results_df['Category'] = categories

    # Aggregate by category
    category_stats = results_df.groupby('Category').agg({
        'R²': ['mean', 'max'],
        'RMSE': ['mean', 'min'],
        'Spearman ρ': ['mean', 'max']
    }).round(4)

    print("\n" + "="*70)
    print("CATEGORY COMPARISON")
    print("="*70)
    print(category_stats)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    category_means = results_df.groupby('Category')['R²'].mean().sort_values(ascending=False)
    category_means.plot(kind='bar', ax=axes[0], color='#1f77b4', edgecolor='k')
    axes[0].set_ylabel('Mean R²', fontsize=12, fontweight='bold')
    axes[0].set_title('Mean R² by Category', fontsize=13, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)

    category_rmse = results_df.groupby('Category')['RMSE'].mean().sort_values()
    category_rmse.plot(kind='bar', ax=axes[1], color='#ff7f0e', edgecolor='k')
    axes[1].set_ylabel('Mean RMSE', fontsize=12, fontweight='bold')
    axes[1].set_title('Mean RMSE by Category', fontsize=13, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)

    category_spearman = results_df.groupby('Category')['Spearman ρ'].mean().sort_values(ascending=False)
    category_spearman.plot(kind='bar', ax=axes[2], color='#2ca02c', edgecolor='k')
    axes[2].set_ylabel('Mean Spearman ρ', fontsize=12, fontweight='bold')
    axes[2].set_title('Mean Spearman ρ by Category', fontsize=13, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved category comparison to: {output_path}")
    plt.close()

    return results_df


def main():
    """Main function to create comprehensive visualization."""
    base_dir = Path("/home1/s9383/Albumin")

    print("="*70)
    print("COMPREHENSIVE VISUALIZATION OF ALL METHODS")
    print("="*70)

    # Load all predictions
    predictions = load_all_predictions(base_dir)

    if not predictions:
        print("No prediction files found!")
        return

    # Create summary table
    results_df = create_summary_table(predictions)

    print("\n" + "="*70)
    print("ALL METHODS COMPARISON (sorted by R²)")
    print("="*70)
    print(results_df.to_string(index=False))

    # Save summary table
    summary_path = base_dir / "all_methods_summary.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary table to: {summary_path}")

    # Create visualizations
    plot_comparison_scatter(predictions, base_dir / "all_methods_comparison_scatter.png")
    plot_bar_comparison(results_df, base_dir / "all_methods_bar_comparison.png")
    results_df = plot_category_comparison(results_df, base_dir / "all_methods_category_comparison.png")

    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total methods evaluated: {len(results_df)}")
    print(f"Best method: {results_df.iloc[0]['Model']} (R² = {results_df.iloc[0]['R²']:.4f})")
    print(f"\nTop 5 methods:")
    for idx, row in results_df.head(5).iterrows():
        print(f"  {row['Model']:<30} R² = {row['R²']:.4f} | RMSE = {row['RMSE']:.4f} | Spearman = {row['Spearman ρ']:.4f}")


if __name__ == "__main__":
    main()
