#!/usr/bin/env python3
"""
Comprehensive visualization of ALL physicochemical methods for albumin partition coefficient prediction
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_all_results(base_dir):
    """Load results from all methods."""

    # Thermodynamic methods
    thermo_models = ['regular-solution', 'flory-huggins', 'delta-G-transfer', 'hybrid-thermo']

    # Group contribution & solubility methods
    gc_models = ['hansen', 'hildebrand', 'unifac', 'combined']

    all_results = {}

    # Load thermodynamic results
    for model in thermo_models:
        csv_path = base_dir / f"thermodynamic_{model}_predictions.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if 'logK_pred' in df.columns:
                y_true = df['logK_exp'].values
                y_pred = df['logK_pred'].values
                mask = ~(np.isnan(y_true) | np.isnan(y_pred))

                if np.sum(mask) > 0:
                    y_true = y_true[mask]
                    y_pred = y_pred[mask]

                    ss_res = np.sum((y_true - y_pred)**2)
                    ss_tot = np.sum((y_true - np.mean(y_true))**2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                    rmse = np.sqrt(np.mean((y_true - y_pred)**2))

                    from scipy.stats import spearmanr
                    spearman = spearmanr(y_true, y_pred).correlation

                    all_results[f'thermo_{model}'] = {
                        'category': 'Thermodynamic',
                        'r2': r2, 'rmse': rmse, 'spearman': spearman,
                        'y_true': y_true, 'y_pred': y_pred
                    }

    # Load group contribution results
    for model in gc_models:
        csv_path = base_dir / f"group_contrib_{model}_predictions.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if 'logK_pred' in df.columns:
                y_true = df['logK_exp'].values
                y_pred = df['logK_pred'].values
                mask = ~(np.isnan(y_true) | np.isnan(y_pred))

                if np.sum(mask) > 0:
                    y_true = y_true[mask]
                    y_pred = y_pred[mask]

                    ss_res = np.sum((y_true - y_pred)**2)
                    ss_tot = np.sum((y_true - np.mean(y_true))**2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                    rmse = np.sqrt(np.mean((y_true - y_pred)**2))

                    from scipy.stats import spearmanr
                    spearman = spearmanr(y_true, y_pred).correlation

                    all_results[f'gc_{model}'] = {
                        'category': 'Group Contribution',
                        'r2': r2, 'rmse': rmse, 'spearman': spearman,
                        'y_true': y_true, 'y_pred': y_pred
                    }

    return all_results


def plot_overall_comparison(all_results, output_dir):
    """Plot overall comparison of all methods."""

    # Categorize results
    categories = defaultdict(list)
    for name, result in all_results.items():
        categories[result['category']].append((name, result))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: R² Comparison - All Methods
    ax = axes[0, 0]
    methods = list(all_results.keys())
    r2_values = [all_results[m]['r2'] for m in methods]
    colors = ['#1b9e77' if 'thermo' in m else '#d95f02' if 'gc' in m else '#7570b3' for m in methods]

    bars = ax.bar(range(len(methods)), r2_values, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('_', ' ').replace('thermo', '').replace('gc', '').title()
                        for m in methods], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('All Physicochemical Methods - R² Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='R² = 0.5')
    ax.legend()

    for bar, val in zip(bars, r2_values):
        height = bar.get_height()
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    # Plot 2: RMSE Comparison
    ax = axes[0, 1]
    rmse_values = [all_results[m]['rmse'] for m in methods]
    bars = ax.bar(range(len(methods)), rmse_values, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('_', ' ').replace('thermo', '').replace('gc', '').title()
                        for m in methods], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('RMSE (log units)', fontsize=12)
    ax.set_title('All Physicochemical Methods - RMSE Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)

    for bar, val in zip(bars, rmse_values):
        height = bar.get_height()
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    # Plot 3: Spearman Correlation
    ax = axes[1, 0]
    spearman_values = [all_results[m]['spearman'] for m in methods]
    bars = ax.bar(range(len(methods)), spearman_values, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('_', ' ').replace('thermo', '').replace('gc', '').title()
                        for m in methods], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Spearman ρ', fontsize=12)
    ax.set_title('Rank Correlation Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    ax.set_ylim([0, 1])
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='ρ = 0.7')
    ax.legend()

    for bar, val in zip(bars, spearman_values):
        height = bar.get_height()
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    # Plot 4: Categorized Summary
    ax = axes[1, 1]

    category_data = []
    category_labels = []
    for cat_name, cat_results in categories.items():
        avg_r2 = np.mean([r[1]['r2'] for r in cat_results if not np.isnan(r[1]['r2'])])
        avg_rmse = np.mean([r[1]['rmse'] for r in cat_results if not np.isnan(r[1]['rmse'])])
        avg_spearman = np.mean([r[1]['spearman'] for r in cat_results if not np.isnan(r[1]['spearman'])])

        category_data.append([avg_r2, avg_rmse, avg_spearman])
        category_labels.append(cat_name)

    category_data = np.array(category_data)
    x = np.arange(len(category_labels))
    width = 0.25

    # Normalize values for comparison
    r2_norm = category_data[:, 0]
    rmse_norm = category_data[:, 1] / np.max(category_data[:, 1]) if np.max(category_data[:, 1]) > 0 else category_data[:, 1]
    spearman_norm = category_data[:, 2]

    ax.bar(x - width, r2_norm, width, label='R²', color='#1b9e77', alpha=0.8)
    ax.bar(x, rmse_norm, width, label='RMSE (norm)', color='#d95f02', alpha=0.8)
    ax.bar(x + width, spearman_norm, width, label='Spearman', color='#7570b3', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(category_labels)
    ax.set_ylabel('Normalized Value', fontsize=12)
    ax.set_title('Category-wise Summary (Normalized)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_dir / 'all_physicochemical_methods_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: all_physicochemical_methods_comparison.png")
    plt.close()


def plot_best_models_by_category(all_results, output_dir):
    """Plot best models from each category."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Find best model from each category
    best_models = {}
    for name, result in all_results.items():
        cat = result['category']
        if cat not in best_models or result['r2'] > best_models[cat]['r2']:
            best_models[cat] = {**result, 'name': name}

    # Plot 1: Scatter plots of best models
    ax = axes[0]
    colors_cat = {'Thermodynamic': '#1b9e77', 'Group Contribution': '#d95f02'}

    for cat, model in best_models.items():
        y_true = model['y_true']
        y_pred = model['y_pred']
        ax.scatter(y_true, y_pred, s=30, c=colors_cat.get(cat, 'gray'),
                   edgecolor='w', alpha=0.6, label=cat)

    # Diagonal line
    all_y_true = np.concatenate([model['y_true'] for model in best_models.values()])
    lo, hi = all_y_true.min(), all_y_true.max()
    ax.plot([lo, hi], [lo, hi], 'k-', lw=1, label='y = x')

    ax.set_xlabel('Experimental log K$_{BSA/w}$', fontsize=12)
    ax.set_ylabel('Predicted log K$_{BSA/w}$', fontsize=12)
    ax.set_title('Best Models from Each Category', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.25)

    # Plot 2: Performance comparison
    ax = axes[1]
    categories = list(best_models.keys())
    r2_vals = [best_models[cat]['r2'] for cat in categories]
    rmse_vals = [best_models[cat]['rmse'] for cat in categories]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, r2_vals, width, label='R²', color='#1b9e77', alpha=0.8)
    bars2 = ax.bar(x + width/2, rmse_vals, width, label='RMSE', color='#d95f02', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Best Model Performance by Category', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'best_models_by_category.png', dpi=150, bbox_inches='tight')
    print(f"Saved: best_models_by_category.png")
    plt.close()


def main():
    """Main visualization function."""
    base_dir = Path("/home1/s9383/Albumin")

    print(f"\n{'='*70}")
    print("COMPREHENSIVE VISUALIZATION: ALL PHYSICOCHEMICAL METHODS")
    print(f"{'='*70}")

    all_results = load_all_results(base_dir)

    print(f"\nLoaded {len(all_results)} method results")

    if all_results:
        # Print summary
        print(f"\n{'Method':<30} {'Category':<25} {'R2':<10} {'RMSE':<10} {'Spearman':<10}")
        print('-'*85)

        # Sort by R²
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['r2'], reverse=True)
        for name, result in sorted_results:
            display_name = name.replace('_', ' ').replace('thermo', 'Thermo: ').replace('gc', 'GC: ')
            print(f"{display_name:<30} {result['category']:<25} {result['r2']:<10.4f} {result['rmse']:<10.4f} {result['spearman']:<10.4f}")

        # Create plots
        plot_overall_comparison(all_results, base_dir)
        plot_best_models_by_category(all_results, base_dir)

        print(f"\n{'='*70}")
        print("TOP 5 METHODS BY R²")
        print(f"{'='*70}")
        for i, (name, result) in enumerate(sorted_results[:5], 1):
            display_name = name.replace('_', ' ').replace('thermo', '').replace('gc', '')
            print(f"{i}. {display_name:<25} R² = {result['r2']:.4f}, RMSE = {result['rmse']:.4f}")


if __name__ == "__main__":
    main()
