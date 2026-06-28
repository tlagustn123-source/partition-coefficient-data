#!/usr/bin/env python3
"""
Visualization script for Thermodynamic Partition Model Results
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def plot_comparison(all_results, output_dir):
    """Plot comparison of all methods."""
    methods = list(all_results.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: R² Comparison
    ax = axes[0, 0]
    r2_values = [all_results[m]['r2'] for m in methods]
    colors = ['#1b9e77' if 'thermo' in m else '#d95f02' for m in methods]
    bars = ax.bar(range(n_methods), r2_values, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels([m.replace('-', ' ').title() for m in methods], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('Model Performance Comparison (R²)', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    ax.set_ylim([0, max(r2_values) * 1.2])

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, r2_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Plot 2: RMSE Comparison
    ax = axes[0, 1]
    rmse_values = [all_results[m]['rmse'] for m in methods]
    bars = ax.bar(range(n_methods), rmse_values, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels([m.replace('-', ' ').title() for m in methods], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('RMSE (log units)', fontsize=12)
    ax.set_title('Model Performance Comparison (RMSE)', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    ax.set_ylim([0, max(rmse_values) * 1.2])

    for bar, val in zip(bars, rmse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Plot 3: Spearman Correlation
    ax = axes[1, 0]
    spearman_values = [all_results[m]['spearman'] for m in methods]
    bars = ax.bar(range(n_methods), spearman_values, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels([m.replace('-', ' ').title() for m in methods], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Spearman ρ', fontsize=12)
    ax.set_title('Rank Correlation Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    ax.set_ylim([0, 1])
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='ρ = 0.7')
    ax.legend()

    for bar, val in zip(bars, spearman_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Plot 4: Overall Score (combined metric)
    ax = axes[1, 1]
    # Combined score: R² - (RMSE / max_RMSE) + Spearman
    max_rmse = max(rmse_values)
    scores = [r2_values[i] - (rmse_values[i]/max_rmse) + spearman_values[i]
              for i in range(n_methods)]
    bars = ax.bar(range(n_methods), scores, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels([m.replace('-', ' ').title() for m in methods], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Combined Score', fontsize=12)
    ax.set_title('Overall Performance Score\n(R² - RMSE/max + Spearman)', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)

    for bar, val in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'thermodynamic_model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: thermodynamic_model_comparison.png")
    plt.close()


def plot_best_model(df_pred, model_type, output_dir):
    """Plot experimental vs predicted for best model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Remove NaN
    df_clean = df_pred.dropna(subset=['logK_exp', 'logK_pred'])
    y_true = df_clean['logK_exp'].values
    y_pred = df_clean['logK_pred'].values

    # Plot 1: Prediction vs Experiment
    ax = axes[0]
    ax.scatter(y_true, y_pred, s=40, c='#1b9e77', edgecolor='w', alpha=0.7)

    # Diagonal line
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'k-', lw=1, label='y = x')

    # Calibration line
    a, b = np.polyfit(y_pred, y_true, 1)
    x_line = np.array([lo, hi])
    y_cal = a * x_line + b
    ax.plot(x_line, y_cal, 'r--', lw=2, label=f'y = {a:.2f}x + {b:.2f}')

    ax.set_xlabel('Experimental log K$_{BSA/w}$', fontsize=12)
    ax.set_ylabel('Predicted log K$_{BSA/w}$', fontsize=12)
    ax.set_title(f'Thermodynamic Model ({model_type})\nR² = {np.corrcoef(y_true, y_pred)[0,1]**2:.3f}', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.25)

    # Plot 2: Residuals
    ax = axes[1]
    residuals = y_pred - y_true
    ax.scatter(y_true, residuals, s=40, c='#2c7fb8', edgecolor='w', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', lw=1)
    ax.set_xlabel('Experimental log K$_{BSA/w}$', fontsize=12)
    ax.set_ylabel('Residuals (pred - exp)', fontsize=12)
    ax.set_title(f'Residual Plot\nMean = {np.mean(residuals):.3f}', fontsize=13)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_dir / f'thermodynamic_{model_type}_fit.png', dpi=150, bbox_inches='tight')
    print(f"Saved: thermodynamic_{model_type}_fit.png")
    plt.close()


def main():
    """Main visualization function."""
    base_dir = Path("/home1/s9383/Albumin")

    # Load all prediction files
    all_results = {}

    # Thermodynamic models
    for model_type in ['regular-solution', 'flory-huggins', 'delta-G-transfer', 'hybrid-thermo']:
        csv_path = base_dir / f"thermodynamic_{model_type}_predictions.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            y_true = df['logK_exp'].values
            y_pred = df['logK_pred'].values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))

            if np.sum(mask) > 0:
                y_true = y_true[mask]
                y_pred = y_pred[mask]

                # Calculate R²
                ss_res = np.sum((y_true - y_pred)**2)
                ss_tot = np.sum((y_true - np.mean(y_true))**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                rmse = np.sqrt(np.mean((y_true - y_pred)**2))

                # Calculate Spearman
                from scipy.stats import spearmanr
                spearman = spearmanr(y_true, y_pred).correlation

                all_results[model_type] = {'r2': r2, 'rmse': rmse, 'spearman': spearman}

    print(f"\nLoaded {len(all_results)} thermodynamic model results")

    if all_results:
        # Plot comparison
        plot_comparison(all_results, base_dir)

        # Plot best model
        best_model = max(all_results.keys(), key=lambda k: all_results[k]['r2'])
        best_csv = base_dir / f"thermodynamic_{best_model}_predictions.csv"
        if best_csv.exists():
            df_pred = pd.read_csv(best_csv)
            plot_best_model(df_pred, best_model, base_dir)

        # Summary
        print(f"\n{'='*60}")
        print("THERMODYNAMIC MODEL SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<25} {'R²':<10} {'RMSE':<10} {'Spearman':<10}")
        print(f"{'-'*60}")
        for model in sorted(all_results.keys(), key=lambda k: all_results[k]['r2'], reverse=True):
            m = all_results[model]
            print(f"{model:<25} {m['r2']:<10.4f} {m['rmse']:<10.4f} {m['spearman']:<10.4f}")


if __name__ == "__main__":
    main()
