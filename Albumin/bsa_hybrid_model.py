#!/usr/bin/env python3
"""
BSA Hybrid PP-LFER Model: Vina Docking + COSMO-RS Predictions

This module implements a hybrid PP-LFER model that combines:
1. AutoDock Vina docking scores (binding affinity)
2. COSMO-RS predictions (solvation thermodynamics)

Model form: logK = a * vina_score + b * cosmo_pred + c
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class HybridBSAModel:
    """
    Hybrid PP-LFER model for BSA partition coefficient prediction.

    Combines Vina docking scores with COSMO-RS predictions.
    """

    def __init__(self, model_type='linear'):
        """Initialize the hybrid model."""
        if model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=0.1)
        else:
            self.model = LinearRegression()

        self.model_type = model_type
        self.fitted = False
        self.feature_names = []

    def fit(self, X, y, feature_names=None):
        """Fit the hybrid model."""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y)

        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, X):
        """Predict logK values."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model.predict(X)

    def get_coefficients(self):
        """Get model coefficients."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            intercept = self.model.intercept_
        else:
            raise ValueError("Model does not have coefficients")

        return dict(zip(self.feature_names, coef)), intercept

    def evaluate(self, y_true, y_pred):
        """Calculate evaluation metrics."""
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = np.array(y_true)[mask]
        y_pred = np.array(y_pred)[mask]

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Calibrated R² (linear fit between pred and exp)
        if len(y_true) > 2:
            a, b = np.polyfit(y_pred, y_true, 1)
            y_cal = a * y_pred + b
            r2_cal = r2_score(y_true, y_cal)
            rmse_cal = np.sqrt(mean_squared_error(y_true, y_cal))
        else:
            r2_cal, rmse_cal, a, b = np.nan, np.nan, np.nan, np.nan

        spearman = spearmanr(y_true, y_pred).correlation
        pearson = pearsonr(y_true, y_pred)[0]

        return {
            'r2': r2,
            'rmse': rmse,
            'r2_cal': r2_cal,
            'rmse_cal': rmse_cal,
            'spearman': spearman,
            'pearson': pearson,
            'cal_slope': a,
            'cal_intercept': b
        }


def load_data(base_dir):
    """Load all required data files."""
    base_dir = Path(base_dir)

    # Load experimental data
    df_exp = pd.read_csv(base_dir / "albumin_data.csv")
    print(f"Loaded {len(df_exp)} experimental data points")

    # Load Vina focused results
    df_vina_focus = pd.read_csv(base_dir / "vina_focused_sudlow1_results.csv")
    print(f"Loaded {len(df_vina_focus)} Vina focused results")

    # Load Vina blind results
    df_vina_blind = pd.read_csv(base_dir / "vina_blind_results.csv")
    print(f"Loaded {len(df_vina_blind)} Vina blind results")

    # Load COSMO-RS results
    cosmo_json = base_dir / "cosmo_xtb/cosmo_work_v4_albumin/results_openrs_albumin_v4.json"
    with open(cosmo_json) as f:
        cosmo_data = json.load(f)
    df_cosmo = pd.DataFrame(cosmo_data['rows'])
    df_cosmo = df_cosmo.rename(columns={'pred': 'cosmo_pred', 'exp': 'exp_cosmo'})
    print(f"Loaded {len(df_cosmo)} COSMO-RS predictions")

    # Merge all data
    df = df_exp[['name', 'logK_exp', 'smiles']].copy()

    # Merge with Vina focused
    df = df.merge(df_vina_focus[['name', 'vina_score', 'logKpw_pred']],
                  on='name', how='left', suffixes=('', '_vina_focus'))

    # Merge with Vina blind
    df = df.merge(df_vina_blind[['name', 'vina_score', 'logKpw_pred']],
                  on='name', how='left', suffixes=('', '_vina_blind'))

    # Rename Vina columns
    df = df.rename(columns={
        'vina_score': 'vina_score_focus',
        'logKpw_pred': 'logKpw_pred_focus',
        'vina_score_vina_blind': 'vina_score_blind',
        'logKpw_pred_vina_blind': 'logKpw_pred_blind'
    })

    # Merge with COSMO-RS
    df = df.merge(df_cosmo[['name', 'cosmo_pred']], on='name', how='left')

    print(f"\nMerged data: {len(df)} compounds")
    print(f"  Complete cases: {df.dropna().shape[0]}")

    return df


def train_hybrid_model(df, features, target='logK_exp'):
    """
    Train hybrid PP-LFER model.

    Args:
        df: DataFrame with all features
        features: List of feature column names
        target: Target column name

    Returns:
        Model, predictions, metrics, clean dataframe
    """
    # Remove NaN values
    df_clean = df.dropna(subset=features + [target]).copy()

    X = df_clean[features].values
    y = df_clean[target].values

    print(f"  Training samples: {len(df_clean)}")
    print(f"  Features: {features}")

    # Train model
    model = HybridBSAModel(model_type='linear')
    model.fit(X, y, feature_names=features)

    # Get coefficients
    coef, intercept = model.get_coefficients()
    print(f"\n  Model coefficients:")
    for name, value in coef.items():
        print(f"    {name}: {value:.6f}")
    print(f"    Intercept: {intercept:.6f}")

    # Predict
    y_pred = model.predict(X)

    # Calculate metrics
    metrics = model.evaluate(y, y_pred)

    return model, y_pred, metrics, df_clean


def compare_all_methods(df):
    """Compare all prediction methods."""
    results = {}

    # Method 1: Vina focused only
    print("\n" + "="*60)
    print("Method 1: Vina Focused Only")
    print("="*60)
    model, y_pred, metrics, df_clean = train_hybrid_model(
        df, features=['vina_score_focus'])
    results['Vina Focused'] = {'metrics': metrics, 'y_pred': y_pred,
                               'df': df_clean, 'model': model}

    # Method 2: Vina blind only
    print("\n" + "="*60)
    print("Method 2: Vina Blind Only")
    print("="*60)
    model, y_pred, metrics, df_clean = train_hybrid_model(
        df, features=['vina_score_blind'])
    results['Vina Blind'] = {'metrics': metrics, 'y_pred': y_pred,
                            'df': df_clean, 'model': model}

    # Method 3: COSMO-RS only
    print("\n" + "="*60)
    print("Method 3: COSMO-RS Only")
    print("="*60)
    model, y_pred, metrics, df_clean = train_hybrid_model(
        df, features=['cosmo_pred'])
    results['COSMO-RS'] = {'metrics': metrics, 'y_pred': y_pred,
                         'df': df_clean, 'model': model}

    # Method 4: Hybrid - Vina Focus + COSMO
    print("\n" + "="*60)
    print("Method 4: Hybrid (Vina Focus + COSMO-RS)")
    print("="*60)
    model, y_pred, metrics, df_clean = train_hybrid_model(
        df, features=['vina_score_focus', 'cosmo_pred'])
    results['Hybrid Focus'] = {'metrics': metrics, 'y_pred': y_pred,
                              'df': df_clean, 'model': model}

    # Method 5: Hybrid - Vina Blind + COSMO
    print("\n" + "="*60)
    print("Method 5: Hybrid (Vina Blind + COSMO-RS)")
    print("="*60)
    model, y_pred, metrics, df_clean = train_hybrid_model(
        df, features=['vina_score_blind', 'cosmo_pred'])
    results['Hybrid Blind'] = {'metrics': metrics, 'y_pred': y_pred,
                              'df': df_clean, 'model': model}

    # Method 6: Hybrid - All three
    print("\n" + "="*60)
    print("Method 6: Hybrid (Vina Focus + Vina Blind + COSMO-RS)")
    print("="*60)
    model, y_pred, metrics, df_clean = train_hybrid_model(
        df, features=['vina_score_focus', 'vina_score_blind', 'cosmo_pred'])
    results['Hybrid All'] = {'metrics': metrics, 'y_pred': y_pred,
                            'df': df_clean, 'model': model}

    return results


def print_summary(results):
    """Print summary comparison table."""
    print("\n" + "="*80)
    print("BSA PARTITION COEFFICIENT MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Method':<25} {'R²':<10} {'Calibrated R²':<15} {'RMSE':<10} {'Spearman':<10}")
    print("-"*80)

    for name, res in results.items():
        m = res['metrics']
        print(f"{name:<25} {m['r2']:<10.4f} {m['r2_cal']:<15.4f} "
              f"{m['rmse']:<10.4f} {m['spearman']:<10.4f}")

    print("-"*80)
    print("\nBest R² models:")
    sorted_by_r2 = sorted(results.items(), key=lambda x: x[1]['metrics']['r2'], reverse=True)
    for i, (name, res) in enumerate(sorted_by_r2, 1):
        m = res['metrics']
        print(f"  {i}. {name}: R² = {m['r2']:.4f}, Calibrated R² = {m['r2_cal']:.4f}")


def plot_comparison(results, output_dir):
    """Plot comparison of all methods."""
    output_dir = Path(output_dir)

    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx]
        df = res['df']
        y_true = df['logK_exp'].values
        y_pred = res['y_pred']
        m = res['metrics']

        # Scatter plot
        ax.scatter(y_true, y_pred, s=35, c='#2c7fb8', edgecolor='w', alpha=0.7)

        # Diagonal line
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], 'k-', lw=1, alpha=0.5)

        ax.set_xlabel('Experimental log K', fontsize=10)
        ax.set_ylabel('Predicted log K', fontsize=10)
        ax.set_title(f'{name}\nR² = {m["r2"]:.3f}, RMSE = {m["rmse"]:.3f}',
                    fontsize=11, fontweight='bold')
        ax.grid(alpha=0.25)

    plt.suptitle('BSA Partition Coefficient: Model Comparison',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    png_path = output_dir / "bsa_model_comparison.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"  Saved comparison plot: {png_path}")
    plt.close()

    # Create R² comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(results.keys())
    r2_values = [results[name]['metrics']['r2'] for name in names]
    r2_cal_values = [results[name]['metrics']['r2_cal'] for name in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, r2_values, width, label='R²', color='#1b9e77')
    bars2 = ax.bar(x + width/2, r2_cal_values, width, label='Calibrated R²', color='#d95f02')

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('BSA Partition Coefficient: R² Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    ax.set_ylim([0, max(max(r2_cal_values), 0.5) * 1.1])

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    png_path = output_dir / "bsa_r2_comparison.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"  Saved R² comparison: {png_path}")
    plt.close()


def save_results(results, output_dir):
    """Save prediction results to CSV."""
    output_dir = Path(output_dir)

    for name, res in results.items():
        df = res['df'].copy()
        df['logK_pred'] = res['y_pred']
        df['residual'] = df['logK_pred'] - df['logK_exp']

        safe_name = name.lower().replace(' ', '_').replace('/', '_')
        csv_path = output_dir / f"bsa_{safe_name}_predictions.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")


def main():
    """Main function to train and evaluate hybrid BSA model."""
    base_dir = Path("/home1/s9383/Albumin")

    print("="*80)
    print("BSA HYBRID PP-LFER MODEL: Vina Docking + COSMO-RS")
    print("="*80)

    # Load all data
    df = load_data(base_dir)

    # Compare all methods
    results = compare_all_methods(df)

    # Print summary
    print_summary(results)

    # Plot comparison
    plot_comparison(results, base_dir)

    # Save results
    save_results(results, base_dir)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
