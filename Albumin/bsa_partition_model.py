#!/usr/bin/env python3
"""
BSA Partition Coefficient Module using PP-LFER Model with AutoDock Vina

This module implements a PP-LFER (Polyparameter Linear Free Energy Relationship)
model for predicting BSA partition coefficients using AutoDock Vina docking scores.

The model form: logK = a * docking_score + b (simple linear regression)
More complex models can include additional molecular descriptors.
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class BSAPartitionModel:
    """
    PP-LFER model for BSA partition coefficient prediction.

    Model form: logK = a * E + s * S + a * A + b * B + v * V + c
    where E, S, A, B, V are molecular descriptors.

    With docking scores: logK = a * docking_score + b
    """

    def __init__(self, model_type='linear'):
        """
        Initialize the BSA partition model.

        Args:
            model_type: 'linear' or 'ridge'
        """
        self.model_type = model_type
        if model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        else:
            self.model = LinearRegression()
        self.coef_ = None
        self.intercept_ = None
        self.fitted = False

    def fit(self, X, y):
        """
        Fit the PP-LFER model.

        Args:
            X: Feature matrix (docking scores or molecular descriptors)
            y: Experimental logK values
        """
        X = np.array(X).reshape(-1, 1) if len(np.array(X).shape) == 1 else np.array(X)
        y = np.array(y)

        self.model.fit(X, y)

        if self.model_type == 'linear':
            self.coef_ = self.model.coef_
            self.intercept_ = self.model.intercept_
        else:
            self.coef_ = self.model.coef_
            self.intercept_ = self.model.intercept_

        self.fitted = True

    def predict(self, X):
        """
        Predict logK values.

        Args:
            X: Feature matrix

        Returns:
            Predicted logK values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X).reshape(-1, 1) if len(np.array(X).shape) == 1 else np.array(X)
        return self.model.predict(X)

    def score(self, X, y):
        """Calculate R² score."""
        return r2_score(y, self.predict(X))

    def evaluate(self, y_true, y_pred):
        """
        Calculate evaluation metrics.

        Returns:
            Dictionary with R², RMSE, Spearman, Pearson
        """
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = np.array(y_true)[mask]
        y_pred = np.array(y_pred)[mask]

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        spearman = spearmanr(y_true, y_pred).correlation
        pearson = pearsonr(y_true, y_pred)[0]

        # Calibrated R² (linear fit between pred and exp)
        if len(y_true) > 2:
            a, b = np.polyfit(y_pred, y_true, 1)
            y_cal = a * y_pred + b
            r2_cal = r2_score(y_true, y_cal)
            rmse_cal = np.sqrt(mean_squared_error(y_true, y_cal))
        else:
            r2_cal, rmse_cal = np.nan, np.nan

        return {
            'r2': r2,
            'rmse': rmse,
            'r2_cal': r2_cal,
            'rmse_cal': rmse_cal,
            'spearman': spearman,
            'pearson': pearson,
            'cal_slope': a if len(y_true) > 2 else np.nan,
            'cal_intercept': b if len(y_true) > 2 else np.nan
        }


def load_data(albumin_csv, vina_csv=None, boltz_csv=None):
    """
    Load experimental and docking data.

    Args:
        albumin_csv: Path to albumin experimental data
        vina_csv: Path to Vina results (optional)
        boltz_csv: Path to Boltz2 results (optional)

    Returns:
        DataFrame with merged data
    """
    df_exp = pd.read_csv(albumin_csv)

    if vina_csv:
        df_vina = pd.read_csv(vina_csv)
        df = df_exp.merge(df_vina, on='name', how='left', suffixes=('', '_vina'))

    if boltz_csv:
        df_boltz = pd.read_csv(boltz_csv)
        df = df.merge(df_boltz, on='name', how='left', suffixes=('', '_boltz'))

    return df


def train_pp_lfer_model(df, feature_cols=['vina_score']):
    """
    Train PP-LFER model with docking scores.

    Args:
        df: DataFrame with experimental and docking data
        feature_cols: Columns to use as features

    Returns:
        Trained model and predictions
    """
    # Remove NaN values
    df_clean = df.dropna(subset=feature_cols + ['logK_exp'])

    X = df_clean[feature_cols].values
    y = df_clean['logK_exp'].values

    # Train model
    model = BSAPartitionModel(model_type='linear')
    model.fit(X, y)

    # Predict
    y_pred = model.predict(X)

    # Calculate metrics
    metrics = model.evaluate(y, y_pred)

    return model, y_pred, metrics, df_clean


def vina_to_logKpw(dG_kcal):
    """
    Convert Vina score to logK using thermodynamic relationship.

    From vina_common.py: Kd_M = exp(dG_kcal / (R * T))
    K_pw = N_SITES / (Kd_M * M_BSA_KG_MOL)
    logK = log10(K_pw)
    """
    R_KCAL = 1.987204e-3  # gas constant, kcal/(mol K)
    TEMP_K = 310.15       # 37 C
    M_BSA_KG_MOL = 66.463 # BSA molar mass
    N_SITES = 1

    Kd_M = math.exp(dG_kcal / (R_KCAL * TEMP_K))
    K_pw = N_SITES / (Kd_M * M_BSA_KG_MOL)
    return math.log10(K_pw)


def main():
    """Main function to train and evaluate BSA partition model."""
    base_dir = Path("/home1/s9383/Albumin")

    # Load data
    albumin_csv = base_dir / "albumin_data.csv"
    vina_focused_csv = base_dir / "vina_focused_sudlow1_results.csv"
    vina_blind_csv = base_dir / "vina_blind_results.csv"
    boltz2_csv = base_dir / "boltz2_results.csv"

    # Load experimental data
    df_exp = pd.read_csv(albumin_csv)
    print(f"Loaded {len(df_exp)} experimental data points")

    # Load Vina focused results
    df_vina_focus = pd.read_csv(vina_focused_csv)
    print(f"Loaded {len(df_vina_focus)} Vina focused results")

    # Load Vina blind results
    df_vina_blind = pd.read_csv(vina_blind_csv)
    print(f"Loaded {len(df_vina_blind)} Vina blind results")

    # Merge data
    df_focus = df_exp.merge(df_vina_focus[['name', 'vina_score', 'logKpw_pred']],
                            on='name', how='inner')
    df_blind = df_exp.merge(df_vina_blind[['name', 'vina_score', 'logKpw_pred']],
                           on='name', how='inner')

    print(f"\nMerged data points:")
    print(f"  Focused: {len(df_focus)}")
    print(f"  Blind: {len(df_blind)}")

    # Train models
    results = {}

    for name, df in [('Focused', df_focus), ('Blind', df_blind)]:
        print(f"\n{'='*60}")
        print(f"Training PP-LFER Model: {name}")
        print(f"{'='*60}")

        model, y_pred, metrics, df_clean = train_pp_lfer_model(df, feature_cols=['vina_score'])

        print(f"\nModel Parameters:")
        print(f"  Slope (a): {model.coef_[0]:.6f}")
        print(f"  Intercept (b): {model.intercept_:.6f}")
        print(f"  Equation: logK = {model.coef_[0]:.6f} * vina_score + {model.intercept_:.6f}")

        print(f"\nPrediction Performance:")
        print(f"  Absolute R²: {metrics['r2']:.4f}")
        print(f"  Calibrated R²: {metrics['r2_cal']:.4f}")
        print(f"  Absolute RMSE: {metrics['rmse']:.4f}")
        print(f"  Calibrated RMSE: {metrics['rmse_cal']:.4f}")
        print(f"  Spearman ρ: {metrics['spearman']:.4f}")
        print(f"  Pearson r: {metrics['pearson']:.4f}")

        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'metrics': metrics,
            'df': df_clean
        }

        # Plot results
        plot_results(df_clean['logK_exp'].values, y_pred, name, metrics, base_dir)

    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY: PP-LFER Model Performance Comparison")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'R²':<10} {'Calibrated R²':<15} {'RMSE':<10}")
    print(f"{'-'*60}")

    for name in ['Focused', 'Blind']:
        m = results[name]['metrics']
        print(f"{name:<20} {m['r2']:<10.4f} {m['r2_cal']:<15.4f} {m['rmse']:<10.4f}")

    # Save predictions
    save_predictions(results, base_dir)


def plot_results(y_true, y_pred, name, metrics, output_dir):
    """Plot experimental vs predicted logK values."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Raw predictions
    ax = axes[0]
    ax.scatter(y_true, y_pred, s=40, c='#1b9e77', edgecolor='w', alpha=0.7)

    # Diagonal line
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'k-', lw=1, label='y = x')

    # Calibrated line
    a, b = metrics['cal_slope'], metrics['cal_intercept']
    x_line = np.array([lo, hi])
    y_cal = a * x_line + b
    ax.plot(x_line, y_cal, 'r--', lw=2, label=f'y = {a:.2f}x + {b:.2f}')

    ax.set_xlabel('Experimental log K$_{BSA/w}$', fontsize=12)
    ax.set_ylabel('Predicted log K$_{BSA/w}$', fontsize=12)
    ax.set_title(f'PP-LFER Model ({name})\nR² = {metrics["r2"]:.3f}, RMSE = {metrics["rmse"]:.3f}', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.25)

    # Plot 2: Residuals
    ax = axes[1]
    residuals = y_pred - y_true
    ax.scatter(y_true, residuals, s=40, c='#2c7fb8', edgecolor='w', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', lw=1)
    ax.set_xlabel('Experimental log K$_{BSA/w}$', fontsize=12)
    ax.set_ylabel('Residuals (pred - exp)', fontsize=12)
    ax.set_title(f'Residual Plot\nMean residual = {np.mean(residuals):.3f}', fontsize=13)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    png_path = output_dir / f"pp_lfer_{name.lower()}_results.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot: {png_path}")
    plt.close()


def save_predictions(results, output_dir):
    """Save prediction results to CSV."""
    for name, res in results.items():
        df = res['df'].copy()
        df['logK_pred_pp_lfer'] = res['y_pred']
        df['residual'] = df['logK_pred_pp_lfer'] - df['logK_exp']

        csv_path = output_dir / f"pp_lfer_{name.lower()}_predictions.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved predictions: {csv_path}")


if __name__ == "__main__":
    from pathlib import Path
    main()
