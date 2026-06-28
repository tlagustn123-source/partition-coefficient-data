#!/usr/bin/env python3
"""
Physicochemical LFER Model for Albumin Partition Coefficient Prediction

This module implements a 1-parameter Linear Free Energy Relationship (1p-LFER)
model for predicting albumin partition coefficients using only physicochemical
parameters calculated from SMILES.

Model form: logK_albumin/w = a × LogP + b (1p-LFER)
         logK_albumin/w = a × LogP + b × MW + c (2p-LFER)

This approach:
- Uses only physicochemical parameters (no docking, no QM calculations)
- Based on Linear Free Energy Relationship theory
- SMILES-only calculation (no experimental data required for prediction

Reference:
- Boiteux et al. (2022) Rapid determination of serum albumin partition
  coefficients using affinity chromatography
- LFER fundamentals for protein-water partition systems
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path

# Try importing RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Some features will be limited.")


class PhysicochemicalLFERModel:
    """
    1p-LFER and 2p-LFER models for albumin partition coefficient prediction.

    The model uses only physicochemical parameters calculated from SMILES:
    - LogP: Octanol-water partition coefficient (hydrophobicity)
    - MW: Molecular weight
    - TPSA: Topological polar surface area
    - HBD, HBA: Hydrogen bond donors/acceptors
    """

    def __init__(self, model_type='1p-lfer'):
        """
        Initialize the LFER model.

        Args:
            model_type: '1p-lfer' (LogP only), '2p-lfer' (LogP + MW),
                       or 'multi-lfer' (multiple descriptors)
        """
        self.model_type = model_type
        self.coef_ = None
        self.intercept_ = None
        self.feature_names = []
        self.fitted = False

    def calculate_logp(self, smiles):
        """Calculate LogP using Wildman-Crippen method."""
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit required for LogP calculation")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        return Crippen.MolLogP(mol)

    def calculate_mw(self, smiles):
        """Calculate molecular weight."""
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit required for MW calculation")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        return Descriptors.MolWt(mol)

    def calculate_tpsa(self, smiles):
        """Calculate topological polar surface area."""
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit required for TPSA calculation")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        return Descriptors.TPSA(mol)

    def calculate_hbd(self, smiles):
        """Calculate number of hydrogen bond donors."""
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit required for HBD calculation")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        return rdMolDescriptors.CalcNumHBD(mol)

    def calculate_hba(self, smiles):
        """Calculate number of hydrogen bond acceptors."""
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit required for HBA calculation")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        return rdMolDescriptors.CalcNumHBA(mol)

    def calculate_descriptors(self, smiles):
        """
        Calculate all physicochemical descriptors from SMILES.

        Returns:
            Dictionary with descriptor values
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit required for descriptor calculation")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                'LogP': np.nan, 'MW': np.nan, 'TPSA': np.nan,
                'HBD': np.nan, 'HBA': np.nan, 'HeavyAtoms': np.nan,
                'RotBonds': np.nan
            }

        return {
            'LogP': Crippen.MolLogP(mol),
            'MW': Descriptors.MolWt(mol),
            'TPSA': Descriptors.TPSA(mol),
            'HBD': rdMolDescriptors.CalcNumHBD(mol),
            'HBA': rdMolDescriptors.CalcNumHBA(mol),
            'HeavyAtoms': mol.GetNumHeavyAtoms(),
            'RotBonds': rdMolDescriptors.CalcNumRotatableBonds(mol)
        }

    def _get_feature_columns(self):
        """Get feature columns based on model type."""
        if self.model_type == '1p-lfer':
            return ['LogP']
        elif self.model_type == '2p-lfer':
            return ['LogP', 'MW']
        elif self.model_type == 'polar-lfer':
            return ['LogP', 'TPSA']
        elif self.model_type == 'hbond-lfer':
            return ['LogP', 'HBD', 'HBA']
        else:  # multi-lfer
            return ['LogP', 'MW', 'TPSA', 'HBD', 'HBA']

    def fit(self, smiles_list, y):
        """
        Fit the LFER model using physicochemical descriptors.

        Args:
            smiles_list: List of SMILES strings
            y: Experimental logK values
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit required for model fitting")

        # Calculate descriptors
        desc_list = []
        valid_y = []
        for smi, y_val in zip(smiles_list, y):
            desc = self.calculate_descriptors(smi)
            if not np.isnan(desc['LogP']):  # Basic validity check
                desc_list.append(desc)
                valid_y.append(y_val)

        # Create feature matrix
        self.feature_names = self._get_feature_columns()
        X = np.array([[d[feat] for feat in self.feature_names] for d in desc_list])
        y_valid = np.array(valid_y)

        # Remove any remaining NaN values
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y_valid = y_valid[mask]

        # Linear regression (closed-form solution)
        X_with_bias = np.column_stack([X, np.ones(X.shape[0])])

        # Calculate coefficients using least squares
        coeffs, _, _, _ = np.linalg.lstsq(X_with_bias, y_valid, rcond=None)

        self.coef_ = coeffs[:-1]
        self.intercept_ = coeffs[-1]
        self.fitted = True

        # Calculate training metrics
        y_pred = X @ self.coef_ + self.intercept_
        r2 = 1 - np.sum((y_valid - y_pred)**2) / np.sum((y_valid - np.mean(y_valid))**2)
        rmse = np.sqrt(np.mean((y_valid - y_pred)**2))

        print(f"\n{self.model_type.upper()} Model Results:")
        print(f"  Training samples: {len(y_valid)}")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Equation: logK = ", end="")
        for i, (coef, name) in enumerate(zip(self.coef_, self.feature_names)):
            sign = " + " if coef >= 0 else " - "
            print(f"{sign}{abs(coef):.4f}×{name}", end="")
        print(f" + {self.intercept_:.4f}")

        return {'r2': r2, 'rmse': rmse, 'n_samples': len(y_valid)}

    def predict(self, smiles):
        """
        Predict logK for a given SMILES.

        Args:
            smiles: SMILES string or list of SMILES

        Returns:
            Predicted logK value(s)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        single = isinstance(smiles, str)
        if single:
            smiles = [smiles]

        preds = []
        for smi in smiles:
            desc = self.calculate_descriptors(smi)
            X = np.array([[desc[feat] for feat in self.feature_names]])
            if np.isnan(X).any():
                preds.append(np.nan)
            else:
                pred = X @ self.coef_ + self.intercept_
                preds.append(pred[0])

        return preds[0] if single else preds

    def evaluate(self, smiles_list, y_true):
        """
        Evaluate model performance.

        Returns:
            Dictionary with R², RMSE, Spearman, Pearson, Calibrated R²
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before evaluation")

        y_pred = self.predict(smiles_list)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        # Calculate metrics
        from scipy.stats import spearmanr, pearsonr

        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        spearman = spearmanr(y_true, y_pred).correlation
        pearson = pearsonr(y_true, y_pred)[0]

        # Calibrated R² (linear fit between pred and exp)
        if len(y_true) > 2:
            a, b = np.polyfit(y_pred, y_true, 1)
            y_cal = a * y_pred + b
            ss_res_cal = np.sum((y_true - y_cal)**2)
            r2_cal = 1 - ss_res_cal / ss_tot if ss_tot > 0 else np.nan
            rmse_cal = np.sqrt(np.mean((y_true - y_cal)**2))
        else:
            a, b = np.nan, np.nan
            r2_cal, rmse_cal = np.nan, np.nan

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


def load_albumin_data(csv_path):
    """Load albumin experimental data."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} compounds from {csv_path}")
    return df


def main():
    """Main function to train and evaluate physicochemical LFER models."""
    base_dir = Path("/home1/s9383/Albumin")

    if not RDKIT_AVAILABLE:
        print("ERROR: RDKit is required but not available.")
        print("Please install RDKit using conda: conda install -c conda-forge rdkit")
        return

    # Load experimental data
    albumin_csv = base_dir / "albumin_data.csv"
    df = load_albumin_data(albumin_csv)

    print(f"\n{'='*70}")
    print("PHYSICOCHEMICAL LFER MODEL FOR ALBUMIN PARTITION COEFFICIENT")
    print(f"{'='*70}")
    print("\nModel Types:")
    print("  1p-LFER:    logK = a × LogP + b")
    print("  2p-LFER:    logK = a × LogP + b × MW + c")
    print("  Polar-LFER: logK = a × LogP + b × TPSA + c")
    print("  HBond-LFER: logK = a × LogP + b × HBD + c × HBA + d")
    print("  Multi-LFER:  logK = a × LogP + b × MW + c × TPSA + d × HBD + e × HBA + f")

    # Test different model types
    model_types = ['1p-lfer', '2p-lfer', 'polar-lfer', 'hbond-lfer', 'multi-lfer']
    results = {}

    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*70}")

        model = PhysicochemicalLFERModel(model_type=model_type)

        # Fit model
        train_metrics = model.fit(df['smiles'].tolist(), df['logK_exp'].tolist())

        # Evaluate
        eval_metrics = model.evaluate(df['smiles'].tolist(), df['logK_exp'].tolist())

        print(f"\nEvaluation Metrics:")
        print(f"  R²: {eval_metrics['r2']:.4f}")
        print(f"  Calibrated R²: {eval_metrics['r2_cal']:.4f}")
        print(f"  RMSE: {eval_metrics['rmse']:.4f}")
        print(f"  Calibrated RMSE: {eval_metrics['rmse_cal']:.4f}")
        print(f"  Spearman ρ: {eval_metrics['spearman']:.4f}")
        print(f"  Pearson r: {eval_metrics['pearson']:.4f}")

        results[model_type] = {
            'model': model,
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics
        }

    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY: PHYSICOCHEMICAL LFER MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'R²':<10} {'Cal R²':<10} {'RMSE':<10} {'Spearman':<10}")
    print(f"{'-'*70}")

    for model_type in model_types:
        m = results[model_type]['eval_metrics']
        print(f"{model_type:<20} {m['r2']:<10.4f} {m['r2_cal']:<10.4f} {m['rmse']:<10.4f} {m['spearman']:<10.4f}")

    # Save predictions for best model
    best_model_type = max(results.keys(),
                         key=lambda k: results[k]['eval_metrics']['r2_cal'])
    best_model = results[best_model_type]['model']

    print(f"\nBest Model: {best_model_type.upper()}")
    print(f"Calibrated R²: {results[best_model_type]['eval_metrics']['r2_cal']:.4f}")

    # Generate predictions
    df_pred = df.copy()
    df_pred['logK_pred'] = best_model.predict(df['smiles'].tolist())
    df_pred['residual'] = df_pred['logK_pred'] - df_pred['logK_exp']

    output_csv = base_dir / f"physicochemical_{best_model_type}_predictions.csv"
    df_pred.to_csv(output_csv, index=False)
    print(f"\nSaved predictions to: {output_csv}")

    return results, best_model_type


if __name__ == "__main__":
    results, best_model = main()
