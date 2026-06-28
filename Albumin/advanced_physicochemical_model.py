#!/usr/bin/env python3
"""
Advanced Physicochemical Model for Albumin Partition Coefficient Prediction

This module implements additional physicochemical methods:

1. Fractional Unbound (fu) Method:
   - Protein binding calculation based on hydrophobicity
   - Uses fu = 1 / (1 + 10^(LogP - C))
   - Related to drug-plasma protein binding

2. Molar Refractivity-Based Method:
   - Polarizability and electronic effects
   - Correlates with molecular volume and refraction index

3. Dielectric Constant Method:
   - Electronic and dielectric properties
   - Dipole moment and permittivity effects

4. Combined Advanced Model:
   - Hybrid of multiple advanced physicochemical approaches

References:
- (Literature on plasma protein binding prediction)
- Abraham, M.H. et al. (1990s)
- (Molar refractivity correlations)
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
    from rdkit.Chem import GraphDescriptors, Lipinski
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Some features will be limited.")


class AdvancedPhysicochemicalModel:
    """
    Advanced physicochemical model for partition coefficient prediction.

    Methods:
    - Fractional unbound (fu) based prediction
    - Molar refractivity based
    - Dielectric/electronic property based
    - Hybrid advanced approach
    """

    # Physical constants
    R_J_PER_MOL_K = 8.314462618
    TEMP_K = 310.15  # Body temperature

    # Albumin properties
    M_ALBUMIN = 66500  # g/mol
    V_ALBUMIN_EST = 45000  # cm³/mol

    def __init__(self, model_type='fractional-unbound'):
        """
        Initialize the advanced physicochemical model.

        Args:
            model_type: 'fractional-unbound', 'molar-refractivity',
                       'dielectric', or 'hybrid-advanced'
        """
        self.model_type = model_type
        self.params_ = {}
        self.fitted = False

    def estimate_pka(self, smiles):
        """
        Estimate pKa using structural indicators.

        This is a simplified approximation. Accurate pKa requires
        specialized software or experimental data.

        Returns estimated acidic and basic pKa values.
        """
        if not RDKIT_AVAILABLE:
            return {'acidic_pka': np.nan, 'basic_pka': np.nan}

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'acidic_pka': np.nan, 'basic_pka': np.nan}

        # Count acidic/basic groups
        carboxyl = sum(1 for atom in mol.GetAtoms()
                      if atom.GetAtomicNum() == 8 and atom.GetTotalNumHs() == 0)
        # Simplified acidic group detection
        acidic_count = 0
        basic_count = 0

        for atom in mol.GetAtoms():
            # Carboxylic acid group
            if atom.GetAtomicNum() == 6:  # Carbon
                neighbors = [n for n in atom.GetNeighbors()]
                oxygens = sum(1 for n in neighbors if n.GetAtomicNum() == 8)
                if oxygens >= 2 and atom.GetTotalNumHs() == 0:
                    acidic_count += 1

            # Amino group
            if atom.GetAtomicNum() == 7:  # Nitrogen
                if atom.GetTotalNumHs() > 0:
                    basic_count += 1

        # Typical pKa values
        acidic_pka = 4.5 if acidic_count > 0 else np.nan
        basic_pka = 9.5 if basic_count > 0 else np.nan

        return {'acidic_pka': acidic_pka, 'basic_pka': basic_pka, 'acidic_count': acidic_count, 'basic_count': basic_count}

    def fractional_unbound_logK(self, smiles, ph=7.4):
        """
        Calculate logK using fractional unbound approach.

        Based on plasma protein binding correlation:
        fu ≈ 1 / (1 + 10^(a*LogP + b))

        For albumin binding:
        logK = -log10(fu) + correction

        At pH 7.4 (physiological)
        """
        if not RDKIT_AVAILABLE:
            return np.nan

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan

        # Get basic properties
        logp = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)

        # Estimate pKa
        pka_data = self.estimate_pka(smiles)
        acidic_pka = pka_data['acidic_pka']
        basic_pka = pka_data['basic_pka']

        # Fractional unbound approximation
        # Based on correlation: fu ~ 1/(1 + 10^(LogP - offset))
        # For albumin binding (empirical)

        # pH effect (if ionizable)
        ionization_correction = 0.0
        if not np.isnan(acidic_pka):
            # Acidic compound: fraction ionized at pH
            f_ionized = 1 / (1 + 10**(acidic_pka - ph))
            ionization_correction = -0.5 * f_ionized  # Ionized form binds less

        if not np.isnan(basic_pka):
            # Basic compound
            f_ionized = 1 / (1 + 10**(ph - basic_pka))
            ionization_correction = -0.3 * f_ionized

        # Fractional unbound (empirical correlation)
        # fu = 1 / (1 + 10^(0.7*LogP - 1.5))
        fu = 1 / (1 + 10**(0.7 * logp - 1.5))

        # Convert fu to logK
        # logK = -log10(fu) + corrections
        logK = -np.log10(fu) + ionization_correction

        # Size correction
        logK += 0.001 * mw

        # Polar surface penalty
        logK -= 0.02 * tpsa

        return logK

    def molar_refractivity_logK(self, smiles):
        """
        Calculate logK using molar refractivity.

        Molar refractivity (MR) relates to:
        - Electronic polarizability
        - Molecular volume
        - Refractive index

        MR correlates with partition coefficient through
        dispersion forces and polarizability effects.
        """
        if not RDKIT_AVAILABLE:
            return np.nan

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan

        # Molar refractivity
        mr = Crippen.MolMR(mol)
        logp = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)

        # Refractivity-based partition
        # Higher MR → higher polarizability → different partition behavior
        logK = 0.15 * mr + 0.4 * logp

        # Size correction
        logK += 0.0005 * mw

        return logK

    def dielectric_property_logK(self, smiles):
        """
        Calculate logK using dielectric/electronic properties.

        Based on:
        - Dipole moment (approximated from structure)
        - Electronic effects
        - Permittivity contribution
        """
        if not RDKIT_AVAILABLE:
            return np.nan

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan

        # Get properties
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)

        # Dipole moment approximation (from TPSA)
        # Higher polar surface → higher dipole → lower partition into hydrophobic phase
        dipole_approx = tpsa / 10  # Rough approximation

        # Dielectric-based partition
        # Penalty for high dipole/polar molecules
        logK = logp - 0.1 * dipole_approx - 0.015 * tpsa

        return logK

    def polarizability_logK(self, smiles):
        """
        Calculate logK using polarizability.

        Molecular polarizability affects:
        - Van der Waals interactions with albumin
        - Dispersion force contributions
        """
        if not RDKIT_AVAILABLE:
            return np.nan

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan

        # Get properties
        logp = Crippen.MolLogP(mol)
        mr = Crippen.MolMR(mol)
        tpsa = Descriptors.TPSA(mol)
        num_heavy = rdMolDescriptors.CalcNumHeavyAtoms(mol)

        # Polarizability correlates with MR and size
        polarizability = mr / num_heavy if num_heavy > 0 else mr

        # Polarizability-based partition
        # Higher polarizability → stronger dispersion interactions with albumin
        logK = logp + 0.05 * mr - 0.02 * tpsa

        return logK

    def hybrid_advanced_logK(self, smiles, ph=7.4):
        """
        Calculate logK using hybrid advanced physicochemical approach.

        Combines:
        - Fractional unbound method
        - Molar refractivity
        - Dielectric properties
        """
        logK_fu = self.fractional_unbound_logK(smiles, ph)
        logK_mr = self.molar_refractivity_logK(smiles)
        logK_di = self.dielectric_property_logK(smiles)

        if np.isnan(logK_fu):
            return logK_mr if not np.isnan(logK_mr) else np.nan

        if np.isnan(logK_mr):
            return logK_fu

        if np.isnan(logK_di):
            return 0.5 * logK_fu + 0.5 * logK_mr

        # Weighted combination
        logK = 0.4 * logK_fu + 0.35 * logK_mr + 0.25 * logK_di

        return logK

    def calculate_logK(self, smiles, ph=7.4):
        """Calculate logK based on model type."""
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit required")

        if self.model_type == 'fractional-unbound':
            return self.fractional_unbound_logK(smiles, ph)
        elif self.model_type == 'molar-refractivity':
            return self.molar_refractivity_logK(smiles)
        elif self.model_type == 'dielectric':
            return self.dielectric_property_logK(smiles)
        elif self.model_type == 'polarizability':
            return self.polarizability_logK(smiles)
        elif self.model_type == 'hybrid-advanced':
            return self.hybrid_advanced_logK(smiles, ph)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, smiles_list, y):
        """Fit the advanced physicochemical model."""
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit required")

        # Calculate theoretical predictions
        y_pred_theory = []
        valid_y = []
        for smi, y_val in zip(smiles_list, y):
            pred = self.calculate_logK(smi)
            if not np.isnan(pred):
                y_pred_theory.append(pred)
                valid_y.append(y_val)

        if len(y_pred_theory) == 0:
            print("No valid predictions could be calculated")
            return

        y_pred = np.array(y_pred_theory)
        y_valid = np.array(valid_y)

        # Apply linear calibration
        a, b = np.polyfit(y_pred, y_valid, 1)

        self.params_['slope'] = a
        self.params_['intercept'] = b
        self.fitted = True

        # Calculate metrics
        y_cal = a * y_pred + b
        r2 = 1 - np.sum((y_valid - y_cal)**2) / np.sum((y_valid - np.mean(y_valid))**2)
        rmse = np.sqrt(np.mean((y_valid - y_cal)**2))

        from scipy.stats import spearmanr, pearsonr
        spearman = spearmanr(y_valid, y_cal).correlation
        pearson = pearsonr(y_valid, y_cal)[0]

        print(f"\n{self.model_type.upper()} Advanced Physicochemical Model Results:")
        print(f"  Valid samples: {len(y_valid)}")
        print(f"  Calibration: logK_exp = {a:.4f} × logK_theory + {b:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Spearman ρ: {spearman:.4f}")
        print(f"  Pearson r: {pearson:.4f}")

        return {'r2': r2, 'rmse': rmse, 'spearman': spearman, 'pearson': pearson}

    def predict(self, smiles, ph=7.4):
        """Predict logK for a given SMILES."""
        single = isinstance(smiles, str)
        if single:
            smiles = [smiles]

        preds = []
        for smi in smiles:
            pred = self.calculate_logK(smi, ph)
            if self.fitted and not np.isnan(pred):
                pred = self.params_['slope'] * pred + self.params_['intercept']
            preds.append(pred)

        return preds[0] if single else preds

    def evaluate(self, smiles_list, y_true):
        """Evaluate model performance."""
        y_pred = self.predict(smiles_list)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) < 2:
            return {'r2': np.nan, 'rmse': np.nan, 'spearman': np.nan, 'pearson': np.nan}

        from scipy.stats import spearmanr, pearsonr

        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        spearman = spearmanr(y_true, y_pred).correlation
        pearson = pearsonr(y_true, y_pred)[0]

        return {
            'r2': r2,
            'rmse': rmse,
            'spearman': spearman,
            'pearson': pearson
        }


def load_albumin_data(csv_path):
    """Load albumin experimental data."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} compounds from {csv_path}")
    return df


def main():
    """Main function to train and evaluate advanced physicochemical models."""
    base_dir = Path("/home1/s9383/Albumin")

    if not RDKIT_AVAILABLE:
        print("ERROR: RDKit is required but not available.")
        return

    # Load experimental data
    albumin_csv = base_dir / "albumin_data.csv"
    df = load_albumin_data(albumin_csv)

    print(f"\n{'='*70}")
    print("ADVANCED PHYSICOCHEMICAL MODEL FOR ALBUMIN PARTITION COEFFICIENT")
    print(f"{'='*70}")
    print("\nAdvanced Physicochemical Models:")
    print("  Fractional Unbound:  Protein binding (fu) based prediction")
    print("  Molar Refractivity:  Polarizability and electronic effects")
    print("  Dielectric:         Electronic and dielectric properties")
    print("  Polarizability:     Molecular polarizability based")
    print("  Hybrid-Advanced:     Combined advanced approach")

    # Test different model types
    model_types = ['fractional-unbound', 'molar-refractivity', 'dielectric',
                   'polarizability', 'hybrid-advanced']
    results = {}

    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*70}")

        model = AdvancedPhysicochemicalModel(model_type=model_type)

        # Fit model
        train_metrics = model.fit(df['smiles'].tolist(), df['logK_exp'].tolist())

        if train_metrics is None:
            continue

        # Evaluate
        eval_metrics = model.evaluate(df['smiles'].tolist(), df['logK_exp'].tolist())

        results[model_type] = {
            'model': model,
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics
        }

    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY: ADVANCED PHYSICOCHEMICAL MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'R²':<10} {'RMSE':<10} {'Spearman':<10}")
    print(f"{'-'*70}")

    for model_type in model_types:
        if model_type in results:
            m = results[model_type]['eval_metrics']
            print(f"{model_type:<25} {m['r2']:<10.4f} {m['rmse']:<10.4f} {m['spearman']:<10.4f}")

    # Save best model predictions
    if results:
        best_model_type = max(results.keys(),
                             key=lambda k: results[k]['eval_metrics']['r2'])
        best_model = results[best_model_type]['model']

        print(f"\nBest Model: {best_model_type.upper()}")
        print(f"R²: {results[best_model_type]['eval_metrics']['r2']:.4f}")

        # Generate predictions
        df_pred = df.copy()
        df_pred['logK_pred'] = best_model.predict(df['smiles'].tolist())
        df_pred['residual'] = df_pred['logK_pred'] - df_pred['logK_exp']

        output_csv = base_dir / f"advanced_physicochem_{best_model_type}_predictions.csv"
        df_pred.to_csv(output_csv, index=False)
        print(f"\nSaved predictions to: {output_csv}")

    return results


if __name__ == "__main__":
    results = main()
