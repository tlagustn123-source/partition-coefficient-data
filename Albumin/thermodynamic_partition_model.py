#!/usr/bin/env python3
"""
Thermodynamic Equation-Based Model for Albumin Partition Coefficient Prediction

This module implements partition coefficient prediction using fundamental
thermodynamic equations and theories:

1. Flory-Huggins Solution Theory:
   - Accounts for molecular size differences and mixing entropy
   - Activity coefficient: ln γ₂ = ln(1 - φ₂) + (1 - 1/m)φ₂ + χφ₂²
   - χ parameter: interaction parameter between solute and solvent

2. Regular Solution Theory:
   - Based on Hildebrand solubility parameters
   - Activity coefficient: ln γ₂ = V₂(δ₁ - δ₂)²/RT
   - δ = sqrt((ΔH_vap - RT)/V_mol): solubility parameter

3. Thermodynamic Cycle:
   - K = exp(-ΔG_transfer/RT)
   - ΔG_transfer = ΔG_solv(albumin) - ΔG_solv(water)

Reference:
- Flory, P.J. (1942). Thermodynamics of High Polymer Solutions.
- Hildebrand, J.H. (1950). Regular Solutions.
- Prausnitz, J.M. (1999). Molecular Thermodynamics of Fluid-Phase Equilibria.
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


class ThermodynamicPartitionModel:
    """
    Thermodynamic equation-based model for partition coefficient prediction.

    Uses fundamental thermodynamic relationships:
    - Flory-Huggins theory for polymer solutions
    - Regular solution theory with solubility parameters
    - Thermodynamic cycle for transfer free energy
    """

    # Physical constants
    R_J_PER_MOL_K = 8.314462618  # Gas constant, J/(mol·K)
    R_CAL_PER_MOL_K = 1.987204e-3  # Gas constant, kcal/(mol·K)
    TEMP_K = 310.15  # Body temperature: 37°C = 310.15 K

    # Solvent properties (water and octanol reference values)
    # Water solubility parameters (cal/cm³)^0.5
    DELTA_WATER = 47.8  # Hildebrand parameter for water
    # Octanol solubility parameters (cal/cm³)^0.5
    DELTA_OCTANOL = 21.1  # Hildebrand parameter for octanol

    # Albumin properties (approximate)
    # Albumin is a large protein (MW ~66.5 kDa)
    # Approximated as a polymer solution
    M_ALBUMIN = 66500  # Molecular weight of BSA (g/mol)
    V_ALBUMIN_EST = 45000  # Approximate molar volume (cm³/mol)

    def __init__(self, model_type='regular-solution'):
        """
        Initialize the thermodynamic model.

        Args:
            model_type: 'regular-solution', 'flory-huggins', 'hybrid-thermo',
                       or 'delta-G-transfer'
        """
        self.model_type = model_type
        self.params_ = {}
        self.fitted = False

    def calculate_molecular_volume(self, smiles):
        """
        Calculate molar volume using group contribution or RDKit estimation.

        For organic molecules, approximate V_m ≈ MW / density
        Typical organic liquid density ≈ 1 g/cm³
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit required for volume calculation")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan

        # Calculate molecular weight
        mw = Descriptors.MolWt(mol)

        # Estimate molar volume using McGowan characteristic volume
        # This is an approximation; actual values vary by compound class
        # For liquids: V_m ≈ MW / ρ, where ρ ≈ 0.8-1.2 g/cm³ for organics

        # Use atom-based volume estimation (simplified)
        # Sum of atomic volumes (characteristic volume approach)
        v_mcgowan = rdMolDescriptors.CalcExactMolWt(mol)  # Using MW as proxy

        # Typical molar volume for organic compounds (cm³/mol)
        # Empirical relation: V_m ≈ 0.7 * MW^(0.9) (for typical organics)
        v_mol = 0.7 * (mw ** 0.9) * 100  # Convert to approximate cm³/mol

        return v_mol

    def calculate_logp_thermodynamic(self, smiles):
        """
        Calculate LogP using thermodynamic principles.

        From regular solution theory:
        ln K_ow = V_solute * (δ_water - δ_solute)² / (RT) -
                  V_solute * (δ_octanol - δ_solute)² / (RT)

        Simplified: LogP ≈ f(δ_solute, V_solute)
        """
        if not RDKIT_AVAILABLE:
            return np.nan

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan

        # Calculate solute solubility parameter (approximation)
        # δ_solute ≈ function of polarity, H-bonding, dispersion
        # Using RDKit descriptors as proxy

        logp_rippen = Crippen.MolLogP(mol)  # Experimental correlation
        tpsa = Descriptors.TPSA(mol)

        # Convert LogP to solubility parameter space (inverse problem)
        # This is a theoretical mapping, not direct calculation
        # Higher LogP → Lower δ (more hydrophobic)

        # Approximate δ from LogP (empirical mapping)
        # δ ≈ δ_water - (logP * factor)
        delta_solute = self.DELTA_WATER - (logp_rippen * 5.0)

        return logp_rippen, delta_solute

    def regular_solution_theory_activity_coefficient(self, v_solute, delta_solute, delta_solvent):
        """
        Calculate activity coefficient using Regular Solution Theory.

        ln γ₂ = V₂(δ₁ - δ₂)² / (RT)

        Where:
        - V₂: molar volume of solute
        - δ₁: solubility parameter of solvent
        - δ₂: solubility parameter of solute
        - R: gas constant
        - T: temperature
        """
        if np.isnan(v_solute) or np.isnan(delta_solute):
            return np.nan

        # Convert from cal to J if needed, or work in cal units
        # Using: R = 1.987 cal/(mol·K)
        r_cal = self.R_CAL_PER_MOL_K * 1000  # Convert to cal/(mol·K)
        temp = self.TEMP_K

        delta_diff = delta_solvent - delta_solute
        ln_gamma = (v_solute * delta_diff**2) / (r_cal * temp)

        return ln_gamma

    def flory_huggins_activity_coefficient(self, phi_solute, m_ratio, chi):
        """
        Calculate activity coefficient using Flory-Huggins theory.

        ln γ₂ = ln(1 - φ₂) + (1 - 1/m)φ₂ + χφ₂²

        Where:
        - φ₂: volume fraction of solute
        - m: ratio of solvent to solute molecular volumes (V_solvent/V_solute)
        - χ: Flory-Huggins interaction parameter

        For dilute solutions (φ₂ << 1):
        ln γ₂ ≈ (1 - 1/m + χ)φ₂ - φ₂²/2
        """
        if np.isnan(phi_solute) or np.isnan(chi):
            return np.nan

        # Prevent numerical issues
        phi = min(phi_solute, 0.99)

        ln_gamma = np.log(1 - phi) + (1 - 1/m_ratio) * phi + chi * phi**2

        return ln_gamma

    def calculate_chi_parameter(self, delta_solute, delta_solvent, v_solute, v_solvent):
        """
        Calculate Flory-Huggins χ parameter from solubility parameters.

        χ = V_solute(δ_solvent - δ_solute)² / (RT)

        Related to enthalpy of mixing.
        """
        r_cal = self.R_CAL_PER_MOL_K * 1000  # cal/(mol·K)
        temp = self.TEMP_K

        delta_diff = delta_solvent - delta_solute
        chi = (v_solute * delta_diff**2) / (r_cal * temp)

        return chi

    def calculate_delta_G_transfer(self, smiles):
        """
        Calculate transfer free energy from water to albumin phase.

        ΔG_transfer = ΔG_albumin - ΔG_water

        Using approximation:
        ΔG_solv ≈ -RT * ln(γ) (from activity coefficient)

        For albumin binding, we approximate using:
        - Hydrophobic effect (related to LogP)
        - Steric factors (molecular size)
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit required")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan

        # Calculate descriptors
        logp = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)

        # Convert to thermodynamic quantities
        # ΔG_transfer (cal/mol) ≈ -RT * ln(K_partition)
        # For albumin: approximation based on hydrophobicity

        # Base contribution from LogP (hydrophobic effect)
        # ΔG ≈ -2.303 * RT * LogP
        delta_g_base = -2.303 * self.R_J_PER_MOL_K * self.TEMP_K * logp  # J/mol

        # Steric/size correction (larger molecules have different entropy)
        delta_g_steric = 0.01 * mw * 100  # Small entropic penalty for size

        # Polar surface area penalty (albumin has polar patches)
        delta_g_polar = 0.05 * tpsa  # Penalty for polar groups

        delta_g_transfer = delta_g_base - delta_g_steric + delta_g_polar

        return delta_g_transfer / 1000  # Convert to kcal/mol

    def calculate_logK_thermodynamic(self, smiles):
        """
        Calculate partition coefficient using thermodynamic equations.

        Method depends on model_type.
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit required")

        if self.model_type == 'regular-solution':
            return self._regular_solution_logK(smiles)
        elif self.model_type == 'flory-huggins':
            return self._flory_huggins_logK(smiles)
        elif self.model_type == 'delta-G-transfer':
            return self._deltaG_transfer_logK(smiles)
        else:
            return self._hybrid_thermo_logK(smiles)

    def _regular_solution_logK(self, smiles):
        """Calculate LogK using Regular Solution Theory."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan

        # Get molecular properties
        v_solute = self.calculate_molecular_volume(smiles)
        logp_rippen, delta_solute = self.calculate_logp_thermodynamic(smiles)

        if np.isnan(v_solute) or np.isnan(delta_solute):
            return np.nan

        # Activity coefficient in water
        ln_gamma_water = self.regular_solution_theory_activity_coefficient(
            v_solute, delta_solute, self.DELTA_WATER
        )

        # Activity coefficient in octanol (reference)
        ln_gamma_octanol = self.regular_solution_theory_activity_coefficient(
            v_solute, delta_solute, self.DELTA_OCTANOL
        )

        # Partition coefficient: K_ow = γ_water / γ_octanol
        # logK_ow = (ln γ_water - ln γ_octanol) / ln(10)

        if np.isnan(ln_gamma_water) or np.isnan(ln_gamma_octanol):
            # Fall back to experimental correlation if calculation fails
            return logp_rippen

        logK = (ln_gamma_water - ln_gamma_octanol) / np.log(10)

        # Convert to albumin-water partition coefficient
        # Albumin is more hydrophobic than octanol
        # Empirical correction based on literature
        logK_albumin = 0.85 * logK + 0.5

        return logK_albumin

    def _flory_huggins_logK(self, smiles):
        """Calculate LogK using Flory-Huggins theory."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan

        # Molecular properties
        v_solute = self.calculate_molecular_volume(smiles)
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)

        if np.isnan(v_solute):
            return np.nan

        # Volume fraction (dilute solution approximation)
        # For albumin (polymer): m = V_albumin / V_solute
        m_ratio = self.V_ALBUMIN_EST / v_solute

        # χ parameter (interaction parameter)
        _, delta_solute = self.calculate_logp_thermodynamic(smiles)
        chi = self.calculate_chi_parameter(delta_solute, self.DELTA_WATER, v_solute, self.V_ALBUMIN_EST)

        # Activity coefficient in albumin phase
        # For dilute: φ ≈ concentration * V / RT
        phi_approx = 0.01  # Dilute approximation

        ln_gamma = self.flory_huggins_activity_coefficient(
            phi_approx, m_ratio, chi
        )

        # Convert to partition coefficient
        # logK ≈ -ln(γ) / ln(10) (simplified)
        logK = -ln_gamma / np.log(10)

        # Apply LogP-based scaling for better prediction
        logK = 0.7 * logp + 0.3 * logK

        return logK

    def _deltaG_transfer_logK(self, smiles):
        """Calculate LogK using transfer free energy."""
        delta_g = self.calculate_delta_G_transfer(smiles)  # kcal/mol

        if np.isnan(delta_g):
            return np.nan

        # K = exp(-ΔG/RT)
        # logK = -ΔG / (RT * ln(10))

        rt_kcal = self.R_CAL_PER_MOL_K * self.TEMP_K
        logK = -delta_g / (rt_kcal * np.log(10))

        return logK

    def _hybrid_thermo_logK(self, smiles):
        """Calculate LogK using hybrid thermodynamic approach."""
        # Combine multiple thermodynamic contributions
        logK_reg = self._regular_solution_logK(smiles)
        logK_dg = self._deltaG_transfer_logK(smiles)

        if np.isnan(logK_reg) or np.isnan(logK_dg):
            return logK_reg if not np.isnan(logK_reg) else logK_dg

        # Weighted average
        logK = 0.6 * logK_reg + 0.4 * logK_dg

        return logK

    def fit(self, smiles_list, y):
        """
        Fit the thermodynamic model.

        For thermodynamic models, we may apply scaling/correction
        to align theoretical predictions with experimental data.

        Args:
            smiles_list: List of SMILES strings
            y: Experimental logK values
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit required")

        # Calculate theoretical predictions
        y_pred_theory = []
        valid_y = []
        for smi, y_val in zip(smiles_list, y):
            pred = self.calculate_logK_thermodynamic(smi)
            if not np.isnan(pred):
                y_pred_theory.append(pred)
                valid_y.append(y_val)

        if len(y_pred_theory) == 0:
            print("No valid predictions could be calculated")
            return

        y_pred = np.array(y_pred_theory)
        y_valid = np.array(valid_y)

        # Apply linear calibration: y_cal = a * y_theory + b
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

        print(f"\n{self.model_type.upper()} Thermodynamic Model Results:")
        print(f"  Valid samples: {len(y_valid)}")
        print(f"  Calibration: logK_exp = {a:.4f} × logK_theory + {b:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Spearman ρ: {spearman:.4f}")
        print(f"  Pearson r: {pearson:.4f}")

        return {'r2': r2, 'rmse': rmse, 'spearman': spearman, 'pearson': pearson}

    def predict(self, smiles):
        """Predict logK for a given SMILES."""
        single = isinstance(smiles, str)
        if single:
            smiles = [smiles]

        preds = []
        for smi in smiles:
            pred = self.calculate_logK_thermodynamic(smi)
            if self.fitted and not np.isnan(pred):
                # Apply calibration
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

        # Calculate metrics
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
            'r2_cal': r2,  # Already calibrated in predict()
            'rmse_cal': rmse,
            'spearman': spearman,
            'pearson': pearson
        }


def load_albumin_data(csv_path):
    """Load albumin experimental data."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} compounds from {csv_path}")
    return df


def main():
    """Main function to train and evaluate thermodynamic models."""
    base_dir = Path("/home1/s9383/Albumin")

    if not RDKIT_AVAILABLE:
        print("ERROR: RDKit is required but not available.")
        return

    # Load experimental data
    albumin_csv = base_dir / "albumin_data.csv"
    df = load_albumin_data(albumin_csv)

    print(f"\n{'='*70}")
    print("THERMODYNAMIC EQUATION-BASED MODEL FOR ALBUMIN PARTITION COEFFICIENT")
    print(f"{'='*70}")
    print("\nThermodynamic Models:")
    print("  Regular Solution:  Activity coefficient from solubility parameters")
    print("  Flory-Huggins:     Polymer solution theory with χ parameter")
    print("  ΔG Transfer:       Transfer free energy calculation")
    print("  Hybrid-Thermo:     Combined thermodynamic approach")

    # Test different model types
    model_types = ['regular-solution', 'flory-huggins', 'delta-G-transfer', 'hybrid-thermo']
    results = {}

    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*70}")

        model = ThermodynamicPartitionModel(model_type=model_type)

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
    print("SUMMARY: THERMODYNAMIC MODEL COMPARISON")
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

        output_csv = base_dir / f"thermodynamic_{best_model_type}_predictions.csv"
        df_pred.to_csv(output_csv, index=False)
        print(f"\nSaved predictions to: {output_csv}")

    return results


if __name__ == "__main__":
    results = main()
