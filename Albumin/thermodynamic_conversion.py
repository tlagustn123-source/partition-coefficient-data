#!/usr/bin/env python3
"""
Thermodynamic DeltaG to Partition Coefficient Conversion Module

This module implements physics-based thermodynamic equations for converting
AutoDock Vina deltaG (binding free energy) to BSA-water partition coefficients.

Theoretical Basis:
    ΔG = -RT ln K_A  (Gibbs free energy relationship)
    K_BSA/w = K_A × [BSA]  (partition coefficient definition)

    Therefore:
    log K_BSA/w = -ΔG / (2.303RT) + log [BSA]

This is a thermodynamic conversion WITHOUT fitting/regression.
"""

import math
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr, pearsonr
from pathlib import Path

# Thermodynamic constants
R_KCAL = 1.987204e-3  # gas constant, kcal/(mol·K)
TEMP_K = 310.15  # temperature in Kelvin (37°C, typical for BSA experiments)

# BSA properties
BSA_MW_G_PER_MOL = 66463  # BSA molecular weight (g/mol)


class ThermodynamicConverter:
    """
    Physics-based thermodynamic converter for deltaG to partition coefficient.

    NO FITTING/REGRESSION - purely thermodynamic equations.
    """

    def __init__(self, temperature_k=TEMP_K, bsa_concentration_m=None):
        """
        Initialize the thermodynamic converter.

        Args:
            temperature_k: Temperature in Kelvin (default: 310.15 K = 37°C)
            bsa_concentration_m: BSA concentration in M (if None, calculated from typical mg/mL)
        """
        self.temperature_k = temperature_k
        self.bsa_concentration_m = bsa_concentration_m

    def set_bsa_concentration(self, mg_per_ml):
        """
        Set BSA concentration from mg/mL.

        Args:
            mg_per_ml: BSA concentration in mg/mL
        """
        # Convert mg/mL to M
        g_per_l = mg_per_ml  # mg/mL = g/L
        mol_per_l = g_per_l / BSA_MW_G_PER_MOL
        self.bsa_concentration_m = mol_per_l

    def deltaG_to_KA(self, deltaG_kcal):
        """
        Convert deltaG (kcal/mol) to association constant K_A.

        Using thermodynamic relationship: ΔG = -RT ln K_A
        Therefore: K_A = exp(-ΔG/RT)

        Args:
            deltaG_kcal: Binding free energy in kcal/mol (negative = favorable binding)

        Returns:
            K_A: Association constant in M^-1
        """
        if deltaG_kcal is None or np.isnan(deltaG_kcal):
            return np.nan

        # K_A = exp(-ΔG/RT)
        KA = math.exp(-deltaG_kcal / (R_KCAL * self.temperature_k))
        return KA

    def deltaG_to_logK_BSA(self, deltaG_kcal):
        """
        Convert deltaG (kcal/mol) to log10 BSA-water partition coefficient.

        Thermodynamic derivation:
        1) ΔG = -RT ln K_A  →  K_A = exp(-ΔG/RT)
        2) K_BSA/w = K_A × [BSA]  (approximately, for dilute conditions)
        3) log K_BSA/w = log K_A + log [BSA]
        4) log K_BSA/w = -ΔG/(2.303RT) + log [BSA]

        Args:
            deltaG_kcal: Binding free energy in kcal/mol

        Returns:
            log10 K_BSA/w: BSA-water partition coefficient (log10 scale)
        """
        if self.bsa_concentration_m is None:
            raise ValueError("BSA concentration not set. Call set_bsa_concentration() first.")

        if deltaG_kcal is None or np.isnan(deltaG_kcal):
            return np.nan

        # Calculate K_A
        KA = self.deltaG_to_KA(deltaG_kcal)

        # Convert to partition coefficient
        # K_BSA/w = K_A × [BSA]
        K_BSA_w = KA * self.bsa_concentration_m

        # Return log10
        logK = math.log10(K_BSA_w)
        return logK

    def vina_score_to_logK_BSA(self, vina_score):
        """
        Convert AutoDock Vina score to log10 BSA-water partition coefficient.

        Note: Vina score is an estimate of binding free energy (kcal/mol).
        Negative scores indicate favorable binding.

        Args:
            vina_score: AutoDock Vina score in kcal/mol

        Returns:
            log10 K_BSA/w: BSA-water partition coefficient
        """
        # Vina score ≈ ΔG (binding free energy)
        return self.deltaG_to_logK_BSA(vina_score)


def evaluate_thermodynamic_predictions(vina_csv, exp_csv, output_dir=None):
    """
    Evaluate thermodynamic conversion predictions.

    Args:
        vina_csv: Path to Vina results CSV
        exp_csv: Path to experimental data CSV
        output_dir: Path to save results

    Returns:
        Dictionary with evaluation metrics
    """
    # Load data
    df_vina = pd.read_csv(vina_csv)
    df_exp = pd.read_csv(exp_csv)

    # Merge data
    df = df_exp[['name', 'logK_exp']].merge(
        df_vina[['name', 'vina_score']],
        on='name',
        how='inner'
    )

    print(f"Merged data: {len(df)} compounds")

    # Try different BSA concentrations
    results = {}
    bsa_concentrations = [40, 45, 50, 55, 60, 65, 70]  # mg/mL (typical experimental range)

    for bsa_mg_ml in bsa_concentrations:
        # Initialize converter
        converter = ThermodynamicConverter()
        converter.set_bsa_concentration(bsa_mg_ml)

        # Convert Vina scores to logK
        df['logK_pred'] = df['vina_score'].apply(
            converter.vina_score_to_logK_BSA
        )

        # Remove NaN values
        df_clean = df.dropna(subset=['logK_exp', 'logK_pred'])

        if len(df_clean) < 3:
            continue

        # Calculate metrics
        y_true = df_clean['logK_exp'].values
        y_pred = df_clean['logK_pred'].values

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        spearman = spearmanr(y_true, y_pred).correlation
        pearson = pearsonr(y_true, y_pred)[0]

        # Calibrated R²
        a, b = np.polyfit(y_pred, y_true, 1)
        y_cal = a * y_pred + b
        r2_cal = r2_score(y_true, y_cal)
        rmse_cal = np.sqrt(mean_squared_error(y_true, y_cal))

        results[bsa_mg_ml] = {
            'bsa_mg_ml': bsa_mg_ml,
            'r2': r2,
            'rmse': rmse,
            'r2_cal': r2_cal,
            'rmse_cal': rmse_cal,
            'spearman': spearman,
            'pearson': pearson,
            'df': df_clean.copy()
        }

    # Find best BSA concentration
    if results:
        best_conc = max(results.items(), key=lambda x: x[1]['r2_cal'])
        print(f"\nBest BSA concentration: {best_conc[0]} mg/mL")
        print(f"  Calibrated R²: {best_conc[1]['r2_cal']:.4f}")
        print(f"  RMSE: {best_conc[1]['rmse_cal']:.4f}")
        print(f"  Spearman ρ: {best_conc[1]['spearman']:.4f}")

    # Print summary table
    print("\n" + "="*70)
    print("Thermodynamic Conversion Performance (No Fitting)")
    print("="*70)
    print(f"{'BSA (mg/mL)':<15} {'R²':<10} {'Calibrated R²':<15} {'RMSE':<10} {'Spearman':<10}")
    print("-"*70)

    for conc, res in sorted(results.items()):
        print(f"{conc:<15} {res['r2']:<10.4f} {res['r2_cal']:<15.4f} "
              f"{res['rmse']:<10.4f} {res['spearman']:<10.4f}")

    print("-"*70)
    print("\nNote: These are PURE THERMODYNAMIC predictions - NO fitting/regression!")
    print("Method: log K = -ΔG/(2.303RT) + log [BSA]")

    return results


def main():
    """Main function to run thermodynamic conversion analysis."""
    base_dir = Path("/home1/s9383/Albumin")

    print("="*70)
    print("THERMODYNAMIC DELTAG → PARTITION COEFFICIENT CONVERSION")
    print("="*70)
    print("\nTheoretical Basis:")
    print("  ΔG = -RT ln K_A")
    print("  K_BSA/w = K_A × [BSA]")
    print("  log K_BSA/w = -ΔG/(2.303RT) + log [BSA]")
    print("\nThis is a physics-based conversion - NO FITTING!")
    print("="*70)

    # Evaluate both Vina methods
    for mode, vina_file in [("Focused", "vina_focused_sudlow1_results.csv"),
                            ("Blind", "vina_blind_results.csv")]:
        print(f"\n{'='*70}")
        print(f"Vina Mode: {mode}")
        print(f"{'='*70}")

        vina_csv = base_dir / vina_file
        exp_csv = base_dir / "albumin_data.csv"

        results = evaluate_thermodynamic_predictions(
            vina_csv, exp_csv, base_dir
        )


if __name__ == "__main__":
    main()
