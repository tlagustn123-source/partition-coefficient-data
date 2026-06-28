#!/usr/bin/env python3
"""
Molecular Property-Based Model for Albumin Partition Coefficient Prediction

This module implements partition coefficient prediction using molecular
properties that can be calculated solely from SMILES:

1. Molar Volume / Parachor Method:
   - McGowan characteristic volume
   - Parachor (surface tension/volume relationship)
   - Molar refractivity

2. Surface Area Method:
   - Topological Polar Surface Area (TPSA)
   - Solvent Accessible Surface Area (SASA) approximation
   - Molecular surface properties

3. Hydrophobic/Polar Balance:
   - Hydrogen bond donor/acceptor counts
   - Rotatable bond count
   - Aromatic ratio

These methods use only SMILES input and RDKit calculations.

References:
- McGowan, J.C. (1985). Characteristic molecular volume.
- Mannhold, R. et al. (1998). Calculation of molecular parachor.
- Ertl, P. et al. (2000). Fast calculation of TPSA.
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


class MolecularPropertyPartitionModel:
    """
    Molecular property-based model for partition coefficient prediction.

    Uses calculated molecular properties from SMILES:
    - Molar volume and parachor
    - Surface area descriptors
    - Hydrophobicity indices
    """

    # Physical constants
    R_J_PER_MOL_K = 8.314462618
    TEMP_K = 310.15  # Body temperature

    # Albumin approximate properties
    M_ALBUMIN = 66500  # g/mol
    V_ALBUMIN_EST = 45000  # cm³/mol

    def __init__(self, model_type='molar-volume'):
        """
        Initialize the molecular property model.

        Args:
            model_type: 'molar-volume', 'parachor', 'surface-area',
                       'hydrophobic-balance', or 'hybrid-property'
        """
        self.model_type = model_type
        self.params_ = {}
        self.fitted = False

    def calculate_mcgowan_volume(self, smiles):
        """
        Calculate McGowan characteristic volume (cm³/mol).

        McGowan volume is based on atomic contributions and accounts for
        the actual molecular volume in solution.

        V_MG = Σ (atomic contributions) - correction for bonds

        Reference: McGowan, J.C. (1985)
        """
        if not RDKIT_AVAILABLE:
            return np.nan

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan

        # Calculate McGowan volume using RDKit
        # This is an approximation using atomic contributions
        mw = Descriptors.MolWt(mol)

        # Atomic volume contributions (simplified)
        # C, H, O, N, halogens typical contributions
        num_atoms = mol.GetNumAtoms()
        num_heavy_atoms = mol.GetNumHeavyAtoms()

        # Bondi-style radii sum approximation
        v_mcgowan = 0.0
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            # Atomic volume contributions (cm³/mol per atom)
            # Based on Bondi radii
            if atomic_num == 1:  # H
                v_mcgowan += 7.24
            elif atomic_num == 6:  # C
                v_mcgowan += 16.35
            elif atomic_num == 7:  # N
                v_mcgowan += 14.39
            elif atomic_num == 8:  # O
                v_mcgowan += 12.43
            elif atomic_num == 9:  # F
                v_mcgowan += 10.93
            elif atomic_num == 17:  # Cl
                v_mcgowan += 19.45
            elif atomic_num == 35:  # Br
                v_mcgowan += 24.52
            elif atomic_num == 53:  # I
                v_mcgowan += 30.15
            else:
                v_mcgowan += 15.0  # Default for other atoms

        # Correction for bonds (simplified)
        num_bonds = mol.GetNumBonds()
        v_mcgowan -= 4.5 * num_bonds * 0.1  # Small correction

        return v_mcgowan

    def calculate_parachor(self, smiles):
        """
        Calculate parachor (surface tension parameter).

        Parachor relates surface tension, density, and molecular weight:
        P = γ^(1/4) * M / ρ

        Where γ is surface tension, M is molecular weight, ρ is density.

        Higher parachor → higher surface activity → different partition behavior.
        """
        if not RDKIT_AVAILABLE:
            return np.nan

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan

        # Calculate molar refractivity as approximation
        # Molar refractivity correlates with parachor
        mr = Crippen.MolMR(mol)

        # Additional contributions
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)

        # Parachor approximation
        # P ≈ MR + correction for surface activity
        parachor = mr * 10 + logp * 20 - tpsa * 0.05

        return parachor

    def calculate_molar_refractivity(self, smiles):
        """Calculate molar refractivity (related to polarizability)."""
        if not RDKIT_AVAILABLE:
            return np.nan

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan

        return Crippen.MolMR(mol)

    def calculate_surface_area_descriptors(self, smiles):
        """
        Calculate surface area related descriptors.

        Returns:
            Dictionary with TPSA, SASA_approx, and polar_fraction
        """
        if not RDKIT_AVAILABLE:
            return {'tpsa': np.nan, 'sasa': np.nan, 'polar_frac': np.nan}

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'tpsa': np.nan, 'sasa': np.nan, 'polar_frac': np.nan}

        # Topological Polar Surface Area
        tpsa = Descriptors.TPSA(mol)

        # Approximate SASA (3D would be more accurate)
        # Using Labute ASA approximation
        try:
            sasa = rdMolDescriptors.CalcLabuteASA(mol)
        except:
            # Fallback to approximation
            sasa = 4 * np.pi * (Descriptors.MolWt(mol) / 100) ** (2/3) * 10

        # Polar fraction (TPSA / Total SA)
        polar_frac = tpsa / sasa if sasa > 0 else 0

        return {'tpsa': tpsa, 'sasa': sasa, 'polar_frac': polar_frac}

    def calculate_hydrophobic_balance(self, smiles):
        """
        Calculate hydrophobic/hydrophilic balance indicators.

        Returns:
            Dictionary with HBD, HBA, rotatable bonds, aromatic ratio
        """
        if not RDKIT_AVAILABLE:
            return {'hbd': np.nan, 'hba': np.nan, 'rot_bonds': np.nan, 'arom_frac': np.nan}

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'hbd': np.nan, 'hba': np.nan, 'rot_bonds': np.nan, 'arom_frac': np.nan}

        # Hydrogen bond donors and acceptors
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)

        # Rotatable bonds
        rot_bonds = Lipinski.NumRotatableBonds(mol)

        # Aromatic atoms fraction
        num_atoms = mol.GetNumAtoms()
        num_aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        arom_frac = num_aromatic / num_atoms if num_atoms > 0 else 0

        return {
            'hbd': hbd,
            'hba': hba,
            'rot_bonds': rot_bonds,
            'arom_frac': arom_frac
        }

    def molar_volume_logK(self, smiles):
        """
        Calculate logK using molar volume-based method.

        Based on:
        logK = a * log(V_m/V_albumin) + b * LogP + c

        Larger molecules have different partition behavior due to
        steric effects and volume exclusion.
        """
        if not RDKIT_AVAILABLE:
            return np.nan

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan

        # Get molecular properties
        v_mcgowan = self.calculate_mcgowan_volume(smiles)
        logp = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)

        if np.isnan(v_mcgowan):
            return np.nan

        # Volume ratio (solute/albumin)
        v_ratio = v_mcgowan / self.V_ALBUMIN_EST

        # Partition coefficient based on volume
        # logK ≈ LogP + correction for size
        logK = logp + 0.5 * np.log10(v_ratio * 1e6)  # Scaled for numerical stability

        # MW correction (steric)
        logK -= 0.001 * mw

        return logK

    def parachor_logK(self, smiles):
        """
        Calculate logK using parachor method.

        Parachor relates to surface activity and partitioning behavior:
        logK = a * parachor + b * LogP + c
        """
        if not RDKIT_AVAILABLE:
            return np.nan

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan

        parachor = self.calculate_parachor(smiles)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)

        if np.isnan(parachor):
            return np.nan

        # Parachor-based partition
        # Higher parachor → higher surface activity
        logK = 0.01 * parachor + 0.6 * logp - 0.02 * tpsa

        return logK

    def surface_area_logK(self, smiles):
        """
        Calculate logK using surface area descriptors.

        Based on:
        - Polar surface area penalty
        - Hydrophobic surface contribution

        logK = a * LogP - b * TPSA + c
        """
        if not RDKIT_AVAILABLE:
            return np.nan

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan

        sa_desc = self.calculate_surface_area_descriptors(smiles)
        logp = Crippen.MolLogP(mol)

        tpsa = sa_desc['tpsa']
        polar_frac = sa_desc['polar_frac']

        if np.isnan(tpsa):
            return np.nan

        # Surface area-based partition
        # Hydrophobic contribution from LogP
        # Penalty for polar surface
        logK = logp - 0.02 * tpsa - 0.5 * polar_frac

        return logK

    def hydrophobic_balance_logK(self, smiles):
        """
        Calculate logK using hydrophobic/hydrophilic balance.

        Considers:
        - H-bonding capacity
        - Molecular flexibility
        - Aromatic content

        logK = a * LogP - b * (HBD + HBA) + c * aromatic_frac + d
        """
        if not RDKIT_AVAILABLE:
            return np.nan

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan

        hb = self.calculate_hydrophobic_balance(smiles)
        logp = Crippen.MolLogP(mol)

        hbd = hb['hbd']
        hba = hb['hba']
        arom_frac = hb['arom_frac']

        # Hydrophobic balance-based partition
        # Penalty for hydrogen bonding
        # Bonus for aromatic (hydrophobic interactions with albumin)
        logK = logp - 0.3 * hbd - 0.2 * hba + 0.5 * arom_frac

        return logK

    def hybrid_property_logK(self, smiles):
        """
        Calculate logK using hybrid molecular property approach.

        Combines multiple property-based methods:
        - Volume contribution
        - Surface area penalty
        - Parachor effect
        - LogP baseline
        """
        logK_mv = self.molar_volume_logK(smiles)
        logK_sa = self.surface_area_logK(smiles)
        logK_pc = self.parachor_logK(smiles)

        if np.isnan(logK_mv):
            return logK_sa if not np.isnan(logK_sa) else np.nan

        if np.isnan(logK_sa):
            return logK_mv

        if np.isnan(logK_pc):
            return 0.5 * logK_mv + 0.5 * logK_sa

        # Weighted combination
        logK = 0.4 * logK_mv + 0.4 * logK_sa + 0.2 * logK_pc

        return logK

    def calculate_logK(self, smiles):
        """Calculate logK based on model type."""
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit required")

        if self.model_type == 'molar-volume':
            return self.molar_volume_logK(smiles)
        elif self.model_type == 'parachor':
            return self.parachor_logK(smiles)
        elif self.model_type == 'surface-area':
            return self.surface_area_logK(smiles)
        elif self.model_type == 'hydrophobic-balance':
            return self.hydrophobic_balance_logK(smiles)
        elif self.model_type == 'hybrid-property':
            return self.hybrid_property_logK(smiles)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, smiles_list, y):
        """
        Fit the molecular property model.

        Applies linear calibration to align predictions with experimental data.
        """
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

        print(f"\n{self.model_type.upper()} Molecular Property Model Results:")
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
            pred = self.calculate_logK(smi)
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
    """Main function to train and evaluate molecular property models."""
    base_dir = Path("/home1/s9383/Albumin")

    if not RDKIT_AVAILABLE:
        print("ERROR: RDKit is required but not available.")
        return

    # Load experimental data
    albumin_csv = base_dir / "albumin_data.csv"
    df = load_albumin_data(albumin_csv)

    print(f"\n{'='*70}")
    print("MOLECULAR PROPERTY-BASED MODEL FOR ALBUMIN PARTITION COEFFICIENT")
    print(f"{'='*70}")
    print("\nMolecular Property Models:")
    print("  Molar Volume:      Volume exclusion and steric effects")
    print("  Parachor:          Surface tension and surface activity")
    print("  Surface Area:      TPSA and polar surface effects")
    print("  Hydrophobic Bal:   H-bonding and aromatic content")
    print("  Hybrid-Property:   Combined molecular property approach")

    # Test different model types
    model_types = ['molar-volume', 'parachor', 'surface-area',
                   'hydrophobic-balance', 'hybrid-property']
    results = {}

    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*70}")

        model = MolecularPropertyPartitionModel(model_type=model_type)

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
    print("SUMMARY: MOLECULAR PROPERTY MODEL COMPARISON")
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

        output_csv = base_dir / f"molecular_property_{best_model_type}_predictions.csv"
        df_pred.to_csv(output_csv, index=False)
        print(f"\nSaved predictions to: {output_csv}")

    return results


if __name__ == "__main__":
    results = main()
