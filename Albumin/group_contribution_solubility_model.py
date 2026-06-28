#!/usr/bin/env/python3
"""
Group Contribution & Solubility Parameter Methods for Albumin Partition Coefficient

This module implements multiple physicochemical approaches based on group contribution
and solubility parameter theories:

1. UNIFAC (UNIQUAC Functional-group Activity Coefficients)
   - Group contribution method for activity coefficients
   - γ_i = γ_i^C (combinatorial) + γ_i^R (residual)

2. Hansen Solubility Parameters
   - δD: Dispersion forces
   - δP: Polar interactions
   - δH: Hydrogen bonding
   - RED (Relative Energy Difference) based prediction

3. Hildebrand Solubility Parameter
   - Single parameter regular solution theory
   - δ = sqrt(δD² + δP² + δH²)

References:
- Fredenslund et al. (1975). UNIFAC group-contribution method.
- Hansen, C.M. (2007). Hansen Solubility Parameters.
- Hildebrand, J.H. (1950). Regular Solutions.
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Try importing RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors, GraphDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available.")


# Hansen Solubility Parameters (Group Contribution Values)
# Data from Hansen (2007) and Van Krevelen (2009)
# Units: (MPa)⁰·⁵ or (J/cm³)⁰·⁵, converted here to (cal/cm³)⁰·⁵
# Conversion: 1 (J/cm³)⁰·⁵ ≈ 0.4889 (cal/cm³)⁰·⁵

HANSEN_GROUP_CONTRIBUTIONS = {
    # Alkyl groups (CH3, CH2, CH, C)
    'CH3': {'deltaD': 16.8, 'deltaP': 0.0, 'deltaH': 0.0, 'volume': 33.5},
    'CH2': {'deltaD': 16.8, 'deltaP': 0.0, 'deltaH': 0.0, 'volume': 16.1},
    'CH': {'deltaD': 16.8, 'deltaP': 0.0, 'deltaH': 0.0, 'volume': 7.0},
    'C':  {'deltaD': 16.8, 'deltaP': 0.0, 'deltaH': 0.0, 'volume': -5.0},

    # Aromatic groups
    'ACH': {'deltaD': 18.0, 'deltaP': 1.0, 'deltaH': 1.0, 'volume': 12.0},  # Aromatic CH
    'AC':  {'deltaD': 18.0, 'deltaP': 1.0, 'deltaH': 1.0, 'volume': 5.0},   # Aromatic C (substituted)

    # Halogenated groups
    'CH2Cl': {'deltaD': 18.2, 'deltaP': 5.8, 'deltaH': 3.5, 'volume': 45.0},
    'CHCl':  {'deltaD': 18.2, 'deltaP': 5.8, 'deltaH': 3.5, 'volume': 35.0},
    'CCl':   {'deltaD': 18.2, 'deltaP': 5.8, 'deltaH': 3.5, 'volume': 25.0},
    'CH2F':  {'deltaD': 12.5, 'deltaP': 8.5, 'deltaH': 4.2, 'volume': 30.0},
    'CF':    {'deltaD': 12.5, 'deltaP': 8.5, 'deltaH': 4.2, 'volume': 20.0},
    'CH2Br': {'deltaD': 19.5, 'deltaP': 6.2, 'deltaH': 3.8, 'volume': 50.0},
    'Br':    {'deltaD': 19.5, 'deltaP': 6.2, 'deltaH': 3.8, 'volume': 30.0},
    'I':     {'deltaD': 22.0, 'deltaP': 5.5, 'deltaH': 2.5, 'volume': 35.0},

    # Oxygen-containing groups
    'OH':   {'deltaD': 17.0, 'deltaP': 8.0, 'deltaH': 18.0, 'volume': 12.0},  # Alcohol OH
    'OH_phe': {'deltaD': 17.0, 'deltaP': 8.0, 'deltaH': 18.0, 'volume': 12.0},  # Phenol OH
    'CH3CO': {'deltaD': 16.5, 'deltaP': 8.5, 'deltaH': 6.5, 'volume': 40.0},  # Acetate CH3
    'CH2CO': {'deltaD': 16.5, 'deltaP': 8.5, 'deltaH': 6.5, 'volume': 28.0},  # Ester CH2
    'COO':   {'deltaD': 16.5, 'deltaP': 8.5, 'deltaH': 6.5, 'volume': 18.0},  # Ester C=O
    'CHO':   {'deltaD': 17.0, 'deltaP': 10.0, 'deltaH': 8.0, 'volume': 22.0},  # Aldehyde
    'COOH':  {'deltaD': 17.5, 'deltaP': 12.0, 'deltaH': 16.0, 'volume': 28.0},  # Carboxylic acid
    'COOC':  {'deltaD': 16.5, 'deltaP': 8.5, 'deltaH': 6.5, 'volume': 25.0},  # Ester linkage
    'O':     {'deltaD': 15.5, 'deltaP': 7.5, 'deltaH': 6.0, 'volume': 8.0},    # Ether O

    # Nitrogen-containing groups
    'NH2':  {'deltaD': 17.5, 'deltaP': 10.0, 'deltaH': 12.0, 'volume': 19.0},  # Primary amine
    'NH':   {'deltaD': 17.5, 'deltaP': 10.0, 'deltaH': 12.0, 'volume': 12.0},  # Secondary amine
    'N':    {'deltaD': 17.5, 'deltaP': 10.0, 'deltaH': 12.0, 'volume': 5.0},   # Tertiary amine
    'CN':   {'deltaD': 18.0, 'deltaP': 12.0, 'deltaH': 5.0, 'volume': 25.0},  # Nitrile
    'NO2':  {'deltaD': 18.5, 'deltaP': 15.0, 'deltaH': 6.0, 'volume': 30.0},  # Nitro

    # Sulfur-containing groups
    'SH':   {'deltaD': 18.5, 'deltaP': 5.5, 'deltaH': 8.0, 'volume': 28.0},  # Thiol
    'S':    {'deltaD': 18.5, 'deltaP': 5.5, 'deltaH': 8.0, 'volume': 15.0},  # Sulfide

    # Special groups
    'Cl':   {'deltaD': 18.2, 'deltaP': 5.8, 'deltaH': 3.5, 'volume': 22.0},  # Aromatic Cl
    'F':    {'deltaD': 12.5, 'deltaP': 8.5, 'deltaH': 4.2, 'volume': 10.0},  # Aromatic F
    'Br_aro':{'deltaD': 19.5, 'deltaP': 6.2, 'deltaH': 3.8, 'volume': 28.0},  # Aromatic Br

    # Heteroaromatic
    'pyridine_N': {'deltaD': 18.0, 'deltaP': 10.0, 'deltaH': 8.0, 'volume': 12.0},
    'furan_O':   {'deltaD': 17.0, 'deltaP': 8.0, 'deltaH': 6.0, 'volume': 10.0},
    'thiophene_S': {'deltaD': 18.5, 'deltaP': 5.5, 'deltaH': 8.0, 'volume': 15.0},
}


class GroupContributionModel:
    """
    Group Contribution Method for calculating physicochemical properties
    and partition coefficients.
    """

    def __init__(self, model_type='hansen'):
        """
        Initialize the model.

        Args:
            model_type: 'hansen', 'hildebrand', 'unifac', or 'combined'
        """
        self.model_type = model_type
        self.params_ = {}
        self.fitted = False

        # Solvent reference values (albumin approximation)
        # Albumin as a "solvent" - approximate solubility parameters
        # Based on protein characteristics
        self.ALBUMIN_HANSEN = {
            'deltaD': 18.0,  # Approximate for proteins
            'deltaP': 12.0,  # High polarity
            'deltaH': 15.0,  # H-bonding capability
        }

        self.WATER_HANSEN = {
            'deltaD': 15.5,
            'deltaP': 16.0,
            'deltaH': 42.3,
        }

        self.OCTANOL_HANSEN = {
            'deltaD': 16.0,
            'deltaP': 5.0,
            'deltaH': 15.0,
        }

    def identify_groups(self, smiles):
        """
        Identify functional groups in molecule for group contribution.

        Returns:
            Dictionary of group counts
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit required")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        groups = defaultdict(int)

        # Get atom and bond information
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        rings = mol.GetRingInfo()

        # Count atoms by element
        atom_counts = defaultdict(int)
        for atom in atoms:
            atom_counts[atom.GetSymbol()] += 1

        # Aromaticity check
        aromatic_atoms = set()
        for atom in atoms:
            if atom.GetIsAromatic():
                aromatic_atoms.add(atom.GetIdx())

        # Basic group identification
        for atom in atoms:
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            degree = atom.GetDegree()
            num_h = atom.GetTotalNumHs()

            # Carbon-based groups
            if symbol == 'C':
                # Aliphatic carbons
                if idx not in aromatic_atoms:
                    if degree == 4 and num_h == 3:
                        groups['CH3'] += 1
                    elif degree == 4 and num_h == 2:
                        groups['CH2'] += 1
                    elif degree == 4 and num_h == 1:
                        groups['CH'] += 1
                    elif degree == 4 and num_h == 0:
                        groups['C'] += 1

                    # Oxygen-containing
                    neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
                    if 'O' in neighbors:
                        if degree == 3:  # Carbonyl carbon
                            if any(n.GetSymbol() == 'O' and n.GetDegree() == 1 for n in atom.GetNeighbors()):
                                if 'C' in neighbors and neighbors.count('C') == 1:
                                    groups['CH3CO'] += 1  # Methyl ketone
                                else:
                                    groups['COO'] += 1   # Ester carbonyl

                # Aromatic carbons
                else:
                    if num_h >= 1:
                        groups['ACH'] += 1
                    else:
                        groups['AC'] += 1

            # Oxygen-containing groups
            elif symbol == 'O':
                degree = atom.GetDegree()
                num_h = atom.GetTotalNumHs()

                if num_h == 1:  # OH group
                    # Check if attached to aromatic ring
                    carbon_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']
                    if carbon_neighbors and carbon_neighbors[0].GetIsAromatic():
                        groups['OH_phe'] += 1
                    else:
                        groups['OH'] += 1

                elif degree == 2:  # Ether or carbonyl oxygen
                    # Check if double-bonded to carbon (carbonyl)
                    for bond in atom.GetBonds():
                        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                            groups['COO'] += 1  # Carbonyl O already counted
                            break
                    else:
                        groups['O'] += 1  # Ether oxygen

            # Nitrogen-containing groups
            elif symbol == 'N':
                degree = atom.GetDegree()
                num_h = atom.GetTotalNumHs()

                if idx in aromatic_atoms:
                    groups['pyridine_N'] += 1
                else:
                    if degree == 1:
                        groups['NH2'] += 1
                    elif degree == 2 and num_h == 1:
                        groups['NH'] += 1
                    elif degree == 3:
                        groups['N'] += 1

                    # Check for nitro
                    for bond in atom.GetBonds():
                        if bond.GetBondType() in [Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.AROMATIC]:
                            neighbor = bond.GetOtherAtom(atom)
                            if neighbor.GetSymbol() == 'O':
                                groups['NO2'] += 1
                                break

            # Halogens
            elif symbol == 'Cl':
                if idx in aromatic_atoms:
                    groups['Cl'] += 1  # Aromatic Cl
                else:
                    # Check if part of CH2Cl, CHCl, or CCl
                    carbon_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']
                    if carbon_neighbors:
                        c_atom = carbon_neighbors[0]
                        h_count = c_atom.GetTotalNumHs()
                        cl_count = sum(1 for n in c_atom.GetNeighbors() if n.GetSymbol() == 'Cl')
                        if h_count == 2 and cl_count == 1:
                            groups['CH2Cl'] += 1
                        elif h_count == 1 and cl_count == 1:
                            groups['CHCl'] += 1
                        elif h_count == 0 and cl_count >= 1:
                            groups['CCl'] += 1
                        else:
                            groups['CH2Cl'] += 1  # Default
                    else:
                        groups['CH2Cl'] += 1

            elif symbol == 'F':
                if idx in aromatic_atoms:
                    groups['F'] += 1  # Aromatic F
                else:
                    # Similar to Cl
                    carbon_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']
                    if carbon_neighbors:
                        c_atom = carbon_neighbors[0]
                        h_count = c_atom.GetTotalNumHs()
                        f_count = sum(1 for n in c_atom.GetNeighbors() if n.GetSymbol() == 'F')
                        if h_count == 2 and f_count == 1:
                            groups['CH2F'] += 1
                        elif h_count == 0 and f_count >= 1:
                            groups['CF'] += 1
                        else:
                            groups['CH2F'] += 1
                    else:
                        groups['CH2F'] += 1

            elif symbol == 'Br':
                if idx in aromatic_atoms:
                    groups['Br_aro'] += 1
                else:
                    carbon_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']
                    if carbon_neighbors:
                        c_atom = carbon_neighbors[0]
                        h_count = c_atom.GetTotalNumHs()
                        if h_count == 2:
                            groups['CH2Br'] += 1
                        else:
                            groups['Br'] += 1
                    else:
                        groups['CH2Br'] += 1

            elif symbol == 'I':
                groups['I'] += 1

            # Sulfur
            elif symbol == 'S':
                degree = atom.GetDegree()
                num_h = atom.GetTotalNumHs()

                if num_h == 1:
                    groups['SH'] += 1
                elif degree == 2:
                    groups['S'] += 1
                elif idx in aromatic_atoms:
                    groups['thiophene_S'] += 1

        # Adjust for overcounting in nitro groups
        if 'NO2' in groups:
            nitro_count = sum(1 for atom in atoms if atom.GetSymbol() == 'N' and
                             any(b.GetBondType() in [Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.AROMATIC]
                                 for b in atom.GetBonds() if b.GetOtherAtom(atom).GetSymbol() == 'O'))
            groups['NO2'] = nitro_count // 2  # NO2 has 1 N and 2 O, counted per N

        return dict(groups)

    def calculate_hansen_parameters(self, smiles):
        """
        Calculate Hansen Solubility Parameters using group contribution.

        Returns:
            Dictionary with deltaD, deltaP, deltaH, and total delta
        """
        groups = self.identify_groups(smiles)

        if not groups:
            return {'deltaD': np.nan, 'deltaP': np.nan, 'deltaH': np.nan, 'delta': np.nan, 'volume': np.nan}

        total_volume = 0.0
        weighted_d = 0.0
        weighted_p = 0.0
        weighted_h = 0.0

        for group, count in groups.items():
            if group in HANSEN_GROUP_CONTRIBUTIONS:
                contrib = HANSEN_GROUP_CONTRIBUTIONS[group]
                vol = contrib['volume'] * count
                total_volume += vol
                weighted_d += contrib['deltaD'] * vol
                weighted_p += contrib['deltaP'] * vol
                weighted_h += contrib['deltaH'] * vol

        if total_volume > 0:
            deltaD = weighted_d / total_volume
            deltaP = weighted_p / total_volume
            deltaH = weighted_h / total_volume
            delta = math.sqrt(deltaD**2 + deltaP**2 + deltaH**2)
        else:
            deltaD = deltaP = deltaH = delta = np.nan

        return {
            'deltaD': deltaD,
            'deltaP': deltaP,
            'deltaH': deltaH,
            'delta': delta,
            'volume': total_volume
        }

    def calculate_hildebrand_parameter(self, smiles):
        """Calculate Hildebrand solubility parameter (single parameter)."""
        hansen = self.calculate_hansen_parameters(smiles)
        return hansen['delta']  # Hildebrand = sqrt(δD² + δP² + δH²)

    def calculate_red(self, smiles, solvent_hansen):
        """
        Calculate Relative Energy Difference (RED) using Hansen parameters.

        RED = Ra / Ro

        Where Ra = sqrt(4(δD1-δD2)² + (δP1-δP2)² + (δH1-δH2)²)
        Ro = interaction radius (approximate)

        Lower RED = better miscibility/solubility
        """
        solute = self.calculate_hansen_parameters(smiles)

        if np.isnan(solute['deltaD']):
            return np.nan

        # Calculate Ra (energy distance)
        deltaD_diff = solute['deltaD'] - solvent_hansen['deltaD']
        deltaP_diff = solute['deltaP'] - solvent_hansen['deltaP']
        deltaH_diff = solute['deltaH'] - solvent_hansen['deltaH']

        Ra = math.sqrt(4 * deltaD_diff**2 + deltaP_diff**2 + deltaH_diff**2)

        # Ro (interaction radius) - approximate for protein-ligand systems
        # Typically 10-30 for organic compounds
        Ro = 20.0  # Approximate value

        RED = Ra / Ro

        return RED

    def unifac_activity_coefficient(self, smiles, solvent='water'):
        """
        Calculate activity coefficient using UNIFAC group contribution method.

        Simplified implementation focusing on residual contribution.

        ln γ_i^R = Σ_k ν_k^i (ln Γ_k - ln Γ_k^i)

        Where:
        - ν_k^i: number of groups k in molecule i
        - Γ_k: group activity coefficient in mixture
        - Γ_k^i: group activity coefficient in pure component i

        This is a simplified version for demonstration.
        """
        groups = self.identify_groups(smiles)

        if not groups:
            return np.nan

        # Get group interaction parameters (simplified)
        # In full UNIFAC, this uses a_mn and a_nm interaction parameters

        # Solvent reference values (group affinity)
        if solvent == 'water':
            solvent_affinity = {'OH': 1.0, 'O': 1.0, 'N': 0.8, 'default': 0.3}
        elif solvent == 'octanol':
            solvent_affinity = {'OH': 0.8, 'O': 0.9, 'CH3': 0.95, 'CH2': 0.95, 'default': 0.7}
        elif solvent == 'albumin':
            # Albumin has diverse interaction sites
            solvent_affinity = {'OH': 0.9, 'O': 0.85, 'N': 0.9, 'CH3': 0.5, 'CH2': 0.5, 'default': 0.6}
        else:
            solvent_affinity = {'default': 0.5}

        # Calculate simplified residual activity coefficient
        ln_gamma_r = 0.0
        total_groups = sum(groups.values())

        for group, count in groups.items():
            affinity = solvent_affinity.get(group, solvent_affinity['default'])
            # Simplified group contribution
            ln_gamma_r += count * (1.0 - affinity) / total_groups

        # Combinatorial part (size/shape contribution)
        # Simplified as function of molecular volume
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            volume = Descriptors.MolWt(mol)  # Using MW as volume proxy
            # ln γ^C ≈ ln(φ_i/φ_j) + 1 - φ_i/φ_j (simplified)
            ln_gamma_c = math.log(volume / 100.0) if volume > 0 else 0
        else:
            ln_gamma_c = 0

        ln_gamma = ln_gamma_c + ln_gamma_r

        return math.exp(ln_gamma)

    def calculate_logK_hansen(self, smiles):
        """Calculate logK using Hansen solubility parameters."""
        # Calculate RED for water and albumin
        red_water = self.calculate_red(smiles, self.WATER_HANSEN)
        red_albumin = self.calculate_red(smiles, self.ALBUMIN_HANSEN)

        if np.isnan(red_water) or np.isnan(red_albumin):
            return np.nan

        # Lower RED = better solubility in that phase
        # Partition to phase with lower RED
        # logK ∝ RED_water - RED_albumin

        # Scale factor for conversion to logK
        logK = 3.0 * (red_water - red_albumin)

        # Base adjustment
        logK += 0.5

        return logK

    def calculate_logK_hildebrand(self, smiles):
        """Calculate logK using Hildebrand solubility parameter (regular solution theory)."""
        delta_solute = self.calculate_hildebrand_parameter(smiles)

        if np.isnan(delta_solute):
            return np.nan

        # Hildebrand parameters (cal/cm³)⁰·⁵
        delta_water = 23.4  # Hildebrand parameter for water
        delta_albumin = 15.0  # Approximate for albumin (more hydrophobic than water)

        # Regular solution theory: ln γ = V(δ_solute - δ_solvent)² / RT
        # For partition: K = γ_water / γ_albumin

        # Get molecular volume (approximation)
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan

        mw = Descriptors.MolWt(mol)
        v_mol = 0.7 * (mw ** 0.9) * 100  # cm³/mol (approximation)

        # RT in cal/mol at 310 K
        rt = 1.987 * 310  # cal/mol

        # Activity coefficients
        ln_gamma_water = v_mol * (delta_solute - delta_water)**2 / rt
        ln_gamma_albumin = v_mol * (delta_solute - delta_albumin)**2 / rt

        # Partition coefficient
        ln_K = ln_gamma_water - ln_gamma_albumin
        logK = ln_K / math.log(10)

        return logK

    def calculate_logK_unifac(self, smiles):
        """Calculate logK using UNIFAC activity coefficients."""
        gamma_water = self.unifac_activity_coefficient(smiles, 'water')
        gamma_albumin = self.unifac_activity_coefficient(smiles, 'albumin')

        if np.isnan(gamma_water) or np.isnan(gamma_albumin):
            return np.nan

        # Partition coefficient: K = γ_water / γ_albumin
        # But we need to invert for affinity
        K = gamma_water / gamma_albumin
        logK = math.log10(K)

        return logK

    def calculate_logK(self, smiles):
        """Calculate logK based on model type."""
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit required")

        if self.model_type == 'hansen':
            return self.calculate_logK_hansen(smiles)
        elif self.model_type == 'hildebrand':
            return self.calculate_logK_hildebrand(smiles)
        elif self.model_type == 'unifac':
            return self.calculate_logK_unifac(smiles)
        elif self.model_type == 'combined':
            # Combine all three methods
            logK_hansen = self.calculate_logK_hansen(smiles)
            logK_hildebrand = self.calculate_logK_hildebrand(smiles)
            logK_unifac = self.calculate_logK_unifac(smiles)

            vals = [v for v in [logK_hansen, logK_hildebrand, logK_unifac] if not np.isnan(v)]
            if vals:
                return np.mean(vals)
            return np.nan
        else:
            return self.calculate_logK_hansen(smiles)

    def fit(self, smiles_list, y):
        """Fit the model (apply calibration to theoretical predictions)."""
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
            print("No valid predictions")
            return None

        y_pred = np.array(y_pred_theory)
        y_valid = np.array(valid_y)

        # Linear calibration
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

        print(f"\n{self.model_type.upper()} Model Results:")
        print(f"  Valid samples: {len(y_valid)}")
        print(f"  Calibration: logK_exp = {a:.4f} × logK_theory + {b:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Spearman ρ: {spearman:.4f}")
        print(f"  Pearson r: {pearson:.4f}")

        return {'r2': r2, 'rmse': rmse, 'spearman': spearman, 'pearson': pearson}

    def predict(self, smiles):
        """Predict logK for given SMILES."""
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
            'r2_cal': r2,
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
    """Main function to test all group contribution and solubility methods."""
    base_dir = Path("/home1/s9383/Albumin")

    if not RDKIT_AVAILABLE:
        print("ERROR: RDKit required")
        return

    albumin_csv = base_dir / "albumin_data.csv"
    df = load_albumin_data(albumin_csv)

    print(f"\n{'='*70}")
    print("GROUP CONTRIBUTION & SOLUBILITY PARAMETER METHODS")
    print(f"{'='*70}")
    print("\nMethods:")
    print("  Hansen:        Hansen Solubility Parameters (δD, δP, δH) + RED")
    print("  Hildebrand:    Hildebrand Parameter (δ) - Regular Solution Theory")
    print("  UNIFAC:        UNIFAC Activity Coefficients")
    print("  Combined:      Average of all three methods")

    model_types = ['hansen', 'hildebrand', 'unifac', 'combined']
    results = {}

    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*70}")

        model = GroupContributionModel(model_type=model_type)
        train_metrics = model.fit(df['smiles'].tolist(), df['logK_exp'].tolist())

        if train_metrics is None:
            continue

        eval_metrics = model.evaluate(df['smiles'].tolist(), df['logK_exp'].tolist())

        results[model_type] = {
            'model': model,
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics
        }

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: GROUP CONTRIBUTION & SOLUBILITY PARAMETER METHODS")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'R²':<10} {'RMSE':<10} {'Spearman':<10}")
    print(f"{'-'*70}")

    for model_type in model_types:
        if model_type in results:
            m = results[model_type]['eval_metrics']
            print(f"{model_type:<20} {m['r2']:<10.4f} {m['rmse']:<10.4f} {m['spearman']:<10.4f}")

    # Save best model
    if results:
        best_model_type = max(results.keys(),
                             key=lambda k: results[k]['eval_metrics']['r2'])
        best_model = results[best_model_type]['model']

        print(f"\nBest Model: {best_model_type.upper()}")
        print(f"R²: {results[best_model_type]['eval_metrics']['r2']:.4f}")

        df_pred = df.copy()
        df_pred['logK_pred'] = best_model.predict(df['smiles'].tolist())
        df_pred['residual'] = df_pred['logK_pred'] - df_pred['logK_exp']

        output_csv = base_dir / f"group_contrib_{best_model_type}_predictions.csv"
        df_pred.to_csv(output_csv, index=False)
        print(f"Saved: {output_csv}")

    return results


if __name__ == "__main__":
    results = main()
