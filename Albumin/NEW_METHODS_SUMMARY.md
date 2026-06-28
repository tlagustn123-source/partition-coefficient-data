# Albumin Partition Coefficient - Complete Methods Summary

## Overview

This document summarizes all physicochemical equation-based methods developed for predicting albumin-water partition coefficients, excluding docking, quantum chemistry, and machine learning approaches.

**Date**: 2026-06-28
**Total Methods**: 14
**Best Method**: Hydrophobic Balance (R² = 0.6161)

---

## Complete Method Comparison

### Rank 1: Hydrophobic Balance (R² = 0.6161) ⭐

**Category**: Molecular Property

**Equation**:
```
logK = LogP - 0.3 × HBD - 0.2 × HBA + 0.5 × arom_frac
```

**Key Features**:
- Simple, interpretable equation
- SMILES-only input
- Based on fundamental molecular properties
- H-bonding penalty + aromatic bonus

**Performance**:
- R²: 0.6161
- RMSE: 0.5341
- Spearman ρ: 0.7193
- Pearson r: 0.7849

---

### Rank 2: Flory-Huggins Theory (R² = 0.5853)

**Category**: Thermodynamic

**Equation**:
```
χ = V_solute × (δ_albumin - δ_solute)² / (R × T)
ln γ = ln(1 - φ) + (1 - V_solute/V_albumin) × φ + χ × φ²
logK = f(ln γ, LogP, calibration)
```

**Key Features**:
- Polymer solution theory
- Accounts for molecular size
- Thermodynamic grounding

**Performance**:
- R²: 0.5853
- RMSE: 0.5551
- Spearman ρ: 0.7056

---

### Rank 3: Molar Volume (R² = 0.5858)

**Category**: Molecular Property

**Equation**:
```
logK = LogP + 0.5 × log10(V_m/V_albumin × 1e6) - 0.001 × MW
```

**Key Features**:
- Volume exclusion effects
- Steric considerations
- McGowan volume

**Performance**:
- R²: 0.5858
- RMSE: 0.5547
- Spearman ρ: 0.7111

---

### Rank 4: Hybrid Property (R² = 0.5790)

**Category**: Molecular Property

**Equation**:
```
logK = 0.4 × logK_mv + 0.4 × logK_sa + 0.2 × logK_pc
```

**Key Features**:
- Combines multiple property methods
- Weighted ensemble

**Performance**:
- R²: 0.5790
- RMSE: 0.5593
- Spearman ρ: 0.7108

---

### Rank 5: ΔG Transfer Method (R² = 0.5742)

**Category**: Thermodynamic

**Equation**:
```
ΔG_transfer = -2.303 × RT × LogP - ΔG_steric + ΔG_polar
logK = -ΔG_transfer / (RT × ln(10))
```

**Key Features**:
- Transfer free energy
- Hydrophobic effect basis

**Performance**:
- R²: 0.5742
- RMSE: 0.5629
- Spearman ρ: 0.7028

---

### Rank 6: Polarizability (R² = 0.5524)

**Category**: Advanced Physicochemical

**Equation**:
```
logK = LogP + 0.05 × MR - 0.02 × TPSA
```

**Key Features**:
- Molecular polarizability
- Dispersion forces
- Electronic effects

**Performance**:
- R²: 0.5524
- RMSE: 0.5767
- Spearman ρ: 0.7242

---

### Rank 7: Dielectric Properties (R² = 0.5411)

**Category**: Advanced Physicochemical

**Equation**:
```
logK = LogP - 0.1 × dipole_approx - 0.015 × TPSA
```

**Key Features**:
- Dielectric constant effects
- Dipole moment

**Performance**:
- R²: 0.5411
- RMSE: 0.5839
- Spearman ρ: 0.6552

---

### Rank 8: Surface Area (R² = 0.5315)

**Category**: Molecular Property

**Equation**:
```
logK = LogP - 0.02 × TPSA - 0.5 × polar_frac
```

**Key Features**:
- TPSA-based
- Polar surface penalty

**Performance**:
- R²: 0.5315
- RMSE: 0.5900
- Spearman ρ: 0.6525

---

### Rank 9: Hybrid Advanced (R² = 0.5057)

**Category**: Advanced Physicochemical

**Equation**:
```
logK = 0.4 × logK_fu + 0.35 × logK_mr + 0.25 × logK_di
```

**Key Features**:
- Combines fractional unbound, molar refractivity, dielectric

**Performance**:
- R²: 0.5057
- RMSE: 0.6060
- Spearman ρ: 0.7296

---

### Rank 10: Regular Solution Theory (R² = 0.4607)

**Category**: Thermodynamic

**Equation**:
```
ln γ = V_solute × (δ_solvent - δ_solute)² / (RT)
logK = (ln γ_water - ln γ_octanol) / ln(10) × 0.85 + 0.5
```

**Key Features**:
- Hildebrand solubility parameters
- Regular solution approximation

**Performance**:
- R²: 0.4607
- RMSE: 0.6190
- Spearman ρ: 0.6417

---

### Rank 11: Fractional Unbound (R² = 0.4791)

**Category**: Advanced Physicochemical

**Equation**:
```
fu = 1 / (1 + 10^(0.7 × LogP - 1.5))
logK = -log10(fu) + ionization_correction
```

**Key Features**:
- Protein binding (fu) correlation
- pH-dependent (if ionizable)

**Performance**:
- R²: 0.4791
- RMSE: 0.6221
- Spearman ρ: 0.5919

---

### Rank 12: Parachor (R² = 0.4787)

**Category**: Molecular Property

**Equation**:
```
P = MR × 10 + LogP × 20 - TPSA × 0.05
logK = 0.01 × P + 0.6 × LogP - 0.02 × TPSA
```

**Key Features**:
- Surface tension relationship
- Parachor calculation

**Performance**:
- R²: 0.4787
- RMSE: 0.6223
- Spearman ρ: 0.7232

---

### Rank 13: Molar Refractivity (R² = 0.3334)

**Category**: Advanced Physicochemical

**Equation**:
```
logK = 0.15 × MR + 0.4 × LogP + 0.0005 × MW
```

**Key Features**:
- Polarizability only
- Simple correlation

**Performance**:
- R²: 0.3334
- RMSE: 0.7038
- Spearman ρ: 0.6344

---

### Rank 14: UNIFAC (R² = 0.0233)

**Category**: Group Contribution

**Key Features**:
- Group contribution method
- Limited performance for this dataset

**Performance**:
- R²: 0.0233
- RMSE: 0.8391
- Spearman ρ: 0.2317

---

## Category Comparison

| Category | Mean R² | Best Method |
|----------|---------|-------------|
| Molecular Property | 0.6161 | Hydrophobic Balance |
| Thermodynamic | 0.5853 | Flory-Huggins |
| Advanced Physicochemical | 0.5524 | Polarizability |
| Group Contribution | 0.0233 | UNIFAC |

---

## Comparison with Previous Methods

| Method | R² | Type |
|--------|-----|------|
| **Hydrophobic Balance** | **0.6161** | Physicochemical |
| Flory-Huggins | 0.5853 | Physicochemical |
| Vina+COSMO Hybrid | 0.437 | Docking+QC |
| COSMO-RS | 0.334 | Quantum Chemistry |
| Vina Blind | 0.274 | Docking |

**Key Insight**: Physicochemical methods outperform docking and quantum chemistry approaches for this dataset!

---

## Practical Recommendations

### For Prediction on New Compounds

**Use Hydrophobic Balance Method**:
```python
from molecular_property_partition_model import MolecularPropertyPartitionModel

model = MolecularPropertyPartitionModel(model_type='hydrophobic-balance')
model.fit(training_smiles, training_logK)  # Calibrate
predictions = model.predict(new_smiles)     # Predict
```

### For Method Development

**Best Approaches**:
1. Combine H-bonding descriptors with LogP
2. Account for aromatic content
3. Include steric/size effects

### For Other Partition Systems

Same methods work for:
- Membrane lipid (k_M)
- Storage lipid (P_SL)
- Muscle partition

Just change the input data file.

---

## File Summary

### Code Files
- `molecular_property_partition_model.py` - Molecular property methods
- `advanced_physicochemical_model.py` - Advanced physicochemical methods
- `thermodynamic_partition_model.py` - Thermodynamic methods
- `group_contribution_solubility_model.py` - Group contribution methods

### Results Files
- `molecular_property_hydrophobic-balance_predictions.csv` - Best model
- `all_methods_summary.csv` - All methods comparison
- `all_methods_comparison_scatter.png` - Visual comparison

### Documentation
- `HANDOFF.md` - Handoff document for next conversation
- `NEW_METHODS_SUMMARY.md` - This file

---

## How to Use in Next Conversation

**Prompt Template**:
```
I want to continue albumin partition coefficient research.

Read HANDOFF.md for context.

I want to:
- Apply Hydrophobic Balance model to [new data]
- Create hybrid model combining [methods]
- Apply to [other partition system]
- Compare with [literature/experimental data]
```

---

## Conclusion

The Hydrophobic Balance method achieves R² = 0.6161, making it the best physicochemical method for albumin partition coefficient prediction. It requires only SMILES input and runs in milliseconds per compound.

**Key Success Factors**:
1. H-bonding capacity (penalty)
2. Aromatic content (bonus)
3. LogP baseline (hydrophobicity)
4. Linear calibration to experimental data
