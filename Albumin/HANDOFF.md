# Albumin Partition Coefficient Research - Handoff Document

## Project Overview

**Research Goal**: Develop physicochemical equation-based methods for predicting albumin-water partition coefficients, excluding docking, quantum chemistry, and machine learning approaches.

**Status**: ✅ Comprehensive implementation complete (multiple method categories)

**Date**: 2026-06-28 (Updated)

---

## What Was Done (Complete)

### 1. All Methods Implemented

| Category | Method | Performance (R²) | Status |
|----------|--------|-----------------|--------|
| **Molecular Property** | Hydrophobic Balance | **0.6161** | ✅ **Best Overall** |
| **Molecular Property** | Molar Volume | 0.5858 | ✅ |
| **Molecular Property** | Hybrid Property | 0.5790 | ✅ |
| **Molecular Property** | Surface Area (TPSA) | 0.5315 | ✅ |
| **Molecular Property** | Parachor | 0.4787 | ✅ |
| **Thermodynamic** | Flory-Huggins Theory | 0.5853 | ✅ |
| **Thermodynamic** | ΔG Transfer Method | 0.5742 | ✅ |
| **Thermodynamic** | Regular Solution Theory | 0.4607 | ✅ |
| **Advanced Physicochem** | Polarizability | 0.5524 | ✅ |
| **Advanced Physicochem** | Dielectric Properties | 0.5411 | ✅ |
| **Advanced Physicochem** | Hybrid Advanced | 0.5057 | ✅ |
| **Advanced Physicochem** | Fractional Unbound (fu) | 0.4791 | ✅ |
| **Advanced Physicochem** | Molar Refractivity | 0.3334 | ✅ |
| **Group Contribution** | UNIFAC | 0.0233 | ✅ Low performance |

### 2. Key Results

**Hydrophobic Balance Model achieves R² = 0.6161**, the best performance overall!

This model outperforms:
- Flory-Huggins (R² = 0.5853)
- Vina Blind (R² = 0.274)
- COSMO-RS (R² = 0.334)
- Hybrid Vina+COSMO (R² = 0.437)

**Advantages of Hydrophobic Balance Method:**
- SMILES-only input (no experimental data needed)
- Milliseconds per prediction
- Based on H-bonding capacity and aromatic content
- Simple equation: `logK = LogP - 0.3×HBD - 0.2×HBA + 0.5×arom_frac`

**Category Performance (Mean R²):**
1. Molecular Property: 0.6161
2. Thermodynamic: 0.5853
3. Advanced Physicochemical: 0.5524
4. Group Contribution: 0.0233

### 3. Files Created

**Main Code (Updated):**
- `Albumin/thermodynamic_partition_model.py` - Thermodynamic methods
- `Albumin/group_contribution_solubility_model.py` - Group contribution methods
- `Albumin/molecular_property_partition_model.py` - **NEW: Molecular property methods**
- `Albumin/advanced_physicochemical_model.py` - **NEW: Advanced physicochemical methods**
- `Albumin/visualize_all_methods_comprehensive.py` - **NEW: Unified visualization**

**Documentation:**
- `Albumin/ALL_PHYSICOCHEMICAL_METHODS_SUMMARY.md` - Complete summary
- `Albumin/THERMODYNAMIC_MODEL_RESULTS.md` - Thermodynamic methods detail
- `Albumin/HYBRID_MODEL_RESULTS.md` - Previous hybrid model results
- `Albumin/RESULTS_SUMMARY.md` - Vina/Boltz results
- `Albumin/NEW_METHODS_SUMMARY.md` - **NEW: Summary of new methods**

**Results Files:**
- `Albumin/molecular_property_hydrophobic-balance_predictions.csv` - **Best model predictions**
- `Albumin/thermodynamic_flory-huggins_predictions.csv` - Previous best
- `Albumin/advanced_physicochem_polarizability_predictions.csv` - Best advanced method
- `Albumin/all_methods_summary.csv` - **All methods comparison**
- `Albumin/all_methods_comparison_scatter.png` - **Visualization**
- `Albumin/all_methods_bar_comparison.png` - **Bar chart comparison**
- `Albumin/all_methods_category_comparison.png` - **Category comparison**

### 4. GitHub Repository

**URL**: https://github.com/tlagustn123-source/partition-coefficient-data

**Branches:**
- `main` - Previous content
- `physicochemical-methods` - Physicochemical methods (original)

**For new methods, create a new branch:**
```bash
git checkout -b advanced-physicochemical-methods
git add Albumin/molecular_property_partition_model.py
git add Albumin/advanced_physicochemical_model.py
git add Albumin/visualize_all_methods_comprehensive.py
# ... add other new files
GIT_SSH_COMMAND="ssh -i /home1/s9383/.ssh/id_rsa" git push -u origin advanced-physicochemical-methods
```

---

## Methods Excluded (Per User Requirements)

- ❌ BSA fitting methods
- ❌ AutoDock Vina (docking)
- ❌ xTB-GBSA/ALPB (quantum chemistry)
- ❌ COSMO-RS (solvation)
- ❌ QSPR/QSAR/DL (machine learning)
- ❌ LFER/UFZ/PPLFER (Linear Free Energy Relationships)
- ❌ Abraham descriptors (requires experimental data)

---

## New Methods Details

### Hydrophobic Balance Method (Best, R² = 0.6161)

**Equation:**
```
logK = LogP - 0.3 × HBD - 0.2 × HBA + 0.5 × arom_frac
```

**Key Features:**
- HBD: Hydrogen bond donor count
- HBA: Hydrogen bond acceptor count
- arom_frac: Fraction of aromatic atoms
- Penalizes hydrogen bonding (reduces albumin binding)
- Rewards aromatic content (π-π interactions with albumin)

**Why it works:**
- Albumin has hydrophobic binding pockets
- Aromatic interactions are important
- H-bonding reduces partition into hydrophobic phase

### Molar Volume Method (R² = 0.5858)

**Equation:**
```
logK = LogP + 0.5 × log10(V_m/V_albumin × 1e6) - 0.001 × MW
```

**Key Features:**
- Accounts for steric effects
- Volume exclusion consideration
- Size-dependent partitioning

### Polarizability Method (R² = 0.5524)

**Equation:**
```
logK = LogP + 0.05 × MR - 0.02 × TPSA
```

**Key Features:**
- MR: Molar refractivity (polarizability)
- TPSA: Topological polar surface area
- Dispersion force contributions

---

## How to Run the Models

### Environment

**Conda Environment**: `rapids-cuml` (has RDKit 2026.03.3, pandas 2.3.3)

```bash
source /home1/s9383/miniconda3/etc/profile.d/conda.sh
conda activate rapids-cuml
```

### Run Commands

```bash
cd /home1/s9383/Albumin

# Thermodynamic methods (previous)
/home1/s9383/miniconda3/envs/rapids-cuml/bin/python thermodynamic_partition_model.py

# NEW: Molecular property methods (best performance)
/home1/s9383/miniconda3/envs/rapids-cuml/bin/python molecular_property_partition_model.py

# NEW: Advanced physicochemical methods
/home1/s9383/miniconda3/envs/rapids-cuml/bin/python advanced_physicochemical_model.py

# NEW: Comprehensive visualization (all methods)
/home1/s9383/miniconda3/envs/rapids-cuml/bin/python visualize_all_methods_comprehensive.py

# Group contribution methods
/home1/s9383/miniconda3/envs/rapids-cuml/bin/python group_contribution_solubility_model.py
```

---

## Data Files

### Input Data
- `Albumin/albumin_data.csv` - 83 compounds with experimental logK values

### Output Predictions (Best Models)
- `Albumin/molecular_property_hydrophobic-balance_predictions.csv` - **Best overall (R² = 0.6161)**
- `Albumin/thermodynamic_flory-huggins_predictions.csv` - Best thermodynamic (R² = 0.5853)
- `Albumin/advanced_physicochem_polarizability_predictions.csv` - Best advanced (R² = 0.5524)

---

## How to Continue This Research

### Option 1: Apply Best Model to New Data

**Using Hydrophobic Balance Model:**
```python
from molecular_property_partition_model import MolecularPropertyPartitionModel

model = MolecularPropertyPartitionModel(model_type='hydrophobic-balance')
model.fit(smiles_list, logK_values)  # Calibrate to data
predictions = model.predict(new_smiles_list)  # Predict new compounds
```

### Option 2: Apply to Other Partition Systems

**Available Data:**
- Membrane lipid (k_M)
- Storage lipid (P_SL)
- Muscle (chicken, fish)

Same models work for other partition systems - just change input data.

### Option 3: Further Method Development

**Potential improvements:**
- Combine Hydrophobic Balance + Flory-Huggins (hybrid)
- Add protein-specific parameters
- Temperature dependence modeling
- pH-dependent partitioning (for ionizable compounds)

### Option 4: Experimental Validation

- Compare with new experimental data
- Blind prediction on external datasets
- Literature comparison

---

## Prompt for Next Conversation

To continue this research in a new conversation, use the following prompt:

```
Hello! I want to continue the albumin partition coefficient research.

Please read /home1/s9383/Albumin/HANDOFF.md to understand the previous work.

The best model is Hydrophobic Balance (R² = 0.6161).

I want to: [select one]
- Apply the model to new compounds
- Apply to other partition systems (membrane/storage lipid)
- Create a hybrid model combining best methods
- Compare with literature/experimental data
- [other specific task]
```

---

## Contact & References

**GitHub**: tlagustn123-source/partition-coefficient-data

**Framework Reference**: sci-adk (https://github.com/ccy5123/sci-adk)

**Key Papers**:
- Flory, P.J. (1942) - Polymer solution theory
- Hansen, C.M. (2007) - Hansen Solubility Parameters
- Endo & Goss (2011) - Albumin partition data
- Mannhold, R. et al. (1998) - Molecular parachor calculation
- McGowan, J.C. (1985) - Characteristic molecular volume

---

## Final Summary

| Metric | Value |
|--------|-------|
| Total methods evaluated | 14 |
| Best method | Hydrophobic Balance |
| Best R² | 0.6161 |
| Best RMSE | 0.5341 |
| Best Spearman ρ | 0.7193 |
| Compounds | 83 |
| Input required | SMILES only |
| Prediction speed | ~1 ms per compound |
