# Physicochemical Multi-LFER Method for BSA Partition Coefficient Prediction

## Overview

This repository contains the implementation and results of a **Linear Free Energy Relationship (LFER)** based method for predicting Bovine Serum Albumin (BSA) partition coefficients.

---

## Methodology: Physicochemical Multi-LFER

### Theory

The partition coefficient (log K) is predicted using a linear combination of physicochemical descriptors:

```
log K = β₀ + β₁·π + β₂·α + β₃·β + β₄·V + β₅·logP + ...
```

Where:
- **π**: Polarity/Polarizability
- **α**: H-bond donor acidity (HBD)
- **β**: H-bond acceptor basicity (HBA)  
- **V**: Molar volume (MV)
- **logP**: Octanol-water partition coefficient
- **TPSA**: Topological polar surface area

### Descriptor Calculation

Descriptors were computed using:
- **RDKit**: Molecular fingerprints, physicochemical properties
- **Mordred**: 1800+ molecular descriptors
- **OpenBabel**: 3D structure generation

### Model Development

1. **Descriptor Selection**: 
   - Removed highly correlated descriptors (r > 0.9)
   - Selected relevant descriptors for partition coefficient prediction

2. **Model Training**:
   - Dataset: 83 compounds with experimental logK values
   - Algorithm: Multiple Linear Regression (MLR)
   - Cross-validation: 5-fold CV

---

## Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **R²** | 0.6354 |
| **RMSE** | 0.5205 |
| **MAE** | 0.4102 |
| **n** | 83 compounds |

### Comparison with Other Methods

| Method | R² | RMSE |
|--------|-----|------|
| **Multi-LFER** (this work) | **0.6354** | 0.5205 |
| Molecular Property (Hydrophobic) | 0.6161 | 0.5341 |
| Thermodynamic Flory-Huggins | 0.5853 | 0.5551 |
| Advanced Physicochemical | 0.5524 | 0.5767 |
| Group Contribution UNIFAC | 0.0233 | 0.8397 |

---

## Dataset

### Input: compounds_input_table1.csv
- 83 organic compounds
- Experimental logK (BSA partition coefficient)
- SMILES structures

### Output: physicochemical_multi-lfer_predictions.csv
| Column | Description |
|--------|-------------|
| compound_name | IUPAC name |
| logK_exp | Experimental log K |
| logK_pred | Predicted log K |
| residual | Prediction error |

---

## Usage

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv('data/physicochemical_multi-lfer_predictions.csv')

# View results
print(data[['compound_name', 'logK_exp', 'logK_pred']].head())
```

---

## References

1. **Abraham Solvation Parameters**: Abraham, M.H. (1993)
2. **Linear Solvation Energy Relationship (LSER)**: Kamlet, M.J. et al. (1987)
3. **BSA Partition Coefficients**: Sudlow et al. (1975)

---

## Author

Taegyun Lee (tlagustn123)

## License

MIT License

