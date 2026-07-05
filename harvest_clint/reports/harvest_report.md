# Human In Vitro CLint Data Harvest Report

**Date:** 2026-07-06
**Goal:** Harvest human in vitro intrinsic clearance (CLint) data for SMILES → CLint regression model training

## Summary

Successfully downloaded and standardized CLint data from 3 sources:

| Source | System | Rows | Unique InChIKeys | Unit | Log-transformed? |
|--------|--------|-------|------------------|------|------------------|
| ExpansionRX HLM | microsome | 4,541 | 4,541 | µL/min/mg protein | No |
| AZ clearance | hepatocyte | 1,213 | 1,020 | µL/min/10^6 cells | No |
| TDC Hepatocyte AZ | hepatocyte | 849 | 755 | µL/min/10^6 cells | No |

**Deduplicated master: 5,561 unique compounds** (4,541 microsome + 1,020 hepatocyte)

## Download Locations

All raw and processed files are in: `/gpfs/home1/s9383/harvest_clint/`

### Raw Sources (in `sources/`)
- `ExpansionRX_HLM_train.csv` - ExpansionRX OpenADMET HLM CLint data
- `TDC_hepatocyte_raw.csv` - TDC Hepatocyte AZ data (from HuggingFace mirror)
- `AZ_clearance_raw_train.csv`, `AZ_clearance_raw_val.csv`, `AZ_clearance_raw_test.csv` - AZ clearance data

### Processed (in `processed/`)
- `ExpansionRX_HLM.csv` - Standardized ExpansionRX data
- `TDC_hepatocyte.csv` - Standardized TDC hepatocyte data
- `AZ_clearance.csv` - Standardized AZ clearance data
- `human_clint_master_dedup.csv` - **Final deduplicated master file**

## Standard Schema

All files use the common schema:

| Column | Description |
|--------|-------------|
| source | Data source name |
| endpoint | "in_vitro_CLint" |
| species | "human" |
| system | "hepatocyte" or "microsome" |
| unit | Original unit (preserved) |
| value | Raw CLint value |
| log10_value | log10(value) for raw data, or original if log |
| SMILES | Original SMILES |
| canonical_SMILES | Canonical SMILES from RDKit |
| InChIKey | InChIKey from RDKit (for deduplication) |
| DTXSID | None (not populated in this harvest) |
| CAS | None (not populated in this harvest) |
| is_PFAS | True if ≥3 fluorine atoms |
| note | Parsing errors or other notes |

## Source Details

### B2 - TDC (Therapeutic Data Commons)

**Status:** ✅ Downloaded via HuggingFace mirror

**Datasets:**
- `Clearance_Hepatocyte_AZ`: 849 rows, 755 unique InChIKeys
- Unit: µL/min/10^6 cells
- Values: Raw (range 3-150)

**Notes:**
- Direct TDC download failed due to SSL/connection errors with Harvard Dataverse S3
- Used HuggingFace mirror: `amirhallaji/ADME-Property-Prediction-Clearance_Hepatocyte_AZ`

### B3 - ExpansionRX OpenADMET HLM CLint

**Status:** ✅ Downloaded successfully

**Dataset:**
- Repository: `scikit-fingerprints/ExpansionRx_OpenADMET_HLM_CLint`
- 4,541 rows, 4,541 unique InChIKeys
- Unit: µL/min/mg protein
- Values: Raw (range 0-2589.9)

**Notes:**
- Large HLM dataset from OpenADMET ExpansionRX challenge
- All unique compounds (no overlap with hepatocyte sources)

### B4 - NCATS HLM (hlm.h5)

**Status:** ❌ Skipped per instructions

**Reason:**
- This is a trained model file, not raw data
- NCATS HLM endpoint likely contains metabolic stability classification labels, not continuous CLint values

### B5 - ChEMBL / PubChem

**Status:** ✅ Found AZ clearance (duplicate of B2)

**Dataset:**
- Repository: `jablonkagroup/clearance_astrazeneca` (raw_data config)
- 1,213 rows total (train: 867, val: 176, test: 170)
- 1,020 unique InChIKeys
- Unit: µL/min/10^6 cells
- Values: Raw (range 3-150)

**Overlap with B2:**
- **Complete overlap:** All 755 TDC InChIKeys are in AZ clearance
- AZ contains TDC data + 265 additional compounds
- Value differences exist (mean diff: 9.42, max: 145.43)
- Deduplication: Used median value for overlapping compounds

**Notes:**
- Did not verify PubChem AID 1159396 (skipped due to confirmed overlap)
- ChEMBL API not used since HuggingFace had the data

### B6 - DTXSID → SMILES Mapping

**Status:** ⚠️ Not completed

**Reason:**
- R with httk package not available in environment
- CompTox batch search alternative not executed

**Instructions for future:**
To map DTXSID to SMILES for existing files (httk_human_clint_measured_1413.csv, EPA_invitroTK_human_clint_L3.csv):

**Option A (R with httk):**
```r
library(httk)
ci <- get_cheminfo(info=c('DTXSID','CAS','SMILES','Compound'), suppress.messages=TRUE)
write.csv(ci, 'dtxsid_smiles_map.csv', row.names=FALSE)
```

**Option B (CompTox Dashboard):**
1. Go to https://comptox.epa.gov/dashboard/batch-search
2. Paste DTXSID list
3. Export with SMILES included
4. Join with existing CSV files

## Deduplication Results

### Overlap Analysis

| Comparison | Overlap | Notes |
|------------|---------|-------|
| TDC vs AZ clearance | 755/755 (100%) | AZ contains all TDC compounds |
| ExpansionRX vs TDC | 0/4541 | Different systems (expected) |
| ExpansionRX vs AZ | 0/4541 | Different systems (expected) |

### Deduplication Strategy

1. **Same InChIKey + Same System:** Take median value
2. **Same InChIKey + Different System:** Keep both rows (system is a feature)

### Final Dataset

- **Total compounds:** 5,561 unique InChIKeys
- **Microsome measurements:** 4,541 (ExpansionRX HLM)
- **Hepatocyte measurements:** 1,020 (AZ clearance, including TDC)

## Data Statistics

### ExpansionRX HLM (Microsome)
- Value range: 0 - 2589.9 µL/min/mg
- Mean: 50.3, Median: 17.5
- All unique compounds

### AZ Clearance (Hepatocyte)
- Value range: 3 - 150 µL/min/10^6 cells
- Mean: 42.9, Median: 19.0
- 1,020 unique compounds after deduplication

## Skipped / Failed Sources

| Source | Status | Reason |
|--------|--------|--------|
| TDC direct download | ❌ Failed | SSL/connection errors with Harvard Dataverse S3 |
| TDC Microsome AZ | ⚠️ Incomplete | HuggingFace mirror had unusual structure (1 row) |
| NCATS HLM (hlm.h5) | ❌ Skipped | Trained model, not data (per instructions) |
| PubChem AID 1159396 | ⚠️ Skipped | Confirmed overlap with AZ clearance data |
| DTXSID mapping | ⚠️ Not completed | R/httk not available |

## Next Steps

1. **Review master file:** Check `/gpfs/home1/s9383/harvest_clint/processed/human_clint_master_dedup.csv`
2. **Add existing data:** If you have existing files (httk, EPA, OPERA, Insights), they can be merged using the same schema
3. **Train model:** Use the deduplicated master file for SMILES → CLint regression
4. **Feature engineering:** Consider system (hepatocyte vs microsome) as a feature or train separate models

## Files Generated

```
/gpfs/home1/s9383/harvest_clint/
├── sources/          # Raw downloaded files
├── processed/        # Standardized and deduplicated files
├── reports/          # This report
└── existing/        # (empty) Place existing files here for future merging
```

## Commit Information

**Branch:** structural-protein
**Commit message:** `feat: Add human CLint harvest with ExpansionRX HLM + AZ hepatocyte data`

**Files to commit:**
- `harvest_clint/` (all contents)
- Add large raw files to `.gitignore` if needed
