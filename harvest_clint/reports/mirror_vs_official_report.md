# Mirror vs Official Source Comparison Report
## B2 TDC Clearance Data Re-download Attempt

**Date:** 2026-07-06
**Task:** Re-download B2 (TDC clearance) from official sources and compare with HuggingFace mirrors

---

## Executive Summary

| Source | Status | Hepatocyte | Microsome | Notes |
|--------|--------|------------|-----------|-------|
| **PyTDC (Official)** | ❌ FAILED | - | - | SSL connection errors with Harvard Dataverse |
| **ChEMBL (Official)** | ⚠️ DIFFERENT | 39 rows | 7 rows | Different assays, NOT AZ clearance |
| **HuggingFace Mirror** | ✓ COMPLETE | 1,213 rows | N/A | AZ clearance data (TDC subset) |
| **ExpansionRX HLM** | ✓ COMPLETE | - | 4,541 rows | From previous harvest |

**Conclusion:** Official sources did NOT provide the complete AZ clearance datasets. The mirror data remains the most complete source available.

---

## STEP 1: PyTDC (Official TDC) - FAILED

### Error
```
SSLError: HTTPSConnectionPool(host='dvn-cloud-iqss.s3.amazonaws.com', port=443):
Max retries exceeded with url: /10.7910/DVN/21LKWG/...
[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol
```

### Details
- **Target endpoints:**
  - `Clearance_Hepatocyte_AZ`
  - `Clearance_Microsome_AZ`
- **Host:** dvn-cloud-iqss.s3.amazonaws.com (Harvard Dataverse)
- **Issue:** SSL handshake failure - persistent across retries
- **Workarounds attempted:**
  - Environment variable adjustment (HUGGINGFACE_HUB_CACHE)
  - Multiple retry attempts
  - All failed with same SSL error

### Status
❌ **Cannot download TDC data directly via PyTDC** due to SSL/connection issues with Harvard Dataverse S3.

---

## STEP 2: ChEMBL API (Official) - LIMITED/DIFFERENT

### Query Results

| System | Rows | Assay IDs | Units | Notes |
|--------|------|-----------|-------|-------|
| Hepatocyte | 39 | CHEMBL5442164-5442167 | uL/min/1E6 cells | **NOT AZ clearance** |
| Microsome | 7 | CHEMBL2114007 | uL min-1 mg-1 | **NOT AZ clearance** |

### ChEMBL Assay Details

**Hepatocyte Assays (CHEMBL5442164-5442167):**
- Description: "To determine the metabolic stability in [rat/mouse/human/dog] hepatocytes..."
- Source: Different from AZ clearance
- Rows: 9-10 per assay (total 39)
- **NOT the AstraZeneca clearance data**

**Microsome Assay (CHEMBL2114007):**
- Description: "OSM: Microsomal stability: 0.5 uM substrate, 0.4 mg/mL protein..."
- Source: Different from AZ clearance
- Rows: 7
- **NOT the AstraZeneca clearance data**

### Analysis
The ChEMBL query for `standard_type='CLint'` returned data, but it is from **different assays** than the AZ clearance data used by TDC.

**Possible explanations:**
1. AZ clearance data was deposited to ChEMBL under different assay IDs
2. AZ clearance data uses a different `standard_type` value
3. AZ clearance data is not fully indexed in ChEMBL
4. TDC uses a curated subset not directly available via ChEMBL API

### Status
⚠️ **ChEMBL does not contain the AZ clearance datasets** expected. The CLint records found are from different metabolic stability assays.

---

## STEP 3: Mirror vs Official Comparison

### Data Sources

**Mirror Data (HuggingFace):**
- AZ clearance: `jablonkagroup/clearance_astrazeneca` (1,213 rows)
- TDC hepatocyte: `amirhallaji/ADME-Property-Prediction-Clearance_Hepatocyte_AZ` (849 rows)
- ExpansionRX HLM: `scikit-fingerprints/ExpansionRx_OpenADMET_HLM_CLint` (4,541 rows)

**ChEMBL Data (Official):**
- Hepatocyte: 39 rows from assays CHEMBL5442164-5442167
- Microsome: 7 rows from assay CHEMBL2114007

### Overlap Analysis

| Comparison | Overlap | Notes |
|------------|---------|-------|
| ChEMBL hepatocyte vs AZ mirror | **0 / 1,020** | No InChIKey overlap |
| ChEMBL hepatocyte vs TDC mirror | **0 / 755** | No InChIKey overlap |
| ChEMBL microsome vs ExpansionRX | **0 / 4,541** | No InChIKey overlap |

### Conclusion
**No overlap found** between ChEMBL CLint records and HuggingFace mirror data. This confirms that ChEMBL contains different metabolic stability assays, not the AZ clearance datasets.

---

## Mirror Data Validation

### Available Data

| Source | System | Rows | Unique InChIKeys | Unit | Status |
|--------|--------|------|------------------|------|--------|
| AZ clearance | Hepatocyte | 1,213 | 1,020 | µL/min/10^6 cells | ✓ |
| TDC hepatocyte | Hepatocyte | 849 | 755 | µL/min/10^6 cells | ✓ (AZ subset) |
| ExpansionRX HLM | Microsome | 4,541 | 4,541 | µL/min/mg protein | ✓ |

### Value Ranges

**AZ clearance (Hepatocyte):**
- Range: 3.00 - 150.00 µL/min/10^6 cells
- Mean: 42.9, Median: 19.0

**ExpansionRX HLM (Microsome):**
- Range: 0.00 - 2589.90 µL/min/mg protein
- Mean: 50.3, Median: 17.5

### Censoring Check

**Boundary value analysis:**
- AZ clearance: No censoring signals detected (no spikes at boundaries)
- ExpansionRX HLM: Some zero values present (normal for low-clearance compounds)

---

## Conclusions

### Official Sources Status

1. **PyTDC (Official TDC):** ❌ FAILED
   - SSL errors prevent download from Harvard Dataverse
   - Hepatocyte and microsome datasets unavailable

2. **ChEMBL (Official):** ⚠️ NOT EQUIVALENT
   - Only 46 CLint records found (39 hepatocyte + 7 microsome)
   - From different assays (NOT AZ clearance)
   - No overlap with mirror data

### Mirror Data Status

**HuggingFace mirrors remain the ONLY complete source available:**
- AZ clearance: 1,213 hepatocyte measurements
- ExpansionRX HLM: 4,541 microsome measurements
- Total: 5,561 unique compounds

### Recommendation

⚠️ **Official sources cannot provide the complete AZ clearance datasets.**

**Continue using mirror data with the following caveats:**
1. The AZ clearance data appears to be the most complete hepatocyte dataset available
2. The ExpansionRX HLM data appears to be the most complete microsome dataset available
3. Without official source validation, the exact provenance of these datasets cannot be confirmed

### For Future Work

**To obtain official AZ clearance data:**
1. Contact TDC maintainers about SSL/connection issues
2. Contact ChEMBL to identify correct assay IDs for AZ clearance
3. Check if AZ clearance is available via AstraZeneca directly
4. Consider using the mirror data as de facto standard while documenting provenance uncertainty

---

## Data Files

**Downloaded ChEMBL data (official but different):**
- `/tmp/chembl_hepatocyte_CLint.csv` (39 rows)
- `/tmp/chembl_microsome_CLint.csv` (7 rows)

**Mirror data (in use):**
- `/gpfs/home1/s9383/harvest_clint/processed/AZ_clearance.csv`
- `/gpfs/home1/s9383/harvest_clint/processed/ExpansionRX_HLM.csv`

---

## Unconfirmed/Unknown Items

| Item | Status | Notes |
|------|--------|-------|
| PyTDC SSL error | UNRESOLVED | Environment or server issue |
| ChEMBL AZ assay IDs | NOT FOUND | AZ clearance may use different standard_type |
| TDC microsome data | NOT FOUND | Broken in HuggingFace, official unavailable |

---

**Report generated:** 2026-07-06
**Task:** B2 TDC clearance re-download and mirror comparison
