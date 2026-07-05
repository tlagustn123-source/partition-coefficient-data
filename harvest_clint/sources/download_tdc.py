#!/usr/bin/env python3
"""
Download TDC clearance data for hepatocyte and microsome systems.
Standardize to common schema.
"""
import sys
import importlib.metadata
from pathlib import Path

# pkg_resources workaround for newer setuptools
class FakePkgResources:
    class Distribution:
        def __init__(self, name):
            self.name = name
            self._version = importlib.metadata.version(name)
        @property
        def version(self):
            return self._version
    @staticmethod
    def get_distribution(name):
        return FakePkgResources.Distribution(name)

sys.modules['pkg_resources'] = FakePkgResources

import pandas as pd
import numpy as np
from tdc.single_pred import ADME
from rdkit import Chem
from rdkit.Chem import Descriptors

def calculate_inchikey(smiles):
    """Calculate InChIKey from SMILES using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "RDKit parsing failed"
        inchikey = Chem.MolToInchIKey(mol)
        canonical = Chem.MolToSmiles(mol, canonical=True)
        return inchikey, None
    except Exception as e:
        return None, f"Error: {str(e)}"

def is_pfas(smiles):
    """Simple PFAS detection based on fluorine count and patterns."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "SMILES parsing failed"
        # Count fluorine atoms
        f_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F')
        # PFAS typically have >= 3 fluorines or specific patterns
        return f_count >= 3, None
    except Exception as e:
        return False, f"Error: {str(e)}"

def process_tdc_dataset(name, system, output_dir):
    """Download and process a TDC clearance dataset."""
    print(f"\n{'='*60}")
    print(f"Processing TDC dataset: {name}")
    print(f"System: {system}")
    print(f"{'='*60}")

    # Download data
    data = ADME(name=name)
    df = data.get_data()

    print(f"\nRaw data shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")

    # Check Y column statistics
    print(f"\nY (clearance value) statistics:")
    print(df['Y'].describe())

    # Check if values are log-transformed
    y_values = df['Y'].dropna()
    y_min = y_values.min()
    y_max = y_values.max()

    is_log = False
    if y_min > 0 and y_max < 10:
        is_log = True
        print(f"\n⚠️  Values appear to be log10-transformed (range: {y_min:.2f} to {y_max:.2f})")
    elif y_min < 0 and y_max < 10:
        is_log = True
        print(f"\n⚠️  Values appear to be log10-transformed (includes negative values)")
    else:
        print(f"\n⚠️  Values appear to be raw (range: {y_min:.2f} to {y_max:.2f})")

    # Determine unit based on system and TDC documentation
    # TDC AZ clearance datasets:
    # - Hepatocyte: µL/min/10^6 cells
    # - Microsome: µL/min/mg protein
    # Values are typically log10 transformed
    if system == 'hepatocyte':
        unit = 'µL/min/10^6 cells'
    else:  # microsome
        unit = 'µL/min/mg protein'

    print(f"Unit: {unit}")
    print(f"Log-transformed: {is_log}")

    # Standardize to common schema
    standardized = []
    parse_errors = 0

    for _, row in df.iterrows():
        smiles = row.get('Drug', row.get('SMILES'))
        y_value = row.get('Y')

        if pd.isna(smiles) or pd.isna(y_value):
            continue

        # Calculate InChIKey and canonical SMILES
        inchikey, error = calculate_inchikey(smiles)
        if inchikey is None:
            parse_errors += 1
            note = error

        # Check PFAS
        is_pfas, pfas_error = is_pfas(smiles)

        # Calculate log10_value if raw, or use existing if log
        if is_log:
            log10_value = y_value
        else:
            log10_value = np.log10(y_value) if y_value > 0 else np.nan

        standardized.append({
            'source': f'TDC_{name}',
            'endpoint': 'in_vitro_CLint',
            'species': 'human',
            'system': system,
            'unit': unit,
            'value': y_value,
            'log10_value': log10_value,
            'SMILES': smiles,
            'canonical_SMILES': Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True) if inchikey else None,
            'InChIKey': inchikey,
            'DTXSID': None,
            'CAS': None,
            'is_PFAS': is_pfas,
            'note': note if inchikey is None else ''
        })

    result_df = pd.DataFrame(standardized)

    print(f"\n{'='*60}")
    print(f"Processed data shape: {result_df.shape}")
    print(f"Parse errors: {parse_errors}")
    print(f"Unique InChIKeys: {result_df['InChIKey'].nunique()}")
    print(f"{'='*60}")

    # Save to CSV
    output_file = output_dir / f'TDC_{system}.csv'
    result_df.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")

    return {
        'name': name,
        'system': system,
        'rows': len(result_df),
        'unique_inchikeys': result_df['InChIKey'].nunique(),
        'unit': unit,
        'is_log': is_log,
        'parse_errors': parse_errors,
        'file': str(output_file)
    }

def main():
    output_dir = Path('/gpfs/home1/s9383/harvest_clint/sources')
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_process = [
        ('Clearance_Hepatocyte_AZ', 'hepatocyte'),
        ('Clearance_Microsome_AZ', 'microsome')
    ]

    results = []
    for name, system in datasets_to_process:
        try:
            result = process_tdc_dataset(name, system, output_dir)
            results.append(result)
        except Exception as e:
            print(f"ERROR processing {name}: {e}")
            results.append({
                'name': name,
                'system': system,
                'error': str(e)
            })

    # Print summary
    print("\n" + "="*60)
    print("TDC DOWNLOAD SUMMARY")
    print("="*60)
    for r in results:
        if 'error' in r:
            print(f"❌ {r['name']} ({r['system']}): {r['error']}")
        else:
            print(f"✓ {r['name']} ({r['system']}): {r['rows']} rows, {r['unique_inchikeys']} unique InChIKeys")

if __name__ == '__main__':
    main()
