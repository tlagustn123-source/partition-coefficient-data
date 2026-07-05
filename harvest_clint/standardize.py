#!/usr/bin/env python3
"""
Standardize all downloaded CLint data to common schema.
"""
import sys
import importlib.metadata
from pathlib import Path

# pkg_resources workaround for newer setuptools
class FakePkgResources:
    class Distribution:
        def __init__(self, name):
            self.name = name
            try:
                self._version = importlib.metadata.version(name)
            except:
                self._version = "0.0.0"
        @property
        def version(self):
            return self._version
    @staticmethod
    def get_distribution(name):
        return FakePkgResources.Distribution(name)

sys.modules['pkg_resources'] = FakePkgResources

import pandas as pd
import numpy as np
from rdkit import Chem

def calculate_inchikey(smiles):
    """Calculate InChIKey from SMILES using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, "RDKit parsing failed"
        inchikey = Chem.MolToInchiKey(mol)
        canonical = Chem.MolToSmiles(mol, canonical=True)
        return inchikey, canonical, None
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def is_pfas(smiles):
    """Simple PFAS detection based on fluorine count."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "SMILES parsing failed"
        f_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F')
        return f_count >= 3, None
    except Exception as e:
        return False, f"Error: {str(e)}"

def standardize_row(smiles, value, source, system, unit, is_log=False):
    """Standardize a single row to common schema."""
    inchikey, canonical, error = calculate_inchikey(smiles)
    pfas, pfas_error = is_pfas(smiles)

    if is_log:
        log10_value = value
    else:
        log10_value = np.log10(value) if value > 0 else np.nan

    return {
        'source': source,
        'endpoint': 'in_vitro_CLint',
        'species': 'human',
        'system': system,
        'unit': unit,
        'value': value,
        'log10_value': log10_value,
        'SMILES': smiles,
        'canonical_SMILES': canonical,
        'InChIKey': inchikey,
        'DTXSID': None,
        'CAS': None,
        'is_PFAS': pfas,
        'note': error if inchikey is None else ''
    }

def process_expansionrx_hlm(input_file, output_dir):
    """Process ExpansionRX HLM dataset."""
    print("\n" + "="*60)
    print("Processing ExpansionRX HLM dataset")
    print("="*60)

    df = pd.read_csv(input_file)
    print(f"Raw shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Check if values are log-transformed
    y_values = df['HLM CLint'].dropna()
    y_min = y_values.min()
    y_max = y_values.max()

    is_log = False
    if y_min > 0 and y_max < 10:
        is_log = True
        print(f"Values appear to be log10-transformed (range: {y_min:.2f} to {y_max:.2f})")
    else:
        print(f"Values appear to be raw (range: {y_min:.2f} to {y_max:.2f})")

    # Unit for HLM: µL/min/mg protein
    unit = 'µL/min/mg protein'

    standardized = []
    parse_errors = 0

    for _, row in df.iterrows():
        smiles = row.get('SMILES')
        value = row.get('HLM CLint')

        if pd.isna(smiles) or pd.isna(value):
            continue

        std_row = standardize_row(
            smiles=smiles,
            value=value,
            source='ExpansionRX_HLM',
            system='microsome',
            unit=unit,
            is_log=is_log
        )

        if std_row['InChIKey'] is None:
            parse_errors += 1

        standardized.append(std_row)

    result_df = pd.DataFrame(standardized)
    print(f"\nProcessed: {len(result_df)} rows")
    print(f"Parse errors: {parse_errors}")
    print(f"Unique InChIKeys: {result_df['InChIKey'].nunique()}")

    output_file = output_dir / 'ExpansionRX_HLM.csv'
    result_df.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")

    return result_df

def process_tdc_hepatocyte(input_file, output_dir):
    """Process TDC Hepatocyte AZ dataset."""
    print("\n" + "="*60)
    print("Processing TDC Hepatocyte AZ dataset")
    print("="*60)

    df = pd.read_csv(input_file)
    print(f"Raw shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Check if values are log-transformed
    y_values = df['Y'].dropna()
    y_min = y_values.min()
    y_max = y_values.max()

    is_log = False
    if y_min > 0 and y_max < 10:
        is_log = True
        print(f"Values appear to be log10-transformed (range: {y_min:.2f} to {y_max:.2f})")
    else:
        print(f"Values appear to be raw (range: {y_min:.2f} to {y_max:.2f})")

    # Unit for hepatocyte: µL/min/10^6 cells
    unit = 'µL/min/10^6 cells'

    standardized = []
    parse_errors = 0

    for _, row in df.iterrows():
        smiles = row.get('Drug')
        value = row.get('Y')

        if pd.isna(smiles) or pd.isna(value):
            continue

        std_row = standardize_row(
            smiles=smiles,
            value=value,
            source='TDC_Hepatocyte_AZ',
            system='hepatocyte',
            unit=unit,
            is_log=is_log
        )

        if std_row['InChIKey'] is None:
            parse_errors += 1

        standardized.append(std_row)

    result_df = pd.DataFrame(standardized)
    print(f"\nProcessed: {len(result_df)} rows")
    print(f"Parse errors: {parse_errors}")
    print(f"Unique InChIKeys: {result_df['InChIKey'].nunique()}")

    output_file = output_dir / 'TDC_hepatocyte.csv'
    result_df.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")

    return result_df

def process_az_clearance(input_files, output_dir):
    """Process AZ clearance dataset (combine train/val/test)."""
    print("\n" + "="*60)
    print("Processing AZ clearance dataset")
    print("="*60)

    dfs = []
    for f in input_files:
        df = pd.read_csv(f)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Raw shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Check if values are log-transformed
    y_values = df['drug_clearance'].dropna()
    y_min = y_values.min()
    y_max = y_values.max()

    is_log = False
    if y_min > 0 and y_max < 10:
        is_log = True
        print(f"Values appear to be log10-transformed (range: {y_min:.2f} to {y_max:.2f})")
    else:
        print(f"Values appear to be raw (range: {y_min:.2f} to {y_max:.2f})")

    # AZ clearance is hepatocyte data
    unit = 'µL/min/10^6 cells'

    standardized = []
    parse_errors = 0

    for _, row in df.iterrows():
        smiles = row.get('SMILES')
        value = row.get('drug_clearance')

        if pd.isna(smiles) or pd.isna(value):
            continue

        std_row = standardize_row(
            smiles=smiles,
            value=value,
            source='AZ_clearance',
            system='hepatocyte',
            unit=unit,
            is_log=is_log
        )

        if std_row['InChIKey'] is None:
            parse_errors += 1

        standardized.append(std_row)

    result_df = pd.DataFrame(standardized)
    print(f"\nProcessed: {len(result_df)} rows")
    print(f"Parse errors: {parse_errors}")
    print(f"Unique InChIKeys: {result_df['InChIKey'].nunique()}")

    output_file = output_dir / 'AZ_clearance.csv'
    result_df.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")

    return result_df

def main():
    sources_dir = Path('/gpfs/home1/s9383/harvest_clint/sources')
    processed_dir = Path('/gpfs/home1/s9383/harvest_clint/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_data = []

    # Process ExpansionRX HLM
    try:
        df_hlm = process_expansionrx_hlm(
            sources_dir / 'ExpansionRX_HLM_train.csv',
            processed_dir
        )
        all_data.append(df_hlm)
    except Exception as e:
        print(f"ERROR processing ExpansionRX: {e}")
        import traceback
        traceback.print_exc()

    # Process TDC Hepatocyte
    try:
        df_hep = process_tdc_hepatocyte(
            sources_dir / 'TDC_hepatocyte_raw.csv',
            processed_dir
        )
        all_data.append(df_hep)
    except Exception as e:
        print(f"ERROR processing TDC Hepatocyte: {e}")
        import traceback
        traceback.print_exc()

    # Process AZ Clearance
    try:
        az_files = [
            sources_dir / 'AZ_clearance_raw_train.csv',
            sources_dir / 'AZ_clearance_raw_val.csv',
            sources_dir / 'AZ_clearance_raw_test.csv'
        ]
        df_az = process_az_clearance(az_files, processed_dir)
        all_data.append(df_az)
    except Exception as e:
        print(f"ERROR processing AZ clearance: {e}")
        import traceback
        traceback.print_exc()

    # Combine all data
    print("\n" + "="*60)
    print("COMBINING ALL DATA")
    print("="*60)

    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        print(f"Total rows: {len(master_df)}")
        print(f"Total unique InChIKeys: {master_df['InChIKey'].nunique()}")

        # Deduplication by InChIKey
        print("\nDeduplicating by InChIKey...")
        print("Rule: Keep all rows, different systems are kept separate")

        master_output = processed_dir / 'human_clint_master.csv'
        master_df.to_csv(master_output, index=False)
        print(f"Saved master file to: {master_output}")

        # Summary statistics
        print("\n" + "="*60)
        print("SUMMARY BY SOURCE")
        print("="*60)
        for source in master_df['source'].unique():
            df_src = master_df[master_df['source'] == source]
            print(f"\n{source}:")
            print(f"  Rows: {len(df_src)}")
            print(f"  Unique InChIKeys: {df_src['InChIKey'].nunique()}")
            print(f"  System: {df_src['system'].unique()}")

if __name__ == '__main__':
    main()
