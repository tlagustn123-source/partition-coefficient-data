#!/usr/bin/env python3
# =============================================================================
# vina_focused_sudlow1.py  --  AutoDock Vina docked AT Sudlow site I
# =============================================================================
#
#   Docks each of the 83 ligands into a box CENTERED on albumin's Sudlow site I
#   (subdomain IIA, anchored on Trp213) -- the SAME site as the pocket-
#   conditioned Boltz test, so Boltz vs Vina is an apples-to-apples comparison.
#
#   Vina score (kcal/mol) -> Kd -> log K_BSA/w  (see vina_common.py).
#
# RUN  (CPU only; needs internet once to fetch the BSA structure)
#   pip install vina numpy pandas scipy scikit-learn matplotlib requests
#   conda install -c conda-forge openbabel        # provides obabel CLI
#   # place albumin_data.csv + vina_common.py next to this script, then:
#   python vina_focused_sudlow1.py
#
#   ~83 ligands x focused box -> roughly tens of minutes to ~1-2 h on a
#   multi-core CPU (exhaustiveness=16).
# =============================================================================

from pathlib import Path
import pandas as pd
import vina_common as vc

HERE = Path(__file__).resolve().parent
BOX_SIZE = [22.5, 22.5, 22.5]      # A; Sudlow I is a large pocket

def main():
    df = pd.read_csv(HERE / "albumin_data.csv")
    assert {"name", "smiles", "logK_exp"} <= set(df.columns)

    clean_pdb = vc.prepare_receptor(HERE)              # download + clean + PDBQT
    center    = vc.find_site1_center(clean_pdb)        # centroid of Trp213
    vc.run_docking("focused_sudlow1", HERE, df,
                   center=center, box_size=BOX_SIZE,
                   exhaustiveness=vc.EXHAUSTIVE_FOCUSED)

if __name__ == "__main__":
    main()
