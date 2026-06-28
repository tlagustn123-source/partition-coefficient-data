#!/usr/bin/env python3
# =============================================================================
# vina_blind.py  --  AutoDock Vina BLIND docking over the whole BSA protein
# =============================================================================
#
#   Box covers the ENTIRE protein, so Vina chooses where the ligand binds
#   (your "let Vina find the site" idea). Same scoring -> Kd -> log K_BSA/w.
#
#   HONEST CAVEAT: albumin is a large protein (~66 kDa). A whole-protein search
#   box is big, so blind docking is SLOWER and LESS RELIABLE than the focused
#   run -- the global optimum over a huge volume is harder to find and the
#   chosen site may not be the experimentally dominant one. Use this mainly to
#   see (a) whether Vina lands in a real pocket at all, and (b) how its score
#   compares to the focused Sudlow-I run.
#
# RUN  (CPU only; needs internet once)
#   pip install vina numpy pandas scipy scikit-learn matplotlib requests
#   conda install -c conda-forge openbabel
#   # place albumin_data.csv + vina_common.py next to this script, then:
#   python vina_blind.py
#
#   Slower than focused (large box, exhaustiveness=32). Expect several hours.
# =============================================================================

from pathlib import Path
import pandas as pd
import vina_common as vc

HERE = Path(__file__).resolve().parent

def main():
    df = pd.read_csv(HERE / "albumin_data.csv")
    assert {"name", "smiles", "logK_exp"} <= set(df.columns)

    clean_pdb       = vc.prepare_receptor(HERE)              # download + clean + PDBQT
    center, box_sz  = vc.whole_protein_box(clean_pdb)        # whole-protein bounding box
    vc.run_docking("blind", HERE, df,
                   center=center, box_size=box_sz,
                   exhaustiveness=vc.EXHAUSTIVE_BLIND)

if __name__ == "__main__":
    main()
