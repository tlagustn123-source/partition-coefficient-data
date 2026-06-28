#!/usr/bin/env python3
# =============================================================================
# Boltz-2  (POCKET-CONDITIONED, Sudlow site I)  ->  K_BSA/w  confound test
# =============================================================================
#
# WHY THIS SCRIPT EXISTS
#   The first (unconstrained) Boltz-2 run let Boltz pick the binding location
#   itself and FAILED (calibrated R2 ~ 0.045, Spearman ~ -0.21 -> no usable
#   signal). Two explanations are possible:
#       (i)  Boltz placed ligands in inconsistent / wrong locations  -> a SITE
#            problem we can fix.
#       (ii) Boltz simply cannot rank these neutral-organic / nonspecific
#            albumin affinities -> an applicability-domain (AD) problem we
#            cannot fix by changing the site.
#
#   This script removes explanation (i) as a confound: it FORCES every ligand
#   into albumin's main drug pocket, Sudlow site I (subdomain IIA, anchored on
#   Trp213), using a pocket constraint with force=true, then re-scores.
#
#   READ THE RESULT LIKE THIS:
#       * still ~0  -> site was NOT the problem -> AD confirmed -> STOP.
#       * jumps up  -> site selection WAS the confound -> multi-site worth doing.
#
# -----------------------------------------------------------------------------
# SETUP  (run once on the GPU machine; internet required first time)
#   # pip install torch --index-url https://download.pytorch.org/whl/cu121
#   pip install boltz -U
#   pip install pandas numpy scipy matplotlib requests pyyaml scikit-learn
#   python -c "import torch; print(torch.cuda.is_available())"   # must be True
#
# RUN
#   # place albumin_data.csv next to this script, then:
#   python boltz2_pocket_sudlow1.py
#   # ~1-3 min/compound -> ~1-4 h for 83 compounds
#
# NOTES / GOTCHAS (verified against Boltz docs + GitHub issues)
#   * force=true is REQUIRED. Without it the pocket constraint is sometimes
#     ignored and the ligand drifts away from the contact (issue #418).
#   * contacts indices are 1-based on the SEQUENCE we feed Boltz (the mature
#     BSA chain), NOT PDB numbering. We auto-verify the anchor really is Trp.
#   * affinity_pred_value is taken as log10(IC50 / 1 uM), lower = stronger.
#     The Spearman/trend result is UNIT-ROBUST and independent of this.
# =============================================================================

import os, json, glob, subprocess, sys, math
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yaml

# ----------------------------- CONFIG ----------------------------------------
HERE          = Path(__file__).resolve().parent
DATA_CSV      = HERE / "albumin_data.csv"          # name, cas, logK_exp, smiles
INPUT_DIR     = HERE / "boltz_pocket_inputs"
OUT_DIR       = HERE / "boltz_pocket_out"
RESULT_CSV    = HERE / "boltz2_pocket_results.csv"
PLOT_PNG      = HERE / "boltz2_pocket_fit.png"

UNIPROT_BSA   = "P02769"        # Bovine serum albumin
STRIP_SIGNAL  = 24              # remove signal(1-18)+pro(19-24) -> mature chain
M_BSA_KG_MOL  = 66.463          # BSA molar mass [kg/mol]; log10 -> 1.823 (the "1.83")
N_SITES       = 1               # single-site default (absolute scale only)
TEMP_K        = 310.15          # 37 C

# --- Sudlow site I pocket constraint ---
SUDLOW1_TRP_1BASED = 213        # mature-BSA Trp213 (subdomain IIA). Auto-verified below.
POCKET_MAX_DIST    = 10.0       # Angstrom (4-20 allowed). Generous: anchor defines a region.
POCKET_FORCE       = True       # MUST be True (see issue #418)

USE_MSA_SERVER     = True
DIFFUSION_SAMPLES  = 1
# -----------------------------------------------------------------------------


def fetch_bsa_sequence() -> str:
    url = f"https://rest.uniprot.org/uniprotkb/{UNIPROT_BSA}.fasta"
    r = requests.get(url, timeout=30); r.raise_for_status()
    seq = "".join(line.strip() for line in r.text.splitlines() if not line.startswith(">"))
    mature = seq[STRIP_SIGNAL:]
    print(f"[BSA] precursor {len(seq)} aa -> mature chain {len(mature)} aa")
    return mature


def resolve_site1_anchor(bsa_seq: str) -> int:
    """
    Return the 1-based index (on the mature sequence we give Boltz) of the
    Sudlow-I tryptophan. Robust to small numbering offsets: if position 213 is
    not W, fall back to the Trp located in the 200-225 window (BSA has only two
    Trp: ~134 in IB and ~213 in IIA = site I).
    """
    trps = [i + 1 for i, a in enumerate(bsa_seq) if a == "W"]   # 1-based
    print(f"[site I] tryptophans found at (1-based): {trps}")
    if 0 < SUDLOW1_TRP_1BASED <= len(bsa_seq) and bsa_seq[SUDLOW1_TRP_1BASED - 1] == "W":
        anchor = SUDLOW1_TRP_1BASED
    else:
        window = [t for t in trps if 200 <= t <= 225]
        anchor = window[0] if window else (trps[-1] if trps else SUDLOW1_TRP_1BASED)
        print(f"[site I] position {SUDLOW1_TRP_1BASED} is not Trp; using {anchor} instead")
    print(f"[site I] anchoring pocket on residue {anchor} ({bsa_seq[anchor-1]}) = Sudlow I")
    return anchor


def write_yaml(idx: int, smiles: str, bsa_seq: str, anchor: int) -> Path:
    """BSA (A) + ligand (B) + Sudlow-I pocket constraint + affinity request."""
    rec = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": bsa_seq}},
            {"ligand":  {"id": "B", "smiles": smiles}},
        ],
        "constraints": [
            {"pocket": {
                "binder": "B",
                "contacts": [["A", anchor]],     # Sudlow I anchor (Trp213)
                "max_distance": POCKET_MAX_DIST,
                "force": POCKET_FORCE,
            }}
        ],
        "properties": [{"affinity": {"binder": "B"}}],
    }
    p = INPUT_DIR / f"cmp{idx:03d}.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(rec, f)
    return p


def run_boltz():
    cmd = ["boltz", "predict", str(INPUT_DIR),
           "--out_dir", str(OUT_DIR),
           "--diffusion_samples", str(DIFFUSION_SAMPLES),
           "--no_kernels"]
    if USE_MSA_SERVER:
        cmd.append("--use_msa_server")
    print("[boltz] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_affinity(idx: int):
    hits = glob.glob(str(OUT_DIR / "**" / f"affinity_cmp{idx:03d}.json"), recursive=True)
    if not hits:
        return None, None
    with open(hits[0]) as f:
        d = json.load(f)
    val  = d.get("affinity_pred_value")
    prob = d.get("affinity_probability_binary")
    return (float(val) if val is not None else None,
            float(prob) if prob is not None else None)


def affinity_to_logKpw(aff_value: float) -> float:
    """affinity_pred_value = log10(IC50/uM) -> Kd[M] -> K_BSA/w = N/(Kd*M_BSA)."""
    Kd_M = (10.0 ** aff_value) * 1e-6
    K_pw = N_SITES / (Kd_M * M_BSA_KG_MOL)
    return math.log10(K_pw)


def main():
    df = pd.read_csv(DATA_CSV)
    assert {"name", "smiles", "logK_exp"} <= set(df.columns), "albumin_data.csv missing columns"
    INPUT_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)

    bsa = fetch_bsa_sequence()
    anchor = resolve_site1_anchor(bsa)

    print(f"[input] writing {len(df)} pocket-conditioned YAMLs (Sudlow I, force={POCKET_FORCE}) ...")
    for i, row in df.iterrows():
        write_yaml(i, row["smiles"], bsa, anchor)

    run_boltz()

    rows = []
    for i, row in df.iterrows():
        aff, prob = parse_affinity(i)
        logKpw = affinity_to_logKpw(aff) if aff is not None else np.nan
        rows.append({"name": row["name"], "logK_exp": row["logK_exp"],
                     "boltz_affinity": aff, "binder_prob": prob,
                     "logKpw_pred": logKpw})
    res = pd.DataFrame(rows)
    res.to_csv(RESULT_CSV, index=False)
    print(f"[save] {RESULT_CSV}")

    ok = res.dropna(subset=["boltz_affinity", "logKpw_pred"])
    if len(ok) < 3:
        print(f"[warn] only {len(ok)} parsed; check Boltz output. Stopping.")
        return

    from scipy.stats import spearmanr, pearsonr
    from sklearn.metrics import r2_score, mean_squared_error

    sp = spearmanr(ok["boltz_affinity"], ok["logK_exp"]).correlation
    pr = pearsonr(-ok["boltz_affinity"], ok["logK_exp"])[0]
    r2   = r2_score(ok["logK_exp"], ok["logKpw_pred"])
    rmse = math.sqrt(mean_squared_error(ok["logK_exp"], ok["logKpw_pred"]))
    a, b = np.polyfit(ok["logKpw_pred"], ok["logK_exp"], 1)
    cal  = a * ok["logKpw_pred"] + b
    r2c  = r2_score(ok["logK_exp"], cal)
    rmsec= math.sqrt(mean_squared_error(ok["logK_exp"], cal))

    print("\n=========== POCKET-CONDITIONED RESULTS (Sudlow I, n=%d) ===========" % len(ok))
    print(f"(1) trend       Spearman rho = {sp:+.3f}   Pearson(-aff,exp) = {pr:+.3f}")
    print(f"(2) absolute    R2 = {r2:.3f}   RMSE = {rmse:.3f} log units")
    print(f"(3) calibrated  R2 = {r2c:.3f}  RMSE = {rmsec:.3f}")
    print("-------------------------------------------------------------------")
    print("  vs unconstrained run: Spearman -0.211, calibrated R2 0.045")
    print("  still ~0  -> site was NOT the problem -> AD confirmed -> STOP")
    print("  jumps up  -> site selection WAS a confound -> try multi-site")
    print("===================================================================\n")

    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(ok["boltz_affinity"], ok["logK_exp"], s=40, c="#7570b3", edgecolor="w")
    ax[0].set_xlabel("Boltz-2 affinity_pred_value (lower = stronger)")
    ax[0].set_ylabel("Experimental log K$_{BSA/w}$")
    ax[0].set_title(f"Pocket-conditioned (Sudlow I)\nSpearman rho = {sp:+.2f}")
    ax[0].grid(alpha=.25)
    lo = min(ok["logK_exp"].min(), ok["logKpw_pred"].min())
    hi = max(ok["logK_exp"].max(), ok["logKpw_pred"].max())
    ax[1].plot([lo, hi], [lo, hi], "k-", lw=1)
    ax[1].scatter(ok["logK_exp"], ok["logKpw_pred"], s=40, c="#2c7fb8", edgecolor="w")
    ax[1].set_xlabel("Experimental log K$_{BSA/w}$")
    ax[1].set_ylabel("Predicted log K$_{BSA/w}$")
    ax[1].set_title(f"Absolute\nR2 = {r2:.2f}, RMSE = {rmse:.2f}")
    ax[1].grid(alpha=.25)
    plt.tight_layout(); plt.savefig(PLOT_PNG, dpi=150, bbox_inches="tight")
    print(f"[save] {PLOT_PNG}")


if __name__ == "__main__":
    main()
