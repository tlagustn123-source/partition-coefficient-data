#!/usr/bin/env python3
# =============================================================================
# Boltz-2  ->  BSA-water partition coefficient (K_BSA/w)  validation pipeline
# =============================================================================
#
# WHAT THIS DOES
#   For each of 83 neutral organic compounds (Endo & Goss 2011 BSA dataset):
#     1. co-fold BSA + ligand with Boltz-2 and predict binding affinity
#     2. convert the predicted affinity (Kd proxy) -> K_BSA/w  using
#            K_BSA/w = n * Ka / M_BSA = n / (Kd * M_BSA)
#     3. compare predicted vs experimental log K_BSA/w (R2, RMSE, Spearman)
#
#   Boltz-2 is a PRE-TRAINED foundation model: we only run inference, we do
#   not train anything here. The "model" is Boltz-2 (frozen); the only thing
#   built around it is the Kd->K_pw conversion + evaluation.
#
# -----------------------------------------------------------------------------
# SETUP  (run once on the GPU machine)
# -----------------------------------------------------------------------------
#   # CUDA-enabled PyTorch must be present first (match your CUDA version):
#   #   pip install torch --index-url https://download.pytorch.org/whl/cu121
#   pip install boltz -U
#   pip install pandas numpy scipy matplotlib requests pyyaml
#
#   # GPU check:
#   python -c "import torch; print(torch.cuda.is_available())"   # must print True
#
# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------
#   # place albumin_data.csv next to this script, then:
#   python boltz2_albumin_pipeline.py
#
#   Internet is required the first time (Boltz-2 weights download from
#   HuggingFace; --use_msa_server queries the public MMseqs2 MSA server).
#   Expect roughly 1-3 min/compound on a single modern GPU -> ~1-4 h total.
#
# -----------------------------------------------------------------------------
# !!! TWO THINGS TO VERIFY / TUNE  (these decide absolute accuracy) !!!
#   (A) AFFINITY UNIT  -- Boltz-2 reports `affinity_pred_value`. Per the Boltz-2
#       docs this is log10(IC50) with IC50 in micromolar (lower = stronger).
#       VERIFY against your installed Boltz-2 version before trusting absolute
#       numbers. The conversion function below states this assumption explicitly.
#       (The rank/Spearman result is unit-robust and does NOT depend on this.)
#   (B) SITE COUNT n  -- albumin has multiple binding sites; Boltz-2 gives ONE
#       dominant-pose affinity. n below is a single-site default. The absolute
#       scale shifts with n; the trend does not.
# =============================================================================

import os, json, glob, subprocess, sys, math
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yaml

# ----------------------------- CONFIG ----------------------------------------
HERE          = Path(__file__).resolve().parent
DATA_CSV      = HERE / "albumin_data.csv"      # columns: name, cas, logK_exp, smiles
INPUT_DIR     = HERE / "boltz_inputs"          # generated YAMLs
OUT_DIR       = HERE / "boltz_out"             # Boltz-2 predictions
RESULT_CSV    = HERE / "boltz2_results.csv"
PLOT_PNG      = HERE / "boltz2_albumin_fit.png"

UNIPROT_BSA   = "P02769"        # Bovine serum albumin (Endo & Goss measured BSA)
STRIP_SIGNAL  = 24              # remove signal(1-18)+pro(19-24) peptide -> mature chain (UniProt P02769)
M_BSA_KG_MOL  = 66.463          # BSA molar mass [kg/mol]
N_SITES       = 1               # (B) adjustable; single-site default
TEMP_K        = 310.15          # 37 C (matches experimental KBSA/w)

USE_MSA_SERVER = True           # query public MMseqs2 server (needs internet)
DIFFUSION_SAMPLES = 1           # 1 is fine for affinity ranking; increase for structure quality
# -----------------------------------------------------------------------------


def fetch_bsa_sequence() -> str:
    """Download BSA sequence from UniProt and return the mature chain."""
    url = f"https://rest.uniprot.org/uniprotkb/{UNIPROT_BSA}.fasta"
    r = requests.get(url, timeout=30); r.raise_for_status()
    seq = "".join(line.strip() for line in r.text.splitlines() if not line.startswith(">"))
    mature = seq[STRIP_SIGNAL:]
    print(f"[BSA] fetched {len(seq)} aa precursor -> mature chain {len(mature)} aa")
    return mature


def write_yaml(idx: int, smiles: str, bsa_seq: str) -> Path:
    """One Boltz-2 input YAML: BSA (chain A) + ligand (chain B) + affinity request."""
    rec = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": bsa_seq}},
            {"ligand":  {"id": "B", "smiles": smiles}},
        ],
        "properties": [{"affinity": {"binder": "B"}}],
    }
    p = INPUT_DIR / f"cmp{idx:03d}.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(rec, f, sort_keys=False)
    return p


def run_boltz():
    """Batch-predict over the whole input directory."""
    cmd = ["boltz", "predict", str(INPUT_DIR),
           "--out_dir", str(OUT_DIR),
           "--diffusion_samples", str(DIFFUSION_SAMPLES)]
    if USE_MSA_SERVER:
        cmd.append("--use_msa_server")
    print("[boltz] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_affinity(idx: int):
    """Find affinity_cmpXXX.json under OUT_DIR and return predicted value + binder prob."""
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
    """
    Convert Boltz-2 affinity_pred_value -> log10(K_BSA/w).

    ASSUMPTION (verify, see header note A):
        affinity_pred_value = log10(IC50 / 1 uM),  lower = stronger binder.
    Steps:
        IC50[M] = 10**(aff_value) * 1e-6 ;  Kd ~= IC50  (proxy)
        K_BSA/w = N_SITES * Ka / M_BSA = N_SITES / (Kd * M_BSA)   [L/kg]
    """
    Kd_M = (10.0 ** aff_value) * 1e-6
    K_pw = N_SITES / (Kd_M * M_BSA_KG_MOL)
    return math.log10(K_pw)


def main():
    df = pd.read_csv(DATA_CSV)
    assert {"name", "smiles", "logK_exp"} <= set(df.columns), "albumin_data.csv missing columns"
    INPUT_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)

    bsa = fetch_bsa_sequence()

    print(f"[input] writing {len(df)} Boltz-2 YAMLs ...")
    for i, row in df.iterrows():
        write_yaml(i, row["smiles"], bsa)

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
        print(f"[warn] only {len(ok)} compounds parsed; check Boltz output. Stopping before metrics.")
        return

    from scipy.stats import spearmanr, pearsonr
    from sklearn.metrics import r2_score, mean_squared_error

    # (1) trend capture -- UNIT-ROBUST: does Boltz rank binders correctly?
    #     stronger binding (lower affinity_pred_value) should mean higher K -> negative corr
    sp = spearmanr(ok["boltz_affinity"], ok["logK_exp"]).correlation
    pr = pearsonr(-ok["boltz_affinity"], ok["logK_exp"])[0]

    # (2) absolute agreement after physical conversion (depends on assumptions A,B)
    r2   = r2_score(ok["logK_exp"], ok["logKpw_pred"])
    rmse = math.sqrt(mean_squared_error(ok["logK_exp"], ok["logKpw_pred"]))

    # (3) calibrated: allow one linear scale+offset -> separates trend from scale
    a, b = np.polyfit(ok["logKpw_pred"], ok["logK_exp"], 1)
    cal  = a * ok["logKpw_pred"] + b
    r2c  = r2_score(ok["logK_exp"], cal)
    rmsec= math.sqrt(mean_squared_error(ok["logK_exp"], cal))

    print("\n================= RESULTS (n=%d parsed) =================" % len(ok))
    print(f"(1) trend       Spearman rho = {sp:+.3f}   Pearson(-aff,exp) = {pr:+.3f}")
    print(f"(2) absolute    R2 = {r2:.3f}   RMSE = {rmse:.3f} log units   (assumptions A,B)")
    print(f"(3) calibrated  R2 = {r2c:.3f}  RMSE = {rmsec:.3f}  (1-param scale+offset)")
    print("========================================================\n")

    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(ok["boltz_affinity"], ok["logK_exp"], s=40, c="#d95f02", edgecolor="w")
    ax[0].set_xlabel("Boltz-2 affinity_pred_value (lower = stronger)")
    ax[0].set_ylabel("Experimental log K$_{BSA/w}$")
    ax[0].set_title(f"Trend (unit-robust)\nSpearman rho = {sp:+.2f}")
    ax[0].grid(alpha=.25)
    lo = min(ok["logK_exp"].min(), ok["logKpw_pred"].min())
    hi = max(ok["logK_exp"].max(), ok["logKpw_pred"].max())
    ax[1].plot([lo, hi], [lo, hi], "k-", lw=1)
    ax[1].scatter(ok["logK_exp"], ok["logKpw_pred"], s=40, c="#2c7fb8", edgecolor="w")
    ax[1].set_xlabel("Experimental log K$_{BSA/w}$")
    ax[1].set_ylabel("Predicted log K$_{BSA/w}$ (converted)")
    ax[1].set_title(f"Absolute (assumptions A,B)\nR2 = {r2:.2f}, RMSE = {rmse:.2f}")
    ax[1].grid(alpha=.25)
    plt.tight_layout(); plt.savefig(PLOT_PNG, dpi=150, bbox_inches="tight")
    print(f"[save] {PLOT_PNG}")


if __name__ == "__main__":
    main()
