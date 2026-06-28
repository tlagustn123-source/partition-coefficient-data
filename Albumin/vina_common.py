#!/usr/bin/env python3
# =============================================================================
# vina_common.py  --  shared logic for the AutoDock Vina -> K_BSA/w experiments
# =============================================================================
#
#   Imported by:
#       vina_focused_sudlow1.py   (box fixed on Sudlow site I)
#       vina_blind.py             (box covers the whole protein)
#
#   Both scripts feed a Vina docking score (kcal/mol) through the SAME chain as
#   the Boltz pipeline so the three approaches are directly comparable:
#
#       Vina dG (kcal/mol)  -- thermodynamics -->  Kd
#                           Kd  ->  Ka = 1/Kd
#       log10 K_BSA/w = log10(Ka) - 1.83     (1.83 = log10 of BSA molar mass)
#                     = log10( N_SITES / (Kd * M_BSA) )
#
#   HONEST CAVEAT: the Vina score is an EMPIRICAL scoring function in
#   kcal/mol-like units; it correlates with, but is not, a calibrated binding
#   free energy. Treating it as dG (above) gives only a ROUGH absolute Kd. The
#   rank/Spearman result is what matters for "does Vina capture the trend?";
#   the calibrated R2 removes any constant scale/offset error in the score->dG
#   assumption, exactly as in the Boltz comparison.
#
# -----------------------------------------------------------------------------
# SETUP  (CPU only -- no GPU needed; internet needed to fetch the BSA structure)
#   pip install vina numpy pandas scipy scikit-learn matplotlib requests
#   # Open Babel CLI is used for structure/ligand prep:
#   #   conda install -c conda-forge openbabel    (recommended)
#   #   or: pip install openbabel-wheel           (provides the obabel CLI)
#   obabel -V        # confirm it is on PATH
# =============================================================================

import os, sys, math, shutil, subprocess, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

# ---- conversion constants (identical to the Boltz pipeline) ----
M_BSA_KG_MOL = 66.463          # log10 -> 1.823  (the "1.83" unit conversion)
N_SITES      = 1
TEMP_K       = 310.15          # 37 C, matches experimental K_BSA/w
R_KCAL       = 1.987204e-3     # gas constant, kcal/(mol K)

PDB_ID       = "4F5S"          # BSA crystal structure (override per-script if desired)
CHAIN        = "A"
EXHAUSTIVE_FOCUSED = 16
EXHAUSTIVE_BLIND   = 32        # blind needs more sampling (large search volume)


# ----------------------------- structure prep --------------------------------
def _require_obabel():
    if shutil.which("obabel") is None:
        sys.exit("ERROR: Open Babel CLI ('obabel') not found on PATH.\n"
                 "  conda install -c conda-forge openbabel   (or)   pip install openbabel-wheel")


def download_pdb(here: Path, pdb_id: str = PDB_ID) -> Path:
    out = here / f"{pdb_id}.pdb"
    if out.exists():
        return out
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"[recv] downloading {url}")
    urllib.request.urlretrieve(url, out)
    return out


def clean_receptor(pdb_in: Path, pdb_out: Path, chain: str = CHAIN) -> Path:
    """Keep only protein ATOM records of one chain; drop HETATM/water/altlocs."""
    kept = 0
    with open(pdb_in) as fi, open(pdb_out, "w") as fo:
        for ln in fi:
            if ln.startswith("ATOM") and (ln[21] == chain) and (ln[16] in (" ", "A")):
                fo.write(ln); kept += 1
        fo.write("END\n")
    print(f"[prep] cleaned receptor chain {chain}: {kept} atoms -> {pdb_out.name}")
    return pdb_out


def prep_receptor_pdbqt(pdb_clean: Path, pdbqt_out: Path) -> Path:
    _require_obabel()
    # -xr = rigid receptor PDBQT ; -p 7.4 = protonate at physiological pH
    subprocess.run(["obabel", str(pdb_clean), "-O", str(pdbqt_out), "-xr", "-p", "7.4"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[prep] receptor PDBQT -> {pdbqt_out.name}")
    return pdbqt_out


def prep_ligand_pdbqt(smiles: str, pdbqt_out: Path) -> bool:
    """SMILES -> 3D -> protonated PDBQT via Open Babel. Returns success."""
    try:
        subprocess.run(["obabel", f"-:{smiles}", "--gen3d", "-O", str(pdbqt_out), "-p", "7.4"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        return pdbqt_out.exists() and pdbqt_out.stat().st_size > 0
    except Exception as e:
        print(f"[lig] FAILED to prep {smiles!r}: {e}")
        return False


# ----------------------------- box helpers -----------------------------------
def _atoms(pdb_clean: Path):
    for ln in open(pdb_clean):
        if ln.startswith("ATOM"):
            yield ln


def residue_centroid(pdb_clean: Path, resseq: int, chain: str = CHAIN):
    pts = []
    for ln in _atoms(pdb_clean):
        if ln[21] == chain and ln[22:26].strip().isdigit() and int(ln[22:26]) == resseq:
            pts.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    if not pts:
        return None
    return np.mean(pts, axis=0)


def find_site1_center(pdb_clean: Path, chain: str = CHAIN):
    """
    Center of Sudlow site I. Prefer residue 213 if it is TRP; otherwise use the
    TRP in the 200-225 window (BSA's site-I tryptophan). Falls back to res 213.
    """
    # collect TRP residues
    trp_res = sorted({int(ln[22:26]) for ln in _atoms(pdb_clean)
                      if ln[17:20].strip() == "TRP" and ln[21] == chain and ln[22:26].strip().lstrip('-').isdigit()})
    print(f"[site I] TRP residues in chain {chain}: {trp_res}")
    target = 213
    if 213 not in trp_res:
        window = [r for r in trp_res if 200 <= r <= 225]
        target = window[0] if window else (trp_res[-1] if trp_res else 213)
        print(f"[site I] res 213 not TRP; using TRP {target}")
    c = residue_centroid(pdb_clean, target, chain)
    if c is None:
        sys.exit(f"ERROR: could not locate residue {target} in {pdb_clean}")
    print(f"[site I] box center on TRP{target}: ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})")
    return c


def whole_protein_box(pdb_clean: Path, pad: float = 6.0):
    pts = np.array([(float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
                    for ln in _atoms(pdb_clean)])
    lo, hi = pts.min(0), pts.max(0)
    center = (lo + hi) / 2.0
    size = (hi - lo) + pad
    print(f"[blind] whole-protein box center ({center[0]:.1f},{center[1]:.1f},{center[2]:.1f}) "
          f"size ({size[0]:.0f},{size[1]:.0f},{size[2]:.0f}) A")
    if max(size) > 40:
        print("[blind] WARNING: large search box -> slower and less reliable than focused docking.")
    return center, size


# ----------------------------- conversion ------------------------------------
def vina_to_logKpw(dG_kcal: float) -> float:
    """Vina score (kcal/mol) -> Kd -> log10 K_BSA/w (same chain as Boltz)."""
    Kd_M = math.exp(dG_kcal / (R_KCAL * TEMP_K))      # dG = RT ln Kd
    K_pw = N_SITES / (Kd_M * M_BSA_KG_MOL)
    return math.log10(K_pw)


# ----------------------------- docking + eval --------------------------------
def run_docking(mode: str, here: Path, df: pd.DataFrame,
                center, box_size, exhaustiveness: int):
    """Dock every ligand at the given box; score, convert, evaluate, plot."""
    from vina import Vina

    receptor_pdbqt = here / "receptor.pdbqt"
    lig_dir = here / f"vina_ligs_{mode}"; lig_dir.mkdir(exist_ok=True)

    v = Vina(sf_name="vina", cpu=0, seed=42, verbosity=0)
    v.set_receptor(rigid_pdbqt_filename=str(receptor_pdbqt))
    v.compute_vina_maps(center=[float(c) for c in center],
                        box_size=[float(s) for s in box_size])

    rows = []
    for i, row in df.iterrows():
        lig_pdbqt = lig_dir / f"lig{i:03d}.pdbqt"
        score = np.nan
        if prep_ligand_pdbqt(row["smiles"], lig_pdbqt):
            try:
                v.set_ligand_from_file(str(lig_pdbqt))
                v.dock(exhaustiveness=exhaustiveness, n_poses=5)
                score = float(v.energies(n_poses=1)[0][0])   # best-pose total (kcal/mol)
            except Exception as e:
                print(f"[dock] {row['name']}: FAILED ({e})")
        logKpw = vina_to_logKpw(score) if np.isfinite(score) else np.nan
        rows.append({"name": row["name"], "logK_exp": row["logK_exp"],
                     "vina_score": score, "logKpw_pred": logKpw})
        print(f"  {i+1:3d}/{len(df)}  {row['name'][:28]:28s}  score={score:7.2f}  logKpw_pred={logKpw:6.2f}")

    res = pd.DataFrame(rows)
    res_csv = here / f"vina_{mode}_results.csv"
    res.to_csv(res_csv, index=False); print(f"[save] {res_csv}")
    _evaluate(mode, here, res)
    return res


def _evaluate(mode: str, here: Path, res: pd.DataFrame):
    ok = res.dropna(subset=["vina_score", "logKpw_pred"])
    if len(ok) < 3:
        print(f"[warn] only {len(ok)} docked; skipping metrics."); return

    from scipy.stats import spearmanr, pearsonr
    from sklearn.metrics import r2_score, mean_squared_error

    # lower (more negative) score = stronger = higher K  -> negative Spearman if correct
    sp   = spearmanr(ok["vina_score"], ok["logK_exp"]).correlation
    pr   = pearsonr(-ok["vina_score"], ok["logK_exp"])[0]
    r2   = r2_score(ok["logK_exp"], ok["logKpw_pred"])
    rmse = math.sqrt(mean_squared_error(ok["logK_exp"], ok["logKpw_pred"]))
    a, b = np.polyfit(ok["logKpw_pred"], ok["logK_exp"], 1)
    cal  = a * ok["logKpw_pred"] + b
    r2c  = r2_score(ok["logK_exp"], cal)
    rmsec= math.sqrt(mean_squared_error(ok["logK_exp"], cal))

    print(f"\n=========== VINA ({mode}) RESULTS (n={len(ok)}) ===========")
    print(f"(1) trend       Spearman rho = {sp:+.3f}   Pearson(-score,exp) = {pr:+.3f}")
    print(f"(2) absolute    R2 = {r2:.3f}   RMSE = {rmse:.3f} log units")
    print(f"(3) calibrated  R2 = {r2c:.3f}  RMSE = {rmsec:.3f}")
    print( "  reference ceilings:  descriptor XGBoost R2~0.70 / pp-LFER R2~0.78")
    print( "  unconstrained Boltz: Spearman -0.211, calibrated R2 0.045")
    print("===========================================================\n")

    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(ok["vina_score"], ok["logK_exp"], s=40, c="#1b9e77", edgecolor="w")
    ax[0].set_xlabel("Vina score (kcal/mol; lower = stronger)")
    ax[0].set_ylabel("Experimental log K$_{BSA/w}$")
    ax[0].set_title(f"Vina {mode} -- trend\nSpearman rho = {sp:+.2f}")
    ax[0].grid(alpha=.25)
    lo = min(ok["logK_exp"].min(), ok["logKpw_pred"].min())
    hi = max(ok["logK_exp"].max(), ok["logKpw_pred"].max())
    ax[1].plot([lo, hi], [lo, hi], "k-", lw=1)
    ax[1].scatter(ok["logK_exp"], ok["logKpw_pred"], s=40, c="#2c7fb8", edgecolor="w")
    ax[1].set_xlabel("Experimental log K$_{BSA/w}$")
    ax[1].set_ylabel("Predicted log K$_{BSA/w}$")
    ax[1].set_title(f"Vina {mode} -- absolute\nR2 = {r2:.2f}, RMSE = {rmse:.2f}")
    ax[1].grid(alpha=.25)
    png = here / f"vina_{mode}_fit.png"
    plt.tight_layout(); plt.savefig(png, dpi=150, bbox_inches="tight")
    print(f"[save] {png}")


def prepare_receptor(here: Path, pdb_id: str = PDB_ID):
    """Download + clean + PDBQT. Returns the cleaned PDB path (for box building)."""
    raw   = download_pdb(here, pdb_id)
    clean = clean_receptor(raw, here / "receptor_clean.pdb")
    prep_receptor_pdbqt(clean, here / "receptor.pdbqt")
    return clean
