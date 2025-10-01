# =============================================================================
# METABRIC RUN — Deep Gaussian Process (DGP) Survival (clean; posterior-H grouping)
# TRAIN-only PC selection + optional GES; True-ELBO (discrete-time) or Cox-PL

#
# Outputs saved under: <BASE>/metabric_results
# =============================================================================

import os, math, warnings, inspect, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import gpytorch
from collections import defaultdict
from gpytorch.means import ConstantMean, LinearMean
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== LaTeX-style look without external LaTeX dependency =====
plt.rcParams.update(
    {
        "text.usetex": False,  # use mathtext (no system TeX needed)
        "font.family": "serif",
        "mathtext.fontset": "stix",  # LaTeX-like glyphs
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
    }
)

from sklearn.model_selection import StratifiedShuffleSplit
import sklearn
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression  # [A]

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index as _lifelines_c
from lifelines.statistics import multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts

# ---- SciPy (optional but recommended) ----
try:
    from scipy.stats import spearmanr, pearsonr  # ### NEW/UPDATED: import pearsonr
except Exception:
    spearmanr = None
    pearsonr = None
try:
    from scipy.stats import kruskal as kruskal_wallis
except Exception:
    kruskal_wallis = None
try:
    from scipy.stats import fisher_exact
except Exception:
    fisher_exact = None
try:
    from scipy.stats import mannwhitneyu
except Exception:
    mannwhitneyu = None


warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------ USER KNOBS ------------------------
SEED = 42
RUN_MULTI_SEEDS = False
MULTI_SEEDS = [42, 43, 44, 45, 46]

# ENDORSE-style split
TEST_FRAC = 0.50
VAL_FRAC = 0.0  # 0 disables validation/early-stop (kept as is)

# Toggles
USE_GES = True  # add ER/Proliferation gene-set scores
GES_METHOD = "ssgsea"  # "ssgsea" or "rankmean"
KMEANS_INIT = True  # k-means init for inducing points

# Expression prefilter (speeds PCA)
GENE_SELECT_MODE = "quantile"  # "quantile" or "count"
GENE_VAR_TOP_Q = 0.95
GENE_N_TOP = 1000

# PCA & survival-based PC selection
TARGET_EVR = 0.95
SURV_TOP_K = 22
MIN_PC = 8
MAX_PC = 40
KM_EXTEND_TO_300 = True  # extend KM curves flat to 300 mo for plot comparability

# DGP architecture
HIDDEN_DIM = 5
NUM_INDUCING = 64

# Training
EPOCHS = 1000
PRINT_EVERY = 50
GRAD_CLIP = 1.0

# ----- Variational Inference -----
MC_TRAIN = 32  # MC samples for E[loglik] during training
KL_WARMUP = int(0.30 * EPOCHS)
VI_DIST = "meanfield"  # "meanfield" or "cholesky"

# Optimizer LRs per parameter group
LR_VAR = 0.02  # variational dists
LR_IND = 0.01  # inducing locations
LR_HYP = 0.003  # kernels + means

# ----- Likelihood choice -----
USE_DISCRETE_TIME = True
DT_BINS = 12
DT_LINK = "logistic"  # or "cloglog"

# Posterior-survival grouping parameters
POSTERIOR_H_MONTHS = 120.0
TAU_HIGH = 0.6
TAU_LOW = 0.75  # changed rom 0.7
P_STAR = 0.75  # changed from 0.7

# Calibration horizons (months)
H_CAL_LIST = [60.0, 120.0]

# PD settings
PD_POINTS = 50
PD_ICE_NSAMP = 128
PD_QUANTILES = (0.10, 0.90)

# Reporting extras
EVAL_MC_SAMPLES = 32
BOOTSTRAP_B = 1000

# Outcome proxy thresholds [A]
EARLY_EVENT_MO = 24.0  # early event if event <= 24 mo
EVENT_FREE_MO = 120.0  # long-term event-free if censored past 120 mo

# File system — update BASE to your environment
BASE = Path(
    # insert your base path here
)
OUTDIR = BASE / # insert your output directory here 
OUTDIR.mkdir(parents=True, exist_ok=True)

P_CLIN_PAT = BASE / "data_clinical_patient.txt"
P_CLIN_SAMP = BASE / "data_clinical_sample.txt"
P_EXPR_MICRO = BASE / "data_mrna_illumina_microarray.txt"


WHITELIST_PATH = BASE / # insert patient ID list here
# -----------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------- helpers -----------------------
def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_table_guess(path, index_col=None):
    try:
        return pd.read_table(
            path, sep="\t", comment="#", index_col=index_col, low_memory=False
        )
    except Exception:
        return pd.read_csv(path, index_col=index_col, low_memory=False)


def ucase(s):
    return str(s).strip().upper()


def is_pos(v):
    return any(tok in ucase(v) for tok in ["POS", "POSITIVE", "1", "TRUE", "YES"])


def is_neg(v):
    return any(tok in ucase(v) for tok in ["NEG", "NEGATIVE", "0", "FALSE", "NO"])


# === EVENT MAPPING: 1 = DECEASED, 0 = LIVING ===
def os_event(v):
    u = ucase(v)
    if "DECEASED" in u or u.startswith("1"):
        return 1
    if "LIVING" in u or u.startswith("0"):
        return 0
    return 1 if ("TRUE" in u or "YES" in u) else 0


def savefig(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def dxy(c):
    return 2 * c - 1


# === C-index helper with auto-orientation ===
def harrell_c(times, scores, events, auto_orient=True):
    t = np.asarray(times, float)
    s = np.asarray(scores, float)
    e = np.asarray(events, int)
    c = float(_lifelines_c(t, s, e))
    if auto_orient and c < 0.5:
        c_flip = float(_lifelines_c(t, -s, e))
        if c_flip > c:
            return c_flip, True
    return c, False


def load_table_guess(path, index_col=None):
    path = Path(path)
    # Excel first
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, index_col=index_col, dtype=str)
    # Text/CSV with robust fallbacks
    try:
        return pd.read_table(
            path,
            sep="\t",
            comment="#",
            index_col=index_col,
            low_memory=False,
            encoding="utf-8",
        )
    except Exception:
        try:
            return pd.read_csv(
                path, index_col=index_col, low_memory=False, encoding="utf-8"
            )
        except UnicodeDecodeError:
            # final fallback for odd encodings
            return pd.read_csv(
                path, index_col=index_col, low_memory=False, encoding="latin1"
            )


def somers_dxy(c):
    return 2.0 * float(c) - 1.0


def orient_risk(times, events, scores):
    c_raw = _lifelines_c(times, scores, events)
    return -scores if c_raw < 0.5 else scores


def bootstrap_cindex(times, events, scores, B=BOOTSTRAP_B, seed=123):
    rng = np.random.default_rng(seed)
    n = len(times)
    stats = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        c, _ = harrell_c(times[idx], scores[idx], events[idx])
        stats.append(c)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(np.mean(stats)), float(lo), float(hi)


def bootstrap_c_and_dxy(times, events, scores, B=BOOTSTRAP_B, seed=123):
    rng = np.random.default_rng(seed)
    n = len(times)
    c_stats = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        c, _ = harrell_c(times[idx], scores[idx], events[idx])
        c_stats.append(c)
    d_stats = [2 * c - 1 for c in c_stats]
    c_lo, c_hi = np.percentile(c_stats, [2.5, 97.5])
    d_lo, d_hi = np.percentile(d_stats, [2.5, 97.5])
    return (
        float(np.mean(c_stats)),
        float(c_lo),
        float(c_hi),
        float(np.mean(d_stats)),
        float(d_lo),
        float(d_hi),
    )


def inv_softplus(x):
    return math.log(math.exp(float(x)) - 1.0)


def kernel_lengthscale_stats(kern):
    ls = kern.base_kernel.lengthscale.detach().flatten().cpu().numpy()
    return {
        "ls_med": float(np.median(ls)),
        "ls_min": float(np.min(ls)),
        "ls_max": float(np.max(ls)),
        "ls_n": int(ls.size),
        "outscale": float(kern.outputscale.detach().cpu().item()),
    }


def admin_censor(t: np.ndarray, e: np.ndarray, max_mo: float = 300.0):
    t = np.asarray(t, dtype=float).copy()
    e = np.asarray(e, dtype=int).copy()
    over = t > max_mo
    e[over] = 0
    t[over] = max_mo
    return t, e


def draw_guides(ax, horiz_at=0.5, vlines=(100, 150, 225), xmax=300):
    ax.axhline(horiz_at, ls="--")
    for v in vlines:
        ax.axvline(v, ls="--")
    ax.set_xlim(0, xmax)


# ---------------- Gene symbol cleaning ----------------
_token = re.compile(r"[A-Z0-9-]+")


def _canon_symbol(s: str) -> str:
    s = str(s).upper().strip()
    # split common multi-gene annotations, keep the first gene
    for sep in ("///", "//", "|", ";", "/", ","):
        if sep in s:
            s = s.split(sep)[0].strip()
    m = _token.search(s)
    return m.group(0) if m else s


def clean_and_collapse_genes(E: pd.DataFrame):
    """Uppercase symbols, split multi-gene fields, drop blanks, collapse dups (median)."""
    before = E.shape[0]
    sym = E.index.to_series().astype(str).apply(_canon_symbol)
    E2 = E.copy()
    E2.index = sym
    E2 = E2[~(E2.index.isna() | (E2.index == ""))]
    # drop obvious controls if present
    mask_affx = ~E2.index.str.startswith("AFFX")
    E2 = E2[mask_affx]
    E2 = E2.groupby(E2.index).median()
    after = E2.shape[0]
    print(
        f"Gene-symbol cleanup: {before} → {after} unique symbols (collapsed dups by median)."
    )
    return E2


# Small alias map to catch common alternatives on microarrays
ALIASES = {
    "AURKA": ["STK15"],
    "BIRC5": ["SURVIVIN"],
    "TOP2A": ["TOPIIA", "TOP2"],
}


def present_genes(genes, index_like):
    idx = set(index_like)
    out = []
    for g in genes:
        if g in idx:
            out.append(g)
        else:
            for alt in ALIASES.get(g, []):
                if alt in idx:
                    out.append(alt)
                    break
    return out


# -------------------  Core pipeline -------------------
def run_single(seed=SEED):
    set_all_seeds(seed)
    rng = np.random.default_rng(seed)

    print("\n====================== METABRIC RUN ======================")
    print(f"=== Seed {seed} ===")
    print("Loading clinical files…")
    pat = load_table_guess(P_CLIN_PAT)
    samp = load_table_guess(P_CLIN_SAMP)

    # key columns
    er_col = next(
        (c for c in ["ER_STATUS", "ER_IHC", "ER_STATUS_BY_IHC"] if c in samp.columns),
        None,
    )
    her2_col = next(
        (
            c
            for c in ["HER2_STATUS", "HER2_STATUS_BY_IHC", "HER2_SNP6"]
            if c in samp.columns
        ),
        None,
    )
    endo_col = next(
        (
            c
            for c in ["HORMONE_THERAPY", "ENDOCRINE_THERAPY", "HORMONE_THERAPY_STATUS"]
            if c in pat.columns
        ),
        None,
    )
    chemo_col = next((c for c in ["CHEMOTHERAPY"] if c in pat.columns), None)
    os_m_col = next(
        (c for c in ["OS_MONTHS", "OVERALL_SURVIVAL_MONTHS"] if c in pat.columns), None
    )
    os_s_col = next(
        (c for c in ["OS_STATUS", "VITAL_STATUS"] if c in pat.columns), None
    )

    # Merge clinical sample/patient tables
    merged = samp.merge(pat, on="PATIENT_ID", how="left", suffixes=("_SAMP", "_PAT"))

    # ======================== WHITELIST FILTER ========================
    def load_id_set(path):
        """
        Returns (whitelist_patient_ids, whitelist_sample_ids) as uppercase strings.
        Accepts .xlsx/.xls/.csv/.tsv/.txt or newline-delimited files.
        """
        path = Path(path)
        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path, dtype=str)
        elif path.suffix.lower() in (".csv", ".tsv"):
            try:
                df = pd.read_csv(path, dtype=str, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(path, dtype=str, encoding="latin1")
        elif path.suffix.lower() == ".txt":
            # try CSV first; if that fails, treat as one-ID-per-line
            try:
                df = pd.read_csv(path, dtype=str, sep=None, engine="python")
            except Exception:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    vals = {line.strip() for line in f if line.strip()}
                return {v.upper() for v in vals}, set()
        else:
            # generic newline-delimited fallback
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                vals = {line.strip() for line in f if line.strip()}
            return {v.upper() for v in vals}, set()

        # normalize headers and values
        df.columns = [str(c).upper().strip() for c in df.columns]
        for c in df.columns:
            df[c] = df[c].astype(str).str.strip()

        # find candidate columns
        pat_cols = [c for c in df.columns if "PATIENT" in c or c == "PATIENT_ID"]
        samp_cols = [c for c in df.columns if "SAMPLE" in c or c == "SAMPLE_ID"]

        wl_pat = set(df[pat_cols[0]].dropna().astype(str)) if pat_cols else set()
        wl_samp = set(df[samp_cols[0]].dropna().astype(str)) if samp_cols else set()

        # if the sheet is a single unnamed column, treat it as PATIENT_IDs
        if not wl_pat and not wl_samp and df.shape[1] >= 1:
            col0 = df.columns[0]
            wl_pat = set(df[col0].dropna().astype(str))

        return ({v.upper() for v in wl_pat}, {v.upper() for v in wl_samp})

    wl_pat, wl_samp = load_id_set(WHITELIST_PATH)

    merged["PATIENT_ID"] = merged["PATIENT_ID"].astype(str).str.strip()
    merged["SAMPLE_ID"] = merged["SAMPLE_ID"].astype(str).str.strip()

    if wl_samp:
        mask_wl = merged["SAMPLE_ID"].isin(wl_samp)
        print(f"[WHITELIST] Matching by SAMPLE_ID")
    else:
        mask_wl = merged["PATIENT_ID"].isin(wl_pat)
        print(f"[WHITELIST] Matching by PATIENT_ID")

    merged_wl = merged.loc[mask_wl].copy()
    print(f"[WHITELIST] Input rows: {len(merged)}")
    print(f"[WHITELIST] Matched rows (before de-dup): {int(mask_wl.sum())}")

    # one sample per patient
    merged_wl = merged_wl.sort_values(["PATIENT_ID", "SAMPLE_ID"]).drop_duplicates(
        "PATIENT_ID", keep="first"
    )
    print(f"[WHITELIST] Unique patients after de-dup: {len(merged_wl)}")

    # Optional debugging of missing IDs
    if wl_pat and not wl_samp:
        missing_pat = sorted(set(wl_pat) - set(merged["PATIENT_ID"]))
        print(f"[WHITELIST] Patient IDs not found in merged: {len(missing_pat)}")
        if len(missing_pat) > 0:
            pd.DataFrame({"missing_patient_id": missing_pat}).to_csv(
                OUTDIR / "missing_patients.csv", index=False
            )
    if wl_samp:
        missing_samp = sorted(set(wl_samp) - set(merged["SAMPLE_ID"]))
        print(f"[WHITELIST] Sample IDs not found in merged: {len(missing_samp)}")
        if len(missing_samp) > 0:
            pd.DataFrame({"missing_sample_id": missing_samp}).to_csv(
                OUTDIR / "missing_samples.csv", index=False
            )

    # Build cohort from whitelist
    cohort = merged_wl.copy()
    # Keep only rows with OS fields present
    cohort = cohort[cohort[os_m_col].notna() & cohort[os_s_col].notna()].copy()

    # Add event/time with explicit coding (1=DECEASED, 0=LIVING)
    cohort["OS_EVENT"] = cohort[os_s_col].map(os_event).astype(int)
    cohort["OS_MONTHS"] = pd.to_numeric(cohort[os_m_col], errors="coerce")

    # final safety de-dup
    cohort = cohort.sort_values(["PATIENT_ID", "SAMPLE_ID"]).drop_duplicates(
        "PATIENT_ID", keep="first"
    )
    dup_counts = cohort.groupby("PATIENT_ID")["SAMPLE_ID"].nunique()
    assert int(dup_counts.max()) == 1, "Found patients with >1 sample after filtering."
    print(f"Filtered cohort size = {len(cohort)}")

    # ------------------- Expression -------------------
    print("Loading microarray expression…")
    expr_raw = load_table_guess(P_EXPR_MICRO, index_col=0)
    expr_raw.columns = [c.split(":")[0].split(".")[0].strip() for c in expr_raw.columns]
    sel_samples = set(cohort["SAMPLE_ID"].astype(str))
    expr_raw = (
        expr_raw[[c for c in expr_raw.columns if c in sel_samples]]
        .copy()
        .apply(pd.to_numeric, errors="coerce")
    )

    # order columns to meta order
    cohort = cohort[cohort["SAMPLE_ID"].isin(expr_raw.columns)].copy()
    order = cohort.sort_values("PATIENT_ID")["SAMPLE_ID"].astype(str).tolist()
    expr_raw = expr_raw[order]

    # Clean symbols + collapse duplicates BEFORE any filtering
    expr_full = clean_and_collapse_genes(expr_raw)
    expr_full = expr_full.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # drop (near) constant genes
    std_raw = expr_full.std(axis=1, skipna=True)
    expr_full = expr_full.loc[std_raw > 1e-8]

    meta = cohort.sort_values("PATIENT_ID").reset_index(drop=True)
    print(
        f"Expression matrix (full): {expr_full.shape[0]} genes × {expr_full.shape[1]} samples"
    )

    # ------------------- Splits -------------------
    y_event_all = meta["OS_EVENT"].to_numpy()
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    ((trval_idx, te_idx),) = sss1.split(np.zeros(len(y_event_all)), y_event_all)

    if VAL_FRAC and VAL_FRAC > 0.0:
        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=VAL_FRAC / (1 - TEST_FRAC), random_state=seed
        )
        ((tr_idx, va_idx),) = sss2.split(
            np.zeros(len(trval_idx)), y_event_all[trval_idx]
        )
        tr_idx = trval_idx[tr_idx]
        va_idx = trval_idx[va_idx]
        HAS_VAL = True
    else:
        tr_idx = trval_idx
        va_idx = np.array([], dtype=int)
        HAS_VAL = False

    col_idx = np.arange(expr_full.shape[1])
    tr_cols = col_idx[tr_idx]
    va_cols = col_idx[va_idx] if HAS_VAL else np.array([], dtype=int)
    te_cols = col_idx[te_idx]

    # ------------------- TRAIN-only gene filter + z-score (for PCA only) -------------------
    var_train = expr_full.iloc[:, tr_cols].var(axis=1, skipna=True)
    if GENE_SELECT_MODE == "quantile":
        thresh = var_train.quantile(GENE_VAR_TOP_Q)
        keep_genes = var_train.index[var_train >= thresh]
    else:
        keep_genes = var_train.sort_values(ascending=False).index[:GENE_N_TOP]
    expr = expr_full.loc[keep_genes].copy()  # <- use this "expr" for PCA

    mu_g = expr.iloc[:, tr_cols].mean(axis=1)
    sd_g = expr.iloc[:, tr_cols].std(axis=1) + 1e-8
    expr = expr.sub(mu_g, axis=0).div(sd_g, axis=0)

    # ------------------- GES (ssGSEA or rank-mean) -------------------
    ER_SET = ["ESR1", "PGR", "FOXA1", "GREB1", "XBP1", "BCL2", "GATA3"]
    PROLIF_SET = ["MKI67", "PCNA", "TOP2A", "CCNB1", "BIRC5", "UBE2C", "CDC20", "AURKA"]

    def ges_rankmean(E_df, genes):
        if E_df.shape[0] == 0:
            return pd.Series(0.0, index=E_df.columns)
        present = present_genes(genes, E_df.index)
        if not present:
            return pd.Series(0.0, index=E_df.columns)
        ranks = E_df.rank(axis=0, method="average")  # low expr -> low rank
        G = float(E_df.shape[0])
        nranks = (ranks - 1.0) / max(G - 1.0, 1.0)  # 0..1
        return nranks.loc[present].mean(axis=0)

    def ges_ssgsea(E_df, genes, alpha=0.25):
        """
        Simple ssGSEA-style statistic (Barbie et al.): per-sample running-sum on ranked genes.
        We weight by rank^alpha (ranks computed on descending expression).
        """
        if E_df.shape[0] == 0:
            return pd.Series(0.0, index=E_df.columns)

        present = present_genes(genes, E_df.index)
        if not present:
            return pd.Series(0.0, index=E_df.columns)

        # high expression -> high rank
        ranks_all = E_df.rank(axis=0, ascending=False, method="average")

        S = set(present)
        out = []
        for col in E_df.columns:
            # sort by rank ascending (best at the end)
            r = ranks_all[col].sort_values(ascending=True)  # 1..G
            idx_is_hit = r.index.to_series().apply(lambda g: g in S).to_numpy()
            rvals = r.to_numpy(dtype=float)
            G = len(rvals)
            Nh = int(idx_is_hit.sum())
            Nm = G - Nh
            if Nh == 0:
                out.append(0.0)
                continue

            # Weighted running sum
            w = (rvals**alpha) * idx_is_hit
            Phit = np.cumsum(w) / (w.sum() + 1e-12)
            Pmiss = np.cumsum((~idx_is_hit).astype(float)) / max(Nm, 1)
            RS = Phit - Pmiss
            # take the maximum deviation
            ES = RS.max() if abs(RS.max()) >= abs(RS.min()) else RS.min()
            out.append(float(ES))
        return pd.Series(out, index=E_df.columns, dtype=float)

    if USE_GES:
        scorer = ges_ssgsea if GES_METHOD.lower() == "ssgsea" else ges_rankmean
        E_tr_full = expr_full.iloc[:, tr_cols]
        E_va_full = expr_full.iloc[:, va_cols] if HAS_VAL else expr_full.iloc[:, []]
        E_te_full = expr_full.iloc[:, te_cols]

        ges_ER_tr = scorer(E_tr_full, ER_SET).to_numpy()[:, None]
        ges_Pro_tr = scorer(E_tr_full, PROLIF_SET).to_numpy()[:, None]
        if HAS_VAL:
            ges_ER_va = scorer(E_va_full, ER_SET).to_numpy()[:, None]
            ges_Pro_va = scorer(E_va_full, PROLIF_SET).to_numpy()[:, None]
        else:
            ges_ER_va = np.zeros((0, 0))
            ges_Pro_va = np.zeros((0, 0))
        ges_ER_te = scorer(E_te_full, ER_SET).to_numpy()[:, None]
        ges_Pro_te = scorer(E_te_full, PROLIF_SET).to_numpy()[:, None]

        ER_present = present_genes(ER_SET, expr_full.index)
        PRO_present = present_genes(PROLIF_SET, expr_full.index)
        print(
            f"GES overlaps (train): ER={len(ER_present)}/{len(ER_SET)}, "
            f"Prolif={len(PRO_present)}/{len(PROLIF_SET)}"
        )
    else:
        ges_ER_tr = ges_Pro_tr = np.zeros((len(tr_cols), 0))
        ges_ER_va = ges_Pro_va = np.zeros((len(va_cols), 0))
        ges_ER_te = ges_Pro_te = np.zeros((len(te_cols), 0))

    # ------------------- PCA (TRAIN-only) -------------------
    print("Running PCA on TRAIN only…")
    Xtr_genes = expr.iloc[:, tr_cols].T.values
    Xva_genes = (
        expr.iloc[:, va_cols].T.values if HAS_VAL else np.zeros((0, expr.shape[0]))
    )
    Xte_genes = expr.iloc[:, te_cols].T.values

    mu_g_train = Xtr_genes.mean(axis=0, keepdims=True)
    Xtr0 = Xtr_genes - mu_g_train
    Xva0 = Xva_genes - mu_g_train if HAS_VAL else Xva_genes
    Xte0 = Xte_genes - mu_g_train

    U, S, Vt = np.linalg.svd(Xtr0, full_matrices=False)
    evr_train = (S**2) / (S**2).sum()
    cum = np.cumsum(evr_train)
    r_evr = int(np.searchsorted(cum, TARGET_EVR) + 1)
    r_evr = max(r_evr, MIN_PC)

    print(
        f"PCA(train): {r_evr} PCs reach {cum[r_evr-1]*100:.1f}% cumulative variance "
        f"(target {TARGET_EVR*100:.1f}%)."
    )

    Ttr_full = Xtr0 @ Vt.T
    Tva_full = Xva0 @ Vt.T if HAS_VAL else Xva0
    Tte_full = Xte0 @ Vt.T
    Ttr = Ttr_full[:, :r_evr]
    Tva = Tva_full[:, :r_evr] if HAS_VAL else Tva_full
    Tte = Tte_full[:, :r_evr]
    pc_names = [f"PC{i+1}" for i in range(r_evr)]
    evr_used = evr_train[:r_evr]

    # ------------------- Rank PCs by survival (TRAIN Only) -------------------
    def rank_pcs_by_cox_z(T_train, time, event, names):
        rows = []
        df_base = pd.DataFrame({"time": time, "event": event})
        for k in range(T_train.shape[1]):
            df = df_base.copy()
            df["x"] = T_train[:, k]
            try:
                cph = CoxPHFitter(penalizer=0.0)
                cph.fit(df, duration_col="time", event_col="event", show_progress=False)
                s = (
                    cph.summary.loc["x"]
                    if "x" in cph.summary.index
                    else cph.summary.iloc[0]
                )
                rows.append(
                    {
                        "pc": names[k],
                        "idx": k,
                        "z": float(s["z"]),
                        "p": float(s["p"]),
                        "coef": float(s["coef"]),
                    }
                )
            except Exception:
                rows.append({"pc": names[k], "idx": k, "z": 0.0, "p": 1.0, "coef": 0.0})
        return pd.DataFrame(rows).sort_values(by="z", key=np.abs, ascending=False)

    time_all = meta["OS_MONTHS"].to_numpy()
    event_all = meta["OS_EVENT"].to_numpy()
    t_tr = time_all[tr_idx]
    e_tr = event_all[tr_idx]
    t_va = time_all[va_idx] if HAS_VAL else np.array([], dtype=float)
    e_va = event_all[va_idx] if HAS_VAL else np.array([], dtype=int)
    t_te = time_all[te_idx]
    e_te = event_all[te_idx]

    pc_rank = rank_pcs_by_cox_z(Ttr, t_tr, e_tr, pc_names)
    pc_rank["evr_train"] = pc_rank["idx"].map(
        {i: evr_used[i] for i in range(len(evr_used))}
    )
    pc_rank.to_csv(OUTDIR / "pc_survival_ranking_train.csv", index=False)

    K = SURV_TOP_K if SURV_TOP_K is not None else min(30, r_evr)
    K = int(np.clip(K, MIN_PC, min(MAX_PC, r_evr)))
    keep_idxs = pc_rank.head(K)["idx"].to_numpy()

    print(
        f"PC selection: using top {len(keep_idxs)} by Cox Z out of {r_evr} PCA PCs "
        f"(bounds [{MIN_PC}, {min(MAX_PC, r_evr)}])."
    )

    Ttr_sel = Ttr[:, keep_idxs]
    Tva_sel = Tva[:, keep_idxs] if HAS_VAL else np.zeros((0, K))
    Tte_sel = Tte[:, keep_idxs]
    sel_names = [pc_names[i] for i in keep_idxs]
    print(f"PCs selected by survival Z: {len(sel_names)}")

    # ------------------- Clinical covariates (TRAIN-fitted encoders) -------------------
    NUMERIC_CLIN = [c for c in ["AGE_AT_DIAGNOSIS", "TUMOR_SIZE"] if c in meta.columns]
    CAT_CLIN = [c for c in ["GRADE", "INFERRED_MENOPAUSAL_STATE"] if c in meta.columns]

    def _train_fitted_ohe_specs(meta_df, tr_idx, cat_cols):
        specs = {}
        for col in cat_cols:
            s_tr = meta_df.iloc[tr_idx][col].astype(str).str.upper()
            d_tr = pd.get_dummies(s_tr, prefix=col)
            if d_tr.shape[1] > 1:
                d_tr = d_tr.iloc[:, 1:]  # drop-first to avoid collinearity
            specs[col] = list(d_tr.columns)
        return specs

    def _apply_ohe_with_specs(meta_df, col, cols_spec):
        s = meta_df[col].astype(str).str.upper()
        d = pd.get_dummies(s, prefix=col)
        if cols_spec:
            d = d.reindex(columns=cols_spec, fill_value=0)
        else:
            d = pd.DataFrame(index=meta_df.index)
        return d

    def build_clin_matrices_train_fitted(meta_df, tr_idx, va_idx, te_idx):
        blocks_all, names = [], []
        # numeric
        for c in NUMERIC_CLIN:
            s_all = pd.to_numeric(meta_df[c], errors="coerce")
            med = pd.to_numeric(meta_df.iloc[tr_idx][c], errors="coerce").median()
            s_all = s_all.fillna(med)
            blocks_all.append(s_all.to_numpy()[:, None])
            names.append(c)
        # categorical
        ohe_specs = _train_fitted_ohe_specs(meta_df, tr_idx, CAT_CLIN)
        for c in CAT_CLIN:
            d_all = _apply_ohe_with_specs(meta_df, c, ohe_specs[c])
            if d_all.shape[1] > 0:
                blocks_all.append(d_all.to_numpy())
                names += list(d_all.columns)
        X_all = np.hstack(blocks_all) if blocks_all else np.zeros((len(meta_df), 0))
        Xc_tr = X_all[tr_idx]
        Xc_va = (
            X_all[va_idx]
            if HAS_VAL
            else np.zeros((0, X_all.shape[1] if X_all.ndim == 2 else 0))
        )
        Xc_te = X_all[te_idx]
        return Xc_tr, Xc_va, Xc_te, names

    Xc_tr, Xc_va, Xc_te, clin_names = build_clin_matrices_train_fitted(
        meta, tr_idx, va_idx, te_idx
    )

    # ------------------- Final matrices + scaling (TRAIN stats) ----------------
    if USE_GES:
        Xtr = np.hstack([Ttr_sel, Xc_tr, ges_ER_tr, ges_Pro_tr])
        Xva = (
            np.hstack([Tva_sel, Xc_va, ges_ER_va, ges_Pro_va])
            if HAS_VAL
            else np.zeros((0, Ttr_sel.shape[1] + Xc_tr.shape[1] + 2))
        )
        Xte = np.hstack([Tte_sel, Xc_te, ges_ER_te, ges_Pro_te])
        feat_names = sel_names + clin_names + ["ges_ER", "ges_Prolif"]
    else:
        Xtr = np.hstack([Ttr_sel, Xc_tr])
        Xva = (
            np.hstack([Tva_sel, Xc_va])
            if HAS_VAL
            else np.zeros((0, Ttr_sel.shape[1] + Xc_tr.shape[1]))
        )
        Xte = np.hstack([Tte_sel, Xc_te])
        feat_names = sel_names + clin_names

    print(
        f"Feature dims — PCs={Ttr_sel.shape[1]}, Clin={Xc_tr.shape[1]}, GES={(2 if USE_GES else 0)}; total={Xtr.shape[1]}"
    )

    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-8
    Xtr_s = (Xtr - mu) / sd
    Xva_s = (Xva - mu) / sd if HAS_VAL else Xva
    Xte_s = (Xte - mu) / sd

    # Save design + outcomes
    pd.DataFrame(Xtr_s, columns=feat_names).to_csv(OUTDIR / "X_train.csv", index=False)
    if HAS_VAL:
        pd.DataFrame(Xva_s, columns=feat_names).to_csv(
            OUTDIR / "X_val.csv", index=False
        )
    pd.DataFrame(Xte_s, columns=feat_names).to_csv(OUTDIR / "X_test.csv", index=False)
    pd.DataFrame({"time": t_tr, "event": e_tr}).to_csv(
        OUTDIR / "y_train.csv", index=False
    )
    if HAS_VAL:
        pd.DataFrame({"time": t_va, "event": e_va}).to_csv(
            OUTDIR / "y_val.csv", index=False
        )
    pd.DataFrame({"time": t_te, "event": e_te}).to_csv(
        OUTDIR / "y_test.csv", index=False
    )

    # tensors
    Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva_s, dtype=torch.float32, device=device) if HAS_VAL else None
    Xte_t = torch.tensor(Xte_s, dtype=torch.float32, device=device)
    t_tr_t = torch.tensor(t_tr, dtype=torch.float32, device=device)
    t_va_t = torch.tensor(t_va, dtype=torch.float32, device=device) if HAS_VAL else None
    t_te_t = torch.tensor(t_te, dtype=torch.float32, device=device)
    e_tr_t = torch.tensor(e_tr, dtype=torch.long, device=device)
    e_va_t = torch.tensor(e_va, dtype=torch.long, device=device) if HAS_VAL else None
    e_te_t = torch.tensor(e_te, dtype=torch.long, device=device)

    # ------------------- DGP (one hidden layer, variational) -------------------
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import VariationalStrategy
    from gpytorch.variational import (
        CholeskyVariationalDistribution,
        MeanFieldVariationalDistribution,
    )
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.means import ZeroMean, LinearMean

    M_inducing = int(min(NUM_INDUCING, Xtr_t.size(0)))

    def make_vs(module, inducing, q, learn_inducing_locations=True, prefer_whiten=True):
        sig = inspect.signature(VariationalStrategy.__init__)
        if "whiten" in sig.parameters:
            return VariationalStrategy(
                module,
                inducing,
                q,
                learn_inducing_locations=learn_inducing_locations,
                whiten=prefer_whiten,
            )
        else:
            return VariationalStrategy(
                module, inducing, q, learn_inducing_locations=learn_inducing_locations
            )

    class HiddenLayer(ApproximateGP):
        def __init__(self, in_dim, hid_dim, Xref, seed):
            batch_shape = torch.Size([hid_dim])
            if KMEANS_INIT:
                with torch.no_grad():
                    km = KMeans(n_clusters=M_inducing, random_state=seed, n_init=10)
                    centers = km.fit(Xref.cpu().numpy()).cluster_centers_
                    inducing_base = torch.from_numpy(centers).to(
                        dtype=Xref.dtype, device=Xref.device
                    )
            else:
                idx = torch.randperm(Xref.size(0))[:M_inducing]
                inducing_base = Xref[idx]
            inducing = inducing_base.unsqueeze(0).expand(hid_dim, -1, -1).contiguous()

            q = (
                MeanFieldVariationalDistribution(M_inducing, batch_shape=batch_shape)
                if VI_DIST.lower() == "meanfield"
                else CholeskyVariationalDistribution(
                    M_inducing, batch_shape=batch_shape
                )
            )
            strat = make_vs(
                self, inducing, q, learn_inducing_locations=True, prefer_whiten=True
            )
            super().__init__(strat)

            self.mean_module = ZeroMean(batch_shape=batch_shape)
            self.covar_module = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=in_dim, batch_shape=batch_shape),
                batch_shape=batch_shape,
            )
            with torch.no_grad():
                raw_ls = torch.full_like(
                    self.covar_module.base_kernel.raw_lengthscale, inv_softplus(1.5)
                )
                self.covar_module.base_kernel.raw_lengthscale.copy_(raw_ls)
                self.covar_module.outputscale.copy_(
                    torch.tensor(1.5, device=Xref.device)
                )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    class OutputLayer(ApproximateGP):
        def __init__(self, hid_dim):
            inducing_h = torch.randn(M_inducing, hid_dim, device=device) * 0.05
            q = (
                MeanFieldVariationalDistribution(M_inducing)
                if VI_DIST.lower() == "meanfield"
                else CholeskyVariationalDistribution(M_inducing)
            )
            strat = make_vs(
                self, inducing_h, q, learn_inducing_locations=True, prefer_whiten=True
            )
            super().__init__(strat)
            self.mean_module = LinearMean(hid_dim)
            self.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=hid_dim))
            with torch.no_grad():
                raw_ls = torch.full_like(
                    self.covar_module.base_kernel.raw_lengthscale, inv_softplus(1.5)
                )
                self.covar_module.base_kernel.raw_lengthscale.copy_(raw_ls)
                self.covar_module.outputscale.copy_(torch.tensor(2.0, device=device))

        def forward(self, h):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(h), self.covar_module(h)
            )

    class DGP(gpytorch.models.deep_gps.DeepGP):
        def __init__(self, in_dim, hid_dim, Xref, seed):
            super().__init__()
            self.hidden = HiddenLayer(in_dim, hid_dim, Xref, seed)
            self.output = OutputLayer(hid_dim)
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            with torch.no_grad():
                self.likelihood.noise = 1e-3

        def forward(self, x):
            h_dist = self.hidden(x)
            h = h_dist.rsample().transpose(-1, -2).contiguous()  # [N, hid_dim]
            return self.output(h)

    # -------- Discrete-time utilities (TRUE ELBO) ----------
    def make_dt_masks(times_np, events_np, edges_np):
        t = torch.tensor(times_np, dtype=torch.float32, device=device)
        e = torch.tensor(events_np, dtype=torch.float32, device=device)
        edges_t = torch.tensor(edges_np, dtype=torch.float32, device=device)
        J = edges_t.numel() - 1
        j_star = torch.bucketize(t, edges_t[1:], right=False)
        j_star = torch.clamp(j_star, 0, J - 1)
        grid = torch.arange(J, device=device)
        at_risk = (grid[None, :] <= j_star[:, None]).float()
        event_mask = torch.zeros_like(at_risk)
        event_mask.scatter_(1, j_star[:, None], e[:, None])
        return at_risk, event_mask

    def expected_dt_ll_mc(model, X, at_risk, event_mask, alpha, S=MC_TRAIN):
        import torch.nn.functional as F

        def logs_logistic(eta):
            logh = -F.softplus(-eta)  # log(sigmoid(eta))
            log1m = -F.softplus(eta)  # log(1 - sigmoid(eta))
            return logh, log1m

        def logs_cloglog(eta):
            eta_cap = torch.where(
                eta > 20.0, torch.tensor(20.0, device=eta.device), eta
            )
            z = torch.exp(eta_cap)
            log1m = -z
            small = z < 1e-4
            logh = torch.empty_like(z)
            logh[small] = torch.log(z[small])
            logh[~small] = torch.log1p(-torch.exp(-z[~small]))
            return logh, log1m

        use_cloglog = DT_LINK.lower() == "cloglog"
        vals = []
        with gpytorch.settings.cholesky_jitter(1e-3):
            for _ in range(S):
                f = model(X).rsample().squeeze()  # [N]
                eta = f[:, None] + alpha[None, :]  # [N,J]
                logh, log1m = logs_cloglog(eta) if use_cloglog else logs_logistic(eta)
                term1 = torch.where(at_risk > 0, log1m, torch.zeros_like(log1m))
                term2 = torch.where(
                    event_mask > 0, (logh - log1m), torch.zeros_like(logh)
                )
                ll = term1.sum() + term2.sum()
                if not torch.isfinite(ll):
                    logh = torch.where(
                        torch.isfinite(logh), logh, torch.zeros_like(logh)
                    )
                    log1m = torch.where(
                        torch.isfinite(log1m), log1m, torch.zeros_like(log1m)
                    )
                    term1 = torch.where(at_risk > 0, log1m, torch.zeros_like(log1m))
                    term2 = torch.where(
                        event_mask > 0, (logh - log1m), torch.zeros_like(logh)
                    )
                    ll = term1.sum() + term2.sum()
                vals.append(ll)
        return torch.stack(vals).mean()

    # -------- Cox partial log-likelihood (pseudo) ----------
    def cox_partial_ll_breslow(risk, time, event):
        order = torch.argsort(time)
        t = time[order]
        e = event[order].float()
        r = risk[order]
        er = torch.exp(r)
        rs_sum = torch.flip(torch.cumsum(torch.flip(er, dims=[0]), dim=0), dims=[0])
        uniq_t, counts = torch.unique_consecutive(t, return_counts=True)
        starts = torch.cat(
            [
                torch.zeros(1, device=t.device, dtype=torch.long),
                torch.cumsum(counts, dim=0)[:-1],
            ]
        )
        gid = torch.repeat_interleave(
            torch.arange(len(uniq_t), device=t.device), counts
        )
        r_event_sum = torch.zeros(len(uniq_t), device=t.device)
        r_event_sum.scatter_add_(0, gid, e * r)
        d_by_t = torch.zeros(len(uniq_t), device=t.device)
        d_by_t.scatter_add_(0, gid, e)
        denom = rs_sum[starts]
        pll = (r_event_sum - d_by_t * torch.log(denom + 1e-12)).sum()
        return pll

    def expected_cox_pll_mc(model, X, time, event, S=MC_TRAIN):
        vals = []
        with gpytorch.settings.cholesky_jitter(1e-3):
            for _ in range(S):
                f = model(X).rsample().squeeze()
                vals.append(cox_partial_ll_breslow(f, time, event))
        return torch.stack(vals, 0).mean()

    # ---- KL utilities ----
    def total_kl_details(model):
        kl_h = model.hidden.variational_strategy.kl_divergence().sum()
        kl_o = model.output.variational_strategy.kl_divergence().sum()
        return kl_h, kl_o

    def beta_schedule(it):
        return min(1.0, float(it) / max(1, KL_WARMUP))

    # Instantiate model
    model = DGP(Xtr_t.size(1), HIDDEN_DIM, Xtr_t, seed).to(device)

    # Discrete-time binning & baseline parameters (TRUE ELBO)
    if USE_DISCRETE_TIME:
        raw_edges = np.quantile(t_tr, np.linspace(0, 1, DT_BINS + 1))
        raw_edges[0] = 0.0
        edges = np.unique(raw_edges)
        tmax = float(np.max(t_tr)) if len(t_tr) else 1.0
        if edges[-1] < tmax - 1e-12:
            edges = np.append(edges, tmax + 1e-6)
        edges[0] = 0.0
        if len(edges) - 1 < max(4, DT_BINS // 2):
            edges = np.linspace(0.0, tmax + 1e-6, DT_BINS + 1)
        bin_edges = edges
        AR_tr, EV_tr = make_dt_masks(t_tr, e_tr, bin_edges)
        if not (torch.isfinite(AR_tr).all() and torch.isfinite(EV_tr).all()):
            raise RuntimeError("Non-finite at_risk/event_mask.")
        alpha_dt = torch.nn.Parameter(torch.zeros(len(bin_edges) - 1, device=device))

    # ---- Optimizer groups ----
    var_params, ind_params = [], []
    from gpytorch.variational import VariationalStrategy as VS
    from gpytorch.variational import (
        CholeskyVariationalDistribution,
        MeanFieldVariationalDistribution,
    )

    for mod in model.modules():
        if isinstance(
            mod, (CholeskyVariationalDistribution, MeanFieldVariationalDistribution)
        ):
            var_params += list(mod.parameters())
        vs = getattr(mod, "variational_strategy", None)
        if isinstance(vs, VS):
            ind_params.append(vs.inducing_points)
    var_ids = {id(p) for p in var_params}
    ind_ids = {id(p) for p in ind_params}
    hyp_params = [p for p in model.parameters() if id(p) not in var_ids | ind_ids]
    optim_groups = [
        {"params": var_params, "lr": LR_VAR},
        {"params": ind_params, "lr": LR_IND},
        {"params": hyp_params, "lr": LR_HYP},
    ]
    if USE_DISCRETE_TIME:
        optim_groups.append({"params": [alpha_dt], "lr": 0.01})
    opt = torch.optim.Adam(optim_groups)

    # ---- MC & utility prediction helpers ----
    @torch.no_grad()
    def predict_risk_mc(X, S=EVAL_MC_SAMPLES, mode=None):
        model.eval()
        Fs = []
        with gpytorch.settings.cholesky_jitter(1e-3):
            for _ in range(S):
                Fs.append(model(X).rsample().squeeze())
        F = torch.stack(Fs, 0)  # [S, N]
        if not USE_DISCRETE_TIME and (mode is None or mode == "logexp"):
            r = torch.logsumexp(F, dim=0) - math.log(F.size(0))
        else:
            r = F.mean(0)
        s = F.std(0)
        return r.cpu().numpy(), s.cpu().numpy()

    @torch.no_grad()
    def posterior_survival_at_H(X, H_months: float, alpha, S=EVAL_MC_SAMPLES):
        if not USE_DISCRETE_TIME:
            return None
        model.eval()
        Fs = []
        with gpytorch.settings.cholesky_jitter(1e-3):
            for _ in range(S):
                Fs.append(model(X).rsample().squeeze())
        F = torch.stack(Fs, 0)  # [S,N]
        eta = F[:, :, None] + alpha[None, None, :]  # [S,N,J]
        if DT_LINK.lower() == "cloglog":
            h = 1.0 - torch.exp(-torch.exp(torch.clamp(eta, max=20.0)))
        else:
            h = torch.sigmoid(eta)
        surv_edges = torch.cumprod(1.0 - h, dim=2)
        j = np.searchsorted(bin_edges, H_months, side="right") - 1
        j = int(np.clip(j, 0, len(bin_edges) - 2))
        S_H = surv_edges[:, :, j]
        return S_H.cpu().numpy()

    @torch.no_grad()
    def neglogS_at_H(X, H_months: float, alpha, S=EVAL_MC_SAMPLES):
        if not USE_DISCRETE_TIME:
            return None
        SH = posterior_survival_at_H(X, H_months, alpha, S=S)  # [S,N]
        nlS = -np.log(np.clip(SH, 1e-10, 1.0))
        return nlS.mean(axis=0)

    # >>> Integrated Brier Score helper (IPCW)  --------------------------
    def brier_ipcw(times, events, preds_S, H):
        times = np.asarray(times, float)
        events = np.asarray(events, int)
        preds_S = np.asarray(preds_S, float)
        kmc = KaplanMeierFitter().fit(times, 1 - events)  # censoring KM, G(t)
        t_minus = np.maximum(times - 1e-8, 0.0)
        G_tminus = kmc.predict(pd.Series(np.minimum(t_minus, float(times.max()))))
        G_tminus = np.asarray(G_tminus, float)
        G_tminus = np.clip(G_tminus, 1e-6, None)
        G_H = kmc.predict(min(H, float(times.max())))
        G_H = float(np.clip(G_H, 1e-6, None))
        yH = (times > H).astype(float)
        w = np.where(times <= H, events / G_tminus, 1.0 / G_H)
        return float(np.mean(w * (yH - preds_S) ** 2))

    # >>> IBS using IPCW (integrate Brier over time up to tau)
    def ibs_ipcw(times, events, S_at_H_callable, tau, grid=None):
        times = np.asarray(times, float)
        events = np.asarray(events, int)
        tau = float(tau)
        if not np.isfinite(tau) or tau <= 0:
            return np.nan

        if grid is None:
            if USE_DISCRETE_TIME and "bin_edges" in globals():
                mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                grid = mids[(mids > 0.0) & (mids <= tau)]
            else:
                qs = np.linspace(0.02, 1.0, 100)
                finite_t = times[np.isfinite(times)]
                if finite_t.size == 0:
                    return np.nan
                grid = np.quantile(finite_t, qs)
                grid = grid[(grid > 0.0) & (grid <= tau)]
        grid = np.unique(np.clip(grid, 1e-6, tau))
        if grid.size == 0:
            return np.nan

        bs_vals = []
        for H in grid:
            S_H = S_at_H_callable(float(H))
            bs_vals.append(brier_ipcw(times, events, S_H, float(H)))
        bs_vals = np.asarray(bs_vals, float)

        ibs = np.trapz(bs_vals, grid) / tau
        return float(ibs)

    # ---- Training ----
    n_events_tr = int(e_tr.sum())
    n_events_va = int(e_va.sum()) if HAS_VAL else 0

    print("Starting training…")
    best_c, best_state = -1.0, None
    for it in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        if USE_DISCRETE_TIME:
            AR_tr, EV_tr = make_dt_masks(t_tr, e_tr, bin_edges)
            ell = expected_dt_ll_mc(model, Xtr_t, AR_tr, EV_tr, alpha_dt, S=MC_TRAIN)
            pll_or_ell = ell
        else:
            pll = expected_cox_pll_mc(model, Xtr_t, t_tr_t, e_tr_t, S=MC_TRAIN)
            pll_or_ell = pll

        kl_h, kl_o = total_kl_details(model)
        kl = kl_h + kl_o
        kl_beta = beta_schedule(it)
        elbo = pll_or_ell - kl_beta * kl
        loss = -elbo
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        if it % PRINT_EVERY == 0 or it == 1:
            n_ev = max(1, n_events_tr)
            ks_out = kernel_lengthscale_stats(model.output.covar_module)
            like_name = "ELL" if USE_DISCRETE_TIME else "PLL"
            msg = (
                f"[{it:04d}] {like_name}/evt={float(pll_or_ell.item())/n_ev:.4f} | "
                f"KL/evt={(kl.item()/n_ev):.4f} | KLhid={(kl_h.item()/n_ev):.4f} | KLout={(kl_o.item()/n_ev):.4f} "
                f"| beta={kl_beta:.3f} | ls_out_med={ks_out['ls_med']:.2f} | outscale={ks_out['outscale']:.2f}"
            )
            if HAS_VAL:
                model.eval()
                with torch.no_grad():
                    if USE_DISCRETE_TIME:
                        AR_va, EV_va = make_dt_masks(t_va, e_va, bin_edges)
                        ell_va = float(
                            expected_dt_ll_mc(
                                model, Xva_t, AR_va, EV_va, alpha_dt, S=MC_TRAIN
                            ).item()
                        )
                        r_va_tmp, _ = predict_risk_mc(Xva_t, S=16)
                        c_va_tmp, _ = harrell_c(t_va, r_va_tmp, e_va)
                        msg += f" | valELL/evt={ell_va/max(1,n_events_va):.4f} | valC(MC,E[f])={c_va_tmp:.3f}"
                    else:
                        pll_va = float(
                            expected_cox_pll_mc(
                                model, Xva_t, t_va_t, e_va_t, S=MC_TRAIN
                            ).item()
                        )
                        r_va_tmp, _ = predict_risk_mc(Xva_t, S=16, mode="logexp")
                        c_va_tmp, _ = harrell_c(t_va, r_va_tmp, e_va)
                        msg += f" | valPLL/evt={pll_va/max(1,n_events_va):.4f} | valC(MC,logexp)={c_va_tmp:.3f}"
                    if c_va_tmp > best_c:
                        best_c = c_va_tmp
                        best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        }
            print(msg)

    if HAS_VAL and (best_state is not None):
        model.load_state_dict(best_state)
        print(f"Best validation C-index (MC, E[f]/logexp) = {best_c:.3f}")
    elif not HAS_VAL:
        print(
            "No validation split (VAL_FRAC=0). Trained for full EPOCHS without early stopping."
        )

    # ------------------- Metrics & Prints -------------------
    r_tr_plot, r_tr_sd = predict_risk_mc(Xtr_t)
    if HAS_VAL:
        r_va_plot, r_va_sd = predict_risk_mc(Xva_t)
    r_te_plot, r_te_sd = predict_risk_mc(Xte_t)

    print(
        "Risk std (E[f]) — "
        + ("train={:.4f}".format(np.std(r_tr_plot)))
        + (f" | val={np.std(r_va_plot):.4f}" if HAS_VAL else "")
        + f" | test={np.std(r_te_plot):.4f}"
    )
    print(
        f"Risk range (E[f]) — test: {float(np.min(r_te_plot)):.4f} → {float(np.max(r_te_plot)):.4f}"
    )

    # === C-index using r(H_C) = E[-log S(H_C)] with H_C = median TRAIN event time ===
    if USE_DISCRETE_TIME:
        if np.any(e_tr == 1):
            H_C = float(np.median(t_tr[e_tr == 1]))
        else:
            H_C = POSTERIOR_H_MONTHS
        r_tr_c = neglogS_at_H(Xtr_t, H_C, alpha_dt, S=EVAL_MC_SAMPLES)
        if HAS_VAL:
            r_va_c = neglogS_at_H(Xva_t, H_C, alpha_dt, S=EVAL_MC_SAMPLES)
        r_te_c = neglogS_at_H(Xte_t, H_C, alpha_dt, S=EVAL_MC_SAMPLES)
    else:
        r_tr_c, _ = predict_risk_mc(Xtr_t, S=EVAL_MC_SAMPLES, mode="logexp")
        if HAS_VAL:
            r_va_c, _ = predict_risk_mc(Xva_t, S=EVAL_MC_SAMPLES, mode="logexp")
        r_te_c, _ = predict_risk_mc(Xte_t, S=EVAL_MC_SAMPLES, mode="logexp")
        H_C = float("nan")

    r_tr_c = orient_risk(t_tr, e_tr, r_tr_c)
    r_te_c = orient_risk(t_te, e_te, r_te_c)
    if HAS_VAL:
        r_va_c = orient_risk(t_va, e_va, r_va_c)

    c_tr = _lifelines_c(t_tr, r_tr_c, e_tr)
    c_te = _lifelines_c(t_te, r_te_c, e_te)
    if HAS_VAL:
        c_va = _lifelines_c(t_va, r_va_c, e_va)

    rows = [
        {"split": "train", "c_index": c_tr, "dxy": dxy(c_tr)},
        {"split": "test", "c_index": c_te, "dxy": dxy(c_te)},
    ]
    if HAS_VAL:
        rows.insert(1, {"split": "val", "c_index": c_va, "dxy": dxy(c_va)})
    pd.DataFrame(rows).to_csv(OUTDIR / "c_index.csv", index=False)

    if HAS_VAL:
        print(
            f"C-index (E[-log S(H_C)], H_C={H_C:.1f} mo) — train={c_tr:.3f} | val={c_va:.3f} | test={c_te:.3f}"
        )
    else:
        print(
            f"C-index (E[-log S(H_C)], H_C={H_C:.1f} mo) — train={c_tr:.3f} | test={c_te:.3f}"
        )

    mean_c, lo_c, hi_c = bootstrap_cindex(t_te, e_te, r_te_c)
    (c_boot_mean, c_boot_lo, c_boot_hi, d_boot_mean, d_boot_lo, d_boot_hi) = (
        bootstrap_c_and_dxy(t_te, e_te, r_te_c)
    )
    print(f"Test C-index bootstrap mean={mean_c:.3f} (95% CI {lo_c:.3f}-{hi_c:.3f})")
    pd.Series(
        {
            "test_c_mean_boot": c_boot_mean,
            "test_c_lo": c_boot_lo,
            "test_c_hi": c_boot_hi,
            "test_dxy_mean_boot": d_boot_mean,
            "test_dxy_lo": d_boot_lo,
            "test_dxy_hi": d_boot_hi,
            "H_C": H_C,
        }
    ).to_csv(OUTDIR / "c_dxy_bootstrap_test.csv")

    if USE_DISCRETE_TIME:
        for H in H_CAL_LIST:
            r_te_H = neglogS_at_H(Xte_t, H, alpha_dt, S=EVAL_MC_SAMPLES)
            c_te_H, flip_H = harrell_c(t_te, r_te_H, e_te)
            print(f"C-index (E[-log S({int(H)})]) test = {c_te_H:.3f}")

    # scatter plot
    t_tr_cens, _ = admin_censor(t_tr, e_tr, 300.0)
    t_te_cens, _ = admin_censor(t_te, e_te, 300.0)
    plt.figure(figsize=(7.5, 4.3))
    plt.scatter(t_tr_cens, r_tr_plot, alpha=0.6, label="train")
    if HAS_VAL:
        t_va_cens, _ = admin_censor(t_va, e_va, 300.0)
        plt.scatter(t_va_cens, r_va_plot, alpha=0.6, label="val")
    plt.scatter(t_te_cens, r_te_plot, alpha=0.6, label="test")
    plt.xlabel("Survival time (months)")
    plt.ylabel(
        "Predicted risk surrogate (E[f])"
        if USE_DISCRETE_TIME
        else "Predicted risk (log E[exp(f)])"
    )
    plt.legend()
    plt.title("Survival vs predicted risk")
    savefig(OUTDIR / "scatter_survival_vs_risk.png")

    # HR per 1 SD risk (test)
    df_hr = pd.DataFrame(
        {
            "time": t_te,
            "event": e_te,
            "risk": (r_te_plot - r_te_plot.mean()) / (r_te_plot.std() + 1e-12),
        }
    )
    cph = CoxPHFitter().fit(df_hr, "time", "event")
    hr_row = (
        cph.summary.loc["risk"] if "risk" in cph.summary.index else cph.summary.iloc[0]
    )
    hr = float(np.exp(hr_row["coef"]))
    lo_key = next(k for k in hr_row.index if "coef lower 95%" in k)
    hi_key = next(k for k in hr_row.index if "coef upper 95%" in k)
    hr_lo = float(np.exp(hr_row[lo_key]))
    hr_hi = float(np.exp(hr_row[hi_key]))
    hr_p = float(hr_row["p"]) if "p" in hr_row else float(hr_row.get("p -log2", np.nan))
    print(
        f"HR per 1 SD risk (test, E[f]) = {hr:.2f} (95% CI {hr_lo:.2f}-{hr_hi:.2f}), p={hr_p:.2e}"
    )

    # ---------------- KM helpers ----------------
    def km_plot_groups(
        t,
        e,
        groups,
        labels,
        title,
        outname,
        admin_cap=True,
        extend_to_300=KM_EXTEND_TO_300,
    ):
        t_in, e_in = admin_censor(t, e, 300.0) if admin_cap else (t, e)

        masks = [(m & np.isfinite(t_in)) for m in groups]
        present = [i for i, m in enumerate(masks) if np.sum(m) > 0]
        if len(present) == 0:
            print(f"[WARN] No groups with data for {outname}.")
            return np.nan

        timeline = np.arange(0.0, 300.0 + 1.0, 1.0) if extend_to_300 else None

        kmfs = []
        plt.figure(figsize=(9.8, 6.8))
        ax = plt.gca()
        for i in present:
            kmf = KaplanMeierFitter()
            kmf.fit(t_in[masks[i]], e_in[masks[i]], label=labels[i], timeline=timeline)
            kmf.plot(ax=ax, ci_show=False, show_censors=False)
            kmfs.append(kmf)

        draw_guides(ax, horiz_at=0.5, vlines=(100, 150, 225), xmax=300)
        plt.xlabel("Time (months)")
        plt.ylabel("Survival probability")
        plt.title(title)

        handles, leg_labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            leg_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
        )

        try:
            add_at_risk_counts(*kmfs, ax=ax)
        except Exception as ex:
            print(f"[WARN] add_at_risk_counts failed: {ex}")

        grp_id = np.zeros_like(t_in, dtype=int) - 1
        for j, i in enumerate(present):
            grp_id[masks[i]] = j
        valid = grp_id >= 0
        pval = np.nan
        if len(np.unique(grp_id[valid])) >= 2:
            pval = float(
                multivariate_logrank_test(
                    event_durations=t_in[valid],
                    groups=grp_id[valid],
                    event_observed=e_in[valid],
                ).p_value
            )
            ax.text(
                0.60,
                0.86,
                f"Log-rank P = {pval:.1e}",
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(fc="white", ec="none", alpha=0.7),
            )
        else:
            ax.text(
                0.58,
                0.86,
                "Log-rank P = n/a (1 group)",
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(fc="white", ec="none", alpha=0.7),
            )

        savefig(OUTDIR / outname)
        return pval

    # ---------------- Posterior survival grouping at H ----------------------
    if USE_DISCRETE_TIME:
        print(
            f"[Posterior-H grouping] H={POSTERIOR_H_MONTHS} mo; thresholds: tau_high={TAU_HIGH}, tau_low={TAU_LOW}, p*={P_STAR}"
        )
        S_H_all = posterior_survival_at_H(
            torch.tensor(
                np.vstack(
                    [Xtr_s, Xva_s if HAS_VAL else np.zeros((0, Xtr_s.shape[1])), Xte_s]
                ),
                dtype=torch.float32,
                device=device,
            ),
            POSTERIOR_H_MONTHS,
            alpha_dt,
            S=EVAL_MC_SAMPLES,
        )
        n_tr, n_va, n_te = (
            Xtr_s.shape[0],
            (Xva_s.shape[0] if HAS_VAL else 0),
            Xte_s.shape[0],
        )
        S_H_tr = S_H_all[:, :n_tr]
        S_H_va = (
            S_H_all[:, n_tr : n_tr + n_va]
            if HAS_VAL
            else np.zeros((EVAL_MC_SAMPLES, 0))
        )
        S_H_te = S_H_all[:, n_tr + n_va :]

        def post_groups(SH):
            p_hi = np.mean(SH <= TAU_HIGH, axis=0)
            p_lo = np.mean(SH >= TAU_LOW, axis=0)
            gH = p_hi >= P_STAR
            gL = p_lo >= P_STAR
            gM = (~gH) & (~gL)
            return gL, gM, gH, p_lo, p_hi

        gL_tr, gM_tr, gH_tr, _, _ = post_groups(S_H_tr)
        gL_te, gM_te, gH_te, _, _ = post_groups(S_H_te)
        print(
            f"[TRAIN] counts Low/Med/High = {int(gL_tr.sum())}/{int(gM_tr.sum())}/{int(gH_tr.sum())}"
        )
        print(
            f"[TEST ] counts Low/Med/High = {int(gL_te.sum())}/{int(gM_te.sum())}/{int(gH_te.sum())}"
        )

        labels = ["Low risk", "Medium risk", "High risk"]
        title_test = (
            f"Test KM - Posterior-H grouping @H={int(POSTERIOR_H_MONTHS)} mo "
            f"($\\tau_H$={TAU_HIGH}, $\\tau_L$={TAU_LOW}, $p^*$={P_STAR})"
        )
        title_all = (
            f"Full cohort KM - Posterior-H grouping @H={int(POSTERIOR_H_MONTHS)} mo "
            f"($\\tau_H$={TAU_HIGH}, $\\tau_L$={TAU_LOW}, $p^*$={P_STAR})"
        )

        p_cap = km_plot_groups(
            t_te,
            e_te,
            [gL_te, gM_te, gH_te],
            labels,
            title_test,
            "KM_test_postH_cap.png",
            admin_cap=True,
        )
        p_raw = km_plot_groups(
            t_te,
            e_te,
            [gL_te, gM_te, gH_te],
            labels,
            title_test,
            "KM_test_postH_raw.png",
            admin_cap=False,
        )
        print(f"Log-rank p (admin-capped)={p_cap:.2e} | (raw)={p_raw:.2e}")

        # full cohort masks
        gL_all = np.zeros(len(meta), dtype=bool)
        gM_all = np.zeros(len(meta), dtype=bool)
        gH_all = np.zeros(len(meta), dtype=bool)
        gL_all[tr_idx] = gL_tr
        gM_all[tr_idx] = gM_tr
        gH_all[tr_idx] = gH_tr
        gL_all[te_idx] = gL_te
        gM_all[te_idx] = gM_te
        gH_all[te_idx] = gH_te
        if HAS_VAL:
            gL_va, gM_va, gH_va, _, _ = post_groups(S_H_va)
            gL_all[va_idx] = gL_va
            gM_all[va_idx] = gM_va
            gH_all[va_idx] = gH_va

        p_all_cap = km_plot_groups(
            time_all,
            event_all,
            [gL_all, gM_all, gH_all],
            labels,
            title_all,
            "KM_all_postH_cap.png",
            admin_cap=True,
        )
        pd.DataFrame(
            [
                {"group": "low", "n": int(gL_all.sum())},
                {"group": "med", "n": int(gM_all.sum())},
                {"group": "high", "n": int(gH_all.sum())},
                {"logrank_p_cap": p_all_cap},
                {
                    "H": int(POSTERIOR_H_MONTHS),
                    "tau_high": TAU_HIGH,
                    "tau_low": TAU_LOW,
                    "p_star": P_STAR,
                },
            ]
        ).to_csv(OUTDIR / "KM_all_postH_summary.csv", index=False)
    else:
        print("Posterior-H grouping skipped — requires USE_DISCRETE_TIME=True.")

    # ---------------- Calibration & Brier (test) ----------------
    def predict_SH_for_split(X_t, H, alpha):
        SH = posterior_survival_at_H(X_t, H, alpha, S=EVAL_MC_SAMPLES)  # [S,N]
        return SH.mean(axis=0)

    if USE_DISCRETE_TIME:
        for H in H_CAL_LIST:
            S_te_H = predict_SH_for_split(Xte_t, H, alpha_dt)
            qs = np.quantile(S_te_H, np.linspace(0, 1, 11))
            rows = []
            for i in range(10):
                lo, hi = qs[i], qs[i + 1]
                idx = (S_te_H >= lo) & ((S_te_H < hi) if i < 9 else (S_te_H <= hi))
                if idx.sum() == 0:
                    continue
                s_pred = float(S_te_H[idx].mean())
                kmf = KaplanMeierFitter().fit(t_te[idx], e_te[idx])
                s_emp = float(kmf.predict(min(H, float(t_te[idx].max()))))
                rows.append(
                    {"bin": i + 1, "n": int(idx.sum()), "pred_S": s_pred, "KM_S": s_emp}
                )
            cal_df = pd.DataFrame(rows)
            print(f"[Calibration @ {int(H)} mo] bin,n,mean_pred,KM:")
            for _, r in cal_df.iterrows():
                print(
                    f"  {int(r['bin']):02d}, n={int(r['n']):3d}, pred={r['pred_S']:.3f}, KM={r['KM_S']:.3f}"
                )
            cal_df.to_csv(
                OUTDIR / f"calibration_deciles_test_{int(H)}mo.csv", index=False
            )
            bs = brier_ipcw(t_te, e_te, S_te_H, H)
            print(f"Brier @ {int(H)} mo (IPCW) = {bs:.4f}")
            pd.Series({"Brier": bs}).to_csv(
                OUTDIR / f"brier_test_{int(H)}mo.csv", index=False
            )

        tau_for_ibs = max(H_CAL_LIST) if len(H_CAL_LIST) > 0 else float(np.nanmax(t_te))

        def _S_at_H_test(H):
            return predict_SH_for_split(Xte_t, H, alpha_dt)

        ibs_test = ibs_ipcw(t_te, e_te, _S_at_H_test, tau_for_ibs)
        print(f"IBS (IPCW) up to τ={int(tau_for_ibs)} mo — test = {ibs_test:.4f}")
        pd.Series({"IBS": ibs_test, "tau": tau_for_ibs}).to_csv(
            OUTDIR / f"ibs_test_tau{int(tau_for_ibs)}.csv"
        )

    # ---------------- Biology sanity checks (test split) ----------------
    rows = []
    if spearmanr is not None and USE_GES:
        nlS120 = neglogS_at_H(Xte_t, 120.0, alpha_dt, S=EVAL_MC_SAMPLES)

        def add_corr(name, a, b):
            rho, p = spearmanr(a, b, nan_policy="omit")
            rows.append({"pair": name, "rho": float(rho), "p": float(p)})

        add_corr("risk(E[f])_vs_prolif", r_te_plot, ges_Pro_te[:, 0])
        add_corr("risk(E[f])_vs_ER", r_te_plot, ges_ER_te[:, 0])
        add_corr("neglogS@120_vs_prolif", nlS120, ges_Pro_te[:, 0])
        add_corr("neglogS@120_vs_ER", nlS120, ges_ER_te[:, 0])
        pd.DataFrame(rows).to_csv(OUTDIR / "bio_correlations_test.csv", index=False)
        print("[BIO] Spearman correlations (saved bio_correlations_test.csv):")
        for r in rows:
            print(f"  {r['pair']}: rho={r['rho']:.3f}, p={r['p']}")
    else:
        print("[BIO] Skipped correlations (SciPy not available or USE_GES=False).")

    # 3) Adjusted Cox: risk + ER + Prolif + Grade3 (test)
    try:
        if "GRADE" in meta.columns:
            gvals = meta["GRADE"].astype(str).str.upper()
            grade3_all = gvals.str.contains("3").astype(int).to_numpy()
        else:
            grade_col = next((c for c in meta.columns if "GRADE" in c.upper()), None)
            if grade_col is not None:
                gvals = meta[grade_col].astype(str).str.upper()
                grade3_all = gvals.str.contains("3").astype(int).to_numpy()
            else:
                grade3_all = np.zeros(len(meta), dtype=int)
        grade3_te = grade3_all[te_idx]

        df_adj = pd.DataFrame(
            {
                "time": t_te,
                "event": e_te,
                "risk_z": (r_te_plot - r_te_plot.mean()) / (r_te_plot.std() + 1e-12),
                "prolif": ges_Pro_te[:, 0] if USE_GES else np.nan,
                "ER": ges_ER_te[:, 0] if USE_GES else np.nan,
                "grade3": grade3_te,
            }
        )
        for c in ["prolif", "ER", "grade3"]:
            if c in df_adj.columns and np.nanstd(df_adj[c].to_numpy()) < 1e-12:
                df_adj.drop(columns=[c], inplace=True)
        cph_adj = CoxPHFitter()
        cph_adj.fit(df_adj, "time", "event")
        cph_adj.summary.to_csv(OUTDIR / "cox_adjusted_bio_test.csv")
        print(
            "[BIO] Adjusted Cox (risk + prolif + ER + grade3) saved to cox_adjusted_bio_test.csv"
        )
    except Exception as ex:
        print(f"[BIO] Adjusted Cox failed: {ex}")

    # 4) Grade3 enrichment in High vs Low/Med (odds ratio)
    try:
        if fisher_exact is not None:
            high_mask = r_te_plot >= np.quantile(r_te_plot, 2 / 3)
            lowmed = ~high_mask
            a = int(np.sum((grade3_te == 1) & high_mask))
            b = int(np.sum((grade3_te == 0) & high_mask))
            c = int(np.sum((grade3_te == 1) & lowmed))
            d = int(np.sum((grade3_te == 0) & lowmed))
            if (a + b) > 0 and (c + d) > 0:
                OR, p = fisher_exact([[a, b], [c, d]])
            else:
                OR, p = np.nan, np.nan
            pd.Series(
                {
                    "OR_high_vs_other": OR,
                    "p": p,
                    "a(grade3&high)": a,
                    "b(!grade3&high)": b,
                    "c(grade3&other)": c,
                    "d(!grade3&other)": d,
                }
            ).to_csv(OUTDIR / "grade3_enrichment_test.csv")
            print(
                f"[BIO] Grade3 enrichment OR={OR if OR==OR else 'nan'}, p={p if p==p else 'nan'} (saved grade3_enrichment_test.csv)"
            )
        else:
            print("[BIO] Skipped grade3 enrichment (SciPy not available).")
    except Exception as ex:
        print(f"[BIO] Grade3 enrichment failed: {ex}")

    # ---------------- Split balance table -------------------
    def split_balance():
        def summarize(idx, name):
            d = {
                "split": name,
                "n": int(len(idx)),
                "events": int(event_all[idx].sum()),
                "censored": int(len(idx) - event_all[idx].sum()),
                "event_rate": float(event_all[idx].mean()),
            }
            for col in ["AGE_AT_DIAGNOSIS", "TUMOR_SIZE"]:
                if col in meta.columns:
                    vals = pd.to_numeric(meta[col].iloc[idx], errors="coerce")
                    d[f"{col}_mean"] = float(vals.mean())
                    d[f"{col}_sd"] = float(vals.std())
            return d

        rows_bal = [summarize(tr_idx, "train"), summarize(te_idx, "test")]
        if HAS_VAL:
            rows_bal.insert(1, summarize(va_idx, "val"))
        return pd.DataFrame(rows_bal)

    bal = split_balance()
    bal.to_csv(OUTDIR / "split_balance.csv", index=False)

    # ---------------- Partial Dependence (auto top-2 by permutation importance) ----------
    def is_continuous(col):
        return np.unique(col).size > 4

    def permutation_importance_topk(
        X_ref,
        times,
        events,
        feature_names,
        k=2,
        S=EVAL_MC_SAMPLES,
        K=50,
        seed=SEED,
        floor_zero=True,
    ):
        rng_loc = np.random.default_rng(seed)
        X_ref = np.asarray(X_ref, dtype=float)
        X_t = torch.tensor(X_ref, dtype=torch.float32, device=device)

        # Baseline (higher-is-better metric: C-index)
        r_base, _ = predict_risk_mc(X_t, S=S)
        c_base, _ = harrell_c(times, r_base, events)

        drops = []
        for j in range(X_ref.shape[1]):
            if not is_continuous(X_ref[:, j]):
                continue
            deltas = []
            for _ in range(K):
                Xp = X_ref.copy()
                Xp[:, j] = rng_loc.permutation(Xp[:, j])
                Xp_t = torch.tensor(Xp, dtype=torch.float32, device=device)
                r_perm, _ = predict_risk_mc(Xp_t, S=max(8, S // 2))
                c_perm, _ = harrell_c(times, r_perm, events)
                deltas.append(c_base - c_perm)  # higher-is-better → base - perm
            delta = float(np.mean(deltas))
            if floor_zero and delta < 0:
                delta = 0.0  # treat small negatives as 0
            drops.append((j, delta))

        if not drops:
            return list(range(min(2, X_ref.shape[1]))), []
        drops.sort(key=lambda x: x[1], reverse=True)
        top_idx = [j for (j, d) in drops[:k]]
        top_info = [(feature_names[j], d) for (j, d) in drops[:k]]
        return top_idx, top_info

    top_idx, top_info = permutation_importance_topk(Xtr_s, t_tr, e_tr, feat_names, k=2)
    print(
        f"[PD] Top features by ΔC-index drop (using E[f] for ranking in PD): {top_info}"
    )

    def pd_curve_ice(feature_idx, feature_name, fname):
        n_tr = Xtr_s.shape[0]
        ns = min(PD_ICE_NSAMP, n_tr)
        ice_idx = rng.choice(n_tr, size=ns, replace=False)
        base_mat = Xtr_s[ice_idx].copy()
        x_min, x_max = Xtr[:, feature_idx].min(), Xtr[:, feature_idx].max()
        grid = np.linspace(x_min, x_max, PD_POINTS)
        pd_mean, pd_lo, pd_hi = [], [], []
        qlo, qhi = PD_QUANTILES
        for v in grid:
            X_ice = base_mat.copy()
            X_ice[:, feature_idx] = (v - float(mu[0, feature_idx])) / float(
                sd[0, feature_idx]
            )
            X_ice_t = torch.tensor(X_ice, dtype=torch.float32, device=device)
            preds, _ = predict_risk_mc(X_ice_t, S=EVAL_MC_SAMPLES)
            pd_mean.append(float(np.mean(preds)))
            pd_lo.append(float(np.quantile(preds, qlo)))
            pd_hi.append(float(np.quantile(preds, qhi)))
        pd_mean = np.asarray(pd_mean)
        pd_lo = np.asarray(pd_lo)
        pd_hi = np.asarray(pd_hi)
        plt.figure(figsize=(5, 3))
        plt.plot(grid, pd_mean, lw=2, label="PD mean (ICE avg)")
        plt.fill_between(
            grid, pd_lo, pd_hi, alpha=0.25, label=f"ICE {int(qlo*100)}-{int(qhi*100)}%"
        )
        plt.xlabel(feature_name)
        plt.ylabel("Predicted risk surrogate (E[f])")
        plt.title(f"PD: {feature_name}")
        plt.legend()
        savefig(OUTDIR / fname)

    # >>> Hazard-scale PD/ICE using E[-log S(H|x)]
    PD_H = POSTERIOR_H_MONTHS  # pick a clinically meaningful horizon

    def pd_curve_ice_hazard(feature_idx, feature_name, fname, H=PD_H):
        n_tr = Xtr_s.shape[0]
        ns = min(PD_ICE_NSAMP, n_tr)
        ice_idx = rng.choice(n_tr, size=ns, replace=False)
        base_mat = Xtr_s[ice_idx].copy()
        x_min, x_max = Xtr[:, feature_idx].min(), Xtr[:, feature_idx].max()
        grid = np.linspace(x_min, x_max, PD_POINTS)
        pd_mean, pd_lo, pd_hi = [], [], []
        qlo, qhi = PD_QUANTILES
        for v in grid:
            X_ice = base_mat.copy()
            X_ice[:, feature_idx] = (v - float(mu[0, feature_idx])) / float(
                sd[0, feature_idx]
            )
            X_ice_t = torch.tensor(X_ice, dtype=torch.float32, device=device)
            vals = neglogS_at_H(X_ice_t, H, alpha_dt, S=EVAL_MC_SAMPLES)  # np [ns]
            pd_mean.append(float(np.mean(vals)))
            pd_lo.append(float(np.quantile(vals, qlo)))
            pd_hi.append(float(np.quantile(vals, qhi)))
        pd_mean, pd_lo, pd_hi = map(np.asarray, (pd_mean, pd_lo, pd_hi))
        plt.figure(figsize=(5, 3))
        plt.plot(grid, pd_mean, lw=2, label="PD mean (ICE avg)")
        plt.fill_between(
            grid, pd_lo, pd_hi, alpha=0.25, label=f"ICE {int(qlo*100)}-{int(qhi*100)}%"
        )
        plt.xlabel(feature_name)
        plt.ylabel(f"Cum. hazard  -log S({int(H)})")
        plt.title(f"PD (hazard scale) @ {int(H)} mo — {feature_name}")
        plt.legend()
        savefig(OUTDIR / fname)

    for j in top_idx:
        safe_name = feat_names[j].replace("/", "_")
        pd_curve_ice(j, feat_names[j], f"PD_{safe_name}.png")
        pd_curve_ice_hazard(j, feat_names[j], f"PD_hazard_{safe_name}_H{int(PD_H)}.png")

    # ---------------- Boxplots (test) ----------------
    def _boxplot(vecs, labels, title, ylabel, fname):
        import numpy as np
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.boxplot([np.asarray(v).ravel() for v in vecs], labels=labels, showfliers=False)
        ax = plt.gca()

        # ----- compute single p-value: Kruskal–Wallis for ≥3 groups, else Mann–Whitney -----
        clean = [np.asarray(v, float)[~np.isnan(v)] for v in vecs]
        have = [len(v) > 0 for v in clean]
        pval = np.nan
        if sum(have) >= 2:
            if len(clean) >= 3 and kruskal_wallis is not None:
                pval = float(kruskal_wallis(*[v for v, h in zip(clean, have) if h]).pvalue)
            elif len(clean) == 2 and 'mannwhitneyu' in globals() and mannwhitneyu is not None:
                pval = float(mannwhitneyu(clean[0], clean[1], alternative="two-sided").pvalue)

        # ----- draw the single p-value text (no pairwise annotations) -----
        if pval == pval:  # not NaN
            # give a little headroom so the text never touches the whiskers
            ymin, ymax = ax.get_ylim()
            span = ymax - ymin
            ax.set_ylim(ymin, ymax + 0.06 * span)

            ax.text(
                0.02, 0.96, f"p = {pval:.2e}",
                transform=ax.transAxes, va="top", ha="left", fontsize=12,
                bbox=dict(fc="white", ec="none", alpha=0.8)
            )

        plt.ylabel(ylabel)
        plt.title(title)
        savefig(OUTDIR / fname)


    if USE_DISCRETE_TIME:
        groups = [gL_te, gM_te, gH_te]
        glabels = ["Low", "Med", "High"]

        _boxplot(
            [ges_ER_te[:, 0][g] for g in groups],
            glabels,
            "ER score by Posterior-H group (test)",
            "ER score",
            "bio_box_ER_by_group_test.png",
        )

        _boxplot(
            [ges_Pro_te[:, 0][g] for g in groups],
            glabels,
            "Proliferation score by Posterior-H group (test)",
            "Proliferation score",
            "bio_box_prolif_by_group_test.png",
        )

        for j in top_idx:
            fname = feat_names[j].replace("/", "_")
            _boxplot(
                [Xte[:, j][g] for g in groups],
                glabels,
                f"{feat_names[j]} by Posterior-H group (test)",
                feat_names[j],
                f"Boxplot_{fname}_byGroup.png",
            )

    if USE_DISCRETE_TIME:
        groups = [gL_te, gM_te, gH_te]
        glabels = ["Low", "Med", "High"]

        _boxplot(
            [ges_ER_te[:, 0][g] for g in groups],
            glabels,
            "ER score by Posterior-H group (test)",
            "ER score",
            "bio_box_ER_by_group_test.png",
        )

        _boxplot(
            [ges_Pro_te[:, 0][g] for g in groups],
            glabels,
            "Proliferation score by Posterior-H group (test)",
            "Proliferation score",
            "bio_box_prolif_by_group_test.png",
        )

        for j in top_idx:
            fname = feat_names[j].replace("/", "_")
            _boxplot(
                [Xte[:, j][g] for g in groups],
                glabels,
                f"{feat_names[j]} by Posterior-H group (test)",
                feat_names[j],
                f"Boxplot_{fname}_byGroup.png",
            )

    # ============================ [A] Outcome proxies ============================
    def make_outcome_flags(
        times, events, early_cut=EARLY_EVENT_MO, long_cut=EVENT_FREE_MO
    ):
        early_event = (events == 1) & (times <= early_cut)
        long_event_free = (events == 0) & (times >= long_cut)
        return early_event.astype(int), long_event_free.astype(int)

    ee_tr, ltf_tr = make_outcome_flags(t_tr, e_tr)
    ee_te, ltf_te = make_outcome_flags(t_te, e_te)
    print(
        f"[A] Outcome proxies — train early_event_24mo={int(ee_tr.sum())}, "
        f"event_free_120mo={int(ltf_tr.sum())}; test early_event_24mo={int(ee_te.sum())}, "
        f"event_free_120mo={int(ltf_te.sum())}"
    )

    # Export per-split flags
    pd.DataFrame(
        {
            "time": t_tr,
            "event": e_tr,
            "early_event_24mo": ee_tr,
            "event_free_120mo": ltf_tr,
        }
    ).to_csv(OUTDIR / "outcome_flags_train.csv", index=False)
    pd.DataFrame(
        {
            "time": t_te,
            "event": e_te,
            "early_event_24mo": ee_te,
            "event_free_120mo": ltf_te,
        }
    ).to_csv(OUTDIR / "outcome_flags_test.csv", index=False)

    # Optional: multivariable logistic regression on TEST for "early_event_24mo"
    try:
        logit = LogisticRegression(max_iter=500, solver="lbfgs")
        logit.fit(Xte_s, ee_te)
        coefs = pd.Series(logit.coef_.ravel(), index=feat_names)
        OR = np.exp(coefs)
        out = pd.DataFrame(
            {"feature": feat_names, "logit_coef": coefs.values, "odds_ratio": OR.values}
        )
        out.sort_values("odds_ratio", ascending=False).to_csv(
            OUTDIR / "logit_early_event_24mo_test.csv", index=False
        )

        # Forest-style bar plot (top 25 by |coef|)
        top25 = out.reindex(
            out["logit_coef"].abs().sort_values(ascending=False).index[:25]
        )
        plt.figure(figsize=(7, 8))
        y = np.arange(len(top25))[::-1]
        plt.barh(y, top25["logit_coef"].values)
        plt.yticks(y, top25["feature"].values)
        plt.xlabel("Logistic coefficient (early event ≤ 24 mo)")
        plt.title("Early event (test) — multivariable logistic coefficients")
        savefig(OUTDIR / "logit_early_event_24mo_forest_top25.png")
    except Exception as ex:
        print(f"[A] Logistic regression failed: {ex}")

    # ================= [B] Interpretability on hazard scale @ H ====================
    HORIZONS = [120.0, float(H_C), 60.0, 30.0]

    # Precompute horizon risks on TEST for r annotations
    nlS_by_H = {
        H: neglogS_at_H(Xte_t, H, alpha_dt, S=EVAL_MC_SAMPLES) for H in HORIZONS
    }

    def raw_pearson(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        if pearsonr is not None and np.isfinite(a).all() and np.isfinite(b).all():
            r, _ = pearsonr(a, b)
            return float(r)
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        C = np.corrcoef(a, b)
        return float(C[0, 1])

    # (B1) Permutation importance for hazard @ H on TEST split (ΔC-index drop)
    def perm_importance_hazard(
        X_ref, times, events, H, S=EVAL_MC_SAMPLES, K=50, seed=SEED, floor_zero=True
    ):
        rng_loc = np.random.default_rng(seed)
        X_ref = np.asarray(X_ref, float)

        # Baseline scores at horizon H
        base = neglogS_at_H(
            torch.tensor(X_ref, dtype=torch.float32, device=device), H, alpha_dt, S=S
        )
        c_base, _ = harrell_c(times, base, events)  # higher-is-better metric

        drops = np.zeros(X_ref.shape[1], float)
        for j in range(X_ref.shape[1]):
            if not is_continuous(X_ref[:, j]):
                continue
            deltas = []
            for _ in range(K):
                Xp = X_ref.copy()
                Xp[:, j] = rng_loc.permutation(Xp[:, j])
                p = neglogS_at_H(
                    torch.tensor(Xp, dtype=torch.float32, device=device),
                    H,
                    alpha_dt,
                    S=max(8, S // 2),
                )
                c_perm, _ = harrell_c(times, p, events)
                deltas.append(c_base - c_perm) 
            delta = float(np.mean(deltas))
            if floor_zero and delta < 0:
                delta = 0.0
            drops[j] = delta
        return drops

    # (B2) Gradient-based importance
    def grad_importance_hazard(X_ref, H, alpha):
        X = torch.tensor(X_ref, dtype=torch.float32, device=device, requires_grad=True)
        f_mean = model(X).mean.squeeze()  # [N]
        j = int(np.searchsorted(bin_edges, H, side="right") - 1)
        j = int(np.clip(j, 0, len(bin_edges) - 2))
        if DT_LINK.lower() == "cloglog":
            eta = f_mean[:, None] + alpha[None, :]  # [N,J]
            h = 1.0 - torch.exp(-torch.exp(torch.clamp(eta, max=20.0)))
        else:
            eta = f_mean[:, None] + alpha[None, :]
            h = torch.sigmoid(eta)
        S_edges = torch.cumprod(1.0 - h, dim=1)
        nlS = -torch.log(torch.clamp(S_edges[:, j], min=1e-10))
        loss = nlS.mean()
        loss.backward()
        g = X.grad.detach().abs().mean(dim=0).cpu().numpy()  # [P]
        return g

    # (B3) ARD from HIDDEN layer (per-feature)
    try:
        ls = (
            model.hidden.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
        )  # (H,1,P) typically
        ls = np.asarray(ls, float)
        if ls.ndim == 3:
            ls_feat = ls.mean(axis=0).squeeze(0)  # [P]
        elif ls.ndim == 2:
            ls_feat = ls.mean(axis=0)
        else:
            ls_feat = ls
        if ls_feat.shape[0] != len(feat_names):
            print(
                f"[WARN] ARD lengthscale shape mismatch ({ls_feat.shape[0]} vs {len(feat_names)}); skipping ARD."
            )
            inv_ls_base = np.zeros(len(feat_names), dtype=float)
        else:
            inv_ls_base = 1.0 / (ls_feat + 1e-8)
    except Exception as ex:
        print(f"[B] ARD extraction failed: {ex}")
        inv_ls_base = np.zeros(len(feat_names), dtype=float)

    def _z(v):
        v = np.asarray(v, float)
        return (v - v.mean()) / (v.std() + 1e-12)

    # Loop horizons: compute consensus, make Top-N risk-oriented with raw r labels
    for H_INT in HORIZONS:
        print(f"[B] Interpretability @ H={int(H_INT)} months")
        perm_all = perm_importance_hazard(Xte_s, t_te, e_te, H_INT)
        try:
            grads_all = grad_importance_hazard(Xte_s, H_INT, alpha_dt)
        except Exception as ex:
            print(f"[B] Gradient importance failed @H={int(H_INT)}: {ex}")
            grads_all = np.zeros(len(feat_names), dtype=float)
        inv_ls = inv_ls_base  # same per model, reused

        consensus = _z(perm_all) + _z(grads_all) + _z(inv_ls)

        # raw r with hazard at H
        r_raw_vec = np.array(
            [raw_pearson(Xte_s[:, j], nlS_by_H[H_INT]) for j in range(len(feat_names))]
        )
        sign_vec = np.sign(np.where(np.isnan(r_raw_vec), 0.0, r_raw_vec))
        signed_consensus = consensus * sign_vec  

        interp_df = pd.DataFrame(
            {
                "feature": feat_names,
                "perm_drop_c": perm_all,
                "grad_abs": grads_all,
                "inv_lengthscale_hidden": inv_ls,
                "consensus_zsum": consensus,
                "r_raw_with_neglogS": r_raw_vec,
                "signed_consensus": signed_consensus,
            }
        ).sort_values("consensus_zsum", ascending=False)
        interp_df.to_csv(OUTDIR / f"interpretability_H{int(H_INT)}.csv", index=False)

        # --- Plot Top-N (importance-only) ---
        topN = 5
        top_idx_order = interp_df.index[:topN]
        top_plot = interp_df.loc[top_idx_order].copy()
        top_plot = top_plot.iloc[::-1]

        plt.figure(figsize=(8, 6))
        y = np.arange(len(top_plot))
        # Use importance; pick ONE of the two lines below:
        # (A) importance-only (nonnegative bars):
        # vals = top_plot["consensus_zsum"].values
        # (B) risk-oriented sign (can be negative if protective):
        vals = top_plot["signed_consensus"].values

        plt.barh(y, vals)
        plt.yticks(y, top_plot["feature"].values)
        plt.xlabel("Consensus Importance Score")
        plt.title(
            f"Top {topN} features — DGP-utilized risk-associated features @ H={int(H_INT)} mo"
        )

        # Keep bars comfortably inside the frame
        ax = plt.gca()
        xmin = min(0.0, float(np.min(vals)))
        xmax = max(0.0, float(np.max(vals)))
        ax.set_xlim(xmin, xmax + 0.08 * (xmax - xmin + 1e-9))

        savefig(OUTDIR / f"interpretability_consensus_top{topN}_H{int(H_INT)}.png")

    # ======================= [B-PC] PC-only importance + PC→gene PNGs =======================
    # Use only PCs actually used by the DGP (sel_names)
    M_TOP_PC = min(6, len(sel_names))

    # For orientation of loadings in plots, we align the PC so that positive loading
    # corresponds to the direction where risk increases at the chosen horizon H.
    genes_pref = expr.index.to_numpy()  # genes used in PCA
    Vt_sel = Vt[keep_idxs, :]  # [K, Gpref]
    Tte_sel_mat = Tte_sel  # [N_test, K]

    # helper to compute raw r and sign for a given PC and horizon
    def pc_raw_r_and_sign(pc_scores, nlS_vec):
        r = raw_pearson(pc_scores, nlS_vec)  # raw Pearson r
        sgn = 1.0 if r >= 0 else -1.0
        return r, sgn

    # loop horizons and PCs, produce risk-oriented PC→gene plots with neutral x-axis label
    pc_rows_for_csv_all = []
    outdir_pc_pngs = OUTDIR / "pc_gene_pngs"
    os.makedirs(outdir_pc_pngs, exist_ok=True)

    # Select top PCs by consensus among PCs only (reuse consensus at H=120 as tie-breaker)
    # Compute a PC-only consensus vector using H=120 components already computed
    # (fall back to absolute correlation if needed)
    pc_feat_idx = np.arange(len(sel_names))
    # reuse previous consensus at H=120 from interp_df saved above
    interp120 = pd.read_csv(OUTDIR / "interpretability_H120.csv")
    pc_mask = interp120["feature"].isin(sel_names)
    pc_consensus = interp120.loc[pc_mask, "consensus_zsum"].to_numpy()
    order_pc = np.argsort(pc_consensus)[::-1]
    top_pc_local = order_pc[:M_TOP_PC]

    for H_INT in HORIZONS:
        nlS_vec = nlS_by_H[H_INT]
        for pos_in_block in top_pc_local:
            pc_name = sel_names[pos_in_block]
            pc_scores_test = Tte_sel_mat[:, pos_in_block]

            r_raw, sign_for_plot = pc_raw_r_and_sign(pc_scores_test, nlS_vec)

            # Orient loadings for display; axis label remains neutral
            load = sign_for_plot * Vt_sel[pos_in_block, :]

            df = pd.DataFrame(
                {
                    "gene": genes_pref,
                    "pc": pc_name,
                    "loading_oriented": load,
                    "abs_loading": np.abs(load),
                    "raw_r_with_neglogS": r_raw,
                    "H": int(H_INT),
                }
            ).sort_values("abs_loading", ascending=False)

            df.to_csv(
                OUTDIR / f"pc_gene_oriented_{pc_name}_H{int(H_INT)}.csv", index=False
            )
            df.head(100).to_csv(
                OUTDIR / f"pc_{pc_name}_top_genes_oriented_H{int(H_INT)}.csv",
                index=False,
            )

            # plot top-20 
            topg = df.head(20).sort_values("loading_oriented")
            plt.figure(figsize=(8.0, 6.0))
            ax = plt.gca()
            y = np.arange(len(topg))
            plt.barh(y, topg["loading_oriented"].values)
            plt.yticks(y, topg["gene"].values)
            plt.xlabel("PC loading (oriented)")
            ax.axvline(0.0, color="k", lw=0.8, ls="--")
            lim = float(np.max(np.abs(topg["loading_oriented"].values))) * 1.10
            if lim <= 0:
                lim = 1.0
            ax.set_xlim(-lim, lim)

            direction = (
                "higher risk if PC increases"
                if r_raw >= 0
                else "lower risk if PC increases"
            )
            plt.title(
                f"{pc_name} — top gene loadings @ H={int(H_INT)} mo (raw r={r_raw:.2f})"
            )

            savefig(outdir_pc_pngs / f"{pc_name}_top20_riskOriented_H{int(H_INT)}.png")

            pc_rows_for_csv_all.append(
                {
                    "pc": pc_name,
                    "H": int(H_INT),
                    "consensus_importance_pc_only_H120": float(
                        pc_consensus[pos_in_block]
                    ),
                    "raw_r_with_neglogS": float(r_raw),
                    "png_path": str(
                        outdir_pc_pngs
                        / f"{pc_name}_top20_riskOriented_H{int(H_INT)}.png"
                    ),
                }
            )

    if pc_rows_for_csv_all:
        pd.DataFrame(pc_rows_for_csv_all).sort_values(
            ["H", "raw_r_with_neglogS"], ascending=[True, False]
        ).to_csv(OUTDIR / "topPC_riskOriented_multiH_summary.csv", index=False)
        print(f"[B-PC] Saved risk-oriented PC→gene PNGs in {outdir_pc_pngs}")

    # ======================= [C] PC→gene back-projection =======================
    pc_sig = pc_rank.head(min(8, len(pc_rank)))  # take top 8 PCs by |Z|

    back_rows = []
    for _, row in pc_sig.iterrows():
        pid = int(row["idx"])  # PC index in 0..r_evr-1
        load = Vt[pid, :]  # length n_genes_prefilter
        df = pd.DataFrame(
            {
                "gene": genes_pref,
                "pc": row["pc"],
                "pc_idx": pid,
                "pc_z_train": float(row["z"]),
                "loading": load,
                "abs_loading": np.abs(load),
            }
        ).sort_values("abs_loading", ascending=False)
        # Save top 100 by absolute loading
        df.head(100).to_csv(OUTDIR / f"pc_{row['pc']}_top_genes.csv", index=False)

        # Plot top-20 barh for this PC (reference, raw loadings)
        topg = df.head(20).sort_values("loading")
        plt.figure(figsize=(7.5, 6))
        y = np.arange(len(topg))
        plt.barh(y, topg["loading"].values)
        plt.yticks(y, topg["gene"].values)
        plt.xlabel("PC loading (raw)")
        plt.title(f"{row['pc']} (|Z|={abs(row['z']):.2f}) — top gene loadings")
        savefig(OUTDIR / f"pc_{row['pc']}_top20_loadings.png")

        back_rows.append(df.assign(rank=np.arange(1, len(df) + 1)))

    if back_rows:
        comb = pd.concat(back_rows, ignore_index=True)
        comb.to_csv(OUTDIR / "pc_to_gene_backprojection_all.csv", index=False)

    # (Optional) Correlate key PCs with outcome proxies on TEST
    try:
        pc_scores_test = pd.DataFrame(Tte[:, :], columns=pc_names)
        pc_scores_test["early_event_24mo"] = ee_te
        pc_scores_test["event_free_120mo"] = ltf_te
        corr_rows = []
        for pc in pc_rank.head(min(8, len(pc_rank)))["pc"]:
            x = pc_scores_test[pc].to_numpy()
            for lab, yv in [("early_event_24mo", ee_te), ("event_free_120mo", ltf_te)]:
                if spearmanr is not None:
                    rho, p = spearmanr(x, yv, nan_policy="omit")
                    corr_rows.append(
                        {"pc": pc, "label": lab, "rho": float(rho), "p": float(p)}
                    )
        if corr_rows:
            pd.DataFrame(corr_rows).to_csv(
                OUTDIR / "pc_vs_outcome_correlations_test.csv", index=False
            )
    except Exception as ex:
        print(f"[C] PC vs outcome correlation failed: {ex}")

    # Save config summary
    summary = {
        "seed": int(seed),
        "N_patients": int(expr.shape[1]),
        "N_genes_prefilter": int(expr.shape[0]),
        "PCA_train_r_evr": int(Ttr.shape[1]),
        "SURV_TOP_K_used": int(Ttr_sel.shape[1]),
        "hidden_dim": int(HIDDEN_DIM),
        "num_inducing": int(NUM_INDUCING),
        "epochs": int(EPOCHS),
        "use_ges": bool(USE_GES),
        "ges_method": GES_METHOD,
        "kmeans_inducing": bool(KMEANS_INIT),
        "gene_select_mode": GENE_SELECT_MODE,
        "gene_var_top_q": GENE_VAR_TOP_Q,
        "c_train": float(c_tr),
        "c_val": float(c_va) if HAS_VAL else float("nan"),
        "c_test": float(c_te),
        "dxy_train": dxy(c_tr),
        "dxy_val": dxy(c_va) if HAS_VAL else float("nan"),
        "dxy_test": dxy(c_te),
        "test_c_boot_mean": c_boot_mean,
        "test_c_boot_lo": lo_c,
        "test_c_boot_hi": hi_c,
        "test_dxy_mean_boot": d_boot_mean,
        "test_dxy_lo": d_boot_lo,
        "test_dxy_hi": d_boot_hi,
        "HR_per1SD_Ef": float(hr),
        "HR_lo": float(hr_lo),
        "HR_hi": float(hr_hi),
        "HR_p": float(hr_p),
        "vi_dist": VI_DIST,
        "mc_train": MC_TRAIN,
        "kl_warmup": KL_WARMUP,
        "lr_var": LR_VAR,
        "lr_ind": LR_IND,
        "lr_hyp": LR_HYP,
        "use_discrete_time": bool(USE_DISCRETE_TIME),
        "dt_bins": int(DT_BINS) if USE_DISCRETE_TIME else int(0),
        "dt_link": DT_LINK,
        "posterior_H": POSTERIOR_H_MONTHS,
        "tau_high": TAU_HIGH,
        "tau_low": TAU_LOW,
        "p_star": P_STAR,
        "H_C_for_Cindex": float(H_C),
        "pd_top_features": (
            "; ".join([f"{n}:{d:.4f}" for (n, d) in top_info]) if top_info else ""
        ),
        "ibs_test_tau": float(tau_for_ibs) if USE_DISCRETE_TIME else float("nan"),
        "ibs_test": float(ibs_test) if USE_DISCRETE_TIME else float("nan"),
        # [A] counts
        "early_event_24mo_train": int(ee_tr.sum()),
        "event_free_120mo_train": int(ltf_tr.sum()),
        "early_event_24mo_test": int(ee_te.sum()),
        "event_free_120mo_test": int(ltf_te.sum()),
    }

    pd.Series(summary).to_csv(OUTDIR / f"run_summary_seed{seed}.csv")

    print(
        f"Somers Dxy — train={dxy(c_tr):.3f}"
        + (f" | val={dxy(c_va):.3f}" if HAS_VAL else "")
        + f" | test={dxy(c_te):.3f}"
    )
    if USE_DISCRETE_TIME:
        print(f"IBS(test; τ={int(tau_for_ibs)} mo) = {ibs_test:.4f}")
    print(f"Saved results to: {OUTDIR}")
    return {
        "seed": seed,
        "c_test": c_te,
        "c_train": c_tr,
        "c_val": (c_va if HAS_VAL else np.nan),
        "hr": hr,
        "hr_lo": hr_lo,
        "hr_hi": hr_hi,
    }


# ------------------- Main -------------------
if __name__ == "__main__":
    set_all_seeds(SEED)
    single_res = run_single(SEED)
    if RUN_MULTI_SEEDS:
        rows = []
        for s in MULTI_SEEDS:
            rows.append(run_single(s))
        df = pd.DataFrame(rows)
        mean = df["c_test"].mean()
        sd = df["c_test"].std()
        print(
            f"\nMulti-seed test C-index (E[-log S(H_C)]): mean={mean:.3f} ± {sd:.3f} over {len(MULTI_SEEDS)} seeds"
        )
        df.to_csv(OUTDIR / "multi_seed_results.csv", index=False)
