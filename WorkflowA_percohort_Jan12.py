# =============================================================================
# METABRIC RUN — with pathway activity PCA→Cox (top-K PCs) integrated
# Now with optional pathway *consistency* PCs residualized vs activity PCs
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
import itertools

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
    from scipy.stats import spearmanr, pearsonr
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
SPLIT_SEED = 42  # fixed train/test split seed (keeps split identical across runs)
RUN_MULTI_SEEDS = True
MULTI_SEEDS = [42, 43, 44, 45, 46]
# ENDORSE-style split
TEST_FRAC = 0.50
VAL_FRAC = 0.0  # 0 disables validation/early-stop

# ---- Feature selection scope ----
PATHWAY_ONLY = False  # << set True to use ONLY pathway PCs in the final design
USE_PATHWAYS = False  # master switch for pathway activity/consistency blocks
USE_CLINICAL = True  # include clinical covariates when PATHWAY_ONLY=False

# Toggles
USE_GES = False  # add ER/Proliferation gene-set scores
GES_METHOD = "ssgsea"  # "ssgsea" or "rankmean"
KMEANS_INIT = True  # k-means init for inducing points

# Expression prefilter (speeds PCA)
GENE_SELECT_MODE = "quantile"  # "quantile" or "count"
GENE_VAR_TOP_Q = 0.95
GENE_N_TOP = 1000

# PCA & survival-based PC selection (GENE-LEVEL)
TARGET_EVR = 0.95
SURV_TOP_K = 18  # set 0 to skip gene PCs entirely
MIN_PC = 8
MAX_PC = 40
KM_EXTEND_TO_300 = True  # extend KM curves flat to 300 mo for plot comparability

# DGP architecture (not used in this front half; kept for continuity)
HIDDEN_DIM = 5
NUM_INDUCING = 64

# Training
EPOCHS = 900
PRINT_EVERY = 50
GRAD_CLIP = 1.0

# ----- Variational Inference -----
MC_TRAIN = 32
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
TAU_LOW = 0.75
P_STAR = 0.75

# Calibration horizons (months)
H_CAL_LIST = [60.0, 120.0]

# PD settings
PD_POINTS = 50
PD_ICE_NSAMP = 128
PD_QUANTILES = (0.10, 0.90)

# Reporting extras
EVAL_MC_SAMPLES = 32
BOOTSTRAP_B = 1000


# --------- Pathway selector knobs ----------
# ACTIVITY → PCA(train) → Cox(PCs on TRAIN) → keep top-K by |Z|
COX_PENALIZER = 0.05
PATHWAY_PC_TOP_K = 30  # top-K activity PCs
PATHWAY_TARGET_EVR = 0.95  # PCA EVR target for activity block

# >>> NEW: consistency block knobs <<<
USE_PATHWAY_CONSISTENCY = False  # turn on/off adding consistency
RESIDUALIZE_CONSISTENCY = False  # orthogonalize C vs kept A on TRAIN
CONSIST_PC_TOP_K = 30  # how many residualized C-PCs to keep by |Z|
CONSIST_TARGET_EVR = 0.95  # PCA EVR target for consistency block

# Preprocessing toggles
WINSORIZE = False
WINSOR_Q = (0.005, 0.995)

# File system — update BASE to your environment
BASE = Path(
your path here )
OUTDIR = BASE / "metabric_results"
OUTDIR.mkdir(parents=True, exist_ok=True)

P_CLIN_PAT = BASE / "data_clinical_patient.txt"
P_CLIN_SAMP = BASE / "data_clinical_sample.txt"
P_EXPR_MICRO = BASE / "data_mrna_illumina_microarray.txt"

# Pathway files (activity & consistency) — rows=pathways, cols=samples
P_PATH_ACTIVITY = (
    BASE / "METABRIC_expression_median_subset_833Samples_pathway_activity.txt"
)
P_PATH_CONSIST = (
    BASE / "METABRIC_expression_median_subset_833Samples_pathway_consistency.txt"
)

WHITELIST_PATH = BASE / "patient file csv goes here"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------- helpers -----------------------
def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_table_guess(path, index_col=None):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, index_col=index_col, dtype=str)
    # try TSV first, then CSV
    try:
        return pd.read_table(
            path,
            sep="\t",
            comment="#",
            index_col=index_col,
            encoding="utf-8",
            low_memory=False,
        )
    except Exception:
        try:
            return pd.read_csv(
                path, index_col=index_col, encoding="utf-8", low_memory=False
            )
        except UnicodeDecodeError:
            return pd.read_csv(
                path, index_col=index_col, encoding="latin1", low_memory=False
            )


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


# === robust C-index helper with auto-orientation ===
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


def detect_prestandardized(E: pd.DataFrame, tol_mean=0.1, tol_std=0.2):
    m = E.mean(axis=1)
    s = E.std(axis=1)
    med_abs_mean = float(np.median(np.abs(m)))
    med_std = float(np.median(s))
    is_z = (med_abs_mean <= tol_mean) and (abs(med_std - 1.0) <= tol_std)
    return {"is_z": is_z, "med_abs_mean": med_abs_mean, "med_std": med_std}


def train_iqr(E: pd.DataFrame, cols_idx):
    if len(cols_idx) == 0:
        return pd.Series(index=E.index, dtype=float)
    q75 = E.iloc[:, cols_idx].quantile(0.75, axis=1)
    q25 = E.iloc[:, cols_idx].quantile(0.25, axis=1)
    return q75 - q25


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
    for sep in ("///", "//", "|", ";", "/", ","):
        if sep in s:
            s = s.split(sep)[0].strip()
    m = _token.search(s)
    return m.group(0) if m else s


def clean_and_collapse_genes(E: pd.DataFrame):
    before = E.shape[0]
    sym = E.index.to_series().astype(str).apply(_canon_symbol)
    E2 = E.copy()
    E2.index = sym
    E2 = E2[~(E2.index.isna() | (E2.index == ""))]
    mask_affx = ~E2.index.str.startswith("AFFX")
    E2 = E2[mask_affx]
    E2 = E2.groupby(E2.index).median()
    after = E2.shape[0]
    print(
        f"Gene-symbol cleanup: {before} → {after} unique symbols (collapsed dups by median)."
    )
    return E2


ALIASES = {"AURKA": ["STK15"], "BIRC5": ["SURVIVIN"], "TOP2A": ["TOPIIA", "TOP2"]}


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


# -------- MB sample ID canonicalization (dashed) --------
_MB_PAT = re.compile(r"(?i)\bMB[._\- ]?(\d{1,5})\b")


def canon_mb_dash(s: str) -> str | None:
    s = str(s).strip()
    m = _MB_PAT.search(s)
    if not m:
        return None
    d = m.group(1)
    if len(d) <= 4:
        d = d.zfill(4)
    return f"MB-{d}"


# ======= small utilities for robust pathway processing =======
def winsorize_rows(M: np.ndarray, q_low=0.005, q_high=0.995):
    if not WINSORIZE:
        return M
    M = M.copy()
    if M.size == 0:
        return M
    lo = np.nanquantile(M, q_low, axis=1, keepdims=True)
    hi = np.nanquantile(M, q_high, axis=1, keepdims=True)
    M = np.where(M < lo, lo, M)
    M = np.where(M > hi, hi, M)
    return M


# --- Robust kernel stats that handle AdditiveKernel / ProductKernel / ScaleKernel ---
import gpytorch, numpy as np


def _collect_scale_leaves(k):
    leaves = []
    if isinstance(k, gpytorch.kernels.ScaleKernel):
        leaves.append(k)
    elif isinstance(
        k, (gpytorch.kernels.AdditiveKernel, gpytorch.kernels.ProductKernel)
    ):
        for sub in k.kernels:
            leaves.extend(_collect_scale_leaves(sub))
    else:
        leaves.append(k)
    return leaves


# --- SAFE kernel stats over sums/products/scale, with back-compat keys ---
def kernel_lengthscale_stats(kern):
    """
    Safe stats for (possibly composite) kernels:
    - Finds a top-level ScaleKernel outputscale if present
    - Aggregates *all* found lengthscales (RBF/Matern/etc.), skipping kernels with none (e.g., Linear)
    Returns dict with ls_med/ls_min/ls_max/ls_n and outscale (may be NaN if no ScaleKernel).
    """
    import numpy as np
    from gpytorch.kernels import ScaleKernel, AdditiveKernel

    ls_list = []
    outscale = None

    def _visit(k):
        nonlocal outscale
        # If wrapped in ScaleKernel, record its outputscale and descend
        if isinstance(k, ScaleKernel):
            try:
                outscale = float(k.outputscale.detach().cpu())
            except Exception:
                pass
            _visit(k.base_kernel)
            return

        # If additive (sum) kernel, visit children
        if isinstance(k, AdditiveKernel):
            kids = getattr(k, "kernels", None) or getattr(k, "_kernels", [])
            for child in kids:
                _visit(child)
            return

        # Try to pull a lengthscale if present (RBF, Matern, RQ, Periodic, etc.)
        if hasattr(k, "lengthscale"):
            try:
                arr = k.lengthscale.detach().flatten().cpu().numpy()
                if arr.size and np.all(np.isfinite(arr)):
                    ls_list.extend(arr.tolist())
            except Exception:
                pass

        # Some kernels hide another base kernel — descend if present
        if hasattr(k, "base_kernel"):
            try:
                _visit(k.base_kernel)
            except Exception:
                pass

    _visit(kern)

    if ls_list:
        arr = np.asarray(ls_list, dtype=float)
        return {
            "ls_med": float(np.nanmedian(arr)),
            "ls_min": float(np.nanmin(arr)),
            "ls_max": float(np.nanmax(arr)),
            "ls_n": int(arr.size),
            "outscale": (outscale if outscale is not None else float("nan")),
        }
    else:
        return {
            "ls_med": float("nan"),
            "ls_min": float("nan"),
            "ls_max": float("nan"),
            "ls_n": 0,
            "outscale": (outscale if outscale is not None else float("nan")),
        }


def train_only_row_zscore(M: np.ndarray, cols: np.ndarray):
    if M.size == 0:
        return M
    if cols.size == 0:
        return M
    mu = np.nanmean(M[:, cols], axis=1, keepdims=True)
    sd = np.nanstd(M[:, cols], axis=1, keepdims=True)
    sd = np.where((sd <= 0.0) | ~np.isfinite(sd), 1.0, sd)
    Z = (M - mu) / sd
    return Z


def _rows_prestandardized(M, cols, tol_mean=0.1, tol_std=0.2):
    if M.size == 0 or cols.size == 0:
        return False
    m = np.nanmedian(M[:, cols], axis=1)
    s = np.nanstd(M[:, cols], axis=1)
    med_abs_mean = float(np.nanmedian(np.abs(m)))
    med_std = float(np.nanmedian(s))
    return (med_abs_mean <= tol_mean) and (abs(med_std - 1.0) <= tol_std)


def cox_rank_features_matrix(
    X: np.ndarray,
    t_tr: np.ndarray,
    e_tr: np.ndarray,
    names: list[str],
    penalizer: float = 0.0,
    drop_constant=True,
) -> pd.DataFrame:
    rows = []
    if X.size == 0:
        return pd.DataFrame(columns=["feat", "idx", "z", "p", "coef"])
    # Align lengths defensively (guards against sample-dropping in upstream feature blocks)
    n = min(int(X.shape[0]), int(len(t_tr)), int(len(e_tr)))
    if n <= 0:
        return pd.DataFrame(columns=["feat", "idx", "z", "p", "coef"])
    if n != X.shape[0] or n != len(t_tr) or n != len(e_tr):
        X = X[:n]
        t_tr = np.asarray(t_tr)[:n]
        e_tr = np.asarray(e_tr)[:n]
    ev = e_tr.astype(bool)
    keep = []
    for j in range(X.shape[1]):
        x = X[:, j]
        v1 = np.nanvar(x[ev]) if ev.any() else np.nanvar(x)
        v0 = np.nanvar(x[~ev]) if (~ev).any() else np.nanvar(x)
        if drop_constant and ((v1 < 1e-12) or (v0 < 1e-12) or (np.nanvar(x) < 1e-12)):
            continue
        keep.append(j)
    if not keep:
        return pd.DataFrame(columns=["feat", "idx", "z", "p", "coef"])

    Xuse = X[:, keep]
    use_names = [names[j] for j in keep]
    df_base = pd.DataFrame({"time": t_tr, "event": e_tr})
    for k in range(Xuse.shape[1]):
        df = df_base.copy()
        df["x"] = Xuse[:, k]
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(df, duration_col="time", event_col="event", show_progress=False)
            s = (
                cph.summary.loc["x"]
                if "x" in cph.summary.index
                else cph.summary.iloc[0]
            )
            rows.append(
                {
                    "feat": use_names[k],
                    "idx": keep[k],
                    "z": float(s["z"]),
                    "p": float(s["p"]),
                    "coef": float(s["coef"]),
                }
            )
        except Exception:
            rows.append(
                {"feat": use_names[k], "idx": keep[k], "z": 0.0, "p": 1.0, "coef": 0.0}
            )
    return pd.DataFrame(rows).sort_values(by="z", key=np.abs, ascending=False)


# -------------------  Core pipeline -------------------
def run_single(seed=SEED):
    set_all_seeds(seed)
    seed_suffix = f"__seed{int(seed)}" if RUN_MULTI_SEEDS else ""
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

    merged = samp.merge(pat, on="PATIENT_ID", how="left", suffixes=("_SAMP", "_PAT"))

    # ======================== WHITELIST FILTER (833 prefiltered IDs) ========================
    def load_id_set(path):
        """
        KEEP NAME: load_id_set

        For your subgroup CSV whitelist (e.g. patients_ERposHER2neg_CHEMO.csv):
        - returns (wl_pat, wl_samp) where wl_pat is the set of PATIENT_IDs
        - wl_samp will be an empty set (so downstream code chooses PATIENT_ID matching)

        Still supports:
        - xlsx/xls, csv/tsv, txt (table or one-ID-per-line)
        """
        path = Path(path)
        suf = path.suffix.lower()

        # ---------- read table / list ----------
        df = None
        if suf in (".xlsx", ".xls"):
            df = pd.read_excel(path, dtype=str)
        elif suf in (".csv", ".tsv"):
            sep = "," if suf == ".csv" else "\t"
            try:
                df = pd.read_csv(path, dtype=str, sep=sep, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(path, dtype=str, sep=sep, encoding="latin1")
        elif suf == ".txt":
            # try as table; if that fails, treat as one ID per line
            try:
                df = pd.read_csv(path, dtype=str, sep=None, engine="python")
            except Exception:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    vals = [line.strip() for line in f if line.strip()]
                wl_pat = {v.strip().upper() for v in vals}
                return wl_pat, set()
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                vals = [line.strip() for line in f if line.strip()]
            wl_pat = {v.strip().upper() for v in vals}
            return wl_pat, set()

        if df is None or df.shape[0] == 0:
            return set(), set()

        df.columns = [str(c).strip().upper() for c in df.columns]
        for c in df.columns:
            df[c] = df[c].astype(str).str.strip()

        # ---------- decide which column holds PATIENT_ID ----------
        if "PATIENT_ID" in df.columns:
            col = "PATIENT_ID"
        else:
            # subgroup CSVs often have a single column like "PATIENT_ID" anyway;
            # if not, just use the first column
            col = df.columns[0]

        vals = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"": np.nan, "NAN": np.nan, "NA": np.nan, "NONE": np.nan})
            .dropna()
            .tolist()
        )
        wl_pat = {v.upper() for v in vals}

        # IMPORTANT: return empty wl_samp so existing downstream code uses PATIENT_ID
        return wl_pat, set()

    wl_pat, wl_samp = load_id_set(WHITELIST_PATH)

    merged["PATIENT_ID"] = merged["PATIENT_ID"].astype(str).str.strip()
    merged["SAMPLE_ID"] = merged["SAMPLE_ID"].astype(str).str.strip()

    if wl_samp:
        mask_wl = merged["SAMPLE_ID"].str.upper().isin({v.upper() for v in wl_samp})
        print(f"[WHITELIST] Matching by SAMPLE_ID")
    else:
        mask_wl = merged["PATIENT_ID"].str.upper().isin({v.upper() for v in wl_pat})
        print(f"[WHITELIST] Matching by PATIENT_ID")

    merged_wl = merged.loc[mask_wl].copy()
    print(f"[WHITELIST] Input rows: {len(merged)}")
    print(f"[WHITELIST] Matched rows (before de-dup): {int(mask_wl.sum())}")

    merged_wl = merged_wl.sort_values(["PATIENT_ID", "SAMPLE_ID"]).drop_duplicates(
        "PATIENT_ID", keep="first"
    )
    print(f"[WHITELIST] Unique patients after de-dup: {len(merged_wl)}")

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

    cohort = merged_wl.copy()
    cohort = cohort[cohort[os_m_col].notna() & cohort[os_s_col].notna()].copy()

    cohort["OS_EVENT"] = cohort[os_s_col].map(os_event).astype(int)
    cohort["OS_MONTHS"] = pd.to_numeric(cohort[os_m_col], errors="coerce")

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

    cohort = cohort[cohort["SAMPLE_ID"].isin(expr_raw.columns)].copy()
    order = cohort.sort_values("PATIENT_ID")["SAMPLE_ID"].astype(str).tolist()
    expr_raw = expr_raw[order]

    expr_full = clean_and_collapse_genes(expr_raw)
    expr_full = expr_full.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    std_raw = expr_full.std(axis=1, skipna=True)
    expr_full = expr_full.loc[std_raw > 1e-8]

    meta = cohort.sort_values("PATIENT_ID").reset_index(drop=True)
    print(
        f"Expression matrix (full): {expr_full.shape[0]} genes × {expr_full.shape[1]} samples"
    )

    # ------------------- Splits -------------------
    print(
        f"[Split] Using fixed SPLIT_SEED={SPLIT_SEED} for train/test (and val) indices across all runs."
    )
    y_event_all = meta["OS_EVENT"].to_numpy()
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_FRAC, random_state=SPLIT_SEED
    )
    ((trval_idx, te_idx),) = sss1.split(np.zeros(len(y_event_all)), y_event_all)

    if VAL_FRAC and VAL_FRAC > 0.0:
        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=VAL_FRAC / (1 - TEST_FRAC), random_state=SPLIT_SEED
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

    # ------------------- Pathway ACTIVITY + (optional) CONSISTENCY -------------------
    if not USE_PATHWAYS:
        if USE_PATHWAY_CONSISTENCY:
            print("\n[Pathways] USE_PATHWAYS=False, so consistency is also skipped.")
        else:
            print("\n[Pathways] Skipping pathway features (USE_PATHWAYS=False).")
        PATHWAY_FEATURES = np.zeros((len(meta), 0), dtype=float)
        pathway_feat_names = []
        CONSIST_FEATURES = np.zeros((len(meta), 0), dtype=float)
        consist_feat_names = []
    else:
        print("\nLoading pathway-level features (activity + consistency)…")

        def _read_pathway_table(p):
            try:
                df = pd.read_table(p, sep="\t", index_col=0, low_memory=False)
            except Exception:
                df = pd.read_csv(p, sep=",", index_col=0, low_memory=False)
            df.columns = [str(c).strip() for c in df.columns]
            return df

        act_df_raw = _read_pathway_table(P_PATH_ACTIVITY)  # rows=pathways, cols=samples
        con_df_raw = _read_pathway_table(P_PATH_CONSIST)  # rows=pathways, cols=samples

        def _collapse_dupe_cols(df: pd.DataFrame) -> pd.DataFrame:
            if df.columns.has_duplicates:
                df = df.groupby(level=0, axis=1).mean()
            return df

        act_df_raw = _collapse_dupe_cols(act_df_raw)
        con_df_raw = _collapse_dupe_cols(con_df_raw)

        def _canon_cols_to_dash(df: pd.DataFrame) -> pd.DataFrame:
            mapping = {}
            for c in df.columns:
                cc = canon_mb_dash(c)
                mapping[c] = cc if cc is not None else c.strip().upper()
            df2 = df.copy()
            df2.columns = [mapping[c] for c in df.columns]
            df2 = _collapse_dupe_cols(df2)
            return df2

        act_canon = _canon_cols_to_dash(act_df_raw)
        con_canon = _canon_cols_to_dash(con_df_raw)

        sample_order = meta["SAMPLE_ID"].astype(str).tolist()
        cohort_dash_order = [canon_mb_dash(s) for s in sample_order]
        cohort_raw_up_order = [s.strip().upper() for s in sample_order]

        canon_pool = set(act_canon.columns) & set(con_canon.columns)
        order_cols = [
            c for c in cohort_dash_order if (c is not None and c in canon_pool)
        ]

        def _standardize_block(df_sub, order_cols):
            train_pos = np.array(
                [
                    order_cols.index(cohort_dash_order[i])
                    for i in tr_idx
                    if cohort_dash_order[i] in order_cols
                ],
                dtype=int,
            )
            M = df_sub.to_numpy(dtype=float)
            if not _rows_prestandardized(M, train_pos):
                M = winsorize_rows(M, WINSOR_Q[0], WINSOR_Q[1])
                M = train_only_row_zscore(M, train_pos)
            row_nan = np.isnan(M).any(1)
            kept_rows = np.array(df_sub.index)[~row_nan]
            M = M[~row_nan]
            return M, kept_rows, train_pos

        def _standardize_block_raw(df_raw, raw_cols):
            train_pos = np.array(
                [raw_cols.index(cohort_raw_up_order[i]) for i in tr_idx], dtype=int
            )
            M = df_raw.reindex(columns=raw_cols).to_numpy(dtype=float)
            if not _rows_prestandardized(M, train_pos):
                M = winsorize_rows(M, WINSOR_Q[0], WINSOR_Q[1])
                M = train_only_row_zscore(M, train_pos)
            row_nan = np.isnan(M).any(1)
            kept_rows = np.array(df_raw.index)[~row_nan]
            M = M[~row_nan]
            return M, kept_rows, train_pos

        def _svd_train_center(Xtr_like, X_all_like):
            mu = (
                Xtr_like.mean(axis=0, keepdims=True)
                if Xtr_like.size
                else np.zeros((1, X_all_like.shape[1]))
            )
            Xtr0 = Xtr_like - mu
            U, S, Vt = (
                np.linalg.svd(Xtr0, full_matrices=False)
                if Xtr0.size
                else (
                    np.zeros((0, 0)),
                    np.zeros((0,)),
                    np.zeros((0, X_all_like.shape[1])),
                )
            )
            return mu, U, S, Vt

        def _split_by_positions(X_all, pos_list):
            ok = [p for p in pos_list if p is not None]
            return X_all[np.array(ok, int), :], np.array(
                [i for i, p in enumerate(pos_list) if p is None], int
            )

        print(f"[Pathways] cohort samples: {len(sample_order)}")
        print(f"[Pathways] (CANON) overlap candidates: {len(order_cols)}")

        using_raw = False
        PATHWAY_FEATURES = np.zeros((len(meta), 0), dtype=float)
        pathway_feat_names = []

        # Placeholders for act PCs per split (needed later if we add consistency)
        Ttr_act_full = Tva_act_full = Tte_act_full = None
        keep_idxs_act = np.array([], dtype=int)

        # ---------- ACTIVITY block (exactly as before) ----------
        if len(order_cols) == 0:
            # RAW fallback
            act_raw_up_cols = {c.strip().upper() for c in act_df_raw.columns}
            con_raw_up_cols = {c.strip().upper() for c in con_df_raw.columns}
            raw_cols = [
                s
                for s in cohort_raw_up_order
                if (s in act_raw_up_cols and s in con_raw_up_cols)
            ]
            if len(raw_cols) == 0:
                print(
                    f"[Pathways] Columns present after normalization: 0 / {len(sample_order)}."
                )
            else:
                using_raw = True
                A, kept_rows_act, train_pos_act = _standardize_block_raw(
                    act_df_raw, raw_cols
                )

                X_all = A.T  # N × P_pathways
                Xtr_act = X_all[tr_idx, :]
                Xva_act = X_all[va_idx, :] if HAS_VAL else np.zeros((0, X_all.shape[1]))
                Xte_act = X_all[te_idx, :]

                mu_act, U_act, S_act, Vt_act = _svd_train_center(Xtr_act, X_all)
                evr_train_act = (S_act**2) / max((S_act**2).sum(), 1e-12)
                cum_act = np.cumsum(evr_train_act)
                r_evr_act = (
                    int(np.searchsorted(cum_act, PATHWAY_TARGET_EVR) + 1)
                    if cum_act.size
                    else 0
                )
                print(
                    f"[Pathways] PCA(train): {r_evr_act} PCs reach {PATHWAY_TARGET_EVR*100:.1f}% EVR."
                )

                Ttr_act_full = (
                    (Xtr_act - mu_act) @ Vt_act.T
                    if Xtr_act.size
                    else np.zeros((Xtr_act.shape[0], 0))
                )
                Tva_act_full = (
                    ((Xva_act - mu_act) @ Vt_act.T)
                    if (HAS_VAL and Vt_act.size)
                    else Xva_act
                )
                Tte_act_full = (Xte_act - mu_act) @ Vt_act.T if Vt_act.size else Xte_act

                pc_names_act = [f"actPC{i+1}" for i in range(Ttr_act_full.shape[1])]
                rank_df_act = (
                    cox_rank_features_matrix(
                        Ttr_act_full,
                        meta["OS_MONTHS"].to_numpy()[tr_idx],
                        meta["OS_EVENT"].to_numpy()[tr_idx],
                        pc_names_act,
                        penalizer=COX_PENALIZER,
                    )
                    if (Ttr_act_full.shape[0] > 0 and Ttr_act_full.shape[1] > 0)
                    else pd.DataFrame(columns=["feat", "idx", "z", "p", "coef"])
                )
                rank_df_act.to_csv(
                    OUTDIR
                    / f"pathway_activityPC_survival_ranking_train{seed_suffix}.csv",
                    index=False,
                )
                Kp = int(min(PATHWAY_PC_TOP_K, Ttr_act_full.shape[1]))
                print(
                    f"[Pathways] Activity PC selection: top {Kp} by Cox Z out of {Ttr_act_full.shape[1]}."
                )
                keep_idxs_act = (
                    rank_df_act.head(Kp)["idx"].to_numpy()
                    if Kp > 0 and len(rank_df_act)
                    else np.array([], int)
                )

                # Final PATHWAY_FEATURES in meta order
                T_all = np.zeros((len(meta), Ttr_act_full.shape[1]), dtype=float)
                T_all[tr_idx] = Ttr_act_full
                if HAS_VAL and Tva_act_full.size:
                    T_all[va_idx] = Tva_act_full
                T_all[te_idx] = Tte_act_full
                PATHWAY_FEATURES = (
                    T_all[:, keep_idxs_act]
                    if keep_idxs_act.size
                    else np.zeros((len(meta), 0))
                )
                pathway_feat_names = [
                    f"pathPC{int(i)+1}" for i in keep_idxs_act.tolist()
                ]
                print(f"[Pathways] Activity PCs kept: {PATHWAY_FEATURES.shape[1]}")

        else:
            # CANON path
            act_sub = act_canon.reindex(columns=order_cols)
            A, kept_rows_act, train_pos_act = _standardize_block(act_sub, order_cols)
            print(
                f"[Pathways][CANON] A shape (pathways×samples): {A.shape}, TRAIN cols: {len(train_pos_act)}"
            )

            X_all = A.T  # N × P_pathways

            col_pos_by_dash = {c: i for i, c in enumerate(order_cols)}

            def _pos_list(idx_list):
                return [
                    col_pos_by_dash.get(cohort_dash_order[j], None) for j in idx_list
                ]

            tr_pos = _pos_list(tr_idx)
            mask_tr_ok = np.array([p is not None for p in tr_pos])
            tr_idx_ok = tr_idx[mask_tr_ok]
            va_pos = _pos_list(va_idx)
            te_pos = _pos_list(te_idx)

            def _take_rows(X, pos_list):
                ok = [p for p in pos_list if p is not None]
                return X[np.array(ok, int), :], np.array(
                    [i for i, p in enumerate(pos_list) if p is None], int
                )

            Xtr_act_raw, _ = _take_rows(X_all, tr_pos)
            Xva_act_raw, _ = (
                _take_rows(X_all, va_pos)
                if HAS_VAL
                else (np.zeros((0, X_all.shape[1])), np.array([], int))
            )
            Xte_act_raw, _ = _take_rows(X_all, te_pos)

            mu_act, U_act, S_act, Vt_act = _svd_train_center(Xtr_act_raw, X_all)
            evr_train_act = (S_act**2) / max((S_act**2).sum(), 1e-12)
            cum_act = np.cumsum(evr_train_act)
            r_evr_act = (
                int(np.searchsorted(cum_act, PATHWAY_TARGET_EVR) + 1)
                if cum_act.size
                else 0
            )
            print(
                f"[Pathways] PCA(train): {r_evr_act} PCs reach {PATHWAY_TARGET_EVR*100:.1f}% EVR."
            )

            Ttr_act_full = (
                (Xtr_act_raw - mu_act) @ Vt_act.T
                if Xtr_act_raw.size
                else np.zeros((Xtr_act_raw.shape[0], 0))
            )
            Tva_act_full = (
                ((Xva_act_raw - mu_act) @ Vt_act.T)
                if (HAS_VAL and Vt_act.size)
                else Xva_act_raw
            )
            Tte_act_full = (
                ((Xte_act_raw - mu_act) @ Vt_act.T) if Vt_act.size else Xte_act_raw
            )

            pc_names_act = [f"actPC{i+1}" for i in range(Ttr_act_full.shape[1])]
            rank_df_act = (
                cox_rank_features_matrix(
                    Ttr_act_full,
                    meta["OS_MONTHS"].to_numpy()[tr_idx_ok],
                    meta["OS_EVENT"].to_numpy()[tr_idx_ok],
                    pc_names_act,
                    penalizer=COX_PENALIZER,
                )
                if (Ttr_act_full.shape[1] > 0 and Ttr_act_full.shape[0] > 0)
                else pd.DataFrame(columns=["feat", "idx", "z", "p", "coef"])
            )
            rank_df_act.to_csv(
                OUTDIR / f"pathway_activityPC_survival_ranking_train{seed_suffix}.csv",
                index=False,
            )
            Kp = int(min(PATHWAY_PC_TOP_K, Ttr_act_full.shape[1]))
            print(
                f"[Pathways] Activity PC selection: top {Kp} by Cox Z out of {Ttr_act_full.shape[1]}."
            )
            keep_idxs_act = (
                rank_df_act.head(Kp)["idx"].to_numpy()
                if Kp > 0 and len(rank_df_act)
                else np.array([], int)
            )

            PATHWAY_FEATURES = np.zeros((len(meta), keep_idxs_act.size), dtype=float)

            def _fill_rows(Tsplit_full, pos_list, idx_array):
                mask_ok = np.array([p is not None for p in pos_list])
                meta_rows = idx_array[mask_ok]
                PATHWAY_FEATURES[meta_rows] = Tsplit_full[:, keep_idxs_act]

            _fill_rows(Ttr_act_full, tr_pos, tr_idx)
            if HAS_VAL and Tva_act_full.size:
                _fill_rows(Tva_act_full, va_pos, va_idx)
            _fill_rows(Tte_act_full, te_pos, te_idx)

            pathway_feat_names = [f"pathPC{int(i)+1}" for i in keep_idxs_act.tolist()]
            print(f"[Pathways] Activity PCs kept: {PATHWAY_FEATURES.shape[1]}")

        # ---------- CONSISTENCY block (optional, with residualization) ----------
        CONSIST_FEATURES = np.zeros((len(meta), 0), dtype=float)
        consist_feat_names = []

        if USE_PATHWAY_CONSISTENCY:
            # Build consistency matrix aligned the same way as activity
            if len(order_cols) == 0 and using_raw:
                # RAW
                C, kept_rows_con, train_pos_con = _standardize_block_raw(
                    con_df_raw, raw_cols
                )
                X_all_c = C.T
                Xtr_c = X_all_c[tr_idx, :]
                Xva_c = (
                    X_all_c[va_idx, :] if HAS_VAL else np.zeros((0, X_all_c.shape[1]))
                )
                Xte_c = X_all_c[te_idx, :]

                mu_c, U_c, S_c, Vt_c = _svd_train_center(Xtr_c, X_all_c)
                evr_train_c = (S_c**2) / max((S_c**2).sum(), 1e-12)
                cum_c = np.cumsum(evr_train_c)
                r_evr_c = (
                    int(np.searchsorted(cum_c, CONSIST_TARGET_EVR) + 1)
                    if cum_c.size
                    else 0
                )
                print(
                    f"[Consistency] PCA(train): {r_evr_c} PCs reach {CONSIST_TARGET_EVR*100:.1f}% EVR."
                )

                Ttr_c_full = (
                    (Xtr_c - mu_c) @ Vt_c.T
                    if Xtr_c.size
                    else np.zeros((Xtr_c.shape[0], 0))
                )
                Tva_c_full = (
                    ((Xva_c - mu_c) @ Vt_c.T) if (HAS_VAL and Vt_c.size) else Xva_c
                )
                Tte_c_full = (Xte_c - mu_c) @ Vt_c.T if Vt_c.size else Xte_c

                # Residualize C vs kept A (on TRAIN)
                if (
                    RESIDUALIZE_CONSISTENCY
                    and keep_idxs_act.size > 0
                    and Ttr_act_full is not None
                ):
                    A_tr_sel = Ttr_act_full[:, keep_idxs_act]
                    # β solves A_tr_sel β ≈ Ttr_c_full  (multi-target)
                    beta, *_ = np.linalg.lstsq(A_tr_sel, Ttr_c_full, rcond=None)
                    Rtr = Ttr_c_full - A_tr_sel @ beta
                    Ava_sel = (
                        Tva_act_full[:, keep_idxs_act]
                        if (HAS_VAL and Tva_act_full is not None and Tva_act_full.size)
                        else np.zeros((Tva_c_full.shape[0], A_tr_sel.shape[1]))
                    )
                    Ate_sel = (
                        Tte_act_full[:, keep_idxs_act]
                        if (Tte_act_full is not None and Tte_act_full.size)
                        else np.zeros((Tte_c_full.shape[0], A_tr_sel.shape[1]))
                    )
                    Rva = (
                        (Tva_c_full - Ava_sel @ beta)
                        if HAS_VAL
                        else np.zeros((0, Rtr.shape[1]))
                    )
                    Rte = Tte_c_full - Ate_sel @ beta
                else:
                    # No residualization or no A PCs: use raw C PCs
                    Rtr, Rva, Rte = (
                        Ttr_c_full,
                        (Tva_c_full if HAS_VAL else np.zeros((0, 0))),
                        Tte_c_full,
                    )

            else:
                # CANON
                con_sub = con_canon.reindex(columns=order_cols)
                C, kept_rows_con, train_pos_con = _standardize_block(
                    con_sub, order_cols
                )

                X_all_c = C.T
                col_pos_by_dash = {c: i for i, c in enumerate(order_cols)}

                def _pos_list(idx_list):
                    return [
                        col_pos_by_dash.get(cohort_dash_order[j], None)
                        for j in idx_list
                    ]

                tr_pos = _pos_list(tr_idx)
                va_pos = _pos_list(va_idx)
                te_pos = _pos_list(te_idx)

                def _take_rows(X, pos_list):
                    ok = [p for p in pos_list if p is not None]
                    return X[np.array(ok, int), :], np.array(
                        [i for i, p in enumerate(pos_list) if p is None], int
                    )

                Xtr_c_raw, _ = _take_rows(X_all_c, tr_pos)
                Xva_c_raw, _ = (
                    _take_rows(X_all_c, va_pos)
                    if HAS_VAL
                    else (np.zeros((0, X_all_c.shape[1])), np.array([], int))
                )
                Xte_c_raw, _ = _take_rows(X_all_c, te_pos)

                mu_c, U_c, S_c, Vt_c = _svd_train_center(Xtr_c_raw, X_all_c)
                evr_train_c = (S_c**2) / max((S_c**2).sum(), 1e-12)
                cum_c = np.cumsum(evr_train_c)
                r_evr_c = (
                    int(np.searchsorted(cum_c, CONSIST_TARGET_EVR) + 1)
                    if cum_c.size
                    else 0
                )
                print(
                    f"[Consistency] PCA(train): {r_evr_c} PCs reach {CONSIST_TARGET_EVR*100:.1f}% EVR."
                )

                Ttr_c_full = (
                    (Xtr_c_raw - mu_c) @ Vt_c.T
                    if Xtr_c_raw.size
                    else np.zeros((Xtr_c_raw.shape[0], 0))
                )
                Tva_c_full = (
                    ((Xva_c_raw - mu_c) @ Vt_c.T)
                    if (HAS_VAL and Vt_c.size)
                    else Xva_c_raw
                )
                Tte_c_full = ((Xte_c_raw - mu_c) @ Vt_c.T) if Vt_c.size else Xte_c_raw

                if (
                    RESIDUALIZE_CONSISTENCY
                    and keep_idxs_act.size > 0
                    and Ttr_act_full is not None
                ):
                    A_tr_sel = Ttr_act_full[:, keep_idxs_act]
                    beta, *_ = np.linalg.lstsq(A_tr_sel, Ttr_c_full, rcond=None)
                    Rtr = Ttr_c_full - A_tr_sel @ beta
                    Ava_sel = (
                        Tva_act_full[:, keep_idxs_act]
                        if (HAS_VAL and Tva_act_full is not None and Tva_act_full.size)
                        else np.zeros((Tva_c_full.shape[0], A_tr_sel.shape[1]))
                    )
                    Ate_sel = (
                        Tte_act_full[:, keep_idxs_act]
                        if (Tte_act_full is not None and Tte_act_full.size)
                        else np.zeros((Tte_c_full.shape[0], A_tr_sel.shape[1]))
                    )
                    Rva = (
                        (Tva_c_full - Ava_sel @ beta)
                        if HAS_VAL
                        else np.zeros((0, Rtr.shape[1]))
                    )
                    Rte = Tte_c_full - Ate_sel @ beta
                else:
                    Rtr, Rva, Rte = (
                        Ttr_c_full,
                        (Tva_c_full if HAS_VAL else np.zeros((0, 0))),
                        Tte_c_full,
                    )

            # Rank residualized consistency PCs by Cox on TRAIN, keep top-Kc
            cons_pc_names = [f"consPC{i+1}" for i in range(Rtr.shape[1])]
            rank_df_cons = (
                cox_rank_features_matrix(
                    Rtr,
                    meta["OS_MONTHS"].to_numpy()[tr_idx_ok],
                    meta["OS_EVENT"].to_numpy()[tr_idx_ok],
                    cons_pc_names,
                    penalizer=COX_PENALIZER,
                )
                if (Rtr.shape[0] > 0 and Rtr.shape[1] > 0)
                else pd.DataFrame(columns=["feat", "idx", "z", "p", "coef"])
            )
            rank_df_cons.to_csv(
                OUTDIR
                / f"pathway_consistencyPC_survival_ranking_train{seed_suffix}.csv",
                index=False,
            )
            Kc = int(min(CONSIST_PC_TOP_K, Rtr.shape[1]))
            keep_idxs_cons = (
                rank_df_cons.head(Kc)["idx"].to_numpy()
                if Kc > 0 and len(rank_df_cons)
                else np.array([], int)
            )
            print(
                f"[Consistency] Residualized PCs kept: {keep_idxs_cons.size} / {Rtr.shape[1]}"
            )

            # Stitch residualized C back to meta order
            CONSIST_FEATURES = np.zeros((len(meta), keep_idxs_cons.size), dtype=float)
            if len(order_cols) == 0 and using_raw:
                # RAW splits: rows already align with tr_idx / va_idx / te_idx
                CONSIST_FEATURES[tr_idx] = Rtr[:, keep_idxs_cons]
                if HAS_VAL and Rva.size:
                    CONSIST_FEATURES[va_idx] = Rva[:, keep_idxs_cons]
                CONSIST_FEATURES[te_idx] = Rte[:, keep_idxs_cons]
            else:
                # CANON: need to map via the same position lists used above
                col_pos_by_dash = {c: i for i, c in enumerate(order_cols)}

                def _pos_list(idx_list):
                    return [
                        col_pos_by_dash.get(cohort_dash_order[j], None)
                        for j in idx_list
                    ]

                tr_pos = _pos_list(tr_idx)
                va_pos = _pos_list(va_idx)
                te_pos = _pos_list(te_idx)

                def _fill_rows_cons(Rsplit, pos_list, idx_array):
                    mask_ok = np.array([p is not None for p in pos_list])
                    meta_rows = idx_array[mask_ok]
                    CONSIST_FEATURES[meta_rows] = Rsplit[:, keep_idxs_cons]

                _fill_rows_cons(Rtr, tr_pos, tr_idx)
                if HAS_VAL and Rva.size:
                    _fill_rows_cons(Rva, va_pos, va_idx)
                _fill_rows_cons(Rte, te_pos, te_idx)

            # Names for residualized consistency features
            consist_feat_names = [f"consRPC{int(i)+1}" for i in keep_idxs_cons.tolist()]

    # ------------------- TRAIN-only gene filter + (maybe) z-score for PCA -------------------
    zinfo = detect_prestandardized(expr_full)
    print(
        f"[Expr check] pre-standardized? {zinfo['is_z']} (median|mean|={zinfo['med_abs_mean']:.3f}, median(std)={zinfo['med_std']:.3f})"
    )

    if GENE_SELECT_MODE == "quantile":
        if zinfo["is_z"]:
            iqr_train = train_iqr(expr_full, tr_cols)
            thresh = iqr_train.quantile(GENE_VAR_TOP_Q)
            keep_genes = iqr_train.index[iqr_train >= thresh]
        else:
            var_train = expr_full.iloc[:, tr_cols].var(axis=1, skipna=True)
            thresh = var_train.quantile(GENE_VAR_TOP_Q)
            keep_genes = var_train.index[var_train >= thresh]
    else:
        if zinfo["is_z"]:
            iqr_train = train_iqr(expr_full, tr_cols)
            keep_genes = iqr_train.sort_values(ascending=False).index[:GENE_N_TOP]
        else:
            var_train = expr_full.iloc[:, tr_cols].var(axis=1, skipna=True)
            keep_genes = var_train.sort_values(ascending=False).index[:GENE_N_TOP]

    expr = expr_full.loc[keep_genes].copy()

    if not zinfo["is_z"]:
        mu_g = expr.iloc[:, tr_cols].mean(axis=1)
        sd_g = expr.iloc[:, tr_cols].std(axis=1)
        sd_g = sd_g.replace([0.0, np.inf, -np.inf], 1.0).fillna(1.0)
        expr = expr.sub(mu_g, axis=0).div(sd_g, axis=0)
    else:
        print("[Expr] Skipping per-gene z-score (matrix already z-scored).")

    # ------------------- GES (ssGSEA or rank-mean) — on FULL cleaned matrix -------------------
    ER_SET = ["ESR1", "PGR", "FOXA1", "GREB1", "XBP1", "BCL2", "GATA3"]
    PROLIF_SET = ["MKI67", "PCNA", "TOP2A", "CCNB1", "BIRC5", "UBE2C", "CDC20", "AURKA"]

    def ges_rankmean(E_df, genes):
        if E_df.shape[0] == 0:
            return pd.Series(0.0, index=E_df.columns)
        present = present_genes(genes, E_df.index)
        if not present:
            return pd.Series(0.0, index=E_df.columns)
        ranks = E_df.rank(axis=0, method="average")
        G = float(E_df.shape[0])
        nranks = (ranks - 1.0) / max(G - 1.0, 1.0)
        return nranks.loc[present].mean(axis=0)

    def ges_ssgsea(E_df, genes, alpha=0.25):
        if E_df.shape[0] == 0:
            return pd.Series(0.0, index=E_df.columns)
        present = present_genes(genes, E_df.index)
        if not present:
            return pd.Series(0.0, index=E_df.columns)
        ranks_all = E_df.rank(axis=0, ascending=False, method="average")
        S = set(present)
        out = []
        for col in E_df.columns:
            r = ranks_all[col].sort_values(ascending=True)  # 1..G
            idx_is_hit = r.index.to_series().apply(lambda g: g in S).to_numpy()
            rvals = r.to_numpy(dtype=float)
            G = len(rvals)
            Nh = int(idx_is_hit.sum())
            Nm = G - Nh
            if Nh == 0:
                out.append(0.0)
                continue
            w = (rvals**alpha) * idx_is_hit
            Phit = np.cumsum(w) / (w.sum() + 1e-12)
            Pmiss = np.cumsum((~idx_is_hit).astype(float)) / max(Nm, 1)
            RS = Phit - Pmiss
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
            f"GES overlaps (train): ER={len(ER_present)}/{len(ER_SET)}, Prolif={len(PRO_present)}/{len(PROLIF_SET)}"
        )
    else:
        ges_ER_tr = ges_Pro_tr = np.zeros((len(tr_cols), 0))
        ges_ER_va = ges_Pro_va = np.zeros((len(va_cols), 0))
        ges_ER_te = ges_Pro_te = np.zeros((len(te_cols), 0))

    # ------------------- PCA on gene expression (TRAIN only) -------------------
    print("Running PCA on TRAIN only (genes)…")
    Xtr_genes = expr.iloc[:, tr_cols].T.values
    Xva_genes = (
        expr.iloc[:, va_cols].T.values if HAS_VAL else np.zeros((0, expr.shape[0]))
    )
    Xte_genes = expr.iloc[:, te_cols].T.values

    mu_g_train = (
        Xtr_genes.mean(axis=0, keepdims=True)
        if Xtr_genes.size
        else np.zeros((1, expr.shape[0]))
    )
    Xtr0 = Xtr_genes - mu_g_train
    Xva0 = Xva_genes - mu_g_train if HAS_VAL else Xva_genes
    Xte0 = Xte_genes - mu_g_train

    U, S, Vt = (
        np.linalg.svd(Xtr0, full_matrices=False)
        if Xtr0.size
        else (
            np.zeros((0, 0)),
            np.zeros((0,)),
            np.zeros((0, Xtr0.shape[1] if Xtr0.ndim == 2 else 0)),
        )
    )
    evr_train = (S**2) / max((S**2).sum(), 1e-12)
    cum = np.cumsum(evr_train)
    r_evr = int(np.searchsorted(cum, TARGET_EVR) + 1) if cum.size else 0
    r_evr = max(r_evr, MIN_PC)
    print(f"PCA(train): {r_evr} PCs reach {TARGET_EVR*100:.1f}% EVR.")

    Ttr_full = Xtr0 @ Vt.T if Xtr0.size else np.zeros((Xtr0.shape[0], 0))
    Tva_full = Xva0 @ Vt.T if (HAS_VAL and Vt.size) else Xva0
    Tte_full = Xte0 @ Vt.T if Vt.size else Xte0
    Ttr = Ttr_full[:, :r_evr]
    Tva = Tva_full[:, :r_evr] if HAS_VAL else Tva_full
    Tte = Tte_full[:, :r_evr]
    pc_names = [f"PC{i+1}" for i in range(r_evr)]
    evr_used = evr_train[:r_evr]

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
    pc_rank.to_csv(OUTDIR / f"pc_survival_ranking_train{seed_suffix}.csv", index=False)

    K = SURV_TOP_K if SURV_TOP_K is not None else min(30, r_evr)
    if K == 0:
        keep_idxs = np.array([], dtype=int)
        print("PC selection: SURV_TOP_K=0 → skipping gene PCs.")
    else:
        K = int(np.clip(K, MIN_PC, min(MAX_PC, r_evr)))
        keep_idxs = pc_rank.head(K)["idx"].to_numpy()
        print(
            f"PC selection: using top {len(keep_idxs)} by Cox Z out of {r_evr} PCA PCs (bounds [{MIN_PC}, {min(MAX_PC, r_evr)}])."
        )

    Ttr_sel = Ttr[:, keep_idxs] if keep_idxs.size else np.zeros((Ttr.shape[0], 0))
    Tva_sel = Tva[:, keep_idxs] if (HAS_VAL and keep_idxs.size) else np.zeros((0, 0))
    Tte_sel = Tte[:, keep_idxs] if keep_idxs.size else np.zeros((Tte.shape[0], 0))
    sel_names = [pc_names[i] for i in keep_idxs] if keep_idxs.size else []
    print(f"Gene PCs selected by survival Z: {len(sel_names)}")

    # ------------------- Clinical covariates (TRAIN-fitted encoders) -------------------
    # ------------------- Clinical covariates (TRAIN-fitted encoders) -------------------
    NUMERIC_CLIN = [c for c in ["AGE_AT_DIAGNOSIS", "TUMOR_SIZE"] if c in meta.columns]
    CAT_CLIN = [c for c in ["GRADE", "INFERRED_MENOPAUSAL_STATE"] if c in meta.columns]

    # If True, will drop the first OHE column *only when there are >=2 columns*.
    DROP_FIRST_OHE = True

    _MISSING_TOKENS = {
        "",
        "NA",
        "N/A",
        "NONE",
        "NULL",
        "NAN",
        "UNKNOWN",
        "UNAVAILABLE",
        "NOT REPORTED",
    }

    def _clean_cat_series(s: pd.Series) -> pd.Series:
        """
        Normalize strings while preserving missing as <NA> (do NOT create 'NAN' category).
        """
        x = s.astype("string")  # preserves <NA>
        x = x.str.strip().str.upper()
        x = x.mask(x.isna() | x.isin(_MISSING_TOKENS), other=pd.NA)
        return x

    def _clean_grade_series(s: pd.Series) -> pd.Series:
        """
        Normalize GRADE into {'1','2','3'} strings; anything else -> <NA>.
        Handles: '1','2','3','GRADE 2','G3','III','II','I', etc.
        """
        x = _clean_cat_series(s)

        # digits anywhere
        dig = x.str.extract(r"([123])", expand=False)

        # roman numerals (only exact tokens)
        roman_map = {"I": "1", "II": "2", "III": "3"}
        roman = x.replace(roman_map)

        out = dig.fillna(roman)
        out = out.where(out.isin(["1", "2", "3"]), other=pd.NA)
        return out

    def _prep_cat(meta_df: pd.DataFrame, col: str) -> pd.Series:
        if col.upper() == "GRADE":
            return _clean_grade_series(meta_df[col])
        else:
            return _clean_cat_series(meta_df[col])

    def _train_fitted_ohe_specs_and_impute(meta_df: pd.DataFrame, tr_idx, cat_cols):
        """
        TRAIN-fitted OHE specs + TRAIN-mode imputation values per categorical column.
        Missing is imputed to TRAIN mode (so it is NOT a separate 'missing' category).
        """
        specs = {}
        fill_vals = {}

        for col in cat_cols:
            s_tr = _prep_cat(meta_df.iloc[tr_idx], col)  # preserves <NA> at this stage

            s_tr_non = s_tr.dropna()
            if len(s_tr_non) == 0:
                fill_val = pd.NA
            else:
                # mode() can return multiple; pick first deterministically
                fill_val = s_tr_non.mode(dropna=True).iloc[0]

            fill_vals[col] = fill_val

            # Impute TRAIN before building dummies
            if fill_val is not pd.NA:
                s_tr_imp = s_tr.fillna(fill_val)
            else:
                s_tr_imp = s_tr  # all-missing case

            d_tr = pd.get_dummies(s_tr_imp, prefix=col, dummy_na=False)

            if DROP_FIRST_OHE and d_tr.shape[1] > 1:
                d_tr = d_tr.iloc[:, 1:]  # drop baseline safely

            specs[col] = list(d_tr.columns)

        return specs, fill_vals

    def _apply_ohe_with_specs(meta_df: pd.DataFrame, col: str, cols_spec, fill_val):
        """
        Apply TRAIN-fitted dummy columns to ALL rows, after imputing missing -> TRAIN mode.
        """
        s_all = _prep_cat(meta_df, col)

        if fill_val is not pd.NA:
            s_all = s_all.fillna(fill_val)

        d_all = pd.get_dummies(s_all, prefix=col, dummy_na=False)

        if DROP_FIRST_OHE and d_all.shape[1] > 1:
            d_all = d_all.iloc[:, 1:]

        if cols_spec:
            d_all = d_all.reindex(columns=cols_spec, fill_value=0)
        else:
            d_all = pd.DataFrame(index=meta_df.index)

        return d_all

    def build_clin_matrices_train_fitted(meta_df, tr_idx, va_idx, te_idx):
        blocks_all, names = [], []

        # numeric: impute using TRAIN median
        for c in NUMERIC_CLIN:
            s_all = pd.to_numeric(meta_df[c], errors="coerce")
            med = pd.to_numeric(meta_df.iloc[tr_idx][c], errors="coerce").median()
            if not np.isfinite(med):
                med = 0.0
            s_all = s_all.fillna(med)
            blocks_all.append(s_all.to_numpy(dtype=float)[:, None])
            names.append(c)

        # categorical: TRAIN-mode impute + TRAIN-fitted specs
        ohe_specs, fill_vals = _train_fitted_ohe_specs_and_impute(
            meta_df, tr_idx, CAT_CLIN
        )
        for c in CAT_CLIN:
            d_all = _apply_ohe_with_specs(meta_df, c, ohe_specs[c], fill_vals[c])
            if d_all.shape[1] > 0:
                blocks_all.append(d_all.to_numpy(dtype=float))
                names += list(d_all.columns)

        # IMPORTANT: force numeric dtype so Xtr.std can't crash with dtype=object
        X_all = (
            np.hstack(blocks_all).astype(float)
            if blocks_all
            else np.zeros((len(meta_df), 0), dtype=float)
        )

        Xc_tr = X_all[tr_idx]
        Xc_va = (
            X_all[va_idx]
            if HAS_VAL
            else np.zeros((0, X_all.shape[1] if X_all.ndim == 2 else 0), dtype=float)
        )
        Xc_te = X_all[te_idx]
        return Xc_tr, Xc_va, Xc_te, names

    Xc_tr, Xc_va, Xc_te, clin_names = build_clin_matrices_train_fitted(
        meta, tr_idx, va_idx, te_idx
    )

    if "GRADE" in CAT_CLIN:
        g_clean = _prep_cat(meta, "GRADE")
        print("[GRADE] cleaned value counts (incl NA):")
        print(g_clean.value_counts(dropna=False))

    # ------------------- Final matrices + scaling (TRAIN stats) ----------------
    if "PATHWAY_FEATURES" not in locals():
        PATHWAY_FEATURES = np.zeros((len(meta), 0), dtype=float)
        pathway_feat_names = []

    def _print_feature_summary(
        n_gene_pc, n_clin, n_ges, n_path_act, n_path_cons, feat_names, head=6
    ):
        total = len(feat_names)
        print(
            f"[Feature summary] GenePCs={n_gene_pc}, Clinical={n_clin}, GES={n_ges}, "
            f"Path(Activity)={n_path_act}, Path(Consistency)={n_path_cons}; TOTAL={total}"
        )
        if total:
            preview = ", ".join(feat_names[:head])
            print(f"[Feature preview] {preview} {'...' if total>head else ''}")

    # NOTE: keep activity and (optional) consistency as separate blocks for clarity
    n_path_act = PATHWAY_FEATURES.shape[1]
    n_path_cons = CONSIST_FEATURES.shape[1]

    if PATHWAY_ONLY:
        blocks_tr = [PATHWAY_FEATURES[tr_idx]]
        blocks_va = [PATHWAY_FEATURES[va_idx]] if HAS_VAL else []
        blocks_te = [PATHWAY_FEATURES[te_idx]]
        feat_names = [*pathway_feat_names]

        if USE_PATHWAY_CONSISTENCY and n_path_cons > 0:
            blocks_tr.append(CONSIST_FEATURES[tr_idx])
            if HAS_VAL:
                blocks_va.append(CONSIST_FEATURES[va_idx])
            blocks_te.append(CONSIST_FEATURES[te_idx])
            feat_names += consist_feat_names

        if blocks_tr[0].shape[1] == 0 and (
            not (USE_PATHWAY_CONSISTENCY and n_path_cons > 0)
        ):
            raise ValueError(
                "[Pathway-only] No pathway PCs were selected. Increase PATHWAY_PC_TOP_K or check alignment."
            )

        n_gene_pc = 0
        n_clin = 0
        n_ges = 0
        Xtr = np.hstack(blocks_tr)
        Xva = (
            np.hstack(blocks_va)
            if HAS_VAL
            else np.zeros((0, Xtr.shape[1] if Xtr.ndim == 2 else 0))
        )
        Xte = np.hstack(blocks_te)
        _print_feature_summary(
            n_gene_pc, n_clin, n_ges, n_path_act, n_path_cons, feat_names
        )

    else:
        blocks_tr = [Ttr_sel, PATHWAY_FEATURES[tr_idx]]
        blocks_va = [Tva_sel, PATHWAY_FEATURES[va_idx]] if HAS_VAL else []
        blocks_te = [Tte_sel, PATHWAY_FEATURES[te_idx]]

        feat_names = []
        feat_names += [*sel_names] if sel_names else []

        if USE_CLINICAL and Xc_tr.shape[1] > 0:
            blocks_tr.insert(1, Xc_tr)
            if HAS_VAL:
                blocks_va.insert(1, Xc_va)
            blocks_te.insert(1, Xc_te)
            feat_names += clin_names

        if USE_GES:
            insert_at = 1 + (1 if USE_CLINICAL and Xc_tr.shape[1] > 0 else 0)
            blocks_tr.insert(insert_at, ges_ER_tr)
            blocks_tr.insert(insert_at + 1, ges_Pro_tr)
            if HAS_VAL:
                blocks_va.insert(insert_at, ges_ER_va)
                blocks_va.insert(insert_at + 1, ges_Pro_va)
            blocks_te.insert(insert_at, ges_ER_te)
            blocks_te.insert(insert_at + 1, ges_Pro_te)
            feat_names += ["ges_ER", "ges_Prolif"]

        # add activity names
        feat_names += pathway_feat_names

        # add residualized consistency (optional)
        if USE_PATHWAY_CONSISTENCY and n_path_cons > 0:
            blocks_tr.append(CONSIST_FEATURES[tr_idx])
            if HAS_VAL:
                blocks_va.append(CONSIST_FEATURES[va_idx])
            blocks_te.append(CONSIST_FEATURES[te_idx])
            feat_names += consist_feat_names

        Xtr = np.hstack(blocks_tr) if len(blocks_tr) else np.zeros((len(tr_idx), 0))
        Xva = (
            np.hstack(blocks_va)
            if (HAS_VAL and len(blocks_va))
            else np.zeros((0, Xtr.shape[1] if Xtr.ndim == 2 else 0))
        )
        Xte = np.hstack(blocks_te) if len(blocks_te) else np.zeros((len(te_idx), 0))

        n_gene_pc = Ttr_sel.shape[1]
        n_clin = Xc_tr.shape[1] if (USE_CLINICAL and Xc_tr.size) else 0
        n_ges = 2 if USE_GES else 0

        expected_cols = n_gene_pc + n_clin + n_ges + n_path_act + n_path_cons
        if expected_cols != len(feat_names):
            print(
                "[ERROR] names/columns mismatch:",
                "expected",
                expected_cols,
                "have",
                len(feat_names),
                "| parts:",
                n_gene_pc,
                n_clin,
                n_ges,
                n_path_act,
                n_path_cons,
                "| names:",
                len(sel_names),
                (len(clin_names) if (USE_CLINICAL and Xc_tr.size) else 0),
                (2 if USE_GES else 0),
                len(pathway_feat_names),
                len(consist_feat_names),
            )
            raise ValueError("Feature-name / matrix-column mismatch after auto-fix.")

        print(
            f"Feature dims — PCs={n_gene_pc}, Clin={n_clin}, GES={n_ges}, PathAct={n_path_act}, PathCons={n_path_cons}; total={expected_cols}"
        )
        _print_feature_summary(
            n_gene_pc, n_clin, n_ges, n_path_act, n_path_cons, feat_names
        )

    # Standardize design using TRAIN mean/std, BUT do NOT standardize binary one-hot columns.
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True)
    sd = np.where((sd <= 0.0) | ~np.isfinite(sd), 1.0, sd)

    # detect binary columns on TRAIN (robust to floats)
    binary_mask = np.zeros((Xtr.shape[1],), dtype=bool)
    for j in range(Xtr.shape[1]):
        col = Xtr[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        u = np.unique(np.round(col, 8))
        if u.size <= 2 and np.all(np.isin(u, [0.0, 1.0])):
            binary_mask[j] = True

    # keep binary columns untouched (no centering, no scaling)
    mu[:, binary_mask] = 0.0
    sd[:, binary_mask] = 1.0

    Xtr_s = (Xtr - mu) / sd
    Xva_s = (Xva - mu) / sd if HAS_VAL else Xva
    Xte_s = (Xte - mu) / sd

    print(
        f"[Scaling] Standardized {int((~binary_mask).sum())} cols; left {int(binary_mask.sum())} binary cols untouched."
    )

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

    # ---- Block index helpers for additive kernels (BL-A) ----
    def block_indices_from_names(
        feat_names, clin_names, pathway_feat_names, consist_feat_names
    ):
        name_to_idx = {n: i for i, n in enumerate(feat_names)}
        # Clinical (exact provided names)
        clin_idx = [name_to_idx[n] for n in clin_names if n in name_to_idx]

        # Pathway activity (e.g., 'pathPC#')
        act_idx = [name_to_idx[n] for n in pathway_feat_names if n in name_to_idx]

        # Residualized consistency (e.g., 'consRPC#')
        cons_idx = [name_to_idx[n] for n in consist_feat_names if n in name_to_idx]

        # Gene PCs (e.g., 'PC#')
        gene_idx = [i for i, n in enumerate(feat_names) if n.startswith("PC")]

        # GES (optional)
        ges_idx = [name_to_idx.get("ges_ER"), name_to_idx.get("ges_Prolif")]
        ges_idx = [i for i in ges_idx if i is not None]

        return dict(
            gene=gene_idx, clin=clin_idx, act=act_idx, cons=cons_idx, ges=ges_idx
        )

    BLOCKS = block_indices_from_names(
        feat_names=feat_names,
        clin_names=clin_names,
        pathway_feat_names=pathway_feat_names,
        consist_feat_names=consist_feat_names,
    )

    # tensors (unchanged)
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
    from gpytorch.kernels import (
        MaternKernel,
        RBFKernel,
        LinearKernel,
        ScaleKernel,
        AdditiveKernel,
    )
    from gpytorch.means import ZeroMean, LinearMean

    M_inducing = int(min(NUM_INDUCING, Xtr_t.size(0)))

    def _inv_softplus(v: float) -> float:
        import math

        return math.log(math.exp(float(v)) - 1.0)

    def _mk_matern52_block(active_idx, batch_shape=None):
        # Return None if the block is empty
        if not active_idx:
            return None
        k = MaternKernel(nu=2.5, ard_num_dims=len(active_idx), batch_shape=batch_shape)
        # tell GPyTorch which input columns this block sees
        k.active_dims = torch.tensor(active_idx, dtype=torch.long)
        return ScaleKernel(k, batch_shape=batch_shape)  # per-block outputscale

    def make_vs(module, inducing, q, learn_inducing_locations=True, prefer_whiten=True):
        # Compat wrapper for whiten kw
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
        """
        BL-A hidden: Additive Matern-5/2(ARD) over named feature blocks via active_dims.
        Falls back to single Matern-5/2(ARD) if BLOCKS is None or empty.
        """

        def __init__(
            self,
            in_dim,
            hid_dim,
            Xref,
            seed,
            blocks=None,
            kmeans_init=True,
            vi_dist="meanfield",
        ):
            batch_shape = torch.Size([hid_dim])

            # --- inducing locations (shared across batch/hidden outputs) ---
            M_inducing = min(int(NUM_INDUCING), Xref.size(0))
            if kmeans_init:
                with torch.no_grad():
                    km = KMeans(
                        n_clusters=M_inducing, random_state=SPLIT_SEED, n_init=10
                    )
                    centers = km.fit(Xref.cpu().numpy()).cluster_centers_
                    inducing_base = torch.from_numpy(centers).to(
                        dtype=Xref.dtype, device=Xref.device
                    )
            else:
                idx = torch.randperm(Xref.size(0))[:M_inducing]
                inducing_base = Xref[idx]
            inducing = inducing_base.unsqueeze(0).expand(hid_dim, -1, -1).contiguous()

            # --- variational distribution & strategy ---
            if vi_dist.lower() == "meanfield":
                q = MeanFieldVariationalDistribution(
                    M_inducing, batch_shape=batch_shape
                )
            else:
                q = CholeskyVariationalDistribution(M_inducing, batch_shape=batch_shape)
            strat = make_vs(
                self, inducing, q, learn_inducing_locations=True, prefer_whiten=True
            )
            super().__init__(strat)

            # --- mean ---
            self.mean_module = ZeroMean(batch_shape=batch_shape)

            # --- covariance (additive over blocks if provided) ---
            if blocks and any(len(v) > 0 for v in blocks.values()):
                pieces = []
                for key in ("gene", "clin", "act", "cons", "ges"):
                    kblk = _mk_matern52_block(
                        blocks.get(key, []), batch_shape=batch_shape
                    )
                    if kblk is not None:
                        pieces.append(kblk)
                if len(pieces) == 0:
                    # fallback to single Matern if nothing valid
                    self.covar_module = ScaleKernel(
                        MaternKernel(
                            nu=2.5, ard_num_dims=in_dim, batch_shape=batch_shape
                        ),
                        batch_shape=batch_shape,
                    )
                elif len(pieces) == 1:
                    self.covar_module = pieces[0]
                else:
                    # Sum over blocks (AdditiveKernel)
                    self.covar_module = pieces[0]
                    for k in pieces[1:]:
                        self.covar_module = self.covar_module + k
            else:
                # fallback: single ARD Matern-5/2 over all inputs
                self.covar_module = ScaleKernel(
                    MaternKernel(nu=2.5, ard_num_dims=in_dim, batch_shape=batch_shape),
                    batch_shape=batch_shape,
                )

            # --- sensible inits ---
            with torch.no_grad():
                # initialize all sub-kernels’ lengthscales/outputscales
                def _init_one(kmod):
                    if isinstance(kmod, ScaleKernel):
                        _init_one(kmod.base_kernel)
                        try:
                            kmod.outputscale.fill_(1.0)
                        except Exception:
                            pass
                    elif isinstance(kmod, gpytorch.kernels.MaternKernel):
                        try:
                            raw_ls = torch.full_like(
                                kmod.raw_lengthscale, _inv_softplus(1.5)
                            )
                            kmod.raw_lengthscale.copy_(raw_ls)
                        except Exception:
                            pass

                _init_one(self.covar_module)

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    class OutputLayer(ApproximateGP):
        """
        Output GP: Scale[ RBF(ARD) + Linear ] with a ZeroMean.
        Wrapping the sum in a top-level ScaleKernel guarantees an outputscale
        for logging, and the RBF provides a lengthscale for stats.
        """

        def __init__(self, hid_dim, vi_dist="meanfield"):
            M_inducing = min(int(NUM_INDUCING), 512)  # cap if needed
            inducing_h = torch.randn(M_inducing, hid_dim, device=device) * 0.05

            if vi_dist.lower() == "meanfield":
                q = MeanFieldVariationalDistribution(M_inducing)
            else:
                q = CholeskyVariationalDistribution(M_inducing)

            strat = make_vs(
                self, inducing_h, q, learn_inducing_locations=True, prefer_whiten=True
            )
            super().__init__(strat)

            # mean
            self.mean_module = ZeroMean()

            # covariance: Scale[ RBF(ARD) + Linear ]
            rbf = RBFKernel(ard_num_dims=hid_dim)
            lin = LinearKernel()
            summed = rbf + lin
            self.covar_module = ScaleKernel(summed)

            # inits
            with torch.no_grad():
                try:
                    rbf.lengthscale.fill_(1.5)  # reasonable ARD init
                except Exception:
                    pass
                self.covar_module.outputscale.fill_(2.0)  # top-level scale

        def forward(self, h):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(h), self.covar_module(h)
            )

    class DGP(gpytorch.models.deep_gps.DeepGP):
        def __init__(
            self, in_dim, hid_dim, Xref, seed, blocks=None, vi_dist="meanfield"
        ):
            super().__init__()
            self.hidden = HiddenLayer(
                in_dim, hid_dim, Xref, seed, blocks=blocks, vi_dist=vi_dist
            )
            self.output = OutputLayer(hid_dim, vi_dist=vi_dist)
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
    model = DGP(
        in_dim=Xtr_t.size(1),
        hid_dim=HIDDEN_DIM,
        Xref=Xtr_t,
        seed=seed,
        blocks=BLOCKS,  # <<< wire in the additive block-wise kernel
        vi_dist=VI_DIST,
    ).to(device)

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
            # ### NEW/UPDATED: no "[flipped]" suffix in print
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
    def draw_guides(ax, horiz_at=0.5, vlines=(), xmax=300):
        """Light guide lines: horizontal at `horiz_at` and optional vertical ticks."""
        ax.axhline(horiz_at, ls="--", lw=1.0, color="gray", alpha=0.6, zorder=2)
        for v in vlines or []:
            ax.axvline(float(v), ls=":", lw=0.9, color="lightgray", alpha=0.6, zorder=1)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0.0, 1.0)

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
        """
        Plots KM curves for boolean group masks, adds:
        - 0.5 horizontal guide
        - half-height dashed verticals at each group's median
        - 'NR' if median not reached (won't crash)
        - legend, number-at-risk table, log-rank P
        Returns the log-rank p-value (float or NaN).
        """
        # cap at 300 mo if requested
        t_in, e_in = admin_censor(t, e, 300.0) if admin_cap else (t, e)

        # ensure masks valid and at least one has data
        masks = [(m & np.isfinite(t_in)) for m in groups]
        present = [i for i, m in enumerate(masks) if np.sum(m) > 0]
        if len(present) == 0:
            print(f"[WARN] No groups with data for {outname}.")
            return np.nan

        timeline = np.arange(0.0, 300.0 + 1.0, 1.0) if extend_to_300 else None

        kmfs = []
        plt.figure(figsize=(9.8, 6.8))
        ax = plt.gca()

        # plot KM curves and remember each curve's color for matching median lines
        curve_colors = []
        for i in present:
            kmf = KaplanMeierFitter()
            kmf.fit(t_in[masks[i]], e_in[masks[i]], label=labels[i], timeline=timeline)
            h_ax = kmf.plot(ax=ax, ci_show=False, show_censors=False)
            # grab the plotted line's color
            line = (
                h_ax.get_lines()[-1]
                if hasattr(h_ax, "get_lines")
                else ax.get_lines()[-1]
            )
            color = line.get_color() if line is not None else None
            curve_colors.append(color)
            kmfs.append(kmf)

        # --- guides and medians ---
        draw_guides(ax, horiz_at=0.5, vlines=(), xmax=300)

        # per-stratum empirical medians (half-height dashed verticals)
        nr_colors = []  # annotate after limits are set
        for kmf, col in zip(kmfs, curve_colors):
            try:
                med = kmf.median_survival_time_
            except Exception:
                med = np.nan  # safety

            if np.isfinite(med):
                ax.vlines(
                    float(med),
                    0.0,
                    0.5,
                    linestyles="--",
                    linewidth=1.1,
                    colors=col,
                    zorder=3,
                )
                # small x-offset to avoid overlap with curve
                x_txt = float(med) + 2.0
                ax.text(
                    x_txt,
                    0.03,
                    f"{med:.0f} mo",
                    rotation=90,
                    va="bottom",
                    ha="right",
                    fontsize=9,
                    alpha=0.85,
                    color=col,
                    zorder=4,
                )
            else:
                nr_colors.append(col)

        # annotate NR near the right edge (after x/y limits are fixed)
        if nr_colors:
            xr = ax.get_xlim()[1]
            for col in nr_colors:
                ax.text(xr - 10, 0.52, "NR", fontsize=9, color=col, alpha=0.9, zorder=4)

        plt.xlabel("Time (months)")
        plt.ylabel("Survival probability")
        plt.title(title)

        # legend
        handles, leg_labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            leg_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
        )

        # number-at-risk rows
        try:
            add_at_risk_counts(*kmfs, ax=ax)
        except Exception as ex:
            print(f"[WARN] add_at_risk_counts failed: {ex}")

        # log-rank p-value
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
                0.92,
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

        # save & close
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

    # ---------------- SAVE PATIENT-LEVEL RISK GROUP ASSIGNMENTS ----------------
    # Helper: write one CSV for a given split
    def _write_groups_csv(meta_df, idx, gL, gM, gH, S_H_split, split_name):
        if len(idx) == 0:
            return
        pid = meta_df.iloc[idx]["PATIENT_ID"].astype(str).values
        sid = meta_df.iloc[idx]["SAMPLE_ID"].astype(str).values

        # group label per person
        group = np.where(gH, "high", np.where(gL, "low", "med"))

        # posterior summaries at H for this split
        Smean = S_H_split.mean(axis=0)  # E[S(H)]
        p_hi = (S_H_split <= TAU_HIGH).mean(axis=0)  # P{S(H) <= tau_high}
        p_lo = (S_H_split >= TAU_LOW).mean(axis=0)  # P{S(H) >= tau_low}

        out = pd.DataFrame(
            {
                "PATIENT_ID": pid,
                "SAMPLE_ID": sid,
                "split": split_name,
                "group": group,  # "low" | "med" | "high"
                "E_S_at_H": Smean,  # posterior mean survival at H
                "P_S_le_tauH": p_hi,
                "P_S_ge_tauL": p_lo,
                "H_months": int(POSTERIOR_H_MONTHS),
                "tau_high": TAU_HIGH,
                "tau_low": TAU_LOW,
                "p_star": P_STAR,
            }
        )
        out.to_csv(OUTDIR / f"posteriorH_groups_{split_name}.csv", index=False)
        print(
            f"[Posterior-H] Saved patient groups → posteriorH_groups_{split_name}.csv"
        )

    # Per-split files
    _write_groups_csv(meta, tr_idx, gL_tr, gM_tr, gH_tr, S_H_tr, "train")
    if HAS_VAL:
        _write_groups_csv(meta, va_idx, gL_va, gM_va, gH_va, S_H_va, "val")
    _write_groups_csv(meta, te_idx, gL_te, gM_te, gH_te, S_H_te, "test")

    # -------- Full cohort file (merged in meta order) --------
    # Build arrays in meta-index order
    N_all = len(meta)
    group_all = np.full(N_all, "med", dtype=object)
    group_all[gL_all] = "low"
    group_all[gH_all] = "high"

    # stitch posterior summaries in meta order using split pieces
    Smean_all = np.full(N_all, np.nan, float)
    p_hi_all = np.full(N_all, np.nan, float)
    p_lo_all = np.full(N_all, np.nan, float)

    Smean_all[tr_idx] = S_H_tr.mean(axis=0)
    p_hi_all[tr_idx] = (S_H_tr <= TAU_HIGH).mean(axis=0)
    p_lo_all[tr_idx] = (S_H_tr >= TAU_LOW).mean(axis=0)

    if HAS_VAL and S_H_va.shape[1] > 0:
        Smean_all[va_idx] = S_H_va.mean(axis=0)
        p_hi_all[va_idx] = (S_H_va <= TAU_HIGH).mean(axis=0)
        p_lo_all[va_idx] = (S_H_va >= TAU_LOW).mean(axis=0)

    Smean_all[te_idx] = S_H_te.mean(axis=0)
    p_hi_all[te_idx] = (S_H_te <= TAU_HIGH).mean(axis=0)
    p_lo_all[te_idx] = (S_H_te >= TAU_LOW).mean(axis=0)

    df_all = pd.DataFrame(
        {
            "PATIENT_ID": meta["PATIENT_ID"].astype(str).values,
            "SAMPLE_ID": meta["SAMPLE_ID"].astype(str).values,
            "split": np.where(
                np.isin(np.arange(N_all), tr_idx),
                "train",
                np.where(
                    np.isin(np.arange(N_all), te_idx),
                    "test",
                    np.where(np.isin(np.arange(N_all), va_idx), "val", "unknown"),
                ),
            ),
            "group": group_all,
            "E_S_at_H": Smean_all,
            "P_S_le_tauH": p_hi_all,
            "P_S_ge_tauL": p_lo_all,
            "H_months": int(POSTERIOR_H_MONTHS),
            "tau_high": TAU_HIGH,
            "tau_low": TAU_LOW,
            "p_star": P_STAR,
        }
    )
    df_all.to_csv(OUTDIR / "posteriorH_groups_full.csv", index=False)
    print("[Posterior-H] Saved patient groups → posteriorH_groups_full.csv")
    # --------------------------------------------------------------------------

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
    _spearmanr = spearmanr  # <-- use the module-level import; don't shadow it
    if (_spearmanr is not None) and USE_GES:
        nlS120 = neglogS_at_H(Xte_t, 120.0, alpha_dt, S=EVAL_MC_SAMPLES)

        def add_corr(name, a, b):
            rho, p = _spearmanr(a, b, nan_policy="omit")
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

    # ====================== PERM IMPORTANCE (single metric) + r annotation ======================

    def permutation_importance_topk(
        X_ref,
        times,
        events,
        feature_names,
        H,
        k=2,
        S=EVAL_MC_SAMPLES,
        K=50,
        seed=SEED,
        floor_zero=True,
    ):
        """
        Permutation importance using ΔC-index drop where the score is -log S(H) (risk at horizon H).
        Returns:
        top_idx: list[int]
        top_info: list[(feature_name, delta_cindex_drop)]
        drops_all: np.ndarray shape [P] with delta per feature (0 for skipped)
        """
        rng_loc = np.random.default_rng(seed)
        X_ref = np.asarray(X_ref, dtype=float)

        # Baseline scores at horizon H
        base = neglogS_at_H(
            torch.tensor(X_ref, dtype=torch.float32, device=device),
            float(H),
            alpha_dt,
            S=S,
        )
        c_base, _ = harrell_c(times, base, events)  # higher-is-better

        drops_all = np.zeros(X_ref.shape[1], dtype=float)

        for j in range(X_ref.shape[1]):
            if not is_continuous(X_ref[:, j]):
                continue
            deltas = []
            for _ in range(K):
                Xp = X_ref.copy()
                Xp[:, j] = rng_loc.permutation(Xp[:, j])
                p = neglogS_at_H(
                    torch.tensor(Xp, dtype=torch.float32, device=device),
                    float(H),
                    alpha_dt,
                    S=max(8, S // 2),
                )
                c_perm, _ = harrell_c(times, p, events)
                deltas.append(c_base - c_perm)  # C-index drop
            delta = float(np.mean(deltas)) if deltas else 0.0
            if floor_zero and delta < 0:
                delta = 0.0
            drops_all[j] = delta

        order = np.argsort(drops_all)[::-1]
        top_idx = order[: int(min(k, len(order)))].tolist()
        top_info = [(feature_names[j], float(drops_all[j])) for j in top_idx]
        return top_idx, top_info, drops_all

    # ---- Example usage: pick horizon you care about (e.g., 120 months) ----
    H_PD = 120.0  # or float(H_C), 60.0, etc.
    top_idx, top_info, drops_all = permutation_importance_topk(
        Xtr_s, t_tr, e_tr, feat_names, H=H_PD, k=2, S=EVAL_MC_SAMPLES, K=50
    )
    print(
        f"[PD] Top features by ΔC-index drop using -logS({int(H_PD)}) (perm): {top_info}"
    )

    # ---- Annotate those top perm features with raw correlation to -log S(H_PD) on TRAIN ----
    # ---- Annotate those top perm features with raw correlation to -log S(H_PD) on TRAIN ----
    nlS_ref = neglogS_at_H(
        torch.tensor(Xtr_s, dtype=torch.float32, device=device),
        float(H_PD),
        alpha_dt,
        S=EVAL_MC_SAMPLES,
    )

    # FIX: define locally so it's always in-scope, and handle torch -> numpy safely
    def raw_pearsonlocal(a, b):
        a = np.asarray(a, dtype=float).ravel()
        if torch.is_tensor(b):
            b = b.detach().cpu().numpy()
        b = np.asarray(b, dtype=float).ravel()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3:
            return np.nan
        a = a[m] - a[m].mean()
        b = b[m] - b[m].mean()
        denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
        return float((a * b).sum() / denom) if denom > 0 else np.nan

    for j, (nm, delta) in zip(top_idx, top_info):
        r = raw_pearsonlocal(Xtr_s[:, j], nlS_ref)
        print(f"    {nm}: delta={delta:.4g}, raw_r_with_neglogS={r:.3g}")

    # --- Robust ARD extraction for composite kernels (Additive/Product/Scale) ---

    def ard_inv_lengthscale_per_feature(kern, P):
        """
        Returns inv_lengthscale vector of shape [P] for a possibly composite kernel.
        - Handles ScaleKernel, AdditiveKernel, ProductKernel
        - Uses active_dims to map sub-kernel ARD to the full feature vector
        - Averages contributions if multiple subkernels touch same dim
        """
        inv = np.zeros(P, dtype=float)
        cnt = np.zeros(P, dtype=float)

        def _to_np_active_dims(ad):
            if ad is None:
                return None
            try:
                # torch tensor
                return np.asarray(ad.detach().cpu().numpy(), dtype=int)
            except Exception:
                return np.asarray(ad, dtype=int)

        def _visit(k):
            # Unwrap ScaleKernel
            if isinstance(k, gpytorch.kernels.ScaleKernel):
                _visit(k.base_kernel)
                return

            # Recurse into composite kernels
            if isinstance(
                k, (gpytorch.kernels.AdditiveKernel, gpytorch.kernels.ProductKernel)
            ):
                kids = getattr(k, "kernels", None) or getattr(k, "_kernels", [])
                for child in kids:
                    _visit(child)
                return

            # Pull lengthscale if present
            if hasattr(k, "lengthscale"):
                try:
                    ls = k.lengthscale.detach().cpu().numpy().reshape(-1).astype(float)
                except Exception:
                    return

                ad = _to_np_active_dims(getattr(k, "active_dims", None))

                # Map to full feature indices
                if ad is None:
                    # If no active_dims, assume it covers all P dims when ARD matches P
                    if ls.size == P:
                        dims = np.arange(P, dtype=int)
                    elif ls.size == 1:
                        dims = np.arange(P, dtype=int)
                        ls = np.repeat(ls, P)
                    else:
                        # Fallback: treat as first ls.size dims
                        dims = np.arange(min(ls.size, P), dtype=int)
                        ls = ls[: dims.size]
                else:
                    dims = ad
                    if ls.size == 1:
                        ls = np.repeat(ls, dims.size)
                    elif ls.size != dims.size:
                        # If mismatch, truncate to min
                        m = min(ls.size, dims.size)
                        dims = dims[:m]
                        ls = ls[:m]

                inv_part = 1.0 / (ls + 1e-8)
                inv[dims] += inv_part
                cnt[dims] += 1.0

            # Some kernels wrap another kernel in base_kernel (rare outside ScaleKernel)
            if hasattr(k, "base_kernel"):
                try:
                    _visit(k.base_kernel)
                except Exception:
                    pass

        _visit(kern)

        nz = cnt > 0
        inv[nz] /= cnt[nz]
        return inv

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

        plt.figure(figsize=(6, 4))
        plt.boxplot(
            [np.asarray(v).ravel() for v in vecs], labels=labels, showfliers=False
        )
        ax = plt.gca()

        # ---- single overall p-value (Kruskal–Wallis for ≥3 groups; Mann–Whitney if exactly 2) ----
        clean = [np.asarray(v, float)[~np.isnan(v)] for v in vecs]
        have = [len(v) > 0 for v in clean]
        pval = np.nan
        if sum(have) >= 2:
            if len(clean) >= 3 and kruskal_wallis is not None:
                pval = float(
                    kruskal_wallis(*[v for v, h in zip(clean, have) if h]).pvalue
                )
            elif (
                len(clean) == 2
                and "mannwhitneyu" in globals()
                and mannwhitneyu is not None
            ):
                pval = float(
                    mannwhitneyu(clean[0], clean[1], alternative="two-sided").pvalue
                )
            elif len(clean) == 2 and kruskal_wallis is not None:
                # fallback: KW on two groups (equivalent ranking test)
                pval = float(kruskal_wallis(clean[0], clean[1]).pvalue)

        if pval == pval:  # not NaN
            ymin, ymax = ax.get_ylim()
            span = ymax - ymin
            ax.set_ylim(ymin, ymax + 0.08 * span)
            ax.text(
                0.02,
                0.96,
                f"p = {pval:.2e}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=12,
                bbox=dict(fc="white", ec="none", alpha=0.8),
            )

        plt.ylabel(ylabel)
        plt.title(title)
        savefig(OUTDIR / fname)

    def _boxplot_npip_with_pairs(vecs, labels, title, ylabel, fname):
        plt.figure(figsize=(6, 4))
        plt.boxplot(
            [np.asarray(v).ravel() for v in vecs], labels=labels, showfliers=False
        )
        ax = plt.gca()

        clean = [np.asarray(v, float)[~np.isnan(v)] for v in vecs]
        have = [len(v) > 0 for v in clean]
        pval = np.nan
        if sum(have) >= 2:
            if len(clean) >= 3 and kruskal_wallis is not None:
                pval = float(
                    kruskal_wallis(*[v for v, h in zip(clean, have) if h]).pvalue
                )
            elif len(clean) == 2 and mannwhitneyu is not None:
                pval = float(
                    mannwhitneyu(clean[0], clean[1], alternative="two-sided").pvalue
                )
            elif len(clean) == 2 and kruskal_wallis is not None:
                pval = float(kruskal_wallis(clean[0], clean[1]).pvalue)

        if pval == pval:  # not NaN
            ymin, ymax = ax.get_ylim()
            span = ymax - ymin
            ax.set_ylim(ymin, ymax + 0.08 * span)
            ax.text(
                0.02,
                0.96,
                f"p = {pval:.2e}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=12,
                bbox=dict(fc="white", ec="none", alpha=0.8),
            )

        # ---- pairwise p-values (MWU) with BH–FDR, drawn as brackets ----
        def _bh_fdr(pvals):
            pvals = np.asarray(pvals, float)
            m = np.sum(~np.isnan(pvals))
            if m == 0:
                return pvals
            order = np.argsort(np.where(np.isnan(pvals), np.inf, pvals))
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(order) + 1)
            q = pvals * m / ranks
            q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
            q_adj = np.empty_like(q)
            q_adj[order] = q_sorted
            return q_adj

        def _p_to_stars(p):
            if not np.isfinite(p):
                return "n/a"
            if p <= 1e-3:
                return "***"
            if p <= 1e-2:
                return "**"
            if p <= 5e-2:
                return "*"
            return "ns"

        def _draw_bracket(x1, x2, y, h, txt):
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, color="k")
            ax.text((x1 + x2) / 2.0, y + h, txt, ha="center", va="bottom", fontsize=10)

        if len(clean) >= 2 and mannwhitneyu is not None:
            pairs = list(itertools.combinations(range(len(clean)), 2))
            p_raw = []
            for i, j in pairs:
                if len(clean[i]) == 0 or len(clean[j]) == 0:
                    p_raw.append(np.nan)
                else:
                    p_raw.append(
                        float(
                            mannwhitneyu(
                                clean[i], clean[j], alternative="two-sided"
                            ).pvalue
                        )
                    )
            p_adj = _bh_fdr(p_raw)

            ymin, ymax = ax.get_ylim()
            data_top = max((np.max(v) if len(v) else ymin) for v in clean)
            span = ymax - ymin if ymax > ymin else 1.0
            step = 0.08 * span
            base = max(ymax, data_top) + 0.06 * span

            order = np.argsort(np.where(np.isnan(p_adj), np.inf, p_adj))
            for k, idx in enumerate(order):
                i, j = pairs[idx]
                x1, x2 = i + 1, j + 1
                y = base + k * step
                _draw_bracket(
                    x1,
                    x2,
                    y,
                    0.02 * span,
                    f"{_p_to_stars(p_adj[idx])} (p={p_adj[idx]:.2e})",
                )

            ax.set_ylim(ymin, base + len(pairs) * step + 0.12 * span)

        plt.ylabel(ylabel)
        plt.title(title)
        savefig(OUTDIR / fname)

    def _boxplot_prolif_with_pairs(vecs, labels, title, ylabel, fname):
        plt.figure(figsize=(6, 4))
        plt.boxplot(
            [np.asarray(v).ravel() for v in vecs], labels=labels, showfliers=False
        )
        ax = plt.gca()

        clean = [np.asarray(v, float)[~np.isnan(v)] for v in vecs]
        have = [len(v) > 0 for v in clean]
        pval = np.nan
        if sum(have) >= 2:
            if len(clean) >= 3 and kruskal_wallis is not None:
                pval = float(
                    kruskal_wallis(*[v for v, h in zip(clean, have) if h]).pvalue
                )
            elif len(clean) == 2 and mannwhitneyu is not None:
                pval = float(
                    mannwhitneyu(clean[0], clean[1], alternative="two-sided").pvalue
                )
            elif len(clean) == 2 and kruskal_wallis is not None:
                pval = float(kruskal_wallis(clean[0], clean[1]).pvalue)

        if pval == pval:  # not NaN
            ymin, ymax = ax.get_ylim()
            span = ymax - ymin
            ax.set_ylim(ymin, ymax + 0.08 * span)
            ax.text(
                0.02,
                0.96,
                f"p = {pval:.2e}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=12,
                bbox=dict(fc="white", ec="none", alpha=0.8),
            )

        # ---- pairwise p-values (MWU) with BH–FDR, drawn as brackets ----
        def _bh_fdr(pvals):
            pvals = np.asarray(pvals, float)
            m = np.sum(~np.isnan(pvals))
            if m == 0:
                return pvals
            order = np.argsort(np.where(np.isnan(pvals), np.inf, pvals))
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(order) + 1)
            q = pvals * m / ranks
            q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
            q_adj = np.empty_like(q)
            q_adj[order] = q_sorted
            return q_adj

        def _p_to_stars(p):
            if not np.isfinite(p):
                return "n/a"
            if p <= 1e-3:
                return "***"
            if p <= 1e-2:
                return "**"
            if p <= 5e-2:
                return "*"
            return "ns"

        def _draw_bracket(x1, x2, y, h, txt):
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, color="k")
            ax.text((x1 + x2) / 2.0, y + h, txt, ha="center", va="bottom", fontsize=10)

        if len(clean) >= 2 and mannwhitneyu is not None:
            pairs = list(itertools.combinations(range(len(clean)), 2))
            p_raw = []
            for i, j in pairs:
                if len(clean[i]) == 0 or len(clean[j]) == 0:
                    p_raw.append(np.nan)
                else:
                    p_raw.append(
                        float(
                            mannwhitneyu(
                                clean[i], clean[j], alternative="two-sided"
                            ).pvalue
                        )
                    )
            p_adj = _bh_fdr(p_raw)

            ymin, ymax = ax.get_ylim()
            data_top = max((np.max(v) if len(v) else ymin) for v in clean)
            span = ymax - ymin if ymax > ymin else 1.0
            step = 0.08 * span
            base = max(ymax, data_top) + 0.06 * span

            order = np.argsort(np.where(np.isnan(p_adj), np.inf, p_adj))
            for k, idx in enumerate(order):
                i, j = pairs[idx]
                x1, x2 = i + 1, j + 1
                y = base + k * step
                _draw_bracket(
                    x1,
                    x2,
                    y,
                    0.02 * span,
                    f"{_p_to_stars(p_adj[idx])} (p={p_adj[idx]:.2e})",
                )

            ax.set_ylim(ymin, base + len(pairs) * step + 0.12 * span)

        plt.ylabel(ylabel)
        plt.title(title)
        savefig(OUTDIR / fname)

    if USE_DISCRETE_TIME:
        groups = [gL_te, gM_te, gH_te]
        glabels = ["Low", "Med", "High"]

        # _boxplot(
        #    [ges_ER_te[:, 0][g] for g in groups],
        #   glabels,
        #  "ER score by Posterior-H group (test)",
        # "ER score",
        # "bio_box_ER_by_group_test.png",
        # )

        # _boxplot_prolif_with_pairs(
        #   [ges_Pro_te[:, 0][g] for g in groups],
        #  glabels,
        # "Proliferation score by Posterior-H group (test)",
        # "Proliferation score",
        # "bio_box_prolif_by_group_test.png",
        # )

        E_te = expr_full.iloc[:, te_cols]  # genes x samples

        # Family-level NPIP (symbols starting with "NPIP")
        npip_mask = E_te.index.to_series().str.startswith("NPIP")
        npip_genes_present = E_te.index[npip_mask].tolist()
        if npip_genes_present:
            npip_family_mean = E_te.loc[npip_genes_present].mean(axis=0).to_numpy()
            _boxplot_npip_with_pairs(
                [npip_family_mean[g] for g in groups],
                glabels,
                "NPIP family (mean expression) by Posterior-H group (test)",
                "NPIP family expression (mean across paralogues)",
                "bio_box_NPIPfamily_by_group_test.png",
            )

        # Specific paralogues
        for gene in ["NPIPB13", "NPIPL3"]:
            if gene in E_te.index:
                vals = E_te.loc[gene].to_numpy()
                _boxplot_npip_with_pairs(
                    [vals[g] for g in groups],
                    glabels,
                    f"{gene} expression by Posterior-H group (test)",
                    f"{gene} log₂ microarray expression",
                    f"bio_box_{gene}_by_group_test.png",
                )
            else:
                print(
                    f"[BIO/NPIP] {gene} not present on this array after cleaning/collapse."
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
    # ================= [B] Interpretability on hazard scale @ H ====================
    # ### FC-integrated: Top-7 on TRAIN and TEST by FC score + backprojections for both
    HORIZONS = [120.0, float(H_C), 60.0, 30.0]
    topN = 10

    # ---------- Precompute horizon risks for orientation (TRAIN + TEST) ----------
    nlS_by_H_te = {
        H: neglogS_at_H(Xte_t, H, alpha_dt, S=EVAL_MC_SAMPLES) for H in HORIZONS
    }
    nlS_by_H_tr = {
        H: neglogS_at_H(Xtr_t, H, alpha_dt, S=EVAL_MC_SAMPLES) for H in HORIZONS
    }

    # ---------- Spearman correlation (rank-based) ----------
    # IMPORTANT: use _spearmanr so we NEVER collide with any 'spearmanr' name inside run_single()
    # SciPy spearmanr was imported at module scope already; don't rebind it here.
    _spearmanr = spearmanr

    def raw_spearman(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        if _spearmanr is not None and np.isfinite(a).all() and np.isfinite(b).all():
            r, _ = _spearmanr(a, b)
            return float(r) if np.isfinite(r) else 0.0
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        ra = np.empty_like(a, dtype=float)
        rb = np.empty_like(b, dtype=float)
        ra[np.argsort(a)] = np.arange(len(a), dtype=float)
        rb[np.argsort(b)] = np.arange(len(b), dtype=float)
        C = np.corrcoef(ra, rb)
        return float(C[0, 1]) if np.isfinite(C[0, 1]) else 0.0

    # (B1) Permutation importance for hazard @ H on TEST split (ΔC-index drop)
    def perm_importance_hazard(
        X_ref, times, events, H, S=EVAL_MC_SAMPLES, K=50, seed=SEED, floor_zero=True
    ):
        rng_loc = np.random.default_rng(seed)
        X_ref = np.asarray(X_ref, float)

        base = neglogS_at_H(
            torch.tensor(X_ref, dtype=torch.float32, device=device), H, alpha_dt, S=S
        )
        c_base, _ = harrell_c(times, base, events)

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
        f_mean = model(X).mean.view(-1)
        j = int(np.searchsorted(bin_edges, H, side="right") - 1)
        j = int(np.clip(j, 0, len(bin_edges) - 2))
        if DT_LINK.lower() == "cloglog":
            eta = f_mean[:, None] + alpha[None, :]
            h = 1.0 - torch.exp(-torch.exp(torch.clamp(eta, max=20.0)))
        else:
            eta = f_mean[:, None] + alpha[None, :]
            h = torch.sigmoid(eta)
        S_edges = torch.cumprod(1.0 - h, dim=1)
        nlS = -torch.log(torch.clamp(S_edges[:, j], min=1e-10))
        loss = nlS.mean()
        loss.backward()
        g = X.grad.detach().abs().mean(dim=0).cpu().numpy()
        return g

    # (B3) ARD from HIDDEN layer (per-feature)
    try:
        P = len(feat_names)
        inv_ls_base = ard_inv_lengthscale_per_feature(model.hidden.covar_module, P)
    except Exception as ex:
        print(f"[B] ARD extraction failed: {ex}")
        inv_ls_base = np.zeros(len(feat_names), dtype=float)

    def _z(v):
        v = np.asarray(v, float)
        return (v - v.mean()) / (v.std() + 1e-12)

    # ===================== (B4) Feature Collapsing (FC) (TRAIN + TEST) =====================

    @torch.no_grad()
    def neglogS_samples_at_H(X_torch, H_months: float, alpha, S=EVAL_MC_SAMPLES):
        if not USE_DISCRETE_TIME:
            raise RuntimeError(
                "FC requires USE_DISCRETE_TIME=True (hazard-based survival)."
            )
        SH = posterior_survival_at_H(X_torch, H_months, alpha, S=S)  # [S, N]
        return -np.log(np.clip(SH, 1e-10, 1.0))

    def _wasserstein1_empirical_per_row(Y0, Y1):
        # Y0,Y1: [S,N] -> per-patient empirical W1 by sorting samples
        Y0s = np.sort(np.asarray(Y0, float), axis=0)
        Y1s = np.sort(np.asarray(Y1, float), axis=0)
        return np.mean(np.abs(Y0s - Y1s), axis=0)  # [N]

    def feature_collapsing_importance(
        X_ref,
        feature_names,
        H,
        alpha,
        S=256,
        skip_noncontinuous=False,  # <-- KEEP FALSE (your request)
        method="wasserstein-1",
        use_crn=True,  # <-- CRN ON
        crn_seed=12345,  # <-- overridden per-horizon below
        n_repeats=3,  # <-- REPEAT-AVERAGING (stability fix)
        repeat_seed_stride=10007,  # <-- makes repeats independent (but reproducible)
    ):
        """
        FC stability fixes:
        (1) CRN (Common Random Numbers): within each repeat, restore RNG state before each Y1
            so Y0 vs Y1 differs ONLY by feature collapse.
        (2) Repeat-averaging: run FC R times with different seeds (still CRN within each run),
            then average the per-feature scores.
        """
        X_ref = np.asarray(X_ref, float)
        P = X_ref.shape[1]
        X0_t = torch.tensor(X_ref, dtype=torch.float32, device=device)

        scores_accum = np.zeros(P, dtype=float)
        R = int(max(1, n_repeats))

        for r in range(R):
            seed_r = int(crn_seed) + r * int(repeat_seed_stride)

            # ---- CRN: capture a single RNG state (per repeat) to reuse for all Y1 draws ----
            if use_crn:
                np.random.seed(seed_r)
                torch.manual_seed(seed_r)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_r)

                # IMPORTANT: capture state BEFORE drawing Y0
                np_state0 = np.random.get_state()
                torch_state0 = torch.get_rng_state()
                cuda_state0 = (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                )

            # Baseline samples Y0 (per repeat)
            Y0 = neglogS_samples_at_H(X0_t, float(H), alpha, S=int(S))  # [S,N]

            scores = np.zeros(P, dtype=float)
            for j in range(P):
                if skip_noncontinuous and (not is_continuous(X_ref[:, j])):
                    scores[j] = 0.0
                    continue

                # Restore RNG state so Y1 uses same randomness as Y0 (CRN)
                if use_crn:
                    np.random.set_state(np_state0)
                    torch.set_rng_state(torch_state0)
                    if cuda_state0 is not None:
                        torch.cuda.set_rng_state_all(cuda_state0)

                Xc = X_ref.copy()
                Xc[:, j] = (
                    0.0  # collapsing to 0 (assumes TRAIN-standardized continuous features)
                )
                Xc_t = torch.tensor(Xc, dtype=torch.float32, device=device)
                Y1 = neglogS_samples_at_H(Xc_t, float(H), alpha, S=int(S))  # [S,N]

                if method.lower() in ("wasserstein-1", "wasserstein1", "wasserstein"):
                    w1_i = _wasserstein1_empirical_per_row(Y0, Y1)  # [N]
                    scores[j] = float(np.mean(w1_i))
                else:
                    raise ValueError(
                        f"Unknown FC method='{method}'. Use 'wasserstein-1'."
                    )

            scores_accum += scores

        scores_final = scores_accum / float(R)
        df = pd.DataFrame(
            {
                "feature": list(feature_names),
                "FC_score": scores_final,
                "FC_method": method,
            }
        )
        return df.sort_values("FC_score", ascending=False).reset_index(drop=True)

    def _plot_topN(df, score_col, title, outpath, topN=7, xlabel=None):
        top = df.head(topN).iloc[::-1].copy()
        plt.figure(figsize=(8, 6))
        y = np.arange(len(top))
        plt.barh(y, top[score_col].values)
        plt.yticks(y, top["feature"].values)
        plt.xlabel(xlabel if xlabel is not None else score_col)
        plt.title(title)
        plt.tight_layout()
        savefig(outpath)
        plt.close()

    # ======================= [B-PC] Gene PC backprojections driven by FC Top-N =======================
    genes_pref = expr.index.to_numpy()
    Vt_sel = Vt[keep_idxs, :]  # rows correspond to selected PCs
    Tte_sel_mat = Tte_sel
    Ttr_sel_mat = Ttr_sel
    pc_name_to_idx = {
        n: i for i, n in enumerate(sel_names)
    }  # PC name -> 0-based index inside selected set

    def pc_raw_r_and_sign(pc_scores, nlS_vec):
        r = raw_spearman(pc_scores, nlS_vec)
        sgn = 1.0 if (np.isfinite(r) and r >= 0) else -1.0
        return r, sgn

    def _backproject_gene_pc(
        pc_name, pc_idx0, H_INT, split_tag, pc_scores, nlS_vec, outdir_pc_pngs
    ):
        outdir_pc_pngs.mkdir(parents=True, exist_ok=True)

        r_raw, sign_for_plot = pc_raw_r_and_sign(pc_scores, nlS_vec)
        load = sign_for_plot * Vt_sel[pc_idx0, :]  # oriented loadings over genes

        df = (
            pd.DataFrame(
                {
                    "gene": genes_pref,
                    "pc": pc_name,
                    "loading_oriented": load,
                    "abs_loading": np.abs(load),
                    "raw_spearman_with_neglogS": r_raw,
                    "H": int(H_INT),
                    "split": split_tag,
                }
            )
            .sort_values("abs_loading", ascending=False)
            .reset_index(drop=True)
        )

        df.to_csv(
            OUTDIR / f"pc_gene_oriented_{pc_name}_H{int(H_INT)}_{split_tag}.csv",
            index=False,
        )
        df.head(100).to_csv(
            OUTDIR / f"pc_{pc_name}_top_genes_oriented_H{int(H_INT)}_{split_tag}.csv",
            index=False,
        )

        # small summary CSV so r is recorded, but never shown in plot titles
        pd.DataFrame(
            [
                {
                    "pc": pc_name,
                    "H": int(H_INT),
                    "split": split_tag,
                    "raw_spearman_with_neglogS": float(r_raw),
                }
            ]
        ).to_csv(
            OUTDIR / f"pc_{pc_name}_summary_H{int(H_INT)}_{split_tag}.csv", index=False
        )

        topg = df.head(20).sort_values("loading_oriented")
        plt.figure(figsize=(8.0, 6.0))
        ax = plt.gca()
        y = np.arange(len(topg))
        plt.barh(y, topg["loading_oriented"].values)
        plt.yticks(y, topg["gene"].values)
        plt.xlabel("PC loading (oriented)")
        ax.axvline(0.0, color="k", lw=0.8, ls="--")
        lim = float(np.max(np.abs(topg["loading_oriented"].values))) * 1.10
        ax.set_xlim(-max(lim, 1.0), max(lim, 1.0))
        plt.title(f"{pc_name} — top gene loadings ({split_tag}) @ H={int(H_INT)} mo")
        savefig(
            outdir_pc_pngs
            / f"{pc_name}_top20_riskOriented_H{int(H_INT)}_{split_tag}.png"
        )
        plt.close()

    # ===================== [B-PATH] Pathway PC backprojections driven by FC Top-N =====================
    PATH_BACKPROJ_TOP_PATHWAYS = 30

    def _parse_pc_num(name: str):
        m = re.search(r"(\d+)", str(name))
        return int(m.group(1)) if m else None

    def _backproject_path_pc_one(
        kind: str,
        fname: str,
        H_INT: float,
        split_tag: str,
        X_split_s: np.ndarray,
        nlS_vec: np.ndarray,
        outdir_sub: Path,
        feat_names: list,
        Vt_act=None,
        kept_rows_act=None,
        Vt_c=None,
        kept_rows_con=None,
    ):
        """
        FIXED: do NOT use locals() checks (they never see outer vars).
        Instead, pass Vt_act/Vt_c + kept_rows_* explicitly (or None).
        """
        outdir_sub.mkdir(parents=True, exist_ok=True)

        kind = str(kind).lower()
        if kind == "activity":
            if Vt_act is None:
                return
            Vt_block = np.asarray(Vt_act, float)
            kept_rows = kept_rows_act
        elif kind == "consistency":
            if Vt_c is None:
                return
            Vt_block = np.asarray(Vt_c, float)
            kept_rows = kept_rows_con
        else:
            raise ValueError("kind must be 'activity' or 'consistency'")

        if kept_rows is None or len(kept_rows) == 0:
            kept_rows = np.array(
                [f"pathway_{i}" for i in range(Vt_block.shape[1])], dtype=object
            )
        else:
            kept_rows = np.asarray(kept_rows, dtype=object)

        pc_num = _parse_pc_num(fname)
        if pc_num is None:
            return
        pc_idx0 = pc_num - 1
        if pc_idx0 < 0 or pc_idx0 >= Vt_block.shape[0]:
            return

        # map feature name -> column index in X_split_s
        name_to_feat_idx = {n: i for i, n in enumerate(feat_names)}
        feat_j = name_to_feat_idx.get(str(fname), None)

        if feat_j is None:
            r_raw = float("nan")
            sign_for_plot = 1.0
        else:
            r_raw = raw_spearman(X_split_s[:, feat_j], nlS_vec)
            sign_for_plot = 1.0 if (np.isfinite(r_raw) and r_raw >= 0) else -1.0

        load = sign_for_plot * np.asarray(Vt_block[pc_idx0, :], float)

        df = (
            pd.DataFrame(
                {
                    "pathway": kept_rows,
                    "pc_feature": str(fname),
                    "pc_index_0based": int(pc_idx0),
                    "loading_oriented": load,
                    "abs_loading": np.abs(load),
                    "raw_spearman_with_neglogS": r_raw,
                    "H": int(H_INT),
                    "split": split_tag,
                }
            )
            .sort_values("abs_loading", ascending=False)
            .reset_index(drop=True)
        )

        df.to_csv(
            outdir_sub / f"{kind}_backproj_{fname}_H{int(H_INT)}_{split_tag}.csv",
            index=False,
        )
        df.head(PATH_BACKPROJ_TOP_PATHWAYS).to_csv(
            outdir_sub
            / f"{kind}_{fname}_top{PATH_BACKPROJ_TOP_PATHWAYS}_H{int(H_INT)}_{split_tag}.csv",
            index=False,
        )

        pd.DataFrame(
            [
                {
                    "pc_feature": str(fname),
                    "kind": str(kind),
                    "H": int(H_INT),
                    "split": split_tag,
                    "raw_spearman_with_neglogS": (
                        float(r_raw) if np.isfinite(r_raw) else float("nan")
                    ),
                }
            ]
        ).to_csv(
            outdir_sub / f"{kind}_{fname}_summary_H{int(H_INT)}_{split_tag}.csv",
            index=False,
        )

        top_df = df.head(min(PATH_BACKPROJ_TOP_PATHWAYS, len(df))).sort_values(
            "loading_oriented"
        )
        plt.figure(figsize=(9.0, 7.0))
        ax = plt.gca()
        y = np.arange(len(top_df))
        plt.barh(y, top_df["loading_oriented"].values)
        plt.yticks(y, top_df["pathway"].astype(str).values)
        plt.xlabel("PC loading (oriented)")
        ax.axvline(0.0, color="k", lw=0.8, ls="--")
        lim = float(np.max(np.abs(top_df["loading_oriented"].values))) * 1.10
        ax.set_xlim(-max(lim, 1.0), max(lim, 1.0))
        plt.title(f"{fname} → {kind} pathways ({split_tag}) @ H={int(H_INT)} mo")
        savefig(
            outdir_sub
            / f"{kind}_{fname}_top{len(top_df)}_H{int(H_INT)}_{split_tag}.png"
        )
        plt.close()

    # ===================== Main horizon loop: consensus + add FC + FC-driven plots/backprojections =====================
    FC_S = 256  # MC Samples
    FC_METHOD = "wasserstein-1"
    FC_USE_CRN = True  # <-- CRN ON
    FC_REPEATS = 3  # <-- Repeat-averaging for stability
    FC_REPEAT_STRIDE = 10007  # <-- Makes repeats independent but reproducible

    # ---- FC sensitivity test (Top-K vs Random) knobs ----
    RUN_FC_SENSITIVITY_TEST = True
    FC_SENS_SCOPES = ("test",)  # scopes to run sensitivity on ("train","test")
    FC_SENS_TOPK = 10
    FC_SENS_RANDOM_REPS = 200
    FC_SENS_MC = int(min(256, EVAL_MC_SAMPLES))
    FC_SENS_SEED_OFFSET = 123456

    def _safe_fname(s):
        s = str(s)
        s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s[:180] if len(s) > 180 else s

    def _fc_sensitivity_topk_vs_random(scope_tag, X_s, t_np, e_np, fc_df, H_INT):
        """Sensitivity: collapse Top-K FC features vs random K features."""
        if (not RUN_FC_SENSITIVITY_TEST) or (fc_df is None):
            return
        if str(scope_tag) not in set(map(str, FC_SENS_SCOPES)):
            return
        try:
            X_s = np.asarray(X_s, float)
            t_np = np.asarray(t_np, float)
            e_np = np.asarray(e_np, int)
            P = int(X_s.shape[1])
            name_to_idx = {str(n): i for i, n in enumerate(feat_names)}
            top_names = (
                fc_df.sort_values("FC_score", ascending=False)
                .head(int(FC_SENS_TOPK))["feature"]
                .astype(str)
                .tolist()
            )
            top_idx = [name_to_idx.get(n, None) for n in top_names]
            top_idx = [i for i in top_idx if i is not None]
            K_eff = int(min(len(top_idx), int(FC_SENS_TOPK), P))
            top_idx = top_idx[:K_eff]
            if K_eff < 1:
                return

            rng = np.random.default_rng(
                int(seed) + int(FC_SENS_SEED_OFFSET) + int(H_INT) * 17
            )

            # Baseline risk = E[-log S(H)]
            X0_t = torch.tensor(X_s, dtype=torch.float32, device=device)
            Y0 = neglogS_samples_at_H(
                X0_t, float(H_INT), alpha_dt, S=int(FC_SENS_MC)
            )  # [S,N]
            risk0 = np.mean(Y0, axis=0)
            c0, _ = harrell_c(t_np, risk0, e_np)
            mean0 = float(np.mean(risk0))

            # Collapse Top-K
            X_top = np.asarray(X_s, float).copy()
            X_top[:, top_idx] = 0.0
            Xtop_t = torch.tensor(X_top, dtype=torch.float32, device=device)
            Y_top = neglogS_samples_at_H(
                Xtop_t, float(H_INT), alpha_dt, S=int(FC_SENS_MC)
            )
            risk_top = np.mean(Y_top, axis=0)
            c_top, _ = harrell_c(t_np, risk_top, e_np)
            mean_top = float(np.mean(risk_top))

            dC_top = float(c0 - c_top)
            dMean_top = float(mean_top - mean0)

            # Random K feature sets
            rand_dC = []
            rand_dMean = []
            for rr in range(int(FC_SENS_RANDOM_REPS)):
                idx_r = rng.choice(P, size=K_eff, replace=False)
                X_r = np.asarray(X_s, float).copy()
                X_r[:, idx_r] = 0.0
                Xr_t = torch.tensor(X_r, dtype=torch.float32, device=device)
                Y_r = neglogS_samples_at_H(
                    Xr_t, float(H_INT), alpha_dt, S=int(FC_SENS_MC)
                )
                risk_r = np.mean(Y_r, axis=0)
                c_r, _ = harrell_c(t_np, risk_r, e_np)
                mean_r = float(np.mean(risk_r))
                rand_dC.append(float(c0 - c_r))
                rand_dMean.append(float(mean_r - mean0))

            rand_dC = np.asarray(rand_dC, float)
            rand_dMean = np.asarray(rand_dMean, float)
            p_dc = float(np.mean(rand_dC >= dC_top)) if rand_dC.size else float("nan")
            p_dm = (
                float(np.mean(rand_dMean >= dMean_top))
                if rand_dMean.size
                else float("nan")
            )

            sens_dir = outdir_fc / "sensitivity"
            sens_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "scope": str(scope_tag),
                        "H_months": float(H_INT),
                        "K": int(K_eff),
                        "MC": int(FC_SENS_MC),
                        "random_reps": int(FC_SENS_RANDOM_REPS),
                        "baseline_cindex": float(c0),
                        "topK_cindex": float(c_top),
                        "deltaC_topK": float(dC_top),
                        "pval_deltaC_rand_ge_topK": float(p_dc),
                        "baseline_mean_nlS": float(mean0),
                        "topK_mean_nlS": float(mean_top),
                        "deltaMean_nlS_topK": float(dMean_top),
                        "pval_deltaMean_rand_ge_topK": float(p_dm),
                    }
                ]
            ).to_csv(
                sens_dir
                / f"sensitivity_summary_{_safe_fname(scope_tag)}_H{int(H_INT)}_K{int(K_eff)}.csv",
                index=False,
            )

            pd.DataFrame(
                {
                    "rand_rep": np.arange(len(rand_dC), dtype=int),
                    "deltaC_random": rand_dC,
                    "deltaMean_nlS_random": rand_dMean,
                }
            ).to_csv(
                sens_dir
                / f"sensitivity_random_{_safe_fname(scope_tag)}_H{int(H_INT)}_K{int(K_eff)}.csv",
                index=False,
            )

            print(
                f"[FC-SENS] {scope_tag} H={int(H_INT)} K={K_eff}: "
                f"ΔC_topK={dC_top:.4f} (p={p_dc:.3f}); "
                f"Δmean(nlS)={dMean_top:.4f} (p={p_dm:.3f})"
            )

        except Exception as ex:
            print(
                f"[FC-SENS] skipped for scope={scope_tag}, H={int(H_INT)} due to: {ex}"
            )

    outdir_fc = OUTDIR / "fc"
    outdir_fc.mkdir(parents=True, exist_ok=True)

    outdir_pc_pngs = OUTDIR / "pc_gene_pngs_fc"
    outdir_pc_pngs.mkdir(parents=True, exist_ok=True)

    outdir_path_pngs = OUTDIR / "pathway_pc_backprojections_fc"
    outdir_path_pngs.mkdir(parents=True, exist_ok=True)

    pc_rows_for_csv_all = []

    for H_INT in HORIZONS:
        print(f"[B] Interpretability @ H={int(H_INT)} months")

        perm_all = perm_importance_hazard(Xte_s, t_te, e_te, H_INT)
        try:
            grads_all = grad_importance_hazard(Xte_s, H_INT, alpha_dt)
        except Exception as ex:
            print(f"[B] Gradient importance failed @H={int(H_INT)}: {ex}")
            grads_all = np.zeros(len(feat_names), dtype=float)

        inv_ls = inv_ls_base
        consensus = _z(perm_all) + _z(grads_all) + _z(inv_ls)

        r_raw_vec = np.array(
            [
                raw_spearman(Xte_s[:, j], nlS_by_H_te[H_INT])
                for j in range(len(feat_names))
            ]
        )
        sign_vec = np.sign(np.where(np.isnan(r_raw_vec), 0.0, r_raw_vec))
        signed_consensus = consensus * sign_vec
        unsigned_consensus = consensus

        interp_df = (
            pd.DataFrame(
                {
                    "feature": feat_names,
                    "perm_drop_c": perm_all,
                    "grad_abs": grads_all,
                    "inv_lengthscale_hidden": inv_ls,
                    "consensus_zsum": consensus,
                    "raw_spearman_with_neglogS_test": r_raw_vec,
                    "signed_consensus": signed_consensus,
                    "unsigned_consensus": unsigned_consensus,
                }
            )
            .sort_values("consensus_zsum", ascending=False)
            .reset_index(drop=True)
        )
        interp_df.to_csv(OUTDIR / f"interpretability_H{int(H_INT)}.csv", index=False)

        # ---- FC TRAIN + FC TEST (CRN + FC_S=256 + REPEAT-AVERAGING) ----
        # horizon-specific seed base so FC is stable within each H and reproducible across runs
        FC_CRN_SEED_H = int(seed) * 100000 + int(H_INT) * 1000 + 17

        try:
            fc_train_df = feature_collapsing_importance(
                Xtr_s,
                feat_names,
                H_INT,
                alpha_dt,
                S=FC_S,
                skip_noncontinuous=False,  # <-- YOUR REQUEST
                method=FC_METHOD,
                use_crn=FC_USE_CRN,
                crn_seed=FC_CRN_SEED_H,
                n_repeats=FC_REPEATS,
                repeat_seed_stride=FC_REPEAT_STRIDE,
            )
            fc_train_df.to_csv(
                outdir_fc / f"feature_collapsing_train_H{int(H_INT)}.csv", index=False
            )
            _plot_topN(
                fc_train_df,
                "FC_score",
                title=f"Top {topN} features — Feature Collapsing (Wasserstein-1) (TRAIN) @ H={int(H_INT)} mo",
                outpath=outdir_fc
                / f"feature_collapsing_top{topN}_train_H{int(H_INT)}.png",
                topN=topN,
                xlabel="Feature Collapsing score (empirical Wasserstein-1; larger = more influence)",
            )
            top_feats_train = fc_train_df["feature"].astype(str).head(topN).tolist()
        except Exception as ex:
            print(f"[FC] TRAIN failed @H={int(H_INT)}: {ex}")
            fc_train_df = None
            top_feats_train = []

        try:
            fc_test_df = feature_collapsing_importance(
                Xte_s,
                feat_names,
                H_INT,
                alpha_dt,
                S=FC_S,
                skip_noncontinuous=False,  # <-- YOUR REQUEST
                method=FC_METHOD,
                use_crn=FC_USE_CRN,
                crn_seed=FC_CRN_SEED_H,
                n_repeats=FC_REPEATS,
                repeat_seed_stride=FC_REPEAT_STRIDE,
            )
            fc_test_df.to_csv(
                outdir_fc / f"feature_collapsing_test_H{int(H_INT)}.csv", index=False
            )
            _plot_topN(
                fc_test_df,
                "FC_score",
                title=f"Top {topN} features — Feature Collapsing (Wasserstein-1) (TEST) @ H={int(H_INT)} mo",
                outpath=outdir_fc
                / f"feature_collapsing_top{topN}_test_H{int(H_INT)}.png",
                topN=topN,
                xlabel="Feature Collapsing score (empirical Wasserstein-1; larger = more influence)",
            )
            top_feats_test = fc_test_df["feature"].astype(str).head(topN).tolist()
        except Exception as ex:
            print(f"[FC] TEST failed @H={int(H_INT)}: {ex}")
            fc_test_df = None
            top_feats_test = []

        # ---- [FC-SENS] Top-K vs Random sensitivity test ----
        _fc_sensitivity_topk_vs_random("train", Xtr_s, t_tr, e_tr, fc_train_df, H_INT)
        _fc_sensitivity_topk_vs_random("test", Xte_s, t_te, e_te, fc_test_df, H_INT)

        print(f"\n[DEBUG] H={int(H_INT)} TRAIN top{topN}:", top_feats_train)
        print(f"[DEBUG] H={int(H_INT)} TEST  top{topN}:", top_feats_test)

        # ---- merged interpretability table (adds FC columns) ----
        try:
            interp_fc = interp_df.copy()

            if fc_train_df is not None:
                m_tr = dict(
                    zip(
                        fc_train_df["feature"].astype(str),
                        fc_train_df["FC_score"].astype(float),
                    )
                )
                interp_fc["fc_score_train"] = (
                    interp_fc["feature"].astype(str).map(m_tr).fillna(0.0)
                )
            else:
                interp_fc["fc_score_train"] = 0.0

            if fc_test_df is not None:
                m_te = dict(
                    zip(
                        fc_test_df["feature"].astype(str),
                        fc_test_df["FC_score"].astype(float),
                    )
                )
                interp_fc["fc_score_test"] = (
                    interp_fc["feature"].astype(str).map(m_te).fillna(0.0)
                )
            else:
                interp_fc["fc_score_test"] = 0.0

            interp_fc["fc_method"] = str(FC_METHOD)
            interp_fc["fc_S"] = int(FC_S)
            interp_fc["fc_use_crn"] = bool(FC_USE_CRN)
            interp_fc["fc_crn_seed_H"] = int(FC_CRN_SEED_H)
            interp_fc["fc_repeats"] = int(FC_REPEATS)
            interp_fc["fc_repeat_stride"] = int(FC_REPEAT_STRIDE)

            interp_fc.to_csv(
                OUTDIR / f"interpretability_withFC_H{int(H_INT)}.csv", index=False
            )
        except Exception as ex:
            print(
                f"[B] Could not write interpretability_withFC_H{int(H_INT)}.csv: {ex}"
            )

        # ---- Backproject gene PCs for TRAIN ----
        try:
            nlS_vec_tr = nlS_by_H_tr[H_INT]
            for fname in top_feats_train:
                if fname in pc_name_to_idx:
                    pc_idx0 = pc_name_to_idx[fname]
                    pc_scores_tr = Ttr_sel_mat[:, pc_idx0]
                    out_sub = outdir_pc_pngs / f"H{int(H_INT)}" / "train"
                    _backproject_gene_pc(
                        fname,
                        pc_idx0,
                        H_INT,
                        "train",
                        pc_scores_tr,
                        nlS_vec_tr,
                        out_sub,
                    )

                    fc_val = (
                        float(
                            fc_train_df.loc[
                                fc_train_df["feature"].astype(str) == fname, "FC_score"
                            ].iloc[0]
                        )
                        if fc_train_df is not None
                        and np.any(fc_train_df["feature"].astype(str) == fname)
                        else float("nan")
                    )
                    pc_rows_for_csv_all.append(
                        {
                            "split": "train",
                            "H": int(H_INT),
                            "pc": fname,
                            "fc_score": fc_val,
                            "fc_method": str(FC_METHOD),
                            "fc_S": int(FC_S),
                            "fc_use_crn": bool(FC_USE_CRN),
                            "fc_crn_seed_H": int(FC_CRN_SEED_H),
                            "fc_repeats": int(FC_REPEATS),
                            "fc_repeat_stride": int(FC_REPEAT_STRIDE),
                            "raw_spearman_with_neglogS": float(
                                raw_spearman(pc_scores_tr, nlS_vec_tr)
                            ),
                        }
                    )
        except Exception as ex:
            print(f"[B-PC] TRAIN gene-PC backprojection failed @H={int(H_INT)}: {ex}")

        # ---- Backproject gene PCs for TEST ----
        try:
            nlS_vec_te = nlS_by_H_te[H_INT]
            for fname in top_feats_test:
                if fname in pc_name_to_idx:
                    pc_idx0 = pc_name_to_idx[fname]
                    pc_scores_te = Tte_sel_mat[:, pc_idx0]
                    out_sub = outdir_pc_pngs / f"H{int(H_INT)}" / "test"
                    _backproject_gene_pc(
                        fname, pc_idx0, H_INT, "test", pc_scores_te, nlS_vec_te, out_sub
                    )

                    fc_val = (
                        float(
                            fc_test_df.loc[
                                fc_test_df["feature"].astype(str) == fname, "FC_score"
                            ].iloc[0]
                        )
                        if fc_test_df is not None
                        and np.any(fc_test_df["feature"].astype(str) == fname)
                        else float("nan")
                    )
                    pc_rows_for_csv_all.append(
                        {
                            "split": "test",
                            "H": int(H_INT),
                            "pc": fname,
                            "fc_score": fc_val,
                            "fc_method": str(FC_METHOD),
                            "fc_S": int(FC_S),
                            "fc_use_crn": bool(FC_USE_CRN),
                            "fc_crn_seed_H": int(FC_CRN_SEED_H),
                            "fc_repeats": int(FC_REPEATS),
                            "fc_repeat_stride": int(FC_REPEAT_STRIDE),
                            "raw_spearman_with_neglogS": float(
                                raw_spearman(pc_scores_te, nlS_vec_te)
                            ),
                        }
                    )
        except Exception as ex:
            print(f"[B-PC] TEST gene-PC backprojection failed @H={int(H_INT)}: {ex}")

        # ---- Backproject pathway PCs for TRAIN ----
        try:
            nlS_vec_tr = nlS_by_H_tr[H_INT]
            out_tr = outdir_path_pngs / f"H{int(H_INT)}" / "train"
            for fname in top_feats_train:
                sf = str(fname)
                if sf.startswith("pathPC"):
                    _backproject_path_pc_one(
                        "activity",
                        sf,
                        H_INT,
                        "train",
                        Xtr_s,
                        nlS_vec_tr,
                        out_tr / "activity",
                        feat_names=feat_names,
                        Vt_act=Vt_act if "Vt_act" in globals() else None,
                        kept_rows_act=(
                            kept_rows_act if "kept_rows_act" in globals() else None
                        ),
                        Vt_c=Vt_c if "Vt_c" in globals() else None,
                        kept_rows_con=(
                            kept_rows_con if "kept_rows_con" in globals() else None
                        ),
                    )
                if sf.startswith("consRPC"):
                    _backproject_path_pc_one(
                        "consistency",
                        sf,
                        H_INT,
                        "train",
                        Xtr_s,
                        nlS_vec_tr,
                        out_tr / "consistency",
                        feat_names=feat_names,
                        Vt_act=Vt_act if "Vt_act" in globals() else None,
                        kept_rows_act=(
                            kept_rows_act if "kept_rows_act" in globals() else None
                        ),
                        Vt_c=Vt_c if "Vt_c" in globals() else None,
                        kept_rows_con=(
                            kept_rows_con if "kept_rows_con" in globals() else None
                        ),
                    )
        except Exception as ex:
            print(
                f"[B-PATH] TRAIN pathway-PC backprojection failed @H={int(H_INT)}: {ex}"
            )

        # ---- Backproject pathway PCs for TEST ----
        try:
            nlS_vec_te = nlS_by_H_te[H_INT]
            out_te = outdir_path_pngs / f"H{int(H_INT)}" / "test"
            for fname in top_feats_test:
                sf = str(fname)
                if sf.startswith("pathPC"):
                    _backproject_path_pc_one(
                        "activity",
                        sf,
                        H_INT,
                        "test",
                        Xte_s,
                        nlS_vec_te,
                        out_te / "activity",
                        feat_names=feat_names,
                        Vt_act=Vt_act if "Vt_act" in globals() else None,
                        kept_rows_act=(
                            kept_rows_act if "kept_rows_act" in globals() else None
                        ),
                        Vt_c=Vt_c if "Vt_c" in globals() else None,
                        kept_rows_con=(
                            kept_rows_con if "kept_rows_con" in globals() else None
                        ),
                    )
                if sf.startswith("consRPC"):
                    _backproject_path_pc_one(
                        "consistency",
                        sf,
                        H_INT,
                        "test",
                        Xte_s,
                        nlS_vec_te,
                        out_te / "consistency",
                        feat_names=feat_names,
                        Vt_act=Vt_act if "Vt_act" in globals() else None,
                        kept_rows_act=(
                            kept_rows_act if "kept_rows_act" in globals() else None
                        ),
                        Vt_c=Vt_c if "Vt_c" in globals() else None,
                        kept_rows_con=(
                            kept_rows_con if "kept_rows_con" in globals() else None
                        ),
                    )
        except Exception as ex:
            print(
                f"[B-PATH] TEST pathway-PC backprojection failed @H={int(H_INT)}: {ex}"
            )

    # Save a compact summary of all FC-driven PC backprojections we produced
    if pc_rows_for_csv_all:
        (
            pd.DataFrame(pc_rows_for_csv_all)
            .sort_values(["split", "H", "fc_score"], ascending=[True, True, False])
            .to_csv(OUTDIR / "topPC_FC_multiH_train_test_summary.csv", index=False)
        )
        print(f"[B-PC] Saved FC-driven gene PC→gene PNGs in {outdir_pc_pngs}")
        print(
            f"[B-PATH] Saved FC-driven pathway-PC backprojections in {outdir_path_pngs}"
        )

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
    # Run exactly the configured seeds once (no accidental double-run of SEED).
    seeds = MULTI_SEEDS if RUN_MULTI_SEEDS else [SEED]
    seeds = list(dict.fromkeys([int(s) for s in seeds]))  # de-dup, preserve order

    print(f"[Run] RUN_MULTI_SEEDS={RUN_MULTI_SEEDS} | seeds={seeds}")

    rows = []
    for s in seeds:
        print(f"[Run] ===== Starting seed {s} =====")
        rows.append(run_single(s))

    if not RUN_MULTI_SEEDS:
        raise SystemExit(0)

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "multi_seed_results.csv", index=False)

    mean = float(df["c_test"].mean()) if "c_test" in df.columns else float("nan")
    sd = float(df["c_test"].std()) if "c_test" in df.columns else float("nan")
    print(
        f"Multi-seed test C-index (E[-log S(H_C)]): mean={mean:.3f} ± {sd:.3f} over {len(seeds)} seeds"
    )
