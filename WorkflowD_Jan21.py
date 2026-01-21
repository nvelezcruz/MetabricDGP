#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# METABRIC RUN — with pathway activity PCA→Cox (top-K PCs) integrated
# Now adapted to multitask GROUP_LABELs (subtype×treatment groups) — NO whitelist
# =============================================================================

import os, math, warnings, inspect, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import gpytorch
from collections import defaultdict
from gpytorch.means import ConstantMean, LinearMean
import torch.nn.functional as F
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    VariationalStrategy,
    MultitaskVariationalStrategy,  # keep only if you still use it elsewhere
    LMCVariationalStrategy,
    MeanFieldVariationalDistribution,
    CholeskyVariationalDistribution,
)

from gpytorch.kernels import (
    MaternKernel,
    ScaleKernel,
    MultitaskKernel,
    RBFKernel,
    LinearKernel,
)
from gpytorch.means import ZeroMean, LinearMean, MultitaskMean

# >>> IMPORTANT for DeepGP class usage below
import gpytorch.models.deep_gps  # provides gpytorch.models.deep_gps.DeepGP

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
RUN_MULTI_SEEDS = True
MULTI_SEEDS = [42, 43, 44, 45, 46]

# Fixed train/test split seed (decoupled from model seed)
SPLIT_SEED = 42

# ENDORSE-style split
TEST_FRAC = 0.50
VAL_FRAC = 0.0  # 0 disables validation/early-stop

# ---- Feature selection scope ----
PATHWAY_ONLY = False  # << set True to use ONLY pathway PCs in the final design
USE_CLINICAL = True  # << include clinical covariates when PATHWAY_ONLY=False

# Toggles
USE_GES = True  # add ER/Proliferation gene-set scores (False for pathway-only run)
GES_METHOD = "ssgsea"  # "ssgsea" or "rankmean"
KMEANS_INIT = True  # k-means init for inducing points
USE_PATHWAYS = (
    False  # pathway activity/consistency features (833-only); set True to enable
)

# Expression prefilter (speeds PCA)
GENE_SELECT_MODE = "quantile"  # "quantile" or "count"
GENE_VAR_TOP_Q = 0.95
GENE_N_TOP = 1000

# PCA & survival-based PC selection (GENE-LEVEL)
TARGET_EVR = 0.95
SURV_TOP_K = 18  # << set 0 to skip gene PCs entirely (safe-handled below)
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

# ----- Imbalance-aware training (Workflow D) -----
USE_GROUP_WEIGHTED_LL = False  # default OFF (as requested)
GROUP_WEIGHT_POWER = 0.5  # 0.5 => 1/sqrt(n_g); 1.0 => 1/n_g
GROUP_WEIGHT_CAP = 3.0
GROUP_WEIGHT_NORMALIZE = True

# ----- Wasserstein FC sensitivity test (Top-K vs Random) -----
RUN_FC_SENSITIVITY_TEST = True
FC_SENS_SCOPES = ("test",)  # ("train","test","all") allowed
FC_SENS_TOPK = 10
FC_SENS_RANDOM_REPS = 50
FC_SENS_MC = 64
FC_SENS_SEED_OFFSET = 2026

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

# Outcome proxy thresholds [A]
EARLY_EVENT_MO = 24.0
EVENT_FREE_MO = 120.0

# --------- Pathway selector knobs ----------
COX_PENALIZER = 0.05
DIAG_COX_PENALIZER = 0.1  # Cox penalizer used for train-only alignment diagnostic
PATHWAY_PC_TOP_K = 30
PATHWAY_TARGET_EVR = 0.95

# Preprocessing toggles
WINSORIZE = False  # default OFF
WINSOR_Q = (0.005, 0.995)

# File system — update BASE to your environment
# File system — ASU supercomputer
BASE = Path(os.environ.get("BASE", "/home/))
OUTDIR = Path(
    os.environ.get(
        "OUTDIR", str(BASE / "outputdir here")
    )
)
OUTDIR.mkdir(parents=True, exist_ok=True)

P_CLIN_PAT = BASE / "data_clinical_patient.txt"
P_CLIN_SAMP = BASE / "data_clinical_sample.txt"
P_EXPR_MICRO = BASE / "data_mrna_illumina_microarray.txt"
P_GROUP_MULTITASK = BASE / "METABRIC_data_clinical_patientmultitask.csv"
P_PATH_ACTIVITY = (
    BASE / "METABRIC_expression_median_subset_833Samples_pathway_activity.txt"
)
P_PATH_CONSIST = (
    BASE / "METABRIC_expression_median_subset_833Samples_pathway_consistency.txt"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------- helpers -----------------------
def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# >>> FIXED: robust reader (CSV read as CSV; TSV as TSV; fallback for "1-column comma header")
# ============================================================
def load_table_guess(path, index_col=None):
    path = Path(path)
    suf = path.suffix.lower()

    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path, index_col=index_col, dtype=str)

    # ---------- CSV: prefer fast C engine ----------
    if suf == ".csv":
        try:
            return pd.read_csv(
                path,
                index_col=index_col,
                dtype=str,
                encoding="utf-8-sig",
                low_memory=False,  # OK with C engine
            )
        except Exception:
            # fallback: python engine (NO low_memory allowed)
            return pd.read_csv(
                path,
                index_col=index_col,
                dtype=str,
                encoding="utf-8-sig",
                engine="python",
            )

    # ---------- TSV / TXT: try tab first ----------
    try:
        df = pd.read_table(
            path,
            sep="\t",
            comment="#",
            index_col=index_col,
            encoding="utf-8-sig",
            dtype=str,
        )

        # If we accidentally read a comma-separated file as a single column, re-read as CSV
        if df.shape[1] == 1:
            only_col = str(df.columns[0])
            if "," in only_col:
                try:
                    return pd.read_csv(
                        path,
                        index_col=index_col,
                        dtype=str,
                        encoding="utf-8-sig",
                        low_memory=False,
                    )
                except Exception:
                    return pd.read_csv(
                        path,
                        index_col=index_col,
                        dtype=str,
                        encoding="utf-8-sig",
                        engine="python",
                    )
        return df

    except Exception:
        # ---------- last resort: delimiter sniffing ----------
        try:
            return pd.read_csv(
                path,
                sep=None,
                engine="python",  # sniffing requires python engine
                index_col=index_col,
                dtype=str,
                encoding="utf-8-sig",
            )
        except UnicodeDecodeError:
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                index_col=index_col,
                dtype=str,
                encoding="latin1",
            )


def _ensure_patient_id_col(df: pd.DataFrame, name="PATIENT_ID") -> pd.DataFrame:
    cols = list(df.columns)
    if name in cols:
        return df

    # normalize for matching (strip + uppercase + remove BOM if present)
    def _norm(c):
        s = str(c)
        s = s.lstrip("\ufeff")  # BOM
        return s.strip().upper()

    m = {_norm(c): c for c in cols}
    if _norm(name) in m:
        df = df.rename(columns={m[_norm(name)]: name})
        return df

    for cand in ["PATIENTID", "PATIENT", "ID", "PATIENT_ID "]:
        if _norm(cand) in m:
            df = df.rename(columns={m[_norm(cand)]: name})
            return df

    raise ValueError(
        f"[Multitask file] Could not find a PATIENT_ID column. Columns = {cols}"
    )


def _pick_col(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def dxy(c):
    return 2 * c - 1


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
    seed_suffix = f"__seed{seed}" if RUN_MULTI_SEEDS else ""
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
    seed_suffix = f"__seed{seed}" if RUN_MULTI_SEEDS else ""
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


def canon_mb_dash(s: str):
    s = str(s).strip()
    m = _MB_PAT.search(s)
    if not m:
        return None
    d = m.group(1)
    if len(d) <= 4:
        d = d.zfill(4)
    return f"MB-{d}"


def sample_key(s: str) -> str:
    s = str(s).strip()
    mb = canon_mb_dash(s)
    if mb is not None:
        return mb
    base = str(s).split(":")[0].strip()
    return base.upper()


# ======= pathway utility (unchanged) =======
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


# ============================================================
# >>> ADDED: PATHWAY RESIDUALIZATION (TRAIN-FIT)
# ============================================================
RESIDUALIZE_PATHWAYS = True  


def residualize_matrix_trainfit(
    Ytr: np.ndarray,
    Ctr: np.ndarray,
    Yva: np.ndarray,
    Cva: np.ndarray,
    Yte: np.ndarray,
    Cte: np.ndarray,
    ridge: float = 1e-6,
):
    if Ytr is None or Ytr.size == 0:
        return Ytr, Yva, Yte, None
    if Ctr is None or Ctr.size == 0:
        return Ytr, Yva, Yte, None

    def _augment(C):
        ones = np.ones((C.shape[0], 1), dtype=float)
        return np.hstack([ones, C.astype(float)])

    CtrA = _augment(Ctr)
    CvaA = _augment(Cva) if (Cva is not None and Cva.size) else None
    CteA = _augment(Cte) if (Cte is not None and Cte.size) else None

    XtX = CtrA.T @ CtrA
    XtX = XtX + ridge * np.eye(XtX.shape[0], dtype=float)
    XtY = CtrA.T @ Ytr.astype(float)
    beta = np.linalg.solve(XtX, XtY)  # (q+1, p)

    Ytr_hat = CtrA @ beta
    Ytr_res = Ytr.astype(float) - Ytr_hat

    if Yva is not None and Yva.size and CvaA is not None:
        Yva_res = Yva.astype(float) - (CvaA @ beta)
    else:
        Yva_res = Yva

    if Yte is not None and Yte.size and CteA is not None:
        Yte_res = Yte.astype(float) - (CteA @ beta)
    else:
        Yte_res = Yte

    return Ytr_res, Yva_res, Yte_res, beta


# =======================
# GROUPING HELPERS (MATCHES YOUR metabricmultitaskgrouptests.py EXACTLY)
# =======================
MISSING_TOKENS = {"", "NA", "N/A", "NULL", "NONE", "NAN", "UNKNOWN", ".", "?"}


def clean_token(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.upper() in MISSING_TOKENS:
        return np.nan
    return s


def canon_str(x):
    s = clean_token(x)
    if s is np.nan:
        return np.nan
    u = s.upper()
    u = re.sub(r"\s+", "", u)
    u = re.sub(r"[^\w]+", "", u)
    u = u.replace("POSITVE", "POSITIVE")
    return u


def standardize_binary(series: pd.Series) -> pd.Series:
    can = series.map(canon_str)
    yes_pat = re.compile(r"^(YES|Y|TRUE|T|1)$")
    no_pat = re.compile(r"^(NO|N|FALSE|F|0)$")
    out = []
    for c in can.tolist():
        if c is np.nan:
            out.append(np.nan)
            continue
        if yes_pat.match(c):
            out.append(True)
            continue
        if no_pat.match(c):
            out.append(False)
            continue
        out.append(np.nan)
    return pd.Series(out, index=series.index, dtype="object")


def standardize_posneg(series: pd.Series) -> pd.Series:
    can = series.map(canon_str)
    pos_pat = re.compile(r"^(POS|POSITIVE|P|1|TRUE|T)$")
    neg_pat = re.compile(r"^(NEG|NEGATIVE|N|0|FALSE|F)$")
    out = []
    for c in can.tolist():
        if c is np.nan:
            out.append(np.nan)
            continue
        if pos_pat.match(c):
            out.append(True)
            continue
        if neg_pat.match(c):
            out.append(False)
            continue
        out.append(np.nan)
    return pd.Series(out, index=series.index, dtype="object")


def subtype(er_pos, her2_pos):
    if her2_pos is True:
        return "HER2pos"
    if (er_pos is True) and (her2_pos is False):
        return "ERposHER2neg"
    if (er_pos is False) and (her2_pos is False):
        return "ERnegHER2neg"
    return "SubtypeUnknown"


# >>> CHANGED: treatment depends ONLY on chemo status (NO hormone therapy criterion)
def tx(chemo):
    if chemo is True:
        return "CHEMO"
    if chemo is False:
        return "NO_CHEMO"
    return "TX_unknown"


# =======================
# >>> BLOCK INFERENCE
# =======================
def infer_blocks_from_feat_names(feat_names: list[str]):
    """
    Returns dict of additive-kernel blocks, in column-index space of x.

    """
    blocks = {k: [] for k in ["gene", "clin", "ges", "act", "cons"]}

    for j, nm in enumerate(feat_names):
        n = str(nm)

        # Gene PCs
        if re.match(r"^PC\d+$", n):
            blocks["gene"].append(j)
            continue

        # GES
        if n in ("ges_ER", "ges_Prolif") or n.startswith("ges_"):
            blocks["ges"].append(j)
            continue

        # Clinical
        if (
            n in ("AGE_AT_DIAGNOSIS", "TUMOR_SIZE")
            or n.startswith("GRADE_")
            or n.startswith("INFERRED_MENOPAUSAL_STATE_")
        ):
            blocks["clin"].append(j)
            continue

        # Path PCs
        if n.startswith(("actPC", "pathActPC", "PATHACTPC", "ACTPC")):
            blocks["act"].append(j)
            continue
        if n.startswith(("consPC", "pathConsPC", "PATHCONSPC", "CONSPC")):
            blocks["cons"].append(j)
            continue

        if n.startswith("pathPC"):
            blocks["act"].append(j)
            continue

        blocks["clin"].append(j)

    return blocks


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
    rng = np.random.default_rng(seed)
    seed_suffix = f"__seed{seed}" if RUN_MULTI_SEEDS else ""

    print("\n====================== METABRIC RUN ======================")
    print(f"=== Seed {seed} ===")
    print("Loading clinical files…")
    pat = load_table_guess(P_CLIN_PAT)
    samp = load_table_guess(P_CLIN_SAMP)

    # >>> ADDED: load multitask grouping table (CSV)
    print("Loading multitask grouping file…")
    mt = load_table_guess(P_GROUP_MULTITASK)

    # >>> FIXED ORDER: strip column names FIRST, then ensure PATIENT_ID exists
    mt.columns = [str(c).strip() for c in mt.columns]
    mt = _ensure_patient_id_col(mt, "PATIENT_ID")

    print(
        f"Patients table rows: {len(pat)}, Samples table rows: {len(samp)}, Multitask rows: {len(mt)}"
    )

    # >>> CHANGE: take ER/HER2/chemo/vital/os columns from multitask file (mt)
    er_col = _pick_col(mt, ["ER_IHC", "ER_STATUS", "ER"])
    her2_col = _pick_col(mt, ["HER2_STATUS", "HER2", "HER2_IHC"])

    chemo_col = _pick_col(mt, ["CHEMOTHERAPY"])
    os_m_col = _pick_col(mt, ["OS_MONTHS", "OVERALL_SURVIVAL_MONTHS"])
    os_s_col = _pick_col(mt, ["OS_STATUS", "VITAL_STATUS"])  # for event mapping
    vital_col = _pick_col(mt, ["VITAL_STATUS"])  # for "Died of other causes" filtering

    print(f"[Grouping(mt)] Using ER column: {er_col}")
    print(f"[Grouping(mt)] Using HER2 column: {her2_col}")
    print(f"[Grouping(mt)] Using CHEMO column: {chemo_col}")
    print(f"[Grouping(mt)] Using OS_MONTHS column: {os_m_col}")
    print(f"[Grouping(mt)] Using OS_STATUS/VITAL_STATUS column: {os_s_col}")
    print(
        f"[Grouping(mt)] Using VITAL_STATUS column for 'other causes' filter: {vital_col}"
    )

    if er_col is None or her2_col is None:
        raise ValueError(
            f"[Multitask grouping] Missing ER/HER2 columns in {P_GROUP_MULTITASK}. "
            f"Need ER_IHC/ER_STATUS/ER and HER2_STATUS/HER2/HER2_IHC. "
            f"Found columns: {list(mt.columns)}"
        )
    if chemo_col is None:
        raise ValueError(
            f"[Multitask grouping] Missing CHEMOTHERAPY column in {P_GROUP_MULTITASK}. "
            f"chemo_col={chemo_col}. Columns: {list(mt.columns)}"
        )
    if os_m_col is None or os_s_col is None:
        raise ValueError(
            f"[Multitask grouping] Missing OS_MONTHS and/or OS_STATUS/VITAL_STATUS columns in {P_GROUP_MULTITASK}. "
            f"os_m_col={os_m_col}, os_s_col={os_s_col}. Columns: {list(mt.columns)}"
        )

    # base merged table: sample+patient (for SAMPLE_ID and clinical vars), then attach multitask grouping
    merged = samp.merge(pat, on="PATIENT_ID", how="left", suffixes=("_SAMP", "_PAT"))
    merged = merged.merge(mt, on="PATIENT_ID", how="left", suffixes=("", "_MT"))
    print(f"Merged sample+patient+multitask rows: {len(merged)}")

    merged["PATIENT_ID"] = merged["PATIENT_ID"].astype(str).str.strip()
    merged["SAMPLE_ID"] = merged["SAMPLE_ID"].astype(str).str.strip()
    merged["SAMPLE_KEY"] = merged["SAMPLE_ID"].map(sample_key)

    print(
        f"\nLoaded {len(merged):,} rows ({merged['PATIENT_ID'].nunique():,} unique PATIENT_IDs)"
    )

    # >>> CHANGE: filter "Died of other causes" using multitask vital status if present
    if vital_col is not None and vital_col in merged.columns:
        v_norm = merged[vital_col].map(clean_token).fillna("").str.strip().str.upper()
        mask_keep_vital = v_norm != "DIED OF OTHER CAUSES"
        dropped_other = int((~mask_keep_vital).sum())
        merged = merged.loc[mask_keep_vital].copy()
        print(
            f"Dropped 'Died of Other Causes' (via {vital_col}): {dropped_other:,} -> remaining {len(merged):,}"
        )
    elif "VITAL_STATUS" in merged.columns:
        v_norm = (
            merged["VITAL_STATUS"].map(clean_token).fillna("").str.strip().str.upper()
        )
        mask_keep_vital = v_norm != "DIED OF OTHER CAUSES"
        dropped_other = int((~mask_keep_vital).sum())
        merged = merged.loc[mask_keep_vital].copy()
        print(
            f"Dropped 'Died of Other Causes' (fallback VITAL_STATUS): {dropped_other:,} -> remaining {len(merged):,}"
        )
    else:
        print(
            "[WARN] No VITAL_STATUS column found; skipping 'Died of Other Causes' drop."
        )

    # >>> CHANGE: grouping variables from multitask columns (mt)
    # >>> CHANGED: ONLY chemo is required for treatment; hormone therapy is NOT used
    merged["_CHEMO"] = standardize_binary(merged[chemo_col])
    merged["_ERPOS"] = standardize_posneg(merged[er_col])
    merged["_HER2P"] = standardize_posneg(merged[her2_col])

    # >>> CHANGED: drop only chemo-unknown (no endo criterion)
    mask_tx_obs = merged["_CHEMO"].notna()
    dropped_tx = int((~mask_tx_obs).sum())
    merged = merged.loc[mask_tx_obs].copy()
    print(
        f"Dropped missing/unmappable chemo: {dropped_tx:,} -> remaining {len(merged):,}"
    )

    mask_sub_obs = merged["_ERPOS"].notna() & merged["_HER2P"].notna()
    dropped_sub = int((~mask_sub_obs).sum())
    merged = merged.loc[mask_sub_obs].copy()
    print(
        f"Dropped missing/unmappable ER or HER2: {dropped_sub:,} -> remaining {len(merged):,}"
    )

    merged["SUBTYPE"] = [
        subtype(e, h) for e, h in zip(merged["_ERPOS"], merged["_HER2P"])
    ]
    # >>> CHANGED: TX uses ONLY chemo
    merged["TX"] = [tx(c) for c in merged["_CHEMO"]]
    merged["GROUP_LABEL"] = merged["SUBTYPE"] + "_" + merged["TX"]

    print("\n================== GROUP COUNTS (PRE-OS FILTER) ==================")
    print(merged["GROUP_LABEL"].value_counts(dropna=False).to_string())

    # OS columns from multitask
    print(f"\n==== {os_s_col} (raw, before OS filter) ====")
    merged[os_s_col] = merged[os_s_col].astype(str).str.strip()
    print(merged[os_s_col].value_counts(dropna=False))

    merged["OS_MONTHS"] = pd.to_numeric(merged[os_m_col], errors="coerce")
    os_ok = merged["OS_MONTHS"].notna() & merged[os_s_col].notna()
    print(f"\nRows with non-missing OS_MONTHS & {os_s_col}: {int(os_ok.sum())}")
    merged = merged.loc[os_ok].copy()
    merged["OS_EVENT"] = merged[os_s_col].map(os_event).astype(int)
    print(f"After OS filter, cohort size: {len(merged)}")

    # >>> sanity count before dedup (UPDATED LABEL)
    target_label = "ERposHER2neg_NO_CHEMO"
    n_target_pre_dedup = int((merged["GROUP_LABEL"] == target_label).sum())
    print(
        f"\n[Sanity] {target_label} count (pre-dedup, pre-expr): {n_target_pre_dedup:,}"
    )

    merged = merged.sort_values(["PATIENT_ID", "SAMPLE_ID"]).drop_duplicates(
        "PATIENT_ID", keep="first"
    )
    print(f"\nAfter PATIENT_ID de-duplication: {len(merged)} unique patients")
    print("\n==== GROUP_LABEL counts (post-dedup) ====")
    print(merged["GROUP_LABEL"].value_counts().to_string())

    # >>> sanity count after dedup
    n_target_post_dedup = int((merged["GROUP_LABEL"] == target_label).sum())
    print(
        f"\n[Sanity] {target_label} count (post-dedup, pre-expr): {n_target_post_dedup:,}"
    )

    cohort = merged.copy()

    cohort["SAMPLE_ID"] = cohort["SAMPLE_ID"].astype(str).str.strip()
    cohort["SAMPLE_KEY"] = cohort["SAMPLE_ID"].map(sample_key)

    print("\nLoading microarray expression…")
    expr_raw = load_table_guess(P_EXPR_MICRO, index_col=0)

    expr_raw.columns = [sample_key(c) for c in expr_raw.columns]
    keep_mask = [(c is not None) and (str(c).strip() != "") for c in expr_raw.columns]
    expr_raw = expr_raw.loc[:, keep_mask].copy()

    if expr_raw.columns.has_duplicates:
        expr_raw = expr_raw.groupby(level=0, axis=1).mean()

    sel_keys = set(cohort["SAMPLE_KEY"].astype(str))
    expr_raw = (
        expr_raw[[c for c in expr_raw.columns if c in sel_keys]]
        .copy()
        .apply(pd.to_numeric, errors="coerce")
    )

    cohort = cohort[cohort["SAMPLE_KEY"].isin(expr_raw.columns)].copy()

    meta = cohort.sort_values("PATIENT_ID").reset_index(drop=True)
    order_keys = meta["SAMPLE_KEY"].astype(str).tolist()
    expr_raw = expr_raw.reindex(columns=order_keys)

    print(
        f"\nAfter expression alignment: cohort {len(meta)} patients / samples, "
        f"expr_raw shape = {expr_raw.shape}"
    )
    print("Group counts after expression alignment:")
    print(meta["GROUP_LABEL"].value_counts().to_string())

    # >>> sanity count after expression alignment
    n_target_post_expr = int((meta["GROUP_LABEL"] == target_label).sum())
    print(
        f"\n[Sanity] {target_label} count (post-expr alignment): {n_target_post_expr:,}"
    )

    expr_full = clean_and_collapse_genes(expr_raw)
    expr_full = expr_full.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    std_raw = expr_full.std(axis=1, skipna=True)
    expr_full = expr_full.loc[std_raw > 1e-8]

    print(
        f"Expression matrix (full cleaned): {expr_full.shape[0]} genes × {expr_full.shape[1]} samples"
    )

    y_event_all = meta["OS_EVENT"].to_numpy()
    groups_all = meta["GROUP_LABEL"].astype(str).to_numpy()
    y_strat = np.array(
        [f"{g}__{e}" for g, e in zip(groups_all, y_event_all)], dtype=str
    )

    print("\n[Splits] OS_EVENT distribution (overall):")
    print(pd.Series(y_event_all).value_counts().to_string())
    print("\n[Splits] GROUP_LABEL distribution (overall):")
    print(pd.Series(groups_all).value_counts().to_string())
    print("\n[Splits] Combined strat label (GROUP__EVENT) counts:")
    print(pd.Series(y_strat).value_counts().to_string())

    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_FRAC, random_state=SPLIT_SEED
    )
    ((trval_idx, te_idx),) = sss1.split(np.zeros(len(y_strat)), y_strat)

    if VAL_FRAC and VAL_FRAC > 0.0:
        sss2 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=VAL_FRAC / (1 - TEST_FRAC),
            random_state=(SPLIT_SEED + 1),
        )
        ((tr_idx, va_idx),) = sss2.split(np.zeros(len(trval_idx)), y_strat[trval_idx])
        tr_idx = trval_idx[tr_idx]
        va_idx = trval_idx[va_idx]
        HAS_VAL = True
    else:
        tr_idx = trval_idx
        va_idx = np.array([], dtype=int)
        HAS_VAL = False

    print(
        f"[Splits] N_total={len(meta)}, N_train={len(tr_idx)}, N_test={len(te_idx)}, N_val={len(va_idx)}"
    )

    # ---- Persist split indices (for reproducibility across seeds/workflows) ----
    splits_dir = OUTDIR / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        splits_dir / f"split_seed{SPLIT_SEED}.npz",
        split_seed=int(SPLIT_SEED),
        tr_idx=tr_idx,
        va_idx=va_idx,
        te_idx=te_idx,
        has_val=bool(HAS_VAL),
    )

    col_idx = np.arange(expr_full.shape[1])
    tr_cols = col_idx[tr_idx]
    va_cols = col_idx[va_idx] if HAS_VAL else np.array([], dtype=int)
    te_cols = col_idx[te_idx]

    PATHWAY_FEATURES = np.zeros((len(meta), 0), dtype=float)
    pathway_feat_names = []
    if USE_PATHWAYS:
        print("\n[Pathways] Loading pathway-level features (activity + consistency)…")
        pass
    else:
        print("[Pathways] USE_PATHWAYS=False → skipping pathway features.")

    zinfo = detect_prestandardized(expr_full)
    print(
        f"\n[Expr check] pre-standardized? {zinfo['is_z']} "
        f"(median|mean|={zinfo['med_abs_mean']:.3f}, median(std)={zinfo['med_std']:.3f})"
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
    else:
        ges_ER_tr = ges_Pro_tr = np.zeros((len(tr_cols), 0))
        ges_ER_va = ges_Pro_va = np.zeros((len(va_cols), 0))
        ges_ER_te = ges_Pro_te = np.zeros((len(te_cols), 0))

    # >>> CONTINUE YOUR SCRIPT FROM HERE UNCHANGED (everything after this line stays identical)

    print("\nRunning PCA on TRAIN only (genes)…")
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
    r_evr = max(r_evr, MIN_PC) if r_evr > 0 else 0

    print(
        f"PCA(train): {r_evr} PCs reach {TARGET_EVR*100:.1f}% cumulative variance (target {TARGET_EVR*100:.1f}%)."
    )

    Ttr_full = Xtr0 @ Vt.T if Xtr0.size else np.zeros((Xtr0.shape[0], 0))
    Tva_full = Xva0 @ Vt.T if (HAS_VAL and Vt.size) else Xva0
    Tte_full = Xte0 @ Vt.T if Vt.size else Xte0
    Ttr = Ttr_full[:, :r_evr] if r_evr else np.zeros((Xtr0.shape[0], 0))
    Tva = (
        Tva_full[:, :r_evr]
        if (HAS_VAL and r_evr)
        else (Tva_full if HAS_VAL else np.zeros((0, 0)))
    )
    Tte = Tte_full[:, :r_evr] if r_evr else np.zeros((Xte0.shape[0], 0))
    pc_names = [f"PC{i+1}" for i in range(r_evr)]
    evr_used = evr_train[:r_evr] if r_evr else np.array([])

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

    pc_rank = (
        rank_pcs_by_cox_z(Ttr, t_tr, e_tr, pc_names)
        if Ttr.shape[1]
        else pd.DataFrame(columns=["pc", "idx", "z", "p", "coef"])
    )
    if len(evr_used):
        pc_rank["evr_train"] = pc_rank["idx"].map(
            {i: evr_used[i] for i in range(len(evr_used))}
        )
    pc_rank.to_csv(OUTDIR / "pc_survival_ranking_train.csv", index=False)

    K = SURV_TOP_K if SURV_TOP_K is not None else min(30, r_evr)
    if (K == 0) or (r_evr == 0):
        keep_idxs = np.array([], dtype=int)
        print("PC selection: skipping gene PCs (SURV_TOP_K=0 or r_evr=0).")
    else:
        K = int(np.clip(K, MIN_PC, min(MAX_PC, r_evr)))
        keep_idxs = (
            pc_rank.head(K)["idx"].to_numpy()
            if len(pc_rank)
            else np.array([], dtype=int)
        )
        print(
            f"PC selection: using top {len(keep_idxs)} by Cox Z out of {r_evr} PCA PCs (bounds [{MIN_PC}, {min(MAX_PC, r_evr)}])."
        )

    Ttr_sel = Ttr[:, keep_idxs] if keep_idxs.size else np.zeros((Ttr.shape[0], 0))
    Tva_sel = Tva[:, keep_idxs] if (HAS_VAL and keep_idxs.size) else np.zeros((0, 0))
    Tte_sel = Tte[:, keep_idxs] if keep_idxs.size else np.zeros((Tte.shape[0], 0))
    sel_names = [pc_names[i] for i in keep_idxs] if keep_idxs.size else []
    print(f"PCs selected by survival Z: {len(sel_names)}")

    # ============================================================
    # ------------------- Clinical covariates (ADJUSTED TO GIVE 5) -------------------
    # ============================================================
    # CHANGE (requested): categorical clinicals now match "first code" behavior:
    #  - missing categories are IMPUTED to TRAIN MODE (no "__MISSING__" dummy)
    #  - val/test unseen categories are mapped to TRAIN MODE
    # Everything else unchanged.
    DROP_FIRST_OHE = True

    _MISSING_TOKENS2 = {
        "",
        "NA",
        "N/A",
        "NONE",
        "NULL",
        "NAN",
        "UNKNOWN",
        "UNAVAILABLE",
        "NOT REPORTED",
        ".",
    }

    def _clean_cat_token(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if s == "" or s.upper() in _MISSING_TOKENS2:
            return np.nan
        return s

    def _canon_cat(x):
        if pd.isna(x):
            return np.nan
        u = str(x).strip().upper()
        u = re.sub(r"\s+", "_", u)
        u = re.sub(r"[^\w]+", "_", u)
        u = re.sub(r"_+", "_", u).strip("_")
        return u if u != "" else np.nan

    def _prep_cat_series(s: pd.Series) -> pd.Series:
        return s.map(_clean_cat_token).map(_canon_cat)

    def _prep_grade_series(s: pd.Series) -> pd.Series:
        s2 = s.map(_clean_cat_token)

        def _g(v):
            if pd.isna(v):
                return np.nan
            txt = str(v).strip().upper()
            m = re.search(r"([1-4])", txt)
            return m.group(1) if m else np.nan

        return s2.map(_g)

    NUMERIC_CLIN = [c for c in ["AGE_AT_DIAGNOSIS", "TUMOR_SIZE"] if c in meta.columns]
    CAT_CLIN = [c for c in ["GRADE", "INFERRED_MENOPAUSAL_STATE"] if c in meta.columns]

    def _train_mode(series: pd.Series):
        s = series.dropna()
        if len(s) == 0:
            return None
        m = s.mode()
        return m.iloc[0] if len(m) else None

    def _train_fitted_cat_levels_and_fill(meta_df, tr_idx, col):
        # cleaned train series
        tr_clean = (
            _prep_grade_series(meta_df.iloc[tr_idx][col])
            if col == "GRADE"
            else _prep_cat_series(meta_df.iloc[tr_idx][col])
        )
        fill = _train_mode(tr_clean)
        if fill is None:
            fill = "UNKNOWN"
        tr_filled = tr_clean.fillna(fill)

        # levels are TRAIN categories ONLY (no missing dummy)
        levels = sorted(pd.unique(tr_filled))

        return levels, fill, tr_clean  # return tr_clean for debug print (pre-impute)

    def _ohe_from_levels(meta_df, col, levels, fill_value, drop_first_ohe: bool):
        s = (
            _prep_grade_series(meta_df[col])
            if col == "GRADE"
            else _prep_cat_series(meta_df[col])
        )

        # impute missing to TRAIN fill
        s = s.fillna(fill_value)

        # map unseen categories to TRAIN fill (keeps column count stable)
        s = s.where(s.isin(levels), other=fill_value)

        cat = pd.Categorical(s, categories=levels)
        d = pd.get_dummies(cat, prefix=col)

        if drop_first_ohe and (d.shape[1] >= 2):
            d = d.iloc[:, 1:]
        return d

    def build_clin_matrices_train_fitted(meta_df, tr_idx, va_idx, te_idx):
        blocks_all = []
        names = []

        # Numeric: impute with TRAIN median (stable)
        for c in NUMERIC_CLIN:
            s_all = pd.to_numeric(meta_df[c], errors="coerce")
            med = pd.to_numeric(meta_df.iloc[tr_idx][c], errors="coerce").median()
            if not np.isfinite(med):
                med = 0.0
            s_all = s_all.fillna(med)
            blocks_all.append(s_all.to_numpy(dtype=float)[:, None])
            names.append(c)

        # Categorical: TRAIN-fitted levels, missing -> TRAIN mode (no missing dummy col)
        for c in CAT_CLIN:
            levels, fill_value, tr_clean_pre = _train_fitted_cat_levels_and_fill(
                meta_df, tr_idx, c
            )

            # Debug prints
            print(f"\n[Clinical] {c} TRAIN cleaned levels (pre-impute):")
            print(tr_clean_pre.value_counts(dropna=False).to_string())
            print(f"[Clinical] {c} TRAIN mode fill value: {fill_value}")
            print(f"[Clinical] {c} TRAIN-fitted levels (after impute): {levels}")

            d_all = _ohe_from_levels(meta_df, c, levels, fill_value, DROP_FIRST_OHE)
            if d_all.shape[1] > 0:
                blocks_all.append(d_all.to_numpy(dtype=float))
                names += list(d_all.columns)

        X_all = (
            np.hstack(blocks_all)
            if blocks_all
            else np.zeros((len(meta_df), 0), dtype=float)
        )
        Xc_tr = X_all[tr_idx]
        Xc_va = X_all[va_idx] if HAS_VAL else np.zeros((0, X_all.shape[1]), dtype=float)
        Xc_te = X_all[te_idx]
        return Xc_tr, Xc_va, Xc_te, names

    Xc_tr, Xc_va, Xc_te, clin_names = build_clin_matrices_train_fitted(
        meta, tr_idx, va_idx, te_idx
    )

    # ------------------- Final matrices + scaling (TRAIN stats) ----------------
    if "PATHWAY_FEATURES" not in locals():
        PATHWAY_FEATURES = np.zeros((len(meta), 0), dtype=float)
        pathway_feat_names = []

    def _print_feature_summary(n_gene_pc, n_clin, n_ges, n_path_pc, feat_names, head=6):
        print(
            f"[Feature summary] GenePCs={n_gene_pc}, Clinical={n_clin}, "
            f"GES={n_ges}, PathwayPCs={n_path_pc}; TOTAL={len(feat_names)}"
        )
        if len(feat_names):
            preview = ", ".join(feat_names[:head])
            print(
                f"[Feature preview] {preview} {'...' if len(feat_names)>head else ''}"
            )

    PATH_TR = PATHWAY_FEATURES[tr_idx]
    PATH_VA = (
        PATHWAY_FEATURES[va_idx]
        if HAS_VAL
        else np.zeros((0, PATHWAY_FEATURES.shape[1]), dtype=float)
    )
    PATH_TE = PATHWAY_FEATURES[te_idx]

    if USE_PATHWAYS and RESIDUALIZE_PATHWAYS and (PATH_TR.size > 0):
        Ctr_parts = [Ttr_sel]
        Cva_parts = [Tva_sel] if HAS_VAL else []
        Cte_parts = [Tte_sel]

        if USE_CLINICAL and Xc_tr.shape[1] > 0:
            Ctr_parts.append(Xc_tr)
            if HAS_VAL:
                Cva_parts.append(Xc_va)
            Cte_parts.append(Xc_te)

        if USE_GES:
            Ctr_parts += [ges_ER_tr, ges_Pro_tr]
            if HAS_VAL:
                Cva_parts += [ges_ER_va, ges_Pro_va]
            Cte_parts += [ges_ER_te, ges_Pro_te]

        Ctr = (
            np.hstack(Ctr_parts)
            if len(Ctr_parts)
            else np.zeros((len(tr_idx), 0), dtype=float)
        )
        Cva = (
            np.hstack(Cva_parts)
            if (HAS_VAL and len(Cva_parts))
            else np.zeros((0, Ctr.shape[1]), dtype=float)
        )
        Cte = (
            np.hstack(Cte_parts)
            if len(Cte_parts)
            else np.zeros((len(te_idx), 0), dtype=float)
        )

        PATH_TR, PATH_VA, PATH_TE, _beta_path = residualize_matrix_trainfit(
            PATH_TR, Ctr, PATH_VA, Cva, PATH_TE, Cte, ridge=1e-6
        )
        print(
            f"[Pathways] Residualized pathway PCs on conditioning matrix C (train-fit)."
        )

    if PATHWAY_ONLY:
        Xtr = PATH_TR
        Xva = PATH_VA if HAS_VAL else np.zeros((0, PATH_TR.shape[1]))
        Xte = PATH_TE
        feat_names = pathway_feat_names

        if Xtr.shape[1] == 0:
            raise ValueError(
                "[Pathway-only] No pathway PCs were selected. "
                "Increase PATHWAY_PC_TOP_K or check pathway table alignment."
            )

        n_gene_pc = 0
        n_clin = 0
        n_ges = 0
        n_path_pc = Xtr.shape[1]
        _print_feature_summary(n_gene_pc, n_clin, n_ges, n_path_pc, feat_names)

    else:
        blocks_tr = [Ttr_sel, PATH_TR]
        blocks_va = [Tva_sel, PATH_VA] if HAS_VAL else []
        blocks_te = [Tte_sel, PATH_TE]

        feat_names = []
        feat_names += [*sel_names] if sel_names else []

        if USE_CLINICAL and Xc_tr.shape[1] > 0:
            blocks_tr.insert(1, Xc_tr)
            if HAS_VAL:
                blocks_va.insert(1, Xc_va)
            blocks_te.insert(1, Xc_te)
            feat_names += clin_names

        if USE_GES:
            off = 1 + (1 if USE_CLINICAL and Xc_tr.shape[1] > 0 else 0)
            blocks_tr.insert(off, ges_ER_tr)
            blocks_tr.insert(off + 1, ges_Pro_tr)
            if HAS_VAL:
                blocks_va.insert(off, ges_ER_va)
                blocks_va.insert(off + 1, ges_Pro_va)
            blocks_te.insert(off, ges_ER_te)
            blocks_te.insert(off + 1, ges_Pro_te)
            feat_names += ["ges_ER", "ges_Prolif"]

        if (not isinstance(pathway_feat_names, list)) or (
            len(pathway_feat_names) != PATH_TR.shape[1]
        ):
            pathway_feat_names = [f"pathPC{j+1}" for j in range(PATH_TR.shape[1])]
        feat_names += pathway_feat_names

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
        n_path_pc = PATH_TR.shape[1]
        expected_cols = n_gene_pc + n_clin + n_ges + n_path_pc
        if expected_cols != len(feat_names):
            raise ValueError("Feature-name / matrix-column mismatch.")

        print(
            f"\nFeature dims — PCs={n_gene_pc}, Clin={n_clin}, GES={n_ges}, Path={n_path_pc}; total={expected_cols}"
        )
        _print_feature_summary(n_gene_pc, n_clin, n_ges, n_path_pc, feat_names)

    BLOCKS = infer_blocks_from_feat_names(feat_names)
    print(
        "\n[Blocks] Inferred additive-kernel blocks (by feature name → column indices):"
    )
    for k in ("gene", "clin", "act", "cons", "ges"):
        print(f"  {k:>4s}: {len(BLOCKS.get(k, []))} dims")

    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True)
    sd = np.where((sd <= 0.0) | ~np.isfinite(sd), 1.0, sd)
    Xtr_s = (Xtr - mu) / sd
    Xva_s = (Xva - mu) / sd if HAS_VAL else Xva
    Xte_s = (Xte - mu) / sd

    Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva_s, dtype=torch.float32, device=device) if HAS_VAL else None
    Xte_t = torch.tensor(Xte_s, dtype=torch.float32, device=device)

    t_tr_t = torch.tensor(t_tr, dtype=torch.float32, device=device)
    t_va_t = torch.tensor(t_va, dtype=torch.float32, device=device) if HAS_VAL else None
    t_te_t = torch.tensor(t_te, dtype=torch.float32, device=device)

    e_tr_t = torch.tensor(e_tr, dtype=torch.long, device=device)
    e_va_t = torch.tensor(e_va, dtype=torch.long, device=device) if HAS_VAL else None
    e_te_t = torch.tensor(e_te, dtype=torch.long, device=device)

    CANON_GROUPS = sorted(meta["GROUP_LABEL"].astype(str).unique().tolist())
    print("\n[Groups] Inferred GROUP_LABEL order (index = task id):")
    for i, g in enumerate(CANON_GROUPS):
        print(f"  {i:2d} -> {g}")

    NUM_GROUPS = len(CANON_GROUPS)
    print(f"[INFO] NUM_GROUPS set to {NUM_GROUPS} inferred from GROUP_LABELs.")

    group_to_idx = {g: i for i, g in enumerate(CANON_GROUPS)}
    g_all = meta["GROUP_LABEL"].astype(str).map(group_to_idx).to_numpy(dtype=int)
    g_tr = g_all[tr_idx]
    g_te = g_all[te_idx]
    g_va = g_all[va_idx] if HAS_VAL else np.array([], dtype=int)

    g_tr_t = torch.tensor(g_tr, dtype=torch.long, device=device)
    g_te_t = torch.tensor(g_te, dtype=torch.long, device=device)
    g_va_t = torch.tensor(g_va, dtype=torch.long, device=device) if HAS_VAL else None

    # >>> CONTINUE YOUR SCRIPT FROM HERE UNCHANGED (everything after this line stays identical)

    print("\nRunning PCA on TRAIN only (genes)…")
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
    r_evr = max(r_evr, MIN_PC) if r_evr > 0 else 0

    print(
        f"PCA(train): {r_evr} PCs reach {TARGET_EVR*100:.1f}% cumulative variance (target {TARGET_EVR*100:.1f}%)."
    )

    Ttr_full = Xtr0 @ Vt.T if Xtr0.size else np.zeros((Xtr0.shape[0], 0))
    Tva_full = Xva0 @ Vt.T if (HAS_VAL and Vt.size) else Xva0
    Tte_full = Xte0 @ Vt.T if Vt.size else Xte0
    Ttr = Ttr_full[:, :r_evr] if r_evr else np.zeros((Xtr0.shape[0], 0))
    Tva = (
        Tva_full[:, :r_evr]
        if (HAS_VAL and r_evr)
        else (Tva_full if HAS_VAL else np.zeros((0, 0)))
    )
    Tte = Tte_full[:, :r_evr] if r_evr else np.zeros((Xte0.shape[0], 0))
    pc_names = [f"PC{i+1}" for i in range(r_evr)]
    evr_used = evr_train[:r_evr] if r_evr else np.array([])

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

    pc_rank = (
        rank_pcs_by_cox_z(Ttr, t_tr, e_tr, pc_names)
        if Ttr.shape[1]
        else pd.DataFrame(columns=["pc", "idx", "z", "p", "coef"])
    )
    if len(evr_used):
        pc_rank["evr_train"] = pc_rank["idx"].map(
            {i: evr_used[i] for i in range(len(evr_used))}
        )
    pc_rank.to_csv(OUTDIR / "pc_survival_ranking_train.csv", index=False)

    K = SURV_TOP_K if SURV_TOP_K is not None else min(30, r_evr)
    if (K == 0) or (r_evr == 0):
        keep_idxs = np.array([], dtype=int)
        print("PC selection: skipping gene PCs (SURV_TOP_K=0 or r_evr=0).")
    else:
        K = int(np.clip(K, MIN_PC, min(MAX_PC, r_evr)))
        keep_idxs = (
            pc_rank.head(K)["idx"].to_numpy()
            if len(pc_rank)
            else np.array([], dtype=int)
        )
        print(
            f"PC selection: using top {len(keep_idxs)} by Cox Z out of {r_evr} PCA PCs (bounds [{MIN_PC}, {min(MAX_PC, r_evr)}])."
        )

    Ttr_sel = Ttr[:, keep_idxs] if keep_idxs.size else np.zeros((Ttr.shape[0], 0))
    Tva_sel = Tva[:, keep_idxs] if (HAS_VAL and keep_idxs.size) else np.zeros((0, 0))
    Tte_sel = Tte[:, keep_idxs] if keep_idxs.size else np.zeros((Tte.shape[0], 0))
    sel_names = [pc_names[i] for i in keep_idxs] if keep_idxs.size else []
    print(f"PCs selected by survival Z: {len(sel_names)}")

    # ============================================================
    # ------------------- Clinical covariates (ADJUSTED TO GIVE 5) -------------------
    # ============================================================
    # CHANGE (requested): categorical clinicals now match "first code" behavior:
    #  - missing categories are IMPUTED to TRAIN MODE (no "__MISSING__" dummy)
    #  - val/test unseen categories are mapped to TRAIN MODE
    # Everything else unchanged.
    DROP_FIRST_OHE = True

    _MISSING_TOKENS2 = {
        "",
        "NA",
        "N/A",
        "NONE",
        "NULL",
        "NAN",
        "UNKNOWN",
        "UNAVAILABLE",
        "NOT REPORTED",
        ".",
    }

    def _clean_cat_token(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if s == "" or s.upper() in _MISSING_TOKENS2:
            return np.nan
        return s

    def _canon_cat(x):
        if pd.isna(x):
            return np.nan
        u = str(x).strip().upper()
        u = re.sub(r"\s+", "_", u)
        u = re.sub(r"[^\w]+", "_", u)
        u = re.sub(r"_+", "_", u).strip("_")
        return u if u != "" else np.nan

    def _prep_cat_series(s: pd.Series) -> pd.Series:
        return s.map(_clean_cat_token).map(_canon_cat)

    def _prep_grade_series(s: pd.Series) -> pd.Series:
        s2 = s.map(_clean_cat_token)

        def _g(v):
            if pd.isna(v):
                return np.nan
            txt = str(v).strip().upper()
            m = re.search(r"([1-4])", txt)
            return m.group(1) if m else np.nan

        return s2.map(_g)

    NUMERIC_CLIN = [c for c in ["AGE_AT_DIAGNOSIS", "TUMOR_SIZE"] if c in meta.columns]
    CAT_CLIN = [c for c in ["GRADE", "INFERRED_MENOPAUSAL_STATE"] if c in meta.columns]

    def _train_mode(series: pd.Series):
        s = series.dropna()
        if len(s) == 0:
            return None
        m = s.mode()
        return m.iloc[0] if len(m) else None

    def _train_fitted_cat_levels_and_fill(meta_df, tr_idx, col):
        # cleaned train series
        tr_clean = (
            _prep_grade_series(meta_df.iloc[tr_idx][col])
            if col == "GRADE"
            else _prep_cat_series(meta_df.iloc[tr_idx][col])
        )
        fill = _train_mode(tr_clean)
        if fill is None:
            fill = "UNKNOWN"
        tr_filled = tr_clean.fillna(fill)

        # levels are TRAIN categories ONLY (no missing dummy)
        levels = sorted(pd.unique(tr_filled))

        return levels, fill, tr_clean  # return tr_clean for debug print (pre-impute)

    def _ohe_from_levels(meta_df, col, levels, fill_value, drop_first_ohe: bool):
        s = (
            _prep_grade_series(meta_df[col])
            if col == "GRADE"
            else _prep_cat_series(meta_df[col])
        )

        # impute missing to TRAIN fill
        s = s.fillna(fill_value)

        # map unseen categories to TRAIN fill (keeps column count stable)
        s = s.where(s.isin(levels), other=fill_value)

        cat = pd.Categorical(s, categories=levels)
        d = pd.get_dummies(cat, prefix=col)

        if drop_first_ohe and (d.shape[1] >= 2):
            d = d.iloc[:, 1:]
        return d

    def build_clin_matrices_train_fitted(meta_df, tr_idx, va_idx, te_idx):
        blocks_all = []
        names = []

        # Numeric: impute with TRAIN median (stable)
        for c in NUMERIC_CLIN:
            s_all = pd.to_numeric(meta_df[c], errors="coerce")
            med = pd.to_numeric(meta_df.iloc[tr_idx][c], errors="coerce").median()
            if not np.isfinite(med):
                med = 0.0
            s_all = s_all.fillna(med)
            blocks_all.append(s_all.to_numpy(dtype=float)[:, None])
            names.append(c)

        # Categorical: TRAIN-fitted levels, missing -> TRAIN mode (no missing dummy col)
        for c in CAT_CLIN:
            levels, fill_value, tr_clean_pre = _train_fitted_cat_levels_and_fill(
                meta_df, tr_idx, c
            )

            # Debug prints
            print(f"\n[Clinical] {c} TRAIN cleaned levels (pre-impute):")
            print(tr_clean_pre.value_counts(dropna=False).to_string())
            print(f"[Clinical] {c} TRAIN mode fill value: {fill_value}")
            print(f"[Clinical] {c} TRAIN-fitted levels (after impute): {levels}")

            d_all = _ohe_from_levels(meta_df, c, levels, fill_value, DROP_FIRST_OHE)
            if d_all.shape[1] > 0:
                blocks_all.append(d_all.to_numpy(dtype=float))
                names += list(d_all.columns)

        X_all = (
            np.hstack(blocks_all)
            if blocks_all
            else np.zeros((len(meta_df), 0), dtype=float)
        )
        Xc_tr = X_all[tr_idx]
        Xc_va = X_all[va_idx] if HAS_VAL else np.zeros((0, X_all.shape[1]), dtype=float)
        Xc_te = X_all[te_idx]
        return Xc_tr, Xc_va, Xc_te, names

    Xc_tr, Xc_va, Xc_te, clin_names = build_clin_matrices_train_fitted(
        meta, tr_idx, va_idx, te_idx
    )

    # ------------------- Final matrices + scaling (TRAIN stats) ----------------
    if "PATHWAY_FEATURES" not in locals():
        PATHWAY_FEATURES = np.zeros((len(meta), 0), dtype=float)
        pathway_feat_names = []

    def _print_feature_summary(n_gene_pc, n_clin, n_ges, n_path_pc, feat_names, head=6):
        print(
            f"[Feature summary] GenePCs={n_gene_pc}, Clinical={n_clin}, "
            f"GES={n_ges}, PathwayPCs={n_path_pc}; TOTAL={len(feat_names)}"
        )
        if len(feat_names):
            preview = ", ".join(feat_names[:head])
            print(
                f"[Feature preview] {preview} {'...' if len(feat_names)>head else ''}"
            )

    PATH_TR = PATHWAY_FEATURES[tr_idx]
    PATH_VA = (
        PATHWAY_FEATURES[va_idx]
        if HAS_VAL
        else np.zeros((0, PATHWAY_FEATURES.shape[1]), dtype=float)
    )
    PATH_TE = PATHWAY_FEATURES[te_idx]

    if USE_PATHWAYS and RESIDUALIZE_PATHWAYS and (PATH_TR.size > 0):
        Ctr_parts = [Ttr_sel]
        Cva_parts = [Tva_sel] if HAS_VAL else []
        Cte_parts = [Tte_sel]

        if USE_CLINICAL and Xc_tr.shape[1] > 0:
            Ctr_parts.append(Xc_tr)
            if HAS_VAL:
                Cva_parts.append(Xc_va)
            Cte_parts.append(Xc_te)

        if USE_GES:
            Ctr_parts += [ges_ER_tr, ges_Pro_tr]
            if HAS_VAL:
                Cva_parts += [ges_ER_va, ges_Pro_va]
            Cte_parts += [ges_ER_te, ges_Pro_te]

        Ctr = (
            np.hstack(Ctr_parts)
            if len(Ctr_parts)
            else np.zeros((len(tr_idx), 0), dtype=float)
        )
        Cva = (
            np.hstack(Cva_parts)
            if (HAS_VAL and len(Cva_parts))
            else np.zeros((0, Ctr.shape[1]), dtype=float)
        )
        Cte = (
            np.hstack(Cte_parts)
            if len(Cte_parts)
            else np.zeros((len(te_idx), 0), dtype=float)
        )

        PATH_TR, PATH_VA, PATH_TE, _beta_path = residualize_matrix_trainfit(
            PATH_TR, Ctr, PATH_VA, Cva, PATH_TE, Cte, ridge=1e-6
        )
        print(
            f"[Pathways] Residualized pathway PCs on conditioning matrix C (train-fit)."
        )

    if PATHWAY_ONLY:
        Xtr = PATH_TR
        Xva = PATH_VA if HAS_VAL else np.zeros((0, PATH_TR.shape[1]))
        Xte = PATH_TE
        feat_names = pathway_feat_names

        if Xtr.shape[1] == 0:
            raise ValueError(
                "[Pathway-only] No pathway PCs were selected. "
                "Increase PATHWAY_PC_TOP_K or check pathway table alignment."
            )

        n_gene_pc = 0
        n_clin = 0
        n_ges = 0
        n_path_pc = Xtr.shape[1]
        _print_feature_summary(n_gene_pc, n_clin, n_ges, n_path_pc, feat_names)

    else:
        blocks_tr = [Ttr_sel, PATH_TR]
        blocks_va = [Tva_sel, PATH_VA] if HAS_VAL else []
        blocks_te = [Tte_sel, PATH_TE]

        feat_names = []
        feat_names += [*sel_names] if sel_names else []

        if USE_CLINICAL and Xc_tr.shape[1] > 0:
            blocks_tr.insert(1, Xc_tr)
            if HAS_VAL:
                blocks_va.insert(1, Xc_va)
            blocks_te.insert(1, Xc_te)
            feat_names += clin_names

        if USE_GES:
            off = 1 + (1 if USE_CLINICAL and Xc_tr.shape[1] > 0 else 0)
            blocks_tr.insert(off, ges_ER_tr)
            blocks_tr.insert(off + 1, ges_Pro_tr)
            if HAS_VAL:
                blocks_va.insert(off, ges_ER_va)
                blocks_va.insert(off + 1, ges_Pro_va)
            blocks_te.insert(off, ges_ER_te)
            blocks_te.insert(off + 1, ges_Pro_te)
            feat_names += ["ges_ER", "ges_Prolif"]

        if (not isinstance(pathway_feat_names, list)) or (
            len(pathway_feat_names) != PATH_TR.shape[1]
        ):
            pathway_feat_names = [f"pathPC{j+1}" for j in range(PATH_TR.shape[1])]
        feat_names += pathway_feat_names

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
        n_path_pc = PATH_TR.shape[1]
        expected_cols = n_gene_pc + n_clin + n_ges + n_path_pc
        if expected_cols != len(feat_names):
            raise ValueError("Feature-name / matrix-column mismatch.")

        print(
            f"\nFeature dims — PCs={n_gene_pc}, Clin={n_clin}, GES={n_ges}, Path={n_path_pc}; total={expected_cols}"
        )
        _print_feature_summary(n_gene_pc, n_clin, n_ges, n_path_pc, feat_names)

    BLOCKS = infer_blocks_from_feat_names(feat_names)
    print(
        "\n[Blocks] Inferred additive-kernel blocks (by feature name → column indices):"
    )
    for k in ("gene", "clin", "act", "cons", "ges"):
        print(f"  {k:>4s}: {len(BLOCKS.get(k, []))} dims")

    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True)
    sd = np.where((sd <= 0.0) | ~np.isfinite(sd), 1.0, sd)
    Xtr_s = (Xtr - mu) / sd
    Xva_s = (Xva - mu) / sd if HAS_VAL else Xva
    Xte_s = (Xte - mu) / sd

    Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva_s, dtype=torch.float32, device=device) if HAS_VAL else None
    Xte_t = torch.tensor(Xte_s, dtype=torch.float32, device=device)

    t_tr_t = torch.tensor(t_tr, dtype=torch.float32, device=device)
    t_va_t = torch.tensor(t_va, dtype=torch.float32, device=device) if HAS_VAL else None
    t_te_t = torch.tensor(t_te, dtype=torch.float32, device=device)

    e_tr_t = torch.tensor(e_tr, dtype=torch.long, device=device)
    e_va_t = torch.tensor(e_va, dtype=torch.long, device=device) if HAS_VAL else None
    e_te_t = torch.tensor(e_te, dtype=torch.long, device=device)

    CANON_GROUPS = sorted(meta["GROUP_LABEL"].astype(str).unique().tolist())
    print("\n[Groups] Inferred GROUP_LABEL order (index = task id):")
    for i, g in enumerate(CANON_GROUPS):
        print(f"  {i:2d} -> {g}")

    NUM_GROUPS = len(CANON_GROUPS)
    print(f"[INFO] NUM_GROUPS set to {NUM_GROUPS} inferred from GROUP_LABELs.")

    group_to_idx = {g: i for i, g in enumerate(CANON_GROUPS)}
    g_all = meta["GROUP_LABEL"].astype(str).map(group_to_idx).to_numpy(dtype=int)
    g_tr = g_all[tr_idx]
    g_te = g_all[te_idx]
    g_va = g_all[va_idx] if HAS_VAL else np.array([], dtype=int)

    g_tr_t = torch.tensor(g_tr, dtype=torch.long, device=device)
    g_te_t = torch.tensor(g_te, dtype=torch.long, device=device)
    g_va_t = torch.tensor(g_va, dtype=torch.long, device=device) if HAS_VAL else None

    # >>> CONTINUE YOUR SCRIPT FROM HERE UNCHANGED (everything after this line stays identical)

    print("\nRunning PCA on TRAIN only (genes)…")
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
    r_evr = max(r_evr, MIN_PC) if r_evr > 0 else 0

    print(
        f"PCA(train): {r_evr} PCs reach {TARGET_EVR*100:.1f}% cumulative variance (target {TARGET_EVR*100:.1f}%)."
    )

    Ttr_full = Xtr0 @ Vt.T if Xtr0.size else np.zeros((Xtr0.shape[0], 0))
    Tva_full = Xva0 @ Vt.T if (HAS_VAL and Vt.size) else Xva0
    Tte_full = Xte0 @ Vt.T if Vt.size else Xte0
    Ttr = Ttr_full[:, :r_evr] if r_evr else np.zeros((Xtr0.shape[0], 0))
    Tva = (
        Tva_full[:, :r_evr]
        if (HAS_VAL and r_evr)
        else (Tva_full if HAS_VAL else np.zeros((0, 0)))
    )
    Tte = Tte_full[:, :r_evr] if r_evr else np.zeros((Xte0.shape[0], 0))
    pc_names = [f"PC{i+1}" for i in range(r_evr)]
    evr_used = evr_train[:r_evr] if r_evr else np.array([])

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

    pc_rank = (
        rank_pcs_by_cox_z(Ttr, t_tr, e_tr, pc_names)
        if Ttr.shape[1]
        else pd.DataFrame(columns=["pc", "idx", "z", "p", "coef"])
    )
    if len(evr_used):
        pc_rank["evr_train"] = pc_rank["idx"].map(
            {i: evr_used[i] for i in range(len(evr_used))}
        )
    pc_rank.to_csv(OUTDIR / "pc_survival_ranking_train.csv", index=False)

    K = SURV_TOP_K if SURV_TOP_K is not None else min(30, r_evr)
    if (K == 0) or (r_evr == 0):
        keep_idxs = np.array([], dtype=int)
        print("PC selection: skipping gene PCs (SURV_TOP_K=0 or r_evr=0).")
    else:
        K = int(np.clip(K, MIN_PC, min(MAX_PC, r_evr)))
        keep_idxs = (
            pc_rank.head(K)["idx"].to_numpy()
            if len(pc_rank)
            else np.array([], dtype=int)
        )
        print(
            f"PC selection: using top {len(keep_idxs)} by Cox Z out of {r_evr} PCA PCs (bounds [{MIN_PC}, {min(MAX_PC, r_evr)}])."
        )

    Ttr_sel = Ttr[:, keep_idxs] if keep_idxs.size else np.zeros((Ttr.shape[0], 0))
    Tva_sel = Tva[:, keep_idxs] if (HAS_VAL and keep_idxs.size) else np.zeros((0, 0))
    Tte_sel = Tte[:, keep_idxs] if keep_idxs.size else np.zeros((Tte.shape[0], 0))
    sel_names = [pc_names[i] for i in keep_idxs] if keep_idxs.size else []
    print(f"PCs selected by survival Z: {len(sel_names)}")

    # ============================================================
    # ------------------- Clinical covariates (ADJUSTED TO GIVE 5) -------------------
    # ============================================================
    # CHANGE (requested): categorical clinicals now match "first code" behavior:
    #  - missing categories are IMPUTED to TRAIN MODE (no "__MISSING__" dummy)
    #  - val/test unseen categories are mapped to TRAIN MODE
    # Everything else unchanged.
    DROP_FIRST_OHE = True

    _MISSING_TOKENS2 = {
        "",
        "NA",
        "N/A",
        "NONE",
        "NULL",
        "NAN",
        "UNKNOWN",
        "UNAVAILABLE",
        "NOT REPORTED",
        ".",
    }

    def _clean_cat_token(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if s == "" or s.upper() in _MISSING_TOKENS2:
            return np.nan
        return s

    def _canon_cat(x):
        if pd.isna(x):
            return np.nan
        u = str(x).strip().upper()
        u = re.sub(r"\s+", "_", u)
        u = re.sub(r"[^\w]+", "_", u)
        u = re.sub(r"_+", "_", u).strip("_")
        return u if u != "" else np.nan

    def _prep_cat_series(s: pd.Series) -> pd.Series:
        return s.map(_clean_cat_token).map(_canon_cat)

    def _prep_grade_series(s: pd.Series) -> pd.Series:
        s2 = s.map(_clean_cat_token)

        def _g(v):
            if pd.isna(v):
                return np.nan
            txt = str(v).strip().upper()
            m = re.search(r"([1-4])", txt)
            return m.group(1) if m else np.nan

        return s2.map(_g)

    NUMERIC_CLIN = [c for c in ["AGE_AT_DIAGNOSIS", "TUMOR_SIZE"] if c in meta.columns]
    CAT_CLIN = [c for c in ["GRADE", "INFERRED_MENOPAUSAL_STATE"] if c in meta.columns]

    def _train_mode(series: pd.Series):
        s = series.dropna()
        if len(s) == 0:
            return None
        m = s.mode()
        return m.iloc[0] if len(m) else None

    def _train_fitted_cat_levels_and_fill(meta_df, tr_idx, col):
        # cleaned train series
        tr_clean = (
            _prep_grade_series(meta_df.iloc[tr_idx][col])
            if col == "GRADE"
            else _prep_cat_series(meta_df.iloc[tr_idx][col])
        )
        fill = _train_mode(tr_clean)
        if fill is None:
            fill = "UNKNOWN"
        tr_filled = tr_clean.fillna(fill)

        # levels are TRAIN categories ONLY (no missing dummy)
        levels = sorted(pd.unique(tr_filled))

        return levels, fill, tr_clean  # return tr_clean for debug print (pre-impute)

    def _ohe_from_levels(meta_df, col, levels, fill_value, drop_first_ohe: bool):
        s = (
            _prep_grade_series(meta_df[col])
            if col == "GRADE"
            else _prep_cat_series(meta_df[col])
        )

        # impute missing to TRAIN fill
        s = s.fillna(fill_value)

        # map unseen categories to TRAIN fill (keeps column count stable)
        s = s.where(s.isin(levels), other=fill_value)

        cat = pd.Categorical(s, categories=levels)
        d = pd.get_dummies(cat, prefix=col)

        if drop_first_ohe and (d.shape[1] >= 2):
            d = d.iloc[:, 1:]
        return d

    def build_clin_matrices_train_fitted(meta_df, tr_idx, va_idx, te_idx):
        blocks_all = []
        names = []

        # Numeric: impute with TRAIN median (stable)
        for c in NUMERIC_CLIN:
            s_all = pd.to_numeric(meta_df[c], errors="coerce")
            med = pd.to_numeric(meta_df.iloc[tr_idx][c], errors="coerce").median()
            if not np.isfinite(med):
                med = 0.0
            s_all = s_all.fillna(med)
            blocks_all.append(s_all.to_numpy(dtype=float)[:, None])
            names.append(c)

        # Categorical: TRAIN-fitted levels, missing -> TRAIN mode (no missing dummy col)
        for c in CAT_CLIN:
            levels, fill_value, tr_clean_pre = _train_fitted_cat_levels_and_fill(
                meta_df, tr_idx, c
            )

            # Debug prints
            print(f"\n[Clinical] {c} TRAIN cleaned levels (pre-impute):")
            print(tr_clean_pre.value_counts(dropna=False).to_string())
            print(f"[Clinical] {c} TRAIN mode fill value: {fill_value}")
            print(f"[Clinical] {c} TRAIN-fitted levels (after impute): {levels}")

            d_all = _ohe_from_levels(meta_df, c, levels, fill_value, DROP_FIRST_OHE)
            if d_all.shape[1] > 0:
                blocks_all.append(d_all.to_numpy(dtype=float))
                names += list(d_all.columns)

        X_all = (
            np.hstack(blocks_all)
            if blocks_all
            else np.zeros((len(meta_df), 0), dtype=float)
        )
        Xc_tr = X_all[tr_idx]
        Xc_va = X_all[va_idx] if HAS_VAL else np.zeros((0, X_all.shape[1]), dtype=float)
        Xc_te = X_all[te_idx]
        return Xc_tr, Xc_va, Xc_te, names

    Xc_tr, Xc_va, Xc_te, clin_names = build_clin_matrices_train_fitted(
        meta, tr_idx, va_idx, te_idx
    )

    # ------------------- Final matrices + scaling (TRAIN stats) ----------------
    if "PATHWAY_FEATURES" not in locals():
        PATHWAY_FEATURES = np.zeros((len(meta), 0), dtype=float)
        pathway_feat_names = []

    def _print_feature_summary(n_gene_pc, n_clin, n_ges, n_path_pc, feat_names, head=6):
        print(
            f"[Feature summary] GenePCs={n_gene_pc}, Clinical={n_clin}, "
            f"GES={n_ges}, PathwayPCs={n_path_pc}; TOTAL={len(feat_names)}"
        )
        if len(feat_names):
            preview = ", ".join(feat_names[:head])
            print(
                f"[Feature preview] {preview} {'...' if len(feat_names)>head else ''}"
            )

    PATH_TR = PATHWAY_FEATURES[tr_idx]
    PATH_VA = (
        PATHWAY_FEATURES[va_idx]
        if HAS_VAL
        else np.zeros((0, PATHWAY_FEATURES.shape[1]), dtype=float)
    )
    PATH_TE = PATHWAY_FEATURES[te_idx]

    if USE_PATHWAYS and RESIDUALIZE_PATHWAYS and (PATH_TR.size > 0):
        Ctr_parts = [Ttr_sel]
        Cva_parts = [Tva_sel] if HAS_VAL else []
        Cte_parts = [Tte_sel]

        if USE_CLINICAL and Xc_tr.shape[1] > 0:
            Ctr_parts.append(Xc_tr)
            if HAS_VAL:
                Cva_parts.append(Xc_va)
            Cte_parts.append(Xc_te)

        if USE_GES:
            Ctr_parts += [ges_ER_tr, ges_Pro_tr]
            if HAS_VAL:
                Cva_parts += [ges_ER_va, ges_Pro_va]
            Cte_parts += [ges_ER_te, ges_Pro_te]

        Ctr = (
            np.hstack(Ctr_parts)
            if len(Ctr_parts)
            else np.zeros((len(tr_idx), 0), dtype=float)
        )
        Cva = (
            np.hstack(Cva_parts)
            if (HAS_VAL and len(Cva_parts))
            else np.zeros((0, Ctr.shape[1]), dtype=float)
        )
        Cte = (
            np.hstack(Cte_parts)
            if len(Cte_parts)
            else np.zeros((len(te_idx), 0), dtype=float)
        )

        PATH_TR, PATH_VA, PATH_TE, _beta_path = residualize_matrix_trainfit(
            PATH_TR, Ctr, PATH_VA, Cva, PATH_TE, Cte, ridge=1e-6
        )
        print(
            f"[Pathways] Residualized pathway PCs on conditioning matrix C (train-fit)."
        )

    if PATHWAY_ONLY:
        Xtr = PATH_TR
        Xva = PATH_VA if HAS_VAL else np.zeros((0, PATH_TR.shape[1]))
        Xte = PATH_TE
        feat_names = pathway_feat_names

        if Xtr.shape[1] == 0:
            raise ValueError(
                "[Pathway-only] No pathway PCs were selected. "
                "Increase PATHWAY_PC_TOP_K or check pathway table alignment."
            )

        n_gene_pc = 0
        n_clin = 0
        n_ges = 0
        n_path_pc = Xtr.shape[1]
        _print_feature_summary(n_gene_pc, n_clin, n_ges, n_path_pc, feat_names)

    else:
        blocks_tr = [Ttr_sel, PATH_TR]
        blocks_va = [Tva_sel, PATH_VA] if HAS_VAL else []
        blocks_te = [Tte_sel, PATH_TE]

        feat_names = []
        feat_names += [*sel_names] if sel_names else []

        if USE_CLINICAL and Xc_tr.shape[1] > 0:
            blocks_tr.insert(1, Xc_tr)
            if HAS_VAL:
                blocks_va.insert(1, Xc_va)
            blocks_te.insert(1, Xc_te)
            feat_names += clin_names

        if USE_GES:
            off = 1 + (1 if USE_CLINICAL and Xc_tr.shape[1] > 0 else 0)
            blocks_tr.insert(off, ges_ER_tr)
            blocks_tr.insert(off + 1, ges_Pro_tr)
            if HAS_VAL:
                blocks_va.insert(off, ges_ER_va)
                blocks_va.insert(off + 1, ges_Pro_va)
            blocks_te.insert(off, ges_ER_te)
            blocks_te.insert(off + 1, ges_Pro_te)
            feat_names += ["ges_ER", "ges_Prolif"]

        if (not isinstance(pathway_feat_names, list)) or (
            len(pathway_feat_names) != PATH_TR.shape[1]
        ):
            pathway_feat_names = [f"pathPC{j+1}" for j in range(PATH_TR.shape[1])]
        feat_names += pathway_feat_names

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
        n_path_pc = PATH_TR.shape[1]
        expected_cols = n_gene_pc + n_clin + n_ges + n_path_pc
        if expected_cols != len(feat_names):
            raise ValueError("Feature-name / matrix-column mismatch.")

        print(
            f"\nFeature dims — PCs={n_gene_pc}, Clin={n_clin}, GES={n_ges}, Path={n_path_pc}; total={expected_cols}"
        )
        _print_feature_summary(n_gene_pc, n_clin, n_ges, n_path_pc, feat_names)

    BLOCKS = infer_blocks_from_feat_names(feat_names)
    print(
        "\n[Blocks] Inferred additive-kernel blocks (by feature name → column indices):"
    )
    for k in ("gene", "clin", "act", "cons", "ges"):
        print(f"  {k:>4s}: {len(BLOCKS.get(k, []))} dims")

    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True)
    sd = np.where((sd <= 0.0) | ~np.isfinite(sd), 1.0, sd)
    Xtr_s = (Xtr - mu) / sd
    Xva_s = (Xva - mu) / sd if HAS_VAL else Xva
    Xte_s = (Xte - mu) / sd

    Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva_s, dtype=torch.float32, device=device) if HAS_VAL else None
    Xte_t = torch.tensor(Xte_s, dtype=torch.float32, device=device)

    t_tr_t = torch.tensor(t_tr, dtype=torch.float32, device=device)
    t_va_t = torch.tensor(t_va, dtype=torch.float32, device=device) if HAS_VAL else None
    t_te_t = torch.tensor(t_te, dtype=torch.float32, device=device)

    e_tr_t = torch.tensor(e_tr, dtype=torch.long, device=device)
    e_va_t = torch.tensor(e_va, dtype=torch.long, device=device) if HAS_VAL else None
    e_te_t = torch.tensor(e_te, dtype=torch.long, device=device)

    CANON_GROUPS = sorted(meta["GROUP_LABEL"].astype(str).unique().tolist())
    print("\n[Groups] Inferred GROUP_LABEL order (index = task id):")
    for i, g in enumerate(CANON_GROUPS):
        print(f"  {i:2d} -> {g}")

    NUM_GROUPS = len(CANON_GROUPS)
    print(f"[INFO] NUM_GROUPS set to {NUM_GROUPS} inferred from GROUP_LABELs.")

    group_to_idx = {g: i for i, g in enumerate(CANON_GROUPS)}
    g_all = meta["GROUP_LABEL"].astype(str).map(group_to_idx).to_numpy(dtype=int)
    g_tr = g_all[tr_idx]
    g_te = g_all[te_idx]
    g_va = g_all[va_idx] if HAS_VAL else np.array([], dtype=int)

    g_tr_t = torch.tensor(g_tr, dtype=torch.long, device=device)
    g_te_t = torch.tensor(g_te, dtype=torch.long, device=device)
    g_va_t = torch.tensor(g_va, dtype=torch.long, device=device) if HAS_VAL else None

    # =================== OPTION 4: Multitask DGP (shared + multitask OUTPUT) ===================
    # =================== OPTION 4: Multitask SHARED + Private OUTPUT per group ===================

    # ---- set THESE explicitly ----
    ICM_RANK = 6  # will be clamped to <= NUM_GROUPS below

    NUM_INDUCING_SHARED = NUM_INDUCING
    NUM_INDUCING_OUT = NUM_INDUCING

    # Shared representation dimensionality
    SHARED_DIM = HIDDEN_DIM

    def _mk_matern52_block(active_dims, batch_shape):
        if active_dims is None or len(active_dims) == 0:
            return None
        active_dims = list(map(int, active_dims))
        base = MaternKernel(
            nu=2.5,
            ard_num_dims=len(active_dims),
            batch_shape=batch_shape,
            active_dims=active_dims,
        )
        return ScaleKernel(base, batch_shape=batch_shape)

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
                module,
                inducing,
                q,
                learn_inducing_locations=learn_inducing_locations,
            )

    class SharedDimICM(gpytorch.models.ApproximateGP):
        """
        One shared latent dimension h:
        multitask GP across NUM_GROUPS via LMC/ICM rank=ICM_RANK.
        Input: x in R^{in_dim}
        Output: f_h(x) is multitask over groups.
        """

        def __init__(self, in_dim, Xref, seed, num_groups, rank, blocks=None):
            num_groups = int(num_groups)
            rank = int(min(max(1, rank), num_groups))

            # latent functions live in batch dim
            batch_shape = torch.Size([rank])
            M = int(min(NUM_INDUCING_SHARED, Xref.size(0)))

            if KMEANS_INIT and M >= 2:
                with torch.no_grad():
                    km = KMeans(n_clusters=M, random_state=seed, n_init=10)
                    centers = km.fit(Xref.detach().cpu().numpy()).cluster_centers_
                    inducing_base = torch.from_numpy(centers).to(
                        dtype=Xref.dtype, device=Xref.device
                    )
            else:
                idx = torch.randperm(Xref.size(0), device=Xref.device)[:M]
                inducing_base = Xref[idx]

            inducing = inducing_base.unsqueeze(0).expand(rank, -1, -1).contiguous()

            q = (
                MeanFieldVariationalDistribution(M, batch_shape=batch_shape)
                if VI_DIST.lower() == "meanfield"
                else CholeskyVariationalDistribution(M, batch_shape=batch_shape)
            )

            base_vs = make_vs(
                self, inducing, q, learn_inducing_locations=True, prefer_whiten=True
            )

            lmc_vs = LMCVariationalStrategy(
                base_vs,
                num_tasks=num_groups,
                num_latents=rank,
                latent_dim=-1,
            )
            super().__init__(lmc_vs)

            self.mean_module = ZeroMean(batch_shape=batch_shape)

            # ---- covariance: additive blocks (if any) else single ARD Matern ----
            if blocks and any(len(v) > 0 for v in blocks.values()):
                pieces = []
                for key in ("gene", "clin", "act", "cons", "ges"):
                    kblk = _mk_matern52_block(
                        blocks.get(key, []), batch_shape=batch_shape
                    )
                    if kblk is not None:
                        pieces.append(kblk)

                if len(pieces) == 0:
                    self.covar_module = ScaleKernel(
                        MaternKernel(
                            nu=2.5, ard_num_dims=in_dim, batch_shape=batch_shape
                        ),
                        batch_shape=batch_shape,
                    )
                else:
                    cov = pieces[0]
                    for k in pieces[1:]:
                        cov = cov + k
                    self.covar_module = cov
            else:
                self.covar_module = ScaleKernel(
                    MaternKernel(nu=2.5, ard_num_dims=in_dim, batch_shape=batch_shape),
                    batch_shape=batch_shape,
                )

            with torch.no_grad():

                def _init_one(kmod):
                    if isinstance(kmod, ScaleKernel):
                        _init_one(kmod.base_kernel)
                        try:
                            kmod.outputscale.fill_(1.5)
                        except Exception:
                            pass
                    elif isinstance(kmod, gpytorch.kernels.MaternKernel):
                        try:
                            kmod.raw_lengthscale.copy_(
                                torch.full_like(kmod.raw_lengthscale, inv_softplus(1.5))
                            )
                        except Exception:
                            pass
                    elif hasattr(kmod, "kernels"):
                        for kk in kmod.kernels:
                            _init_one(kk)

                _init_one(self.covar_module)

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    class SharedLayerMultitask(torch.nn.Module):
        """
        Shared representation s(x) in R^{SHARED_DIM}, where each latent dim is
        a multitask (group-correlated) GP across NUM_GROUPS.

        forward(x, g_idx) returns s_sel: [N, SHARED_DIM], selecting the task g_idx
        per sample from each multitask latent dimension.
        """

        def __init__(
            self, in_dim, shared_dim, Xref, seed, num_groups, rank, blocks=None
        ):
            super().__init__()
            self.shared_dim = int(shared_dim)
            self.num_groups = int(num_groups)

            self.gps = torch.nn.ModuleList(
                [
                    SharedDimICM(
                        in_dim=in_dim,
                        Xref=Xref,
                        seed=seed + 1000 * h,
                        num_groups=self.num_groups,
                        rank=rank,
                        blocks=blocks,
                    )
                    for h in range(self.shared_dim)
                ]
            )

        def forward(self, x, g_idx):
            g_idx = g_idx.to(device=x.device, dtype=torch.long).view(-1)
            feats = []
            for gp_h in self.gps:
                # dist_h is multitask over groups
                dist_h = gp_h(x)
                f_all = dist_h.rsample()  # expected shape [N, G]
                if (
                    f_all.dim() == 2
                    and f_all.shape[0] == self.num_groups
                    and f_all.shape[1] == x.shape[0]
                ):
                    # occasionally comes out [G, N]
                    f_all = f_all.transpose(0, 1).contiguous()
                f_sel = _select_task(f_all, g_idx, num_tasks=self.num_groups)  # [N]
                feats.append(f_sel)
            s = torch.stack(feats, dim=1)  # [N, SHARED_DIM]
            return s

    class OutputLayerPrivate(gpytorch.models.ApproximateGP):
        """
        Private output GP per group (independent heads):
        batch_shape = [NUM_GROUPS]
        Input: shared representation s in R^{SHARED_DIM}
        Output: f_all for ALL groups, no cross-group mixing.
        """

        def __init__(self, in_dim, Xref, seed, num_groups):
            num_groups = int(num_groups)
            batch_shape = torch.Size([num_groups])

            M = int(min(NUM_INDUCING_OUT, Xref.size(0)))

            # initialize inducing in representation space (we don't have s-ref yet),
            # so just small Gaussian; it will learn locations.
            inducing = (
                torch.randn(num_groups, M, in_dim, device=Xref.device, dtype=Xref.dtype)
                * 0.05
            )

            q = (
                MeanFieldVariationalDistribution(M, batch_shape=batch_shape)
                if VI_DIST.lower() == "meanfield"
                else CholeskyVariationalDistribution(M, batch_shape=batch_shape)
            )

            vs = make_vs(
                self, inducing, q, learn_inducing_locations=True, prefer_whiten=True
            )
            super().__init__(vs)

            self.mean_module = ZeroMean(batch_shape=batch_shape)
            self.covar_module = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=in_dim, batch_shape=batch_shape),
                batch_shape=batch_shape,
            )

            with torch.no_grad():
                try:
                    self.covar_module.base_kernel.raw_lengthscale.copy_(
                        torch.full_like(
                            self.covar_module.base_kernel.raw_lengthscale,
                            inv_softplus(1.5),
                        )
                    )
                except Exception:
                    try:
                        self.covar_module.base_kernel.lengthscale.fill_(1.5)
                    except Exception:
                        pass
                try:
                    self.covar_module.outputscale.fill_(2.0)
                except Exception:
                    pass

        def forward(self, s):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(s), self.covar_module(s)
            )

    class DGP(gpytorch.models.deep_gps.DeepGP):
        def __init__(self, in_dim, Xref, seed, num_groups, icm_rank, blocks=None):
            super().__init__()
            self.num_groups = int(num_groups)

            self.shared = SharedLayerMultitask(
                in_dim=in_dim,
                shared_dim=SHARED_DIM,
                Xref=Xref,
                seed=seed,
                num_groups=self.num_groups,
                rank=icm_rank,
                blocks=blocks,
            )

            self.output = OutputLayerPrivate(
                in_dim=SHARED_DIM,
                Xref=Xref,
                seed=seed,
                num_groups=self.num_groups,
            )

            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            with torch.no_grad():
                self.likelihood.noise = 1e-3

        def forward(self, x, g_idx):
            # shared representation for each sample, selecting its group in the shared multitask layer
            s = self.shared(x, g_idx)  # [N, SHARED_DIM]

            # private output layer returns a batch MVN over groups
            dist_b = self.output(s)  # batch_shape=[G], event_shape=[N]

            # convert to multitask MVN so rsample() is consistently [N, G]
            try:
                dist_mt = (
                    gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                        dist_b
                    )
                )
                return dist_mt
            except Exception:
                # fallback: callers must handle [G, N] samples
                return dist_b

    # =================== Discrete-time utilities (unchanged) ===================

    def _select_task(
        f_all: torch.Tensor, g_idx: torch.Tensor, num_tasks: int | None = None
    ) -> torch.Tensor:
        """
        Select per-sample task output given integer task indices.

        Supports shapes:
        - [N, G]
        - [G, N]
        - [S, N, G]
        - [S, G, N]
        - [N], [N,1] (scalar output): returns squeezed
        """
        if not torch.is_tensor(f_all):
            f_all = torch.as_tensor(f_all)
        if not torch.is_tensor(g_idx):
            g_idx = torch.as_tensor(g_idx)
        g_idx = g_idx.to(device=f_all.device, dtype=torch.long).view(-1)  # [N]

        # Scalar-output model → nothing to select
        if f_all.dim() == 1:
            return f_all
        if f_all.dim() == 2 and (f_all.size(1) == 1 or f_all.size(0) == 1):
            return f_all.squeeze(-1).squeeze(0)

        if f_all.dim() == 2:
            N0, N1 = f_all.shape
            if N0 == g_idx.numel():  # [N, G]
                return f_all.gather(1, g_idx.view(-1, 1)).squeeze(1)
            if N1 == g_idx.numel():  # [G, N]
                return f_all.gather(0, g_idx.view(1, -1)).squeeze(0)
            return f_all.squeeze(-1)

        if f_all.dim() == 3:
            S0, S1, S2 = f_all.shape
            N = g_idx.numel()

            if S1 == N:  # [S, N, G]
                return f_all.gather(2, g_idx.view(1, N, 1).expand(S0, N, 1)).squeeze(2)
            if S2 == N:  # [S, G, N]
                return f_all.gather(1, g_idx.view(1, 1, N).expand(S0, 1, N)).squeeze(1)

            # squeeze singleton task dim if present
            if S2 == 1:
                return f_all.squeeze(2)
            if S1 == 1:
                return f_all.squeeze(1)

            raise RuntimeError(
                f"Unexpected f_all shape {tuple(f_all.shape)} for g_idx shape {tuple(g_idx.shape)}"
            )

        raise RuntimeError(f"Unexpected f_all dim {f_all.dim()}")

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

    def expected_dt_ll_mc(
        model, X, g_idx, at_risk, event_mask, alpha, w_by_group=None, S=MC_TRAIN
    ):

        def logs_logistic(eta):
            logh = -F.softplus(-eta)  # log(sigmoid(eta))
            log1m = -F.softplus(eta)  # log(1 - sigmoid(eta))
            return logh, log1m

        def logs_cloglog(eta):
            eta_cap = torch.clamp(eta, max=20.0)
            z = torch.exp(eta_cap)
            log1m = -z
            small = z < 1e-4
            logh = torch.empty_like(z)
            logh[small] = torch.log(z[small])
            logh[~small] = torch.log1p(-torch.exp(-z[~small]))
            return logh, log1m

        g_idx = g_idx.to(device=X.device, dtype=torch.long)

        # Optional group weights to counter cohort imbalance (full-batch DT hazards)
        w_i = None
        if w_by_group is not None:
            w_i = w_by_group[g_idx].to(device=X.device, dtype=X.dtype)  # [N]

        use_cloglog = DT_LINK.lower() == "cloglog"
        vals = []
        with gpytorch.settings.cholesky_jitter(1e-3):
            for _ in range(S):
                f_all = model(X, g_idx).rsample()  # multitask sample
                f = _select_task(f_all, g_idx)  # [N]
                eta = f[:, None] + alpha[None, :]  # [N, J]

                logh, log1m = logs_cloglog(eta) if use_cloglog else logs_logistic(eta)

                term1 = torch.where(at_risk > 0, log1m, torch.zeros_like(log1m))
                term2 = torch.where(
                    event_mask > 0, (logh - log1m), torch.zeros_like(logh)
                )

                if w_i is not None:
                    term1 = term1 * w_i[:, None]
                    term2 = term2 * w_i[:, None]

                ll = term1.sum() + term2.sum()

                # Keep loss scale comparable to unweighted full-batch sum
                if w_i is not None:
                    ll = ll * (w_i.numel() / torch.clamp(w_i.sum(), min=1e-12))

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
                    if w_i is not None:
                        term1 = term1 * w_i[:, None]
                        term2 = term2 * w_i[:, None]
                    ll = term1.sum() + term2.sum()

                    if w_i is not None:
                        ll = ll * (w_i.numel() / torch.clamp(w_i.sum(), min=1e-12))

                vals.append(ll)

        return torch.stack(vals).mean()

    def _rsample_f_all_NG(model, X, g_idx, num_groups, jitter=1e-3):
        """
        Sample multitask latent f for all N points and G groups.
        Returns f_all shaped [N, G] (or [N,1] for scalar-output models).
        """
        with gpytorch.settings.cholesky_jitter(jitter):
            # Prefer multitask call signature (model(X, g_idx)), but fall back if needed.
            try:
                dist = model(X, g_idx)
            except TypeError:
                dist = model(X)
            f_all = dist.rsample()

        # Normalize shapes
        if f_all.dim() == 1:
            f_all = f_all.unsqueeze(1)  # [N] -> [N,1]

        # If returned as [G, N], transpose to [N, G]
        if (
            f_all.dim() == 2
            and f_all.shape[0] == num_groups
            and f_all.shape[1] == X.shape[0]
        ):
            f_all = f_all.transpose(0, 1).contiguous()

        return f_all

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

    def expected_cox_pll_mc(model, X, g_idx, time, event, S=MC_TRAIN):
        vals = []
        with gpytorch.settings.cholesky_jitter(1e-3):
            for _ in range(S):
                f = model(X, g_idx).rsample().squeeze()
                vals.append(cox_partial_ll_breslow(f, time, event))
        return torch.stack(vals, 0).mean()

    def total_kl_details(model):
        # Shared layer = SHARED_DIM separate multitask GPs
        kl_s = torch.tensor(0.0, device=next(model.parameters()).device)
        for gp_h in model.shared.gps:
            vs = getattr(gp_h, "variational_strategy", None)
            if vs is None:
                continue
            kl_s = kl_s + vs.kl_divergence().sum()

        # Output layer = batched private heads (batch_shape=[G])
        kl_o = model.output.variational_strategy.kl_divergence().sum()
        return kl_s, kl_o

    def beta_schedule(it):
        return min(1.0, float(it) / max(1, KL_WARMUP))

    # =================== Instantiate model (needs group index tensors) ===================

    # ---- Training ----
    n_events_tr = int(e_tr.sum())
    n_events_va = int(e_va.sum()) if HAS_VAL else 0

    # FIX 1: clamp rank
    icm_rank_eff = int(min(ICM_RANK, NUM_GROUPS))
    model = DGP(
        in_dim=int(Xtr_t.size(1)),
        Xref=Xtr_t,
        seed=seed,
        num_groups=NUM_GROUPS,
        icm_rank=icm_rank_eff,
        blocks=BLOCKS,
    ).to(device)

    # =================== Discrete-time binning & baseline alpha_dt ===================
    alpha_dt = None  # define regardless
    w_by_group = None
    if USE_GROUP_WEIGHTED_LL:
        with torch.no_grad():
            counts = torch.bincount(
                g_tr_t.detach().to("cpu"), minlength=NUM_GROUPS
            ).float()

            present = counts > 0
            # Optional: guard against tiny groups before power
            # min_count = float(GROUP_WEIGHT_MIN_COUNT) 
            # eff_counts = torch.where(present, torch.clamp(counts, min=min_count), counts)
            eff_counts = torch.where(present, counts, torch.ones_like(counts))

            w = eff_counts.pow(-float(GROUP_WEIGHT_POWER))  # 1 / (n_g^alpha)
            w = torch.where(
                present, w, torch.zeros_like(w)
            )  # absent groups get 0 weight (unused anyway)

            if GROUP_WEIGHT_CAP is not None:
                w = torch.clamp(w, max=float(GROUP_WEIGHT_CAP))

            if GROUP_WEIGHT_NORMALIZE:
                mean_w = torch.clamp(w[present].mean(), min=1e-12)
                w = w / mean_w

        w_by_group = w.to(device)

        try:
            print(
                "[Imbalance] Group counts (train):", counts.numpy().astype(int).tolist()
            )
            print("[Imbalance] Group weights (train):", w.numpy().tolist())
        except Exception:
            pass
    else:
        w_by_group = None
    bin_edges = None

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

    # =================== Optimizer groups (collect multitask inducing points) ===================
    var_params, ind_params = [], []

    # (imports moved to top of file) aliases kept for readability
    VS = VariationalStrategy
    MFVD = MeanFieldVariationalDistribution
    CVD = CholeskyVariationalDistribution
    MTVS = MultitaskVariationalStrategy

    for mod in model.modules():
        if isinstance(mod, (CVD, MFVD)):
            var_params += list(mod.parameters())

        vs = getattr(mod, "variational_strategy", None)
        if vs is None:
            continue

        # Case 1: plain VariationalStrategy
        if isinstance(vs, VS) and hasattr(vs, "inducing_points"):
            ind_params.append(vs.inducing_points)

        # Case 2: MultitaskVariationalStrategy wrapping a base variational strategy
        if isinstance(vs, MTVS):
            bvs = getattr(vs, "base_variational_strategy", None)
            if bvs is not None and hasattr(bvs, "inducing_points"):
                ind_params.append(bvs.inducing_points)

    # FIX 2: dedupe params (prevents double-updates / optimizer group conflicts)
    def _unique_params(params):
        seen = set()
        out = []
        for p in params:
            if id(p) not in seen:
                out.append(p)
                seen.add(id(p))
        return out

    var_params = _unique_params(var_params)
    ind_params = _unique_params(ind_params)

    var_ids = {id(p) for p in var_params}
    ind_ids = {id(p) for p in ind_params}
    hyp_params = [p for p in model.parameters() if id(p) not in (var_ids | ind_ids)]

    optim_groups = [
        {"params": var_params, "lr": LR_VAR},
        {"params": ind_params, "lr": LR_IND},
        {"params": hyp_params, "lr": LR_HYP},
    ]
    if USE_DISCRETE_TIME and (alpha_dt is not None):
        optim_groups.append({"params": [alpha_dt], "lr": 0.01})

    opt = torch.optim.Adam(optim_groups)

    print(
        f"[Optim] #var={len(var_params)}  #ind={len(ind_params)}  #hyp={len(hyp_params)}  "
        f"{' + alpha_dt' if (USE_DISCRETE_TIME and alpha_dt is not None) else ''}"
    )

    # =================== Prediction helpers (accept g_idx) ===================

    @torch.no_grad()
    def predict_risk_mc(X, g_idx, S=64, mode="linear"):
        """
        Returns:
            risk_mean: [N] numpy
            risk_samples: [S, N] numpy
        """
        model.eval()
        g_idx = g_idx.to(device=X.device, dtype=torch.long).view(-1)

        risk_samps = []
        for _ in range(int(S)):
            f_all = _rsample_f_all_NG(model, X, g_idx, num_groups=NUM_GROUPS)  # [N, G]
            f = _select_task(f_all, g_idx, num_tasks=NUM_GROUPS)  # [N]

            if mode == "logexp":
                r = torch.log1p(torch.exp(f))
            else:
                r = f

            risk_samps.append(r.detach().cpu())

        R = torch.stack(risk_samps, 0)  # [S, N]
        return R.mean(0).numpy(), R.numpy()

    @torch.no_grad()
    def posterior_survival_at_H(X, g_idx, H_months: float, alpha, S=EVAL_MC_SAMPLES):
        if not USE_DISCRETE_TIME:
            return None
        if bin_edges is None:
            raise RuntimeError(
                "bin_edges is not defined (USE_DISCRETE_TIME=True required)."
            )

        model.eval()
        g_idx = g_idx.to(device=X.device, dtype=torch.long).view(-1)
        alpha = alpha.to(device=X.device, dtype=X.dtype).view(-1)  # [J]
        J = int(alpha.numel())
        if J < 1:
            raise RuntimeError("alpha has no bins (alpha.numel() < 1).")

        Fs = []
        for _ in range(int(S)):
            f_all = _rsample_f_all_NG(model, X, g_idx, num_groups=NUM_GROUPS)  # [N, G]
            f = _select_task(f_all, g_idx, num_tasks=NUM_GROUPS)  # [N]
            Fs.append(f)

        F_samp = torch.stack(Fs, 0)  # [S, N]

        eta = F_samp[:, :, None] + alpha[None, None, :]  # [S, N, J]

        if DT_LINK.lower() == "cloglog":
            h = 1.0 - torch.exp(-torch.exp(torch.clamp(eta, max=20.0)))
        else:
            h = torch.sigmoid(eta)

        surv_edges = torch.cumprod(1.0 - h, dim=2)  # [S, N, J]

        j = np.searchsorted(bin_edges, float(H_months), side="right") - 1
        j = int(np.clip(j, 0, len(bin_edges) - 2))
        j = int(min(max(j, 0), J - 1))  # ALSO clip to alpha bins

        S_H = surv_edges[:, :, j]  # [S, N]
        return S_H.cpu().numpy()

    @torch.no_grad()
    def neglogS_at_H(X, g_idx, H_months: float, alpha, S=EVAL_MC_SAMPLES):
        if not USE_DISCRETE_TIME:
            return None
        SH = posterior_survival_at_H(X, g_idx, H_months, alpha, S=S)  # [S, N]
        nlS = -np.log(np.clip(SH, 1e-10, 1.0))
        return nlS.mean(axis=0)

    def brier_ipcw(times, events, preds_S, H):
        times = np.asarray(times, float)
        events = np.asarray(events, int)
        preds_S = np.asarray(preds_S, float)

        kmc = KaplanMeierFitter().fit(times, 1 - events)  # censoring KM, G(t)
        t_minus = np.maximum(times - 1e-8, 0.0)
        G_tminus = kmc.predict(np.minimum(t_minus, float(times.max())))
        G_tminus = np.asarray(G_tminus, float)
        G_tminus = np.clip(G_tminus, 1e-6, None)

        G_H = kmc.predict(min(H, float(times.max())))
        G_H = float(np.clip(G_H, 1e-6, None))

        yH = (times > H).astype(float)
        w = np.where(times <= H, events / G_tminus, 1.0 / G_H)
        return float(np.mean(w * (yH - preds_S) ** 2))

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

    def out_kernel_stats_icm(model):
        # model.output.covar_module is a ScaleKernel with batch_shape=[rank]
        k = model.output.covar_module
        try:
            outscale = float(k.outputscale.detach().mean().cpu())
        except Exception:
            outscale = float("nan")

        # Find the RBF part inside (RBF + Linear)
        base = getattr(k, "base_kernel", k)
        rbf = None
        if hasattr(base, "kernels"):  # AdditiveKernel
            for kk in base.kernels:
                if isinstance(kk, gpytorch.kernels.RBFKernel):
                    rbf = kk
                    break
        elif isinstance(base, gpytorch.kernels.RBFKernel):
            rbf = base

        if rbf is None:
            return {"ls_med": float("nan"), "outscale": outscale}

        ls = rbf.lengthscale.detach().cpu().reshape(-1)
        ls_med = float(ls.median()) if ls.numel() else float("nan")
        return {"ls_med": ls_med, "outscale": outscale}

    # =================== Training loop (DT hazards ELBO: ELL - beta*KL) ===================
    print("Starting training…")
    best_c, best_state = -1.0, None

    for it in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()

        if USE_DISCRETE_TIME:
            pll_or_ell = expected_dt_ll_mc(
                model, Xtr_t, g_tr_t, AR_tr, EV_tr, alpha_dt, S=MC_TRAIN
            )

        else:
            pll_or_ell = expected_cox_pll_mc(
                model, Xtr_t, g_tr_t, t_tr_t, e_tr_t, S=MC_TRAIN
            )

        kl_s, kl_o = total_kl_details(model)
        kl = kl_s + kl_o

        kl_beta = beta_schedule(it)
        elbo = pll_or_ell - kl_beta * kl
        loss = -elbo
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        if it % PRINT_EVERY == 0 or it == 1:
            n_ev = max(1, n_events_tr)
            like_name = "ELL" if USE_DISCRETE_TIME else "PLL"
            msg = (
                f"[{it:04d}] {like_name}/evt={float(pll_or_ell.item())/n_ev:.4f} | "
                f"KL/evt={(kl.item()/n_ev):.4f} | KLsh={(kl_s.item()/n_ev):.4f} | "
                f"KLout={(kl_o.item()/n_ev):.4f} | beta={kl_beta:.3f}"
            )
            print(msg)

            if HAS_VAL:
                model.eval()
                with torch.no_grad():
                    if USE_DISCRETE_TIME:
                        AR_va, EV_va = make_dt_masks(t_va, e_va, bin_edges)
                        ell_va = float(
                            expected_dt_ll_mc(
                                model, Xva_t, g_va_t, AR_va, EV_va, alpha_dt, S=MC_TRAIN
                            ).item()
                        )
                        r_va_tmp, _ = predict_risk_mc(Xva_t, g_va_t, S=16)
                        c_va_tmp, _ = harrell_c(t_va, r_va_tmp, e_va)
                        msg += f" | valELL/evt={ell_va/max(1,n_events_va):.4f} | valC(MC,E[f])={c_va_tmp:.3f}"
                    else:
                        pll_va = float(
                            expected_cox_pll_mc(
                                model, Xva_t, g_va_t, t_va_t, e_va_t, S=MC_TRAIN
                            ).item()
                        )
                        r_va_tmp, _ = predict_risk_mc(
                            Xva_t, g_va_t, S=16, mode="logexp"
                        )
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

    # =================== Metrics & Prints ===================
    r_tr_plot, r_tr_sd = predict_risk_mc(Xtr_t, g_tr_t)
    if HAS_VAL:
        r_va_plot, r_va_sd = predict_risk_mc(Xva_t, g_va_t)
    r_te_plot, r_te_sd = predict_risk_mc(Xte_t, g_te_t)

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

        r_tr_c = neglogS_at_H(Xtr_t, g_tr_t, H_C, alpha_dt, S=EVAL_MC_SAMPLES)
        if HAS_VAL:
            r_va_c = neglogS_at_H(Xva_t, g_va_t, H_C, alpha_dt, S=EVAL_MC_SAMPLES)
        r_te_c = neglogS_at_H(Xte_t, g_te_t, H_C, alpha_dt, S=EVAL_MC_SAMPLES)
    else:
        r_tr_c, _ = predict_risk_mc(Xtr_t, g_tr_t, S=EVAL_MC_SAMPLES, mode="logexp")
        if HAS_VAL:
            r_va_c, _ = predict_risk_mc(Xva_t, g_va_t, S=EVAL_MC_SAMPLES, mode="logexp")
        r_te_c, _ = predict_risk_mc(Xte_t, g_te_t, S=EVAL_MC_SAMPLES, mode="logexp")
        H_C = float("nan")

    # orient pooled risk so higher = worse overall
    r_tr_c = orient_risk(t_tr, e_tr, r_tr_c)
    r_te_c = orient_risk(t_te, e_te, r_te_c)
    if HAS_VAL:
        r_va_c = orient_risk(t_va, e_va, r_va_c)

    c_tr = _lifelines_c(t_tr, r_tr_c, e_tr)
    c_te = _lifelines_c(t_te, r_te_c, e_te)
    if HAS_VAL:
        c_va = _lifelines_c(t_va, r_va_c, e_va)

    rows = [
        {"split": "train", "c_index": float(c_tr), "dxy": float(dxy(c_tr))},
        {"split": "test", "c_index": float(c_te), "dxy": float(dxy(c_te))},
    ]
    if HAS_VAL:
        rows.insert(
            1, {"split": "val", "c_index": float(c_va), "dxy": float(dxy(c_va))}
        )
    pd.DataFrame(rows).to_csv(OUTDIR / f"c_index{seed_suffix}.csv", index=False)

    if HAS_VAL:
        print(
            f"C-index (pooled, E[-log S(H_C)], H_C={H_C:.1f} mo) — train={c_tr:.3f} | val={c_va:.3f} | test={c_te:.3f}"
        )
    else:
        print(
            f"C-index (pooled, E[-log S(H_C)], H_C={H_C:.1f} mo) — train={c_tr:.3f} | test={c_te:.3f}"
        )

    # =================== NEW: C-index per subtype×treatment group (GROUP_LABEL) ===================
    def _safe_fname(s: str) -> str:
        s = str(s)
        s = re.sub(r"\s+", "_", s.strip())
        s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
        return s[:180] if len(s) > 180 else s

    def cindex_all_and_by_group(
        t, e, risk, g_idx, group_names, split_name, min_n=30, min_events=5
    ):
        """
        Writes one CSV per split with:
        - pooled C-index (all patients in split)
        - per-group C-index (within each subtype×treatment group)
        Note: pooled risk orientation is done before calling this (higher=worse).
        """
        t = np.asarray(t, float)
        e = np.asarray(e, int)
        r = np.asarray(risk, float)
        g = np.asarray(g_idx, int)
        ok = np.isfinite(t) & np.isfinite(r) & (g >= 0)
        t0, e0, r0, g0 = t[ok], e[ok], r[ok], g[ok]

        out = []
        c_all = _lifelines_c(t0, r0, e0)
        out.append(
            {
                "split": split_name,
                "scope": "all",
                "group_idx": -1,
                "group": "ALL",
                "n": int(len(t0)),
                "events": int(np.sum(e0 == 1)),
                "c_index": float(c_all),
                "dxy": float(dxy(c_all)),
                "skipped": False,
            }
        )

        for gi, gname in enumerate(group_names):
            m = g0 == gi
            n = int(m.sum())
            d = int(np.sum(e0[m] == 1))
            if (n < min_n) or (d < min_events):
                out.append(
                    {
                        "split": split_name,
                        "scope": "by_group",
                        "group_idx": gi,
                        "group": gname,
                        "n": n,
                        "events": d,
                        "c_index": np.nan,
                        "dxy": np.nan,
                        "skipped": True,
                    }
                )
                continue

            # Recommended: orient WITHIN group too (prevents weird <0.5 within group)
            r_g = orient_risk(t0[m], e0[m], r0[m])
            c_g = _lifelines_c(t0[m], r_g, e0[m])
            out.append(
                {
                    "split": split_name,
                    "scope": "by_group",
                    "group_idx": gi,
                    "group": gname,
                    "n": n,
                    "events": d,
                    "c_index": float(c_g),
                    "dxy": float(dxy(c_g)),
                    "skipped": False,
                }
            )

        df = pd.DataFrame(out)
        df.to_csv(
            OUTDIR / f"c_index_{split_name}_all_and_by_group{seed_suffix}.csv",
            index=False,
        )
        return df

    g_tr_np = g_tr_t.detach().cpu().numpy()
    g_te_np = g_te_t.detach().cpu().numpy()

    # =================== Train-only alignment diagnostic (Cox cosine similarity) ===================
    try:
        archdir = OUTDIR / "archdiag"
        archdir.mkdir(parents=True, exist_ok=True)

        # Train split counts
        counts_rows = []
        e_tr_arr = np.asarray(e_tr, int)
        for gi, gname in enumerate(CANON_GROUPS):
            m = g_tr_np == int(gi)
            counts_rows.append(
                {
                    "group_idx": int(gi),
                    "group": str(gname),
                    "n_train": int(m.sum()),
                    "events_train": int(e_tr_arr[m].sum()),
                }
            )
        pd.DataFrame(counts_rows).to_csv(
            archdir / f"archdiag_group_counts_D_seed{seed}.csv", index=False
        )

        # Fit pooled Cox on TRAIN (same feature matrix used by the model)
        Xtr_df = pd.DataFrame(
            np.asarray(Xtr_s, float), columns=[str(n) for n in feat_names]
        )
        Xtr_df["time"] = np.asarray(t_tr, float)
        Xtr_df["event"] = np.asarray(e_tr, int)

        cph_pool = CoxPHFitter(penalizer=float(DIAG_COX_PENALIZER))
        cph_pool.fit(
            Xtr_df, duration_col="time", event_col="event", show_progress=False
        )
        beta_pool = cph_pool.params_.to_numpy(dtype=float)
        denom_pool = float(np.linalg.norm(beta_pool) + 1e-12)

        sim_rows = []
        for gi, gname in enumerate(CANON_GROUPS):
            m = g_tr_np == int(gi)
            n_g = int(m.sum())
            ev_g = int(np.asarray(e_tr, int)[m].sum())
            row = {
                "group_idx": int(gi),
                "group": str(gname),
                "n_train": n_g,
                "events_train": ev_g,
            }
            if (n_g < 30) or (ev_g < 5):
                row.update({"cos_sim_beta_vs_pooled": np.nan, "skipped": True})
                sim_rows.append(row)
                continue
            try:
                df_g = Xtr_df.loc[m, :].copy()
                cph_g = CoxPHFitter(penalizer=float(DIAG_COX_PENALIZER))
                cph_g.fit(
                    df_g, duration_col="time", event_col="event", show_progress=False
                )
                beta_g = cph_g.params_.to_numpy(dtype=float)
                cos = float(
                    np.dot(beta_g, beta_pool)
                    / ((np.linalg.norm(beta_g) + 1e-12) * denom_pool)
                )
                row.update({"cos_sim_beta_vs_pooled": cos, "skipped": False})
            except Exception:
                row.update({"cos_sim_beta_vs_pooled": np.nan, "skipped": True})
            sim_rows.append(row)

        pd.DataFrame(sim_rows).to_csv(
            archdir / f"archdiag_train_cox_similarity_D_seed{seed}.csv", index=False
        )

    except Exception as _e_diag:
        print(f"[archdiag] Skipped Cox alignment diagnostic due to: {_e_diag}")
    if HAS_VAL:
        g_va_np = g_va_t.detach().cpu().numpy()

    _ = cindex_all_and_by_group(t_tr, e_tr, r_tr_c, g_tr_np, CANON_GROUPS, "train")
    _ = cindex_all_and_by_group(t_te, e_te, r_te_c, g_te_np, CANON_GROUPS, "test")
    if HAS_VAL:
        _ = cindex_all_and_by_group(t_va, e_va, r_va_c, g_va_np, CANON_GROUPS, "val")

    # =================== Bootstrap (pooled test) ===================
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
    ).to_csv(OUTDIR / f"c_dxy_bootstrap_test{seed_suffix}.csv")

    if USE_DISCRETE_TIME:
        for H in H_CAL_LIST:
            r_te_H = neglogS_at_H(Xte_t, g_te_t, H, alpha_dt, S=EVAL_MC_SAMPLES)
            c_te_H, _ = harrell_c(t_te, r_te_H, e_te)
            print(f"C-index (E[-log S({int(H)})]) test = {c_te_H:.3f}")

    # ------------------- plots -------------------
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

    # ------------------- HR per 1 SD risk (test) -------------------
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

        curve_colors = []
        for i in present:
            kmf = KaplanMeierFitter()
            kmf.fit(t_in[masks[i]], e_in[masks[i]], label=labels[i], timeline=timeline)
            h_ax = kmf.plot(ax=ax, ci_show=False, show_censors=False)
            line = (
                h_ax.get_lines()[-1]
                if hasattr(h_ax, "get_lines")
                else ax.get_lines()[-1]
            )
            col = line.get_color() if line is not None else None
            curve_colors.append(col)
            kmfs.append(kmf)

        draw_guides(ax, horiz_at=0.5, vlines=(), xmax=300)

        nr_colors = []
        for kmf, col in zip(kmfs, curve_colors):
            try:
                med = kmf.median_survival_time_
            except Exception:
                med = np.nan

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
                ax.text(
                    float(med) + 2.0,
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

        if nr_colors:
            xr = ax.get_xlim()[1]
            for col in nr_colors:
                ax.text(xr - 10, 0.52, "NR", fontsize=9, color=col, alpha=0.9, zorder=4)

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

        savefig(OUTDIR / outname)
        return pval



    # =================== KM within EACH group: Posterior-H tau grouping ONLY ===================
    def km_within_each_group_postH_tau(
        t,
        e,
        S_H,  # <-- MUST be [S, N_total] aligned to t/e/g_idx ordering
        g_idx,
        group_names,
        out_prefix="KM_WITHIN_group_postH_tau",
        admin_cap=True,
        min_n=30,
        min_events=5,
    ):
        t = np.asarray(t, float)
        e = np.asarray(e, int)
        g = np.asarray(g_idx, int)
        SH = np.asarray(S_H, float)

        if SH.ndim != 2 or SH.shape[1] != t.shape[0]:
            raise ValueError("S_H must be shape [S, N] and align with t/e (same N).")

        rows = []
        for gi, gname in enumerate(group_names):
            m = (g == gi) & np.isfinite(t)
            n = int(m.sum())
            d = int(np.sum(e[m] == 1))

            if (n < min_n) or (d < min_events):
                rows.append(
                    {
                        "group_idx": gi,
                        "group": gname,
                        "n": n,
                        "events": d,
                        "skipped": True,
                    }
                )
                continue

            SH_g = SH[:, m]  # [S, n_group]
            p_hi = np.mean(SH_g <= TAU_HIGH, axis=0)  # P{S(H) <= tau_high}
            p_lo = np.mean(SH_g >= TAU_LOW, axis=0)  # P{S(H) >= tau_low}

            gH = p_hi >= P_STAR
            gL = p_lo >= P_STAR
            gM = (~gH) & (~gL)

            labels = ["Low risk", "Medium risk", "High risk"]
            title = (
                f"KM within group: {gname} | Posterior-H tau grouping @H={int(POSTERIOR_H_MONTHS)} mo "
                f"($\\tau_H$={TAU_HIGH}, $\\tau_L$={TAU_LOW}, $p^*$={P_STAR})"
            )
            outname = f"{out_prefix}__{gi:02d}__{_safe_fname(gname)}__cap.png"

            p = km_plot_groups(
                t[m],
                e[m],
                [gL, gM, gH],
                labels,
                title,
                outname,
                admin_cap=admin_cap,
            )

            rows.append(
                {
                    "group_idx": gi,
                    "group": gname,
                    "n": n,
                    "events": d,
                    "n_low": int(gL.sum()),
                    "n_med": int(gM.sum()),
                    "n_high": int(gH.sum()),
                    "logrank_p_cap": float(p) if np.isfinite(p) else np.nan,
                    "skipped": False,
                    "png": outname,
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv(OUTDIR / f"{out_prefix}__summary.csv", index=False)
        return df

    # ---------------- Posterior survival grouping at H  ----------------------
    if USE_DISCRETE_TIME:
        print(
            f"[Posterior-H grouping] H={POSTERIOR_H_MONTHS} mo; thresholds: "
            f"tau_high={TAU_HIGH}, tau_low={TAU_LOW}, p*={P_STAR}"
        )

        # ---- Build X_all_t, g_all_t in the same order for posterior_survival_at_H ----
        if HAS_VAL:
            X_all_t = torch.cat([Xtr_t, Xva_t, Xte_t], dim=0)
            g_all_t = torch.cat([g_tr_t, g_va_t, g_te_t], dim=0)
            n_tr, n_va, n_te = Xtr_t.size(0), Xva_t.size(0), Xte_t.size(0)
        else:
            X_all_t = torch.cat([Xtr_t, Xte_t], dim=0)
            g_all_t = torch.cat([g_tr_t, g_te_t], dim=0)
            n_tr, n_va, n_te = Xtr_t.size(0), 0, Xte_t.size(0)

        S_H_all = posterior_survival_at_H(
            X_all_t, g_all_t, float(POSTERIOR_H_MONTHS), alpha_dt, S=EVAL_MC_SAMPLES
        )  # [S, N_all]

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

        # =================== (A) PER-GROUP KM on TEST, Posterior-H tau grouping ===================
        _ = km_within_each_group_postH_tau(
            t_te,
            e_te,
            S_H_te,  # [S, N_test]
            g_te_np,
            CANON_GROUPS,
            out_prefix="KM_test_WITHIN_group_postH_tau",
            admin_cap=True,
        )

        # =================== (B) PER-GROUP KM on FULL COHORT, Posterior-H tau grouping ===================
        # IMPORTANT: t/e/g must be in the SAME order as X_all_t / S_H_all columns.
        if HAS_VAL:
            t_all_cat = np.concatenate([t_tr, t_va, t_te])
            e_all_cat = np.concatenate([e_tr, e_va, e_te])
            g_all_cat = np.concatenate([g_tr_np, g_va_np, g_te_np])
        else:
            t_all_cat = np.concatenate([t_tr, t_te])
            e_all_cat = np.concatenate([e_tr, e_te])
            g_all_cat = np.concatenate([g_tr_np, g_te_np])

        assert (
            S_H_all.shape[1]
            == t_all_cat.shape[0]
            == e_all_cat.shape[0]
            == g_all_cat.shape[0]
        )

        _ = km_within_each_group_postH_tau(
            t_all_cat,
            e_all_cat,
            S_H_all,  # [S, N_all]
            g_all_cat,
            CANON_GROUPS,
            out_prefix="KM_all_WITHIN_group_postH_tau",
            admin_cap=True,
        )

        # =================== (C) YOUR EXISTING GLOBAL pooled plots (already working) ===================
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

        # Build pooled all-cohort boolean masks in meta order
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
                {
                    "logrank_p_cap": (
                        float(p_all_cap) if np.isfinite(p_all_cap) else np.nan
                    )
                },
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
    # ---------------- SAVE PATIENT-LEVEL RISK GROUP ASSIGNMENTS ----------------
    def _write_groups_csv(meta_df, idx, gL, gM, gH, S_H_split, split_name):
        """
        Saves Posterior-H risk group assignments (low/med/high) for a split.
        Also writes GROUP_LABEL / GROUP_IDX if present in meta_df.
        """
        if idx is None or len(idx) == 0:
            return

        pid = meta_df.iloc[idx]["PATIENT_ID"].astype(str).values
        sid = meta_df.iloc[idx]["SAMPLE_ID"].astype(str).values

        group = np.where(gH, "high", np.where(gL, "low", "med"))

        Smean = S_H_split.mean(axis=0)  # E[S(H)]
        p_hi = (S_H_split <= TAU_HIGH).mean(axis=0)  # P{S(H) <= tau_high}
        p_lo = (S_H_split >= TAU_LOW).mean(axis=0)  # P{S(H) >= tau_low}

        out = pd.DataFrame(
            {
                "PATIENT_ID": pid,
                "SAMPLE_ID": sid,
                "split": split_name,
                "group": group,
                "E_S_at_H": Smean,
                "P_S_le_tauH": p_hi,
                "P_S_ge_tauL": p_lo,
                "H_months": int(POSTERIOR_H_MONTHS),
                "tau_high": TAU_HIGH,
                "tau_low": TAU_LOW,
                "p_star": P_STAR,
            }
        )

        # Optional: subtype×treatment group info if available
        if "GROUP_LABEL" in meta_df.columns:
            out["GROUP_LABEL"] = meta_df.iloc[idx]["GROUP_LABEL"].astype(str).values
        if "GROUP_IDX" in meta_df.columns:
            out["GROUP_IDX"] = meta_df.iloc[idx]["GROUP_IDX"].values

        out.to_csv(OUTDIR / f"posteriorH_groups_{split_name}.csv", index=False)
        print(
            f"[Posterior-H] Saved patient groups → posteriorH_groups_{split_name}.csv"
        )

    # Call per-split writers (unchanged logic; just uses the corrected function above)
    if USE_DISCRETE_TIME:
        _write_groups_csv(meta, tr_idx, gL_tr, gM_tr, gH_tr, S_H_tr, "train")
        if HAS_VAL:
            _write_groups_csv(meta, va_idx, gL_va, gM_va, gH_va, S_H_va, "val")
        _write_groups_csv(meta, te_idx, gL_te, gM_te, gH_te, S_H_te, "test")

        # -------- Full cohort file (merged in meta order) --------
        N_all = len(meta)
        group_all = np.full(N_all, "med", dtype=object)
        group_all[gL_all] = "low"
        group_all[gH_all] = "high"

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

        # FIX: split array that does NOT reference va_idx when HAS_VAL=False
        all_ids = np.arange(N_all)
        split_arr = np.full(N_all, "unknown", dtype=object)
        split_arr[np.isin(all_ids, tr_idx)] = "train"
        split_arr[np.isin(all_ids, te_idx)] = "test"
        if HAS_VAL:
            split_arr[np.isin(all_ids, va_idx)] = "val"

        df_all = pd.DataFrame(
            {
                "PATIENT_ID": meta["PATIENT_ID"].astype(str).values,
                "SAMPLE_ID": meta["SAMPLE_ID"].astype(str).values,
                "split": split_arr,
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

        # Optional: subtype×treatment group info if available
        if "GROUP_LABEL" in meta.columns:
            df_all["GROUP_LABEL"] = meta["GROUP_LABEL"].astype(str).values
        if "GROUP_IDX" in meta.columns:
            df_all["GROUP_IDX"] = meta["GROUP_IDX"].values

        df_all.to_csv(OUTDIR / "posteriorH_groups_full.csv", index=False)
        print("[Posterior-H] Saved patient groups → posteriorH_groups_full.csv")

    # ---------------- Calibration & Brier (test) ----------------
    def predict_SH_for_split(X_t, g_t, H, alpha):
        SH = posterior_survival_at_H(
            X_t, g_t, float(H), alpha, S=EVAL_MC_SAMPLES
        )  # [S,N]
        return SH.mean(axis=0)  # [N]

    if USE_DISCRETE_TIME:
        for H in H_CAL_LIST:
            S_te_H = predict_SH_for_split(Xte_t, g_te_t, H, alpha_dt)

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

        def _S_at_H_test(H_):
            return predict_SH_for_split(Xte_t, g_te_t, float(H_), alpha_dt)

        ibs_test = ibs_ipcw(t_te, e_te, _S_at_H_test, tau_for_ibs)
        print(f"IBS (IPCW) up to τ={int(tau_for_ibs)} mo — test = {ibs_test:.4f}")
        pd.Series({"IBS": ibs_test, "tau": tau_for_ibs}).to_csv(
            OUTDIR / f"ibs_test_tau{int(tau_for_ibs)}.csv"
        )
    # ---------------- Biology sanity checks (test split) ----------------
    rows = []
    if spearmanr is not None and USE_GES:
        # NOTE: neglogS_at_H requires discrete-time + g_idx
        nlS120 = None
        if USE_DISCRETE_TIME:
            nlS120 = neglogS_at_H(Xte_t, g_te_t, 120.0, alpha_dt, S=EVAL_MC_SAMPLES)

        def add_corr(name, a, b):
            rho, p = spearmanr(a, b, nan_policy="omit")
            rows.append({"pair": name, "rho": float(rho), "p": float(p)})

        add_corr("risk(E[f])_vs_prolif", r_te_plot, ges_Pro_te[:, 0])
        add_corr("risk(E[f])_vs_ER", r_te_plot, ges_ER_te[:, 0])

        if nlS120 is not None:
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
                f"[BIO] Grade3 enrichment OR={OR if OR==OR else 'nan'}, p={p if p==p else 'nan'} "
                f"(saved grade3_enrichment_test.csv)"
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
        g_ref,  # <-- NEW
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
        g_ref = np.asarray(g_ref, dtype=int)

        X_t = torch.tensor(X_ref, dtype=torch.float32, device=device)
        g_t = torch.tensor(g_ref, dtype=torch.long, device=device)

        # Baseline (higher-is-better metric: C-index)
        r_base, _ = predict_risk_mc(X_t, g_t, S=S)
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

                r_perm, _ = predict_risk_mc(Xp_t, g_t, S=max(8, S // 2))
                c_perm, _ = harrell_c(times, r_perm, events)
                deltas.append(c_base - c_perm)  # higher-is-better → base - perm

            delta = float(np.mean(deltas))
            if floor_zero and delta < 0:
                delta = 0.0
            drops.append((j, delta))

        if not drops:
            return list(range(min(2, X_ref.shape[1]))), []
        drops.sort(key=lambda x: x[1], reverse=True)
        top_idx = [j for (j, d) in drops[:k]]
        top_info = [(feature_names[j], d) for (j, d) in drops[:k]]
        return top_idx, top_info

    # NOTE: pass the TRAIN group indices aligned with Xtr_s!
    top_idx, top_info = permutation_importance_topk(
        Xtr_s, g_tr, t_tr, e_tr, feat_names, k=2
    )
    print(
        f"[PD] Top features by ΔC-index drop (using E[f] for ranking in PD): {top_info}"
    )

    def pd_curve_ice(feature_idx, feature_name, fname):
        n_tr = Xtr_s.shape[0]
        ns = min(PD_ICE_NSAMP, n_tr)
        ice_idx = rng.choice(n_tr, size=ns, replace=False)

        base_mat = Xtr_s[ice_idx].copy()
        g_ice = g_tr[ice_idx]  # <-- NEW (aligned with base_mat)
        g_ice_t = torch.tensor(g_ice, dtype=torch.long, device=device)

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

            preds, _ = predict_risk_mc(X_ice_t, g_ice_t, S=EVAL_MC_SAMPLES)  # <-- g_idx
            pd_mean.append(float(np.mean(preds)))
            pd_lo.append(float(np.quantile(preds, qlo)))
            pd_hi.append(float(np.quantile(preds, qhi)))

        pd_mean, pd_lo, pd_hi = map(np.asarray, (pd_mean, pd_lo, pd_hi))

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
    PD_H = POSTERIOR_H_MONTHS

    def pd_curve_ice_hazard(feature_idx, feature_name, fname, H=PD_H):
        if not USE_DISCRETE_TIME:
            print("[PD] Hazard-scale PD skipped (USE_DISCRETE_TIME=False).")
            return

        n_tr = Xtr_s.shape[0]
        ns = min(PD_ICE_NSAMP, n_tr)
        ice_idx = rng.choice(n_tr, size=ns, replace=False)

        base_mat = Xtr_s[ice_idx].copy()
        g_ice = g_tr[ice_idx]
        g_ice_t = torch.tensor(g_ice, dtype=torch.long, device=device)

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

            vals = neglogS_at_H(
                X_ice_t, g_ice_t, float(H), alpha_dt, S=EVAL_MC_SAMPLES
            )  # <-- g_idx
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

    # Expects: savefig, OUTDIR, and (optionally) kruskal_wallis/kruskal + mannwhitneyu in globals()

    def _get_kw():
        # supports either `kruskal_wallis` (your alias) or scipy's `kruskal`
        fn = globals().get("kruskal_wallis", None)
        if fn is None:
            fn = globals().get("kruskal", None)
        return fn

    def _get_mwu():
        return globals().get("mannwhitneyu", None)

    def _overall_pvalue(vecs):
        clean = [np.asarray(v, float).ravel() for v in vecs]
        clean = [v[np.isfinite(v)] for v in clean]
        clean = [v for v in clean if v.size > 0]
        if len(clean) < 2:
            return np.nan

        kw = _get_kw()
        mwu = _get_mwu()

        if len(clean) >= 3 and kw is not None:
            return float(kw(*clean).pvalue)

        if len(clean) == 2 and mwu is not None:
            return float(mwu(clean[0], clean[1], alternative="two-sided").pvalue)

        if len(clean) == 2 and kw is not None:
            # fallback (KW with 2 groups is equivalent to a rank test)
            return float(kw(clean[0], clean[1]).pvalue)

        return np.nan

    def _boxplot(vecs, labels, title, ylabel, fname):
        plt.figure(figsize=(6, 4))
        plt.boxplot(
            [np.asarray(v).ravel() for v in vecs], labels=labels, showfliers=False
        )
        ax = plt.gca()

        pval = _overall_pvalue(vecs)
        if np.isfinite(pval):
            ymin, ymax = ax.get_ylim()
            span = (ymax - ymin) if ymax > ymin else 1.0
            ax.set_ylim(ymin, ymax + 0.10 * span)
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

    def _bh_fdr(pvals):
        """Benjamini–Hochberg FDR adjust. Returns array aligned with input (NaNs stay NaN)."""
        p = np.asarray(pvals, float)
        q = np.full_like(p, np.nan, dtype=float)
        valid = np.isfinite(p)
        if not np.any(valid):
            return q

        pv = p[valid]
        m = pv.size
        order = np.argsort(pv)
        ranked = pv[order] * m / (np.arange(1, m + 1))
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        ranked = np.clip(ranked, 0.0, 1.0)

        qv = np.empty_like(pv)
        qv[order] = ranked
        q[valid] = qv
        return q

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

    def _boxplot_npip_with_pairs(vecs, labels, title, ylabel, fname):
        plt.figure(figsize=(6, 4))
        plt.boxplot(
            [np.asarray(v).ravel() for v in vecs], labels=labels, showfliers=False
        )
        ax = plt.gca()

        # ---- overall p-value ----
        pval = _overall_pvalue(vecs)
        if np.isfinite(pval):
            ymin, ymax = ax.get_ylim()
            span = (ymax - ymin) if ymax > ymin else 1.0
            ax.set_ylim(ymin, ymax + 0.10 * span)
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

        # ---- pairwise MWU (BH–FDR) ----
        mwu = _get_mwu()
        clean = [np.asarray(v, float).ravel() for v in vecs]
        clean = [v[np.isfinite(v)] for v in clean]

        def _draw_bracket(x1, x2, y, h, txt):
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, color="k")
            ax.text((x1 + x2) / 2.0, y + h, txt, ha="center", va="bottom", fontsize=10)

        if mwu is not None and len(clean) >= 2:
            pairs = list(itertools.combinations(range(len(clean)), 2))
            p_raw = []
            for i, j in pairs:
                if clean[i].size == 0 or clean[j].size == 0:
                    p_raw.append(np.nan)
                else:
                    p_raw.append(
                        float(mwu(clean[i], clean[j], alternative="two-sided").pvalue)
                    )

            p_adj = _bh_fdr(p_raw)
            finite_idx = [k for k, qv in enumerate(p_adj) if np.isfinite(qv)]

            if finite_idx:
                ymin, ymax = ax.get_ylim()
                data_top = max((np.max(v) if v.size else ymin) for v in clean)
                span = (ymax - ymin) if ymax > ymin else 1.0

                base = max(ymax, data_top) + 0.08 * span
                step = 0.08 * span
                h = 0.02 * span

                finite_idx.sort(key=lambda k: p_adj[k])  # most significant first
                for rank, k in enumerate(finite_idx):
                    i, j = pairs[k]
                    x1, x2 = i + 1, j + 1
                    y = base + rank * step
                    _draw_bracket(
                        x1, x2, y, h, f"{_p_to_stars(p_adj[k])} (q={p_adj[k]:.2e})"
                    )

                ax.set_ylim(ymin, base + (len(finite_idx) + 1) * step)

        plt.ylabel(ylabel)
        plt.title(title)
        savefig(OUTDIR / fname)

    def _boxplot_prolif_with_pairs(vecs, labels, title, ylabel, fname):
        plt.figure(figsize=(6, 4))
        plt.boxplot(
            [np.asarray(v).ravel() for v in vecs], labels=labels, showfliers=False
        )
        ax = plt.gca()

        # ---- overall p-value (same logic as your _boxplot) ----
        clean = [np.asarray(v, float)[~np.isnan(np.asarray(v, float))] for v in vecs]
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

        if np.isfinite(pval):
            ymin, ymax = ax.get_ylim()
            span = ymax - ymin if ymax > ymin else 1.0
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
            """
            Benjamini–Hochberg FDR correction.
            Robust to NaNs/infs: only finite p-values are adjusted; others remain NaN.
            """
            p = np.asarray(pvals, float)
            out = np.full_like(p, np.nan, dtype=float)
            finite = np.isfinite(p)
            m = int(finite.sum())
            if m == 0:
                return out

            pv = p[finite]
            order = np.argsort(pv)
            pv_sorted = pv[order]
            ranks = np.arange(1, m + 1, dtype=float)

            q_sorted = pv_sorted * m / ranks
            q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
            q_sorted = np.clip(q_sorted, 0.0, 1.0)

            q = np.empty_like(pv_sorted)
            q[order] = q_sorted
            out[finite] = q
            return out

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
            q_adj = _bh_fdr(p_raw)

            ymin, ymax = ax.get_ylim()
            span = (ymax - ymin) if (ymax > ymin) else 1.0
            data_top = max((np.max(v) if len(v) else ymin) for v in clean)
            step = 0.08 * span
            base = max(ymax, data_top) + 0.06 * span

            finite_idx = np.where(np.isfinite(q_adj))[0]
            order = finite_idx[np.argsort(q_adj[finite_idx])]  # smallest q first

            for k, idx in enumerate(order):
                i, j = pairs[idx]
                x1, x2 = i + 1, j + 1
                y = base + k * step
                _draw_bracket(
                    x1,
                    x2,
                    y,
                    0.02 * span,
                    f"{_p_to_stars(q_adj[idx])} (q={q_adj[idx]:.2e})",
                )

            ax.set_ylim(ymin, base + (len(order) + 1) * step + 0.08 * span)

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

        _boxplot_prolif_with_pairs(
            [ges_Pro_te[:, 0][g] for g in groups],
            glabels,
            "Proliferation score by Posterior-H group (test)",
            "Proliferation score",
            "bio_box_prolif_by_group_test.png",
        )

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

    # ================= [B] Interpretability on hazard scale @ H  (WASSERSTEIN FC ONLY; SPEARMAN SIGN) ====================
    HORIZONS = [120.0, float(H_C), 60.0, 30.0]
    HORIZONS = [H for H in HORIZONS if np.isfinite(H) and H > 0]

    # ----------------- Spearman (raw) -----------------
    def raw_spearman(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3:
            return 0.0
        a = a[m]
        b = b[m]
        if spearmanr is not None:
            rho, _ = spearmanr(a, b, nan_policy="omit")
            return float(rho) if np.isfinite(rho) else 0.0

        # fallback (rank-corr) if SciPy unavailable
        def _rank(x):
            x = np.asarray(x, float)
            order = np.argsort(x, kind="mergesort")
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(x), dtype=float)
            xs = x[order]
            i = 0
            while i < len(xs):
                j = i + 1
                while j < len(xs) and xs[j] == xs[i]:
                    j += 1
                if j - i > 1:
                    ranks[order[i:j]] = (i + (j - 1)) / 2.0
                i = j
            return ranks

        ra = _rank(a)
        rb = _rank(b)
        if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
            return 0.0
        return float(np.corrcoef(ra, rb)[0, 1])

    def _safe_fname(s):
        s = str(s)
        s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s[:180] if len(s) > 180 else s

    # ----------------- Precompute mean -logS(H) on TEST for sign annotations -----------------
    # Supports both neglogS_at_H(X_t, g_t, H, alpha_dt, ...) and neglogS_at_H(X_t, H, alpha_dt, ...)

    try:
        nlS_by_H = {
            H: neglogS_at_H(Xte_t, g_te_t, H, alpha_dt, S=EVAL_MC_SAMPLES)
            for H in HORIZONS
        }
    except TypeError:
        nlS_by_H = {
            H: neglogS_at_H(Xte_t, H, alpha_dt, S=EVAL_MC_SAMPLES) for H in HORIZONS
        }

    # Also precompute TRAIN / VAL / ALL for per-scope FC sign + backprojections
    def _neglogS_mean_at_H_split(X_t, g_t, H):
        try:
            return neglogS_at_H(X_t, g_t, H, alpha_dt, S=EVAL_MC_SAMPLES)
        except TypeError:
            return neglogS_at_H(X_t, H, alpha_dt, S=EVAL_MC_SAMPLES)

    nlS_by_H_tr = {H: _neglogS_mean_at_H_split(Xtr_t, g_tr_t, H) for H in HORIZONS}
    if HAS_VAL:
        nlS_by_H_va = {H: _neglogS_mean_at_H_split(Xva_t, g_va_t, H) for H in HORIZONS}
        nlS_by_H_all = {
            H: np.concatenate([nlS_by_H_tr[H], nlS_by_H_va[H], nlS_by_H[H]], axis=0)
            for H in HORIZONS
        }
    else:
        nlS_by_H_va = {H: np.zeros((0,), dtype=float) for H in HORIZONS}
        nlS_by_H_all = {
            H: np.concatenate([nlS_by_H_tr[H], nlS_by_H[H]], axis=0) for H in HORIZONS
        }

    # ========================= Wasserstein Feature Collapse (CRN + repeat-avg) =========================
    FC_S = int(min(256, EVAL_MC_SAMPLES))  # MC samples used inside FC (keep 256)
    FC_METHOD = "wasserstein-1"
    FC_USE_CRN = True
    FC_REPEATS = 3
    FC_REPEAT_STRIDE = 10007
    FC_MIN_N_PER_GROUP = 30  # skip tiny groups for per-group FC

    def _wasserstein1_empirical_per_row(Y0, Y1):
        # Y0,Y1: [S,N] -> per-patient empirical W1 by sorting samples
        Y0s = np.sort(np.asarray(Y0, float), axis=0)
        Y1s = np.sort(np.asarray(Y1, float), axis=0)
        return np.mean(np.abs(Y0s - Y1s), axis=0)  # [N]

    _alpha_t = torch.tensor(alpha_dt, device=device, dtype=torch.float32)

    def _hazard_from_eta(eta):
        if DT_LINK.lower() == "cloglog":
            return 1.0 - torch.exp(-torch.exp(torch.clamp(eta, max=20.0)))
        return torch.sigmoid(eta)

    def _to_np_int(x):
        try:
            return x.detach().cpu().numpy().astype(int)
        except Exception:
            return np.asarray(x, int)

    def _to_torch_long(x_np):
        return torch.tensor(np.asarray(x_np, int), device=device, dtype=torch.long)

    def neglogS_samples_at_H_mt(X_ref_s, g_ref_t, H, S=FC_S):
        """
        Returns nlS samples: shape [S, N] where each row is one MC draw of -log S(H).

        IMPORTANT (multitask): model(X, g) typically returns a multitask distribution whose
        rsample shape may be [S, N, G] or [S, G, N]. We must select the per-patient task
        via _select_task before converting to hazards.
        """
        X_ref_s = np.asarray(X_ref_s, float)
        X = torch.tensor(X_ref_s, device=device, dtype=torch.float32)

        j = int(np.searchsorted(bin_edges, H, side="right") - 1)
        j = int(np.clip(j, 0, len(bin_edges) - 2))

        with torch.no_grad():
            try:
                dist = model(X, g_ref_t)
            except TypeError:
                dist = model(X)

            f_all = dist.rsample(
                torch.Size([int(S)])
            )  # e.g. [S,N,G] or [S,G,N] or [S,N]
            f_s = _select_task(f_all, g_ref_t)  # -> [S,N] (or [N] if scalar)
            if f_s.ndim == 1:
                f_s = f_s[None, :]  # [1,N] for consistency

            eta = f_s[:, :, None] + _alpha_t[None, None, :]  # [S,N,B]
            h = _hazard_from_eta(eta)  # [S,N,B]
            S_edges = torch.cumprod(1.0 - h, dim=2)  # [S,N,B]
            nlS_s = -torch.log(torch.clamp(S_edges[:, :, j], min=1e-10))  # [S,N]
            return nlS_s.detach().cpu().numpy()

    def feature_collapsing_importance_mt(
        X_ref_s,
        g_ref_np,
        feature_names,
        H,
        S=FC_S,
        skip_noncontinuous=False,  # KEEP FALSE
        method=FC_METHOD,
        use_crn=FC_USE_CRN,
        crn_seed=12345,
        n_repeats=FC_REPEATS,
        repeat_seed_stride=FC_REPEAT_STRIDE,
    ):
        """
        Wasserstein feature collapse:
        Score_j = mean_i W1( Y0[:,i], Y1[:,i] ), where Y0/Y1 are nlS(H) samples.

        Stability:
        (1) CRN within a repeat: restore RNG state before each Y1 so baseline vs collapsed differ only by feature.
        (2) Repeat-avg across repeats with different seeds.
        """
        X_ref_s = np.asarray(X_ref_s, float)
        P = int(X_ref_s.shape[1])
        g_ref_np = np.asarray(g_ref_np, int)
        g_ref_t = _to_torch_long(g_ref_np)

        scores_accum = np.zeros(P, dtype=float)
        dmean_accum = np.zeros(
            P, dtype=float
        )  # mean over patients of (E[Y1]-E[Y0]) per feature
        R = int(max(1, n_repeats))

        for r in range(R):
            seed_r = int(crn_seed) + r * int(repeat_seed_stride)

            if use_crn:
                np.random.seed(seed_r)
                torch.manual_seed(seed_r)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_r)

                np_state0 = np.random.get_state()
                torch_state0 = torch.get_rng_state()
                cuda_state0 = (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                )

            Y0 = neglogS_samples_at_H_mt(X_ref_s, g_ref_t, float(H), S=int(S))  # [S,N]
            meanY0 = np.mean(Y0, axis=0)  # [N]

            scores = np.zeros(P, dtype=float)
            dmean = np.zeros(P, dtype=float)

            for j in range(P):
                if not np.isfinite(X_ref_s[:, j]).any():
                    continue
                if skip_noncontinuous and (not is_continuous(X_ref_s[:, j])):
                    continue

                if use_crn:
                    np.random.set_state(np_state0)
                    torch.set_rng_state(torch_state0)
                    if cuda_state0 is not None:
                        torch.cuda.set_rng_state_all(cuda_state0)

                Xc = X_ref_s.copy()
                Xc[:, j] = 0.0

                Y1 = neglogS_samples_at_H_mt(Xc, g_ref_t, float(H), S=int(S))  # [S,N]
                meanY1 = np.mean(Y1, axis=0)  # [N]

                if method.lower() in ("wasserstein-1", "wasserstein1", "wasserstein"):
                    w1_i = _wasserstein1_empirical_per_row(Y0, Y1)  # [N]
                    scores[j] = float(np.mean(w1_i))
                    dmean[j] = float(np.mean(meanY1 - meanY0))
                else:
                    raise ValueError(
                        f"Unknown FC method='{method}'. Use 'wasserstein-1'."
                    )

            scores_accum += scores
            dmean_accum += dmean

        scores_final = scores_accum / float(R)
        dmean_final = dmean_accum / float(R)

        df = pd.DataFrame(
            {
                "feature": list(feature_names),
                "FC_score": scores_final,
                "FC_delta_mean_nlS": dmean_final,  # >0 => collapsing increases predicted -logS(H) on avg
                "FC_method": str(method),
                "FC_S": int(S),
                "FC_use_crn": bool(use_crn),
                "FC_crn_seed": int(crn_seed),
                "FC_repeats": int(R),
                "FC_repeat_stride": int(repeat_seed_stride),
            }
        )
        return df.sort_values("FC_score", ascending=False).reset_index(drop=True)

    def _plot_topN_fc(df, score_col, title, outpath, topN=10, xlabel=None):
        top = df.head(int(topN)).iloc[::-1].copy()
        plt.figure(figsize=(8, 6))
        y = np.arange(len(top))
        plt.barh(y, top[score_col].values)
        plt.yticks(y, top["feature"].values)
        plt.xlabel(xlabel if xlabel is not None else score_col)
        plt.title(title)
        plt.tight_layout()
        savefig(outpath)
        plt.close()

    # ======================= [B-FC] Run Wasserstein FC for TEST, TRAIN, ALL + per-group + FC-driven PC backproj =======================
    outdir_fc = OUTDIR / "fc"
    outdir_fc.mkdir(parents=True, exist_ok=True)

    outdir_pc_pngs = OUTDIR / "pc_gene_pngs_fc"
    os.makedirs(outdir_pc_pngs, exist_ok=True)

    # PC block (selected gene PCs)
    genes_pref = expr.index.to_numpy()
    Vt_sel = Vt[keep_idxs, :]  # rows correspond to selected PCs
    pc_name_to_idx = {
        str(n): i for i, n in enumerate(sel_names)
    }  # PC name -> idx in selected block

    # group names if available
    try:
        GROUP_NAMES = list(CANON_GROUPS)
    except Exception:
        _g_tmp = _to_np_int(g_te_t)
        _K = int(np.max(_g_tmp)) + 1 if _g_tmp.size else 0
        GROUP_NAMES = [f"group_{k}" for k in range(_K)]

    # prepare ALL cohort matrices (standardized X and selected PC scores)
    if HAS_VAL:
        Xall_s = np.concatenate([Xtr_s, Xva_s, Xte_s], axis=0)
        gall_np = np.concatenate(
            [_to_np_int(g_tr_t), _to_np_int(g_va_t), _to_np_int(g_te_t)], axis=0
        )
        Tall_sel = np.concatenate([Ttr_sel, Tva_sel, Tte_sel], axis=0)
    else:
        Xall_s = np.concatenate([Xtr_s, Xte_s], axis=0)
        gall_np = np.concatenate([_to_np_int(g_tr_t), _to_np_int(g_te_t)], axis=0)
        Tall_sel = np.concatenate([Ttr_sel, Tte_sel], axis=0)

    # cache FC dfs so later sections can reuse (e.g., H=120)
    fc_cache = {}  # (scope, H) -> df
    pc_rows_for_csv_all = []

    def _backproject_gene_pc_from_scores(
        pc_name, pc_idx0, H_INT, scope_tag, pc_scores, nlS_vec
    ):
        rho = raw_spearman(pc_scores, nlS_vec)
        sign_for_plot = 1.0 if (np.isfinite(rho) and rho >= 0) else -1.0
        load = sign_for_plot * np.asarray(Vt_sel[pc_idx0, :], float)

        df = (
            pd.DataFrame(
                {
                    "gene": genes_pref,
                    "pc": str(pc_name),
                    "loading_oriented": load,
                    "abs_loading": np.abs(load),
                    "raw_spearman_with_neglogS": float(rho),
                    "H": int(H_INT),
                    "scope": str(scope_tag),
                }
            )
            .sort_values("abs_loading", ascending=False)
            .reset_index(drop=True)
        )

        df.to_csv(
            OUTDIR
            / f"pc_gene_oriented_{pc_name}_H{int(H_INT)}_{_safe_fname(scope_tag)}.csv",
            index=False,
        )
        df.head(100).to_csv(
            OUTDIR
            / f"pc_{pc_name}_top_genes_oriented_H{int(H_INT)}_{_safe_fname(scope_tag)}.csv",
            index=False,
        )

        topg = df.head(20).sort_values("loading_oriented")
        plt.figure(figsize=(8.0, 6.0))
        ax = plt.gca()
        y = np.arange(len(topg))
        plt.barh(y, topg["loading_oriented"].values)
        plt.yticks(y, topg["gene"].astype(str).values)
        plt.xlabel("PC loading (oriented)")
        ax.axvline(0.0, color="k", lw=0.8, ls="--")
        lim = float(np.max(np.abs(topg["loading_oriented"].values))) * 1.10
        lim = max(lim, 1.0)
        ax.set_xlim(-lim, lim)
        plt.title(f"{pc_name} — top gene loadings ({scope_tag}) @ H={int(H_INT)} mo")
        png_path = (
            Path(outdir_pc_pngs)
            / f"H{int(H_INT)}"
            / _safe_fname(scope_tag)
            / f"{pc_name}_top20_riskOriented_H{int(H_INT)}_{_safe_fname(scope_tag)}.png"
        )
        png_path.parent.mkdir(parents=True, exist_ok=True)
        savefig(png_path)
        plt.close()
        return float(rho), str(png_path)

    def _run_fc_scope(scope_tag, X_s, g_np, t_np, e_np, nlS_vec_scope, Tsel_mat, H_INT):
        # horizon-specific seed base so FC is stable within each H and reproducible across runs
        FC_CRN_SEED_H = int(seed) * 100000 + int(H_INT) * 1000 + 17

        df = feature_collapsing_importance_mt(
            X_s,
            g_np,
            feat_names,
            H_INT,
            S=FC_S,
            skip_noncontinuous=False,  # YOUR REQUEST
            method=FC_METHOD,
            use_crn=FC_USE_CRN,
            crn_seed=FC_CRN_SEED_H,
            n_repeats=FC_REPEATS,
            repeat_seed_stride=FC_REPEAT_STRIDE,
        )

        # Keep Spearman correlations for reference/annotation (but do NOT create signed FC scores)
        r_raw_vec = np.array(
            [raw_spearman(X_s[:, j], nlS_vec_scope) for j in range(len(feat_names))],
            dtype=float,
        )
        df["raw_spearman_with_neglogS"] = r_raw_vec

        df.to_csv(
            outdir_fc
            / f"feature_collapsing_{_safe_fname(scope_tag)}_H{int(H_INT)}{seed_suffix}.csv",
            index=False,
        )

        # ---- [FC-SENS] Top-K vs Random sensitivity test ----
        if RUN_FC_SENSITIVITY_TEST and (str(scope_tag) in set(FC_SENS_SCOPES)):
            try:
                rng_sens = np.random.default_rng(
                    int(seed) + int(FC_SENS_SEED_OFFSET) + int(H_INT) * 17
                )

                name_to_idx = {str(n): i for i, n in enumerate(feat_names)}
                top_names = (
                    df.sort_values("FC_score", ascending=False)
                    .head(int(FC_SENS_TOPK))["feature"]
                    .astype(str)
                    .tolist()
                )
                top_idx = [name_to_idx.get(n, None) for n in top_names]
                top_idx = [i for i in top_idx if i is not None]

                P = int(X_s.shape[1]) if hasattr(X_s, "shape") else int(len(feat_names))
                K_eff = int(min(len(top_idx), int(FC_SENS_TOPK), P))
                top_idx = top_idx[:K_eff]

                if K_eff >= 1:
                    g_t = torch.tensor(
                        np.asarray(g_np, int), device=device, dtype=torch.long
                    )

                    # Use common random numbers (paired draws) across baseline/topK/random
                    # to reduce Monte Carlo noise in deltas and improve stability.
                    _sens_seed = int(seed) + int(FC_SENS_SEED_OFFSET) + int(H_INT) * 17
                    try:
                        torch.manual_seed(int(_sens_seed))
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(int(_sens_seed))
                    except Exception:
                        pass
                    _torch_state0 = torch.get_rng_state()
                    _cuda_state0 = (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    )

                    # Baseline predictions (risk = E[-log S(H)])
                    Y0 = neglogS_samples_at_H_mt(
                        X_s, g_t, float(H_INT), S=int(FC_SENS_MC)
                    )  # [S,N]
                    risk0 = np.mean(Y0, axis=0)
                    c0, _ = harrell_c(
                        np.asarray(t_np, float), risk0, np.asarray(e_np, int)
                    )
                    mean0 = float(np.mean(risk0))

                    # Collapse Top-K features
                    X_top = np.asarray(X_s, float).copy()
                    X_top[:, top_idx] = 0.0
                    # CRN restore so Top-K collapse uses same random draws as baseline
                    try:
                        torch.set_rng_state(_torch_state0)
                        if _cuda_state0 is not None:
                            torch.cuda.set_rng_state_all(_cuda_state0)
                    except Exception:
                        pass

                    Y_top = neglogS_samples_at_H_mt(
                        X_top, g_t, float(H_INT), S=int(FC_SENS_MC)
                    )
                    risk_top = np.mean(Y_top, axis=0)
                    c_top, _ = harrell_c(
                        np.asarray(t_np, float), risk_top, np.asarray(e_np, int)
                    )
                    mean_top = float(np.mean(risk_top))

                    dC_top = float(
                        c0 - c_top
                    )  # >0 => performance drops when collapsing Top-K
                    dMean_top = float(
                        mean_top - mean0
                    )  # >0 => predicted risk increases

                    # Random K feature sets
                    rand_dC = []
                    rand_dMean = []
                    risk_rand_list = (
                        []
                    )  # store per-rep risk vectors (for by-group deltas)
                    for rr in range(int(FC_SENS_RANDOM_REPS)):
                        idx_r = rng_sens.choice(P, size=K_eff, replace=False)
                        X_r = np.asarray(X_s, float).copy()
                        X_r[:, idx_r] = 0.0
                        # CRN restore so each random collapse is paired against baseline
                        try:
                            torch.set_rng_state(_torch_state0)
                            if _cuda_state0 is not None:
                                torch.cuda.set_rng_state_all(_cuda_state0)
                        except Exception:
                            pass

                        Y_r = neglogS_samples_at_H_mt(
                            X_r, g_t, float(H_INT), S=int(FC_SENS_MC)
                        )
                        risk_r = np.mean(Y_r, axis=0)
                        risk_rand_list.append(np.asarray(risk_r, float).copy())
                        c_r, _ = harrell_c(
                            np.asarray(t_np, float), risk_r, np.asarray(e_np, int)
                        )
                        mean_r = float(np.mean(risk_r))
                        rand_dC.append(float(c0 - c_r))
                        rand_dMean.append(float(mean_r - mean0))

                    rand_dC = np.asarray(rand_dC, float)
                    rand_dMean = np.asarray(rand_dMean, float)

                    p_dc = (
                        float(np.mean(rand_dC >= dC_top))
                        if rand_dC.size
                        else float("nan")
                    )
                    p_dm = (
                        float(np.mean(rand_dMean >= dMean_top))
                        if rand_dMean.size
                        else float("nan")
                    )

                    sens_dir = outdir_fc / "sensitivity"
                    sens_dir.mkdir(parents=True, exist_ok=True)

                    summ = pd.DataFrame(
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
                    )
                    summ.to_csv(
                        sens_dir
                        / f"sensitivity_summary_{_safe_fname(scope_tag)}_H{int(H_INT)}_K{int(K_eff)}{seed_suffix}.csv",
                        index=False,
                    )

                    raw = pd.DataFrame(
                        {
                            "rand_rep": np.arange(len(rand_dC), dtype=int),
                            "deltaC_random": rand_dC,
                            "deltaMean_nlS_random": rand_dMean,
                        }
                    )
                    raw.to_csv(
                        sens_dir
                        / f"sensitivity_random_{_safe_fname(scope_tag)}_H{int(H_INT)}_K{int(K_eff)}{seed_suffix}.csv",
                        index=False,
                    )

                    # ---- By-group sensitivity (within-group C-index) ----
                    by_rows = []
                    by_raw_rows = []
                    t_arr = np.asarray(t_np, float)
                    e_arr = np.asarray(e_np, int)
                    g_arr = np.asarray(g_np, int)

                    # (using risk_rand_list computed above)

                    for gi, gname in enumerate(CANON_GROUPS):
                        m = g_arr == int(gi)
                        n_g = int(np.sum(m))
                        ev_g = int(np.sum(e_arr[m] == 1))

                        row = {
                            "scope": str(scope_tag),
                            "group_idx": int(gi),
                            "group": str(gname),
                            "H_months": float(H_INT),
                            "K": int(K_eff),
                            "MC": int(FC_SENS_MC),
                            "random_reps": int(FC_SENS_RANDOM_REPS),
                            "n": n_g,
                            "events": ev_g,
                        }

                        if (n_g < 30) or (ev_g < 5):
                            row.update(
                                {
                                    "baseline_cindex": np.nan,
                                    "topK_cindex": np.nan,
                                    "deltaC_topK": np.nan,
                                    "pval_deltaC_rand_ge_topK": np.nan,
                                    "baseline_mean_nlS": np.nan,
                                    "topK_mean_nlS": np.nan,
                                    "deltaMean_nlS_topK": np.nan,
                                    "pval_deltaMean_nlS_rand_ge_topK": np.nan,
                                    "skipped": True,
                                }
                            )
                            by_rows.append(row)
                            continue

                        # Baseline within group
                        r0g = orient_risk(t_arr[m], e_arr[m], risk0[m])
                        c0g = float(_lifelines_c(t_arr[m], r0g, e_arr[m]))
                        mean0g = float(np.mean(risk0[m]))

                        # Top-K within group
                        rTopg = orient_risk(t_arr[m], e_arr[m], risk_top[m])
                        cTopg = float(_lifelines_c(t_arr[m], rTopg, e_arr[m]))
                        meanTopg = float(np.mean(risk_top[m]))

                        dC_g = float(c0g - cTopg)
                        dMean_g = float(meanTopg - mean0g)

                        # Random deltas within group
                        rand_dC_g = []
                        rand_dMean_g = []
                        for rr in range(len(risk_rand_list)):
                            rrisk = np.asarray(risk_rand_list[rr], float)
                            rrg = orient_risk(t_arr[m], e_arr[m], rrisk[m])
                            c_rg = float(_lifelines_c(t_arr[m], rrg, e_arr[m]))
                            mean_rg = float(np.mean(rrisk[m]))
                            rand_dC_g.append(float(c0g - c_rg))
                            rand_dMean_g.append(float(mean_rg - mean0g))
                            by_raw_rows.append(
                                {
                                    "scope": str(scope_tag),
                                    "group_idx": int(gi),
                                    "group": str(gname),
                                    "H_months": float(H_INT),
                                    "K": int(K_eff),
                                    "rand_rep": int(rr),
                                    "deltaC_random": float(c0g - c_rg),
                                    "deltaMean_nlS_random": float(mean_rg - mean0g),
                                }
                            )

                        rand_dC_g = np.asarray(rand_dC_g, float)
                        rand_dMean_g = np.asarray(rand_dMean_g, float)

                        p_dc_g = (
                            float(np.mean(rand_dC_g >= dC_g))
                            if rand_dC_g.size
                            else float("nan")
                        )
                        p_dm_g = (
                            float(np.mean(rand_dMean_g >= dMean_g))
                            if rand_dMean_g.size
                            else float("nan")
                        )

                        row.update(
                            {
                                "baseline_cindex": float(c0g),
                                "topK_cindex": float(cTopg),
                                "deltaC_topK": float(dC_g),
                                "pval_deltaC_rand_ge_topK": float(p_dc_g),
                                "baseline_mean_nlS": float(mean0g),
                                "topK_mean_nlS": float(meanTopg),
                                "deltaMean_nlS_topK": float(dMean_g),
                                "pval_deltaMean_nlS_rand_ge_topK": float(p_dm_g),
                                "skipped": False,
                            }
                        )
                        by_rows.append(row)

                    by_summ = pd.DataFrame(by_rows)
                    by_summ.to_csv(
                        sens_dir
                        / f"sensitivity_summary_bygroup_{_safe_fname(scope_tag)}_H{int(H_INT)}_K{int(K_eff)}{seed_suffix}.csv",
                        index=False,
                    )
                    by_raw = pd.DataFrame(by_raw_rows)
                    by_raw.to_csv(
                        sens_dir
                        / f"sensitivity_random_bygroup_{_safe_fname(scope_tag)}_H{int(H_INT)}_K{int(K_eff)}{seed_suffix}.csv",
                        index=False,
                    )

                    print(
                        f"[FC-SENS] {scope_tag} H={int(H_INT)} K={K_eff}: "
                        f"ΔC_topK={dC_top:.4f} (p={p_dc:.3f}); "
                        f"Δmean(nlS)={dMean_top:.4f} (p={p_dm:.3f})"
                    )

            except Exception as _e:
                print(
                    f"[FC-SENS] skipped for scope={scope_tag}, H={H_INT} due to: {_e}"
                )

        # ---- PLOT TOP10 USING REGULAR FC_score ONLY ----
        _plot_topN_fc(
            df,
            "FC_score",
            title=f"Top 10 features — Feature Collapsing (Wasserstein-1) ({scope_tag}) @ H={int(H_INT)} mo",
            outpath=outdir_fc
            / f"feature_collapsing_top10_{_safe_fname(scope_tag)}_H{int(H_INT)}{seed_suffix}.png",
            topN=10,
            xlabel="Feature Collapsing score (empirical Wasserstein-1; larger = more influence)",
        )

        fc_cache[(str(scope_tag), int(H_INT))] = df

        # FC-driven PC backprojections: ONLY PCs in top10 features
        top_feats = df["feature"].astype(str).head(10).tolist()
        for fname in top_feats:
            if fname in pc_name_to_idx:
                pc_idx0 = pc_name_to_idx[fname]
                pc_scores = np.asarray(Tsel_mat[:, pc_idx0], float)
                rho, png_path = _backproject_gene_pc_from_scores(
                    fname, pc_idx0, H_INT, scope_tag, pc_scores, nlS_vec_scope
                )
                fc_val = float(
                    df.loc[df["feature"].astype(str) == fname, "FC_score"].iloc[0]
                )
                pc_rows_for_csv_all.append(
                    {
                        "scope": str(scope_tag),
                        "H": int(H_INT),
                        "pc": str(fname),
                        "fc_score": float(fc_val),
                        "raw_spearman_with_neglogS": float(rho),
                        "png_path": str(png_path),
                        "FC_method": str(FC_METHOD),
                        "FC_S": int(FC_S),
                        "FC_use_crn": bool(FC_USE_CRN),
                        "FC_repeats": int(FC_REPEATS),
                        "FC_repeat_stride": int(FC_REPEAT_STRIDE),
                    }
                )

        # Per-group FC within this scope
        g_np = np.asarray(g_np, int)
        K = int(np.max(g_np) + 1) if g_np.size else 0
        group_rows = []
        for gi in range(min(K, len(GROUP_NAMES))):
            m = g_np == gi
            n = int(m.sum())
            gname = GROUP_NAMES[gi]
            if n < FC_MIN_N_PER_GROUP:
                group_rows.append(
                    {"group_idx": gi, "group": gname, "n": n, "skipped": True}
                )
                continue

            scope_g = f"{scope_tag}__group{gi:02d}__{_safe_fname(gname)}"
            try:
                df_g = feature_collapsing_importance_mt(
                    X_s[m],
                    g_np[m],
                    feat_names,
                    H_INT,
                    S=FC_S,
                    skip_noncontinuous=False,
                    method=FC_METHOD,
                    use_crn=FC_USE_CRN,
                    crn_seed=FC_CRN_SEED_H + 999 * gi,  # deterministic per-group offset
                    n_repeats=FC_REPEATS,
                    repeat_seed_stride=FC_REPEAT_STRIDE,
                )

                nlS_g = np.asarray(nlS_vec_scope[m], float)
                r_g = np.array(
                    [raw_spearman(X_s[m, j], nlS_g) for j in range(len(feat_names))],
                    dtype=float,
                )
                df_g["raw_spearman_with_neglogS"] = r_g

                df_g.to_csv(
                    outdir_fc
                    / f"feature_collapsing_{_safe_fname(scope_g)}_H{int(H_INT)}{seed_suffix}.csv",
                    index=False,
                )

                # ---- PLOT TOP10 USING REGULAR FC_score ONLY ----
                _plot_topN_fc(
                    df_g,
                    "FC_score",
                    title=f"Top 10 — Feature Collapsing (W1) ({scope_tag}) | {gname} @ H={int(H_INT)} mo",
                    outpath=outdir_fc
                    / f"feature_collapsing_top10_{_safe_fname(scope_g)}_H{int(H_INT)}{seed_suffix}.png",
                    topN=10,
                    xlabel="Feature Collapsing score (empirical Wasserstein-1; larger = more influence)",
                )

                # FC-driven PC backprojections inside this group (top10 only)
                top_feats_g = df_g["feature"].astype(str).head(10).tolist()
                for fname in top_feats_g:
                    if fname in pc_name_to_idx:
                        pc_idx0 = pc_name_to_idx[fname]
                        pc_scores_g = np.asarray(Tsel_mat[m, pc_idx0], float)
                        rho, png_path = _backproject_gene_pc_from_scores(
                            fname, pc_idx0, H_INT, scope_g, pc_scores_g, nlS_g
                        )
                        fc_val = float(
                            df_g.loc[
                                df_g["feature"].astype(str) == fname, "FC_score"
                            ].iloc[0]
                        )
                        pc_rows_for_csv_all.append(
                            {
                                "scope": str(scope_g),
                                "H": int(H_INT),
                                "pc": str(fname),
                                "fc_score": float(fc_val),
                                "raw_spearman_with_neglogS": float(rho),
                                "png_path": str(png_path),
                                "FC_method": str(FC_METHOD),
                                "FC_S": int(FC_S),
                                "FC_use_crn": bool(FC_USE_CRN),
                                "FC_repeats": int(FC_REPEATS),
                                "FC_repeat_stride": int(FC_REPEAT_STRIDE),
                            }
                        )

                group_rows.append(
                    {
                        "group_idx": gi,
                        "group": gname,
                        "n": n,
                        "skipped": False,
                        "csv": str(
                            outdir_fc
                            / f"feature_collapsing_{_safe_fname(scope_g)}_H{int(H_INT)}{seed_suffix}.csv"
                        ),
                    }
                )
            except Exception as ex:
                print(
                    f"[FC] Per-group failed: scope={scope_tag}, group={gname}, H={int(H_INT)} : {ex}"
                )
                group_rows.append(
                    {"group_idx": gi, "group": gname, "n": n, "skipped": True}
                )

        pd.DataFrame(group_rows).to_csv(
            outdir_fc
            / f"feature_collapsing_{_safe_fname(scope_tag)}__per_group_summary_H{int(H_INT)}{seed_suffix}.csv",
            index=False,
        )

        return df

    # Run FC across horizons for TRAIN / TEST / ALL
    for H_INT in HORIZONS:
        print(f"[D] Wasserstein Feature Collapse @ H={int(H_INT)} months")

        # TRAIN
        _run_fc_scope(
            "train",
            Xtr_s,
            _to_np_int(g_tr_t),
            np.asarray(t_tr, float),
            np.asarray(e_tr, int),
            np.asarray(nlS_by_H_tr[H_INT], float),
            np.asarray(Ttr_sel, float),
            H_INT,
        )

        # TEST
        _run_fc_scope(
            "test",
            Xte_s,
            _to_np_int(g_te_t),
            np.asarray(t_te, float),
            np.asarray(e_te, int),
            np.asarray(nlS_by_H[H_INT], float),
            np.asarray(Tte_sel, float),
            H_INT,
        )

        # ALL
        _run_fc_scope(
            "all",
            Xall_s,
            gall_np,
            np.asarray(time_all, float),
            np.asarray(event_all, int),
            np.asarray(nlS_by_H_all[H_INT], float),
            np.asarray(Tall_sel, float),
            H_INT,
        )

    # Save a compact summary of all FC-driven PC backprojections we produced
    if pc_rows_for_csv_all:
        (
            pd.DataFrame(pc_rows_for_csv_all)
            .sort_values(["scope", "H", "fc_score"], ascending=[True, True, False])
            .to_csv(OUTDIR / "topPC_FC_multiH_scopes_summary.csv", index=False)
        )
        print(f"[B-PC] Saved FC-driven PC→gene PNGs in {outdir_pc_pngs}")

    # ======================= [C] PC→gene back-projection (quick view; FC-top PCs only) =======================
    # Replace old pc_rank-based quickview with "FC-top PCs @ H=120 on TEST"
    try:
        H_Q = 120
        if int(H_Q) not in [int(h) for h in HORIZONS]:
            H_Q = int(HORIZONS[0])

        fc120 = fc_cache.get(("test", int(H_Q)), None)
        if fc120 is None:
            fc120_path = (
                outdir_fc / f"feature_collapsing_test_H{int(H_Q)}{seed_suffix}.csv"
            )
            if fc120_path.exists():
                fc120 = pd.read_csv(fc120_path)

        pc_sig = None
        if fc120 is not None:
            pc_sig = (
                fc120.loc[fc120["feature"].astype(str).isin(list(map(str, sel_names)))]
                .head(min(8, len(sel_names)))
                .reset_index(drop=True)
            )

        if pc_sig is not None and len(pc_sig) > 0:
            back_rows = []
            for _, row in pc_sig.iterrows():
                pc_name = str(row["feature"])
                pc_idx0 = pc_name_to_idx.get(pc_name, None)
                if pc_idx0 is None:
                    continue
                rho = raw_spearman(
                    np.asarray(Tte_sel[:, pc_idx0], float),
                    np.asarray(nlS_by_H[int(H_Q)], float),
                )
                sign_for_plot = 1.0 if (np.isfinite(rho) and rho >= 0) else -1.0

                load = sign_for_plot * np.asarray(Vt_sel[pc_idx0, :], float)
                df = pd.DataFrame(
                    {
                        "gene": genes_pref,
                        "pc": pc_name,
                        "pc_idx0": int(pc_idx0),
                        "fc_score_test_H120": float(row["FC_score"]),
                        "loading_oriented": load,
                        "abs_loading": np.abs(load),
                        "raw_spearman_with_neglogS_test": float(rho),
                    }
                ).sort_values("abs_loading", ascending=False)

                df.head(100).to_csv(
                    OUTDIR / f"pc_{pc_name}_top_genes_FCtop.csv", index=False
                )

                topg = df.head(20).sort_values("loading_oriented")
                plt.figure(figsize=(7.5, 6))
                y = np.arange(len(topg))
                plt.barh(y, topg["loading_oriented"].values)
                plt.yticks(y, topg["gene"].astype(str).values)
                plt.xlabel("PC loading (oriented)")
                plt.title(f"{pc_name} (FC-top) — top gene loadings @ H={int(H_Q)} mo")
                savefig(OUTDIR / f"pc_{pc_name}_top20_loadings_FCtop.png")

                back_rows.append(df.assign(rank=np.arange(1, len(df) + 1)))

            if back_rows:
                comb = pd.concat(back_rows, ignore_index=True)
                comb.to_csv(OUTDIR / "pc_to_gene_backprojection_FCtop.csv", index=False)
    except Exception as ex:
        print(f"[C] FC-top PC→gene quickview failed: {ex}")

    # PC vs outcome correlations on TEST (FC-top PCs only)
    try:
        H_Q = 120
        if int(H_Q) not in [int(h) for h in HORIZONS]:
            H_Q = int(HORIZONS[0])

        fc120 = fc_cache.get(("test", int(H_Q)), None)
        if fc120 is None:
            fc120_path = (
                outdir_fc / f"feature_collapsing_test_H{int(H_Q)}{seed_suffix}.csv"
            )
            if fc120_path.exists():
                fc120 = pd.read_csv(fc120_path)

        pcs_top = []
        if fc120 is not None:
            pcs_top = (
                fc120.loc[fc120["feature"].astype(str).isin(list(map(str, sel_names)))]
                .head(min(8, len(sel_names)))["feature"]
                .astype(str)
                .tolist()
            )

        if pcs_top:
            pc_scores_test = pd.DataFrame(Tte[:, :], columns=pc_names)
            pc_scores_test["early_event_24mo"] = ee_te
            pc_scores_test["event_free_120mo"] = ltf_te
            corr_rows = []
            for pc in pcs_top:
                if pc not in pc_scores_test.columns:
                    continue
                x = pc_scores_test[pc].to_numpy()
                for lab, yv in [
                    ("early_event_24mo", ee_te),
                    ("event_free_120mo", ltf_te),
                ]:
                    if spearmanr is not None:
                        rho, p = spearmanr(x, yv, nan_policy="omit")
                        corr_rows.append(
                            {"pc": pc, "label": lab, "rho": float(rho), "p": float(p)}
                        )
            if corr_rows:
                pd.DataFrame(corr_rows).to_csv(
                    OUTDIR / "pc_vs_outcome_correlations_test_FCtop.csv", index=False
                )
    except Exception as ex:
        print(f"[C] PC vs outcome correlation (FC-top) failed: {ex}")

    # ======================= [run summary + return] (unchanged) =======================
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
# ------------------- Main -------------------
if __name__ == "__main__":
    seeds = MULTI_SEEDS if RUN_MULTI_SEEDS else [SEED]
    seeds = list(dict.fromkeys([int(s) for s in seeds]))  # de-dup, preserve order

    print(f"\n[Run] RUN_MULTI_SEEDS={RUN_MULTI_SEEDS} | seeds={seeds}\n")

    rows = []
    for s in seeds:
        print(f"\n[Run] ===== Starting seed {s} =====")
        rows.append(run_single(s))

    # If not multi-seed, we're done
    if not RUN_MULTI_SEEDS:
        raise SystemExit(0)

    # ---------------- Multi-seed aggregation ----------------
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "multi_seed_results.csv", index=False)
    mean = float(df["c_test"].mean())
    sd = float(df["c_test"].std())
    print(
        f"\nMulti-seed pooled test C-index: mean={mean:.3f} ± {sd:.3f} over {len(seeds)} seeds"
    )

    # -------- Aggregate C-index by group (train/test/val) --------
    def _agg_csv(pattern: str, out_name: str):
        files = [OUTDIR / pattern.format(seed=s) for s in seeds]
        frames = []
        for s, fp in zip(seeds, files):
            if not fp.exists():
                print(f"[WARN] Missing {fp}")
                continue
            d = pd.read_csv(fp)
            d["seed"] = int(s)
            frames.append(d)
        if not frames:
            return
        all_df = pd.concat(frames, ignore_index=True)
        key_cols = [
            c for c in ["split", "scope", "group_idx", "group"] if c in all_df.columns
        ]
        num_cols = [
            c
            for c in all_df.columns
            if c not in key_cols + ["seed"] and pd.api.types.is_numeric_dtype(all_df[c])
        ]
        out = (
            all_df.groupby(key_cols, dropna=False)[num_cols]
            .agg(["mean", "std"])
            .reset_index()
        )
        out.columns = [
            "_".join([c for c in col if c]) if isinstance(col, tuple) else col
            for col in out.columns
        ]
        out.to_csv(OUTDIR / out_name, index=False)

    _agg_csv(
        "c_index_train_all_and_by_group__seed{seed}.csv",
        "c_index_train_all_and_by_group__mean_sd.csv",
    )
    _agg_csv(
        "c_index_test_all_and_by_group__seed{seed}.csv",
        "c_index_test_all_and_by_group__mean_sd.csv",
    )
    if VAL_FRAC and VAL_FRAC > 0.0:
        _agg_csv(
            "c_index_val_all_and_by_group__seed{seed}.csv",
            "c_index_val_all_and_by_group__mean_sd.csv",
        )

    # -------- Aggregate FC sensitivity (pooled and by-group) --------
    fc_sens_dir = OUTDIR / "fc" / "sensitivity"
    if fc_sens_dir.exists():

        def _agg_fc(prefix: str, out_name: str):
            frames = []
            for s in seeds:
                for fp in fc_sens_dir.glob(f"{prefix}*__seed{s}.csv"):
                    d = pd.read_csv(fp)
                    d["seed"] = int(s)
                    d["_file"] = fp.name
                    frames.append(d)
            if not frames:
                return
            all_df = pd.concat(frames, ignore_index=True)
            key_cols = [
                c
                for c in ["scope", "H_months", "K", "group_idx", "group"]
                if c in all_df.columns
            ]
            num_cols = [
                c
                for c in all_df.columns
                if c not in key_cols + ["seed", "_file"]
                and pd.api.types.is_numeric_dtype(all_df[c])
            ]
            out = (
                all_df.groupby(key_cols, dropna=False)[num_cols]
                .agg(["mean", "std"])
                .reset_index()
            )
            out.columns = [
                "_".join([c for c in col if c]) if isinstance(col, tuple) else col
                for col in out.columns
            ]
            out.to_csv(fc_sens_dir / out_name, index=False)

        _agg_fc("sensitivity_summary_", "sensitivity_summary__mean_sd.csv")
        _agg_fc(
            "sensitivity_summary_bygroup_", "sensitivity_summary_bygroup__mean_sd.csv"
        )

    # -------- Aggregate alignment diagnostic (cosine similarity) --------
    archdir = OUTDIR / "archdiag"
    if archdir.exists():
        frames = []
        for s in seeds:
            fp = archdir / f"archdiag_train_cox_similarity_D_seed{s}.csv"
            if fp.exists():
                d = pd.read_csv(fp)
                d["seed"] = int(s)
                frames.append(d)

        if frames:
            all_df = pd.concat(frames, ignore_index=True)
            if "skipped" in all_df.columns:
                all_df = all_df[all_df["skipped"] == False].copy()

            if "group" in all_df.columns and "cos_sim_beta_vs_pooled" in all_df.columns:
                agg = (
                    all_df.groupby(["group", "group_idx"], dropna=False)[
                        "cos_sim_beta_vs_pooled"
                    ]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )
                agg.rename(
                    columns={
                        "mean": "cos_sim_mean",
                        "std": "cos_sim_sd",
                        "count": "n_seeds_used",
                    },
                    inplace=True,
                )
                agg.to_csv(
                    archdir / "archdiag_train_cox_similarity_D__mean_sd.csv",
                    index=False,
                )

    print(
        "\n[Multi-seed] Saved mean±SD summaries for C-index, FC sensitivity, and alignment diagnostic."
    )
