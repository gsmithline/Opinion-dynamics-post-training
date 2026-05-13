"""Per-feature label-spread diagnostic for the sliver-training experiment.

For each candidate demographic feature in the LABELED subset, compute:
  - subgroup sizes
  - per-subgroup mean of y_label
  - spread (max - min) of subgroup means across non-tiny subgroups

Features with large spread are good candidates for sliver-training: training
on one subgroup creates a distinguishable signal in the spillover plot.
Features with tiny spread mean smoking labels are roughly uniform across that
feature, so the spillover experiment would be uninformative there.

Also reports the unlabeled-side feature distribution and per-subgroup
unlabeled count, so we can later pick the matching demographically-similar
unlabeled group S' (used in the spillover readout).

Run from repo root:
    python tools/feature_spread_diagnostic.py
"""
import pickle
from collections import Counter

import numpy as np
import pandas as pd


# pandas 2.2 unpickling shim (same pattern as run_free_fj.py)
_orig_sd_init = pd.StringDtype.__init__
def _sd_init(self, storage=None, na_value=None, *args, **kwargs):
    try:
        _orig_sd_init(self, storage=storage)
    except TypeError:
        _orig_sd_init(self)
pd.StringDtype.__init__ = _sd_init
from pandas.core.arrays.string_ import StringArray
_orig_ss = getattr(StringArray, "__setstate__", None)
def _ss(self, state):
    try:
        if _orig_ss:
            return _orig_ss(self, state)
    except Exception:
        pass
    def _find(x):
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, (tuple, list)):
            for e in x:
                r = _find(e)
                if r is not None:
                    return r
        return None
    arr = _find(state)
    if arr is None:
        raise ValueError("no ndarray in state")
    StringArray.__init__(self, arr.astype(object))
StringArray.__setstate__ = _ss


TARGET = "relation_to_smoking"
PROFILES = f"pokec_dataset/lcc_profiles_{TARGET}.pk"
YLAB = "pokec_dataset/parametric_params/y_label2163.pk"
YUNL = "pokec_dataset/parametric_params/y_unlabel_label2163.pk"

MIN_GROUP_SIZE = 30   # ignore tiny subgroups when ranking by spread
MAX_CARDINALITY = 2000  # lifted from 50; per-group MIN filter does real work
# Features the LM actually sees in the prompt (llm_predictor.py PROMPT_COLS).
PROMPT_COLS = {"age", "gender", "relation_to_alcohol"}


def _is_hashable_col(s: pd.Series) -> bool:
    try:
        _ = s.head(20).apply(lambda x: hash(x) if x is not None else 0)
        return True
    except Exception:
        return False


def summarize_feature(df_lab: pd.DataFrame, y: np.ndarray, col: str,
                      df_unl: pd.DataFrame) -> dict | None:
    sub = pd.DataFrame({col: df_lab[col].values, "_y": y})
    grouped = sub.groupby(col, dropna=False)["_y"].agg(["count", "mean"]).reset_index()
    grouped = grouped[grouped["count"] >= MIN_GROUP_SIZE].copy()
    if len(grouped) < 2:
        return None
    grouped = grouped.sort_values("mean")
    means = grouped["mean"].values.astype(float)
    counts = grouped["count"].values.astype(float)
    weights = counts / counts.sum()
    weighted_mean = float(np.average(means, weights=weights))
    weighted_var = float(np.average((means - weighted_mean) ** 2, weights=weights))
    # Unlabeled-side counts per subgroup (for picking S' later).
    unl_counts = df_unl[col].value_counts(dropna=False)
    grouped["unl_count"] = grouped[col].map(unl_counts).fillna(0).astype(int)
    return {
        "col": col,
        "n_groups": int(len(grouped)),
        "spread": float(means.max() - means.min()),
        "weighted_std": float(np.sqrt(weighted_var)),
        "top_groups": grouped.tail(3).to_dict("records"),
        "bottom_groups": grouped.head(3).to_dict("records"),
    }


def main() -> None:
    df = pickle.load(open(PROFILES, "rb"))
    y_lab = pickle.load(open(YLAB, "rb"))
    y_unl = pickle.load(open(YUNL, "rb"))

    n_lab = len(y_lab)
    n_unl = len(y_unl)
    n_tot = n_lab + n_unl
    assert len(df) == n_tot, f"len(df)={len(df)} != n_lab+n_unl={n_tot}"

    df_lab = df.iloc[:n_lab].reset_index(drop=True)
    df_unl = df.iloc[n_lab:].reset_index(drop=True)
    y_lab_arr = np.asarray(y_lab, dtype=float)
    y_unl_arr = np.asarray(y_unl, dtype=float)

    print(f"# rows  total={n_tot}  labeled={n_lab}  unlabeled={n_unl}")
    print(f"# mean y_label   = {y_lab_arr.mean():.4f}   std={y_lab_arr.std():.4f}")
    print(f"# mean y_unlabel = {y_unl_arr.mean():.4f}   std={y_unl_arr.std():.4f}")
    print(f"# columns in df: {list(df.columns)}")
    print()

    # Pick candidate features: cardinality 2..50, hashable.
    cand: list[tuple[str, int]] = []
    for col in df.columns:
        if col in ("user_id", TARGET):
            continue
        try:
            s = df_lab[col]
            if not _is_hashable_col(s):
                continue
            nu = s.nunique(dropna=False)
        except Exception:
            continue
        if 2 <= nu <= MAX_CARDINALITY:
            cand.append((col, int(nu)))
    cand.sort(key=lambda x: x[1])
    print(f"# candidate features (cardinality 2..50): {len(cand)}")
    for c, n in cand:
        print(f"   {c}  (nunique={n})")
    print()

    results = []
    for col, _ in cand:
        try:
            r = summarize_feature(df_lab, y_lab_arr, col, df_unl)
        except Exception as e:
            print(f"   SKIP {col}: {e}")
            continue
        if r is None:
            continue
        results.append(r)

    results.sort(key=lambda r: r["spread"], reverse=True)
    print("# RANKED BY SPREAD (max subgroup mean - min subgroup mean), labeled side")
    print(f"# global mean y_label = {y_lab_arr.mean():.4f}")
    print()
    for r in results:
        in_prompt = "PROMPT" if r["col"] in PROMPT_COLS else "unobserved"
        print(f"=== {r['col']}  [{in_prompt}]  n_groups>={MIN_GROUP_SIZE}: "
              f"{r['n_groups']}  spread={r['spread']:.4f}  "
              f"w_std={r['weighted_std']:.4f} ===")
        print("  low-mean subgroups (potential S, pulls non-S down):")
        for g in r["bottom_groups"]:
            print(f"    {g}")
        print("  high-mean subgroups (potential S, pulls non-S up):")
        for g in r["top_groups"]:
            print(f"    {g}")
        print()


if __name__ == "__main__":
    main()
