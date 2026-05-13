"""Platform-free FJ baseline: pure peer dynamics from innate, no LM, no platform.

Equivalent to setting platform_sus = 0 in the standard FJ + platform loop.
Since x_zero = innate in every round, the trajectory converges to the FJ
equilibrium with innate as the stubbornness anchor after the first inner-loop
fixed point, and remains there.

In our current FJ formulation this also coincides with the 'oracle predictor'
baseline (predict innate exactly), because oracle predictions = innate so
x_zero = innate in both cases.

Output: pokec_dataset/results/llm_<RUN_TAG>_trajectory.pk  (shape (N, T+1))
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

# pandas 2.2 shim for older pickled profiles (same as run_untuned_llm.py).
_orig_sd_init = pd.StringDtype.__init__
def _sd_init(self, storage=None, na_value=None, *args, **kwargs):
    try: _orig_sd_init(self, storage=storage)
    except TypeError: _orig_sd_init(self)
pd.StringDtype.__init__ = _sd_init
from pandas.core.arrays.string_ import StringArray
_orig_ss = getattr(StringArray, "__setstate__", None)
def _ss(self, state):
    try:
        if _orig_ss: return _orig_ss(self, state)
    except Exception: pass
    def _find(x):
        if isinstance(x, np.ndarray): return x
        if isinstance(x, (tuple, list)):
            for e in x:
                r = _find(e)
                if r is not None: return r
        return None
    arr = _find(state)
    if arr is None: raise ValueError("no ndarray in state")
    StringArray.__init__(self, arr.astype(object))
StringArray.__setstate__ = _ss

RUN_TAG  = os.environ.get("RUN_TAG", "fj_free_egta")
OUT_DIR  = os.environ.get("OUT_DIR", "pokec_dataset/results")
RETRAIN_T = int(os.environ.get("RETRAIN_T", 30))
FJ_K = 100

TARGET = "relation_to_smoking"
PROFILES = "pokec_dataset/lcc_profiles_" + TARGET + ".pk"
GRAPH    = "pokec_dataset/lcc_graph_" + TARGET + ".pk"
YLAB     = "pokec_dataset/parametric_params/y_label2163.pk"
YUNL     = "pokec_dataset/parametric_params/y_unlabel_label2163.pk"
PEER_PK  = "pokec_dataset/parametric_params/hetero_peer_sus2163.pkl"


def main() -> None:
    df       = pickle.load(open(PROFILES, "rb"))
    network  = pickle.load(open(GRAPH,    "rb"))
    y_lab    = pickle.load(open(YLAB,     "rb"))
    y_unl    = pickle.load(open(YUNL,     "rb"))
    peer_sus = pickle.load(open(PEER_PK,  "rb"))

    innate = np.array(y_lab + y_unl)
    agent_num = len(innate)
    print(f"[free] agent_num={agent_num}  T={RETRAIN_T}  tag={RUN_TAG}")

    nodelist = df["user_id"].values
    adj_mat = nx.to_numpy_array(network, nodelist=nodelist)
    weight_mat = adj_mat.copy()
    degs_inv = 1.0 / np.sum(adj_mat, axis=0)
    degs_inv[np.isinf(degs_inv)] = 0.0
    degs_inv[degs_inv > 1.1] = 0.0
    W_norm = weight_mat * degs_inv[:, None]

    # platform_sus = 0 -> x_zero = innate always; no platform signal needed.
    traj = np.zeros((agent_num, RETRAIN_T + 1))
    traj[:, 0] = innate.copy()

    for t in range(RETRAIN_T):
        x_zero = innate.copy()
        x_temp = x_zero.copy()
        for _ in range(FJ_K):
            x_temp = peer_sus * x_zero + (1.0 - peer_sus) * (W_norm @ x_temp)
        traj[:, t + 1] = x_temp
        if (t + 1) % 5 == 0 or t == 0:
            print(f"  t={t+1:2d}  mean={x_temp.mean():.4f}  std={x_temp.std():.4f}")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(OUT_DIR) / f"llm_{RUN_TAG}_trajectory.pk"
    pickle.dump(traj, open(out_path, "wb"))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
