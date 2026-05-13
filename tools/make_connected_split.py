"""BFS from a seed node to produce a connected labeled subset of size ~80%.

Saves a numpy int64 array of row indices (into df.iloc) for the labeled subset.
Runner scripts pick this up via LABELED_SPLIT env var.

Usage:
    python tools/make_connected_split.py
    # or with custom params:
    python tools/make_connected_split.py --frac 0.8 --seed-strategy highest-degree \
        --out pokec_dataset/parametric_params/connected_split.pkl
"""
from __future__ import annotations

import argparse
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd

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

DF_PATH = "pokec_dataset/lcc_profiles_relation_to_smoking.pk"
G_PATH  = "pokec_dataset/lcc_graph_relation_to_smoking.pk"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.8,
                    help="Target fraction of nodes in labeled subset. Default 0.8.")
    ap.add_argument("--seed-strategy", choices=["highest-degree", "random"],
                    default="highest-degree",
                    help="How to pick the BFS seed. Default: highest-degree node.")
    ap.add_argument("--seed-rng", type=int, default=0,
                    help="RNG seed for random seed-strategy. Default 0.")
    ap.add_argument("--out", type=Path,
                    default=Path("pokec_dataset/parametric_params/connected_split.pkl"))
    args = ap.parse_args()

    df = pickle.load(open(DF_PATH, "rb"))
    G  = pickle.load(open(G_PATH,  "rb"))
    n_total = len(df)
    n_target = int(round(n_total * args.frac))
    print(f"total agents: {n_total}, target labeled: {n_target}")

    # Map user_id -> row index in df.
    user_ids = df["user_id"].tolist()
    user_id_to_row = {uid: i for i, uid in enumerate(user_ids)}
    G_in_df = G.subgraph(set(user_ids))  # restrict to agents we have rows for
    print(f"graph restricted to df agents: {G_in_df.number_of_nodes()} nodes, "
          f"{G_in_df.number_of_edges()} edges")

    # Pick seed.
    if args.seed_strategy == "highest-degree":
        degs = dict(G_in_df.degree())
        seed = max(degs, key=degs.get)
        print(f"seed (highest-degree): {seed} with degree {degs[seed]}")
    else:
        rng = np.random.default_rng(args.seed_rng)
        seed = user_ids[int(rng.integers(0, len(user_ids)))]
        print(f"seed (random): {seed}")

    # BFS until we've visited n_target nodes.
    visited = []
    visited_set = set()
    queue = deque([seed])
    while queue and len(visited) < n_target:
        node = queue.popleft()
        if node in visited_set:
            continue
        visited_set.add(node)
        visited.append(node)
        for nb in G_in_df.neighbors(node):
            if nb not in visited_set:
                queue.append(nb)
    print(f"BFS reached: {len(visited)} nodes")

    if len(visited) < n_target:
        # Graph component smaller than target; pick another seed in unreached set.
        remaining = [uid for uid in user_ids if uid not in visited_set]
        while remaining and len(visited) < n_target:
            seed2 = remaining[0]
            queue = deque([seed2])
            while queue and len(visited) < n_target:
                node = queue.popleft()
                if node in visited_set:
                    continue
                visited_set.add(node)
                visited.append(node)
                for nb in G_in_df.neighbors(node):
                    if nb not in visited_set:
                        queue.append(nb)
            remaining = [uid for uid in user_ids if uid not in visited_set]
        print(f"after multi-seed BFS: {len(visited)} nodes")

    # Convert to df row indices (sorted; order of indices doesn't matter for split semantics).
    labeled_rows = sorted([user_id_to_row[uid] for uid in visited[:n_target]])
    labeled_rows = np.array(labeled_rows, dtype=np.int64)

    # Verify connectivity of the chosen labeled subset.
    sub = G_in_df.subgraph(set(visited[:n_target]))
    comps = sorted([len(c) for c in nx.connected_components(sub)], reverse=True)
    print(f"labeled subgraph: {len(comps)} components, largest sizes: {comps[:5]}")
    print(f"fraction of labeled in largest component: {comps[0] / len(labeled_rows):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(labeled_rows, f)
    print(f"wrote {args.out}  ({labeled_rows.size} row indices)")


if __name__ == "__main__":
    main()
