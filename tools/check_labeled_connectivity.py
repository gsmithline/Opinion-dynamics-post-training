"""Diagnostic: is our labeled subset (first 80% of df rows) a connected subgraph?

The Wu et al. paper's V' is a connected subgraph; their spillover theorem
(Proposition 4) assumes that. Our split is positional (df.iloc[:n]), not
graph-aware. This script checks whether the positional split happens to be
mostly connected anyway.
"""
import pickle

import networkx as nx
import numpy as np
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
    df = pickle.load(open(DF_PATH, "rb"))
    G  = pickle.load(open(G_PATH,  "rb"))
    n_total = len(df)
    n = int(n_total * 0.8)
    print(f"total agents: {n_total}")
    print(f"labeled (first {n}): {n} agents")
    print(f"unlabeled (remaining): {n_total - n} agents")
    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    labeled_ids = set(df["user_id"].iloc[:n].tolist())
    unlabeled_ids = set(df["user_id"].iloc[n:].tolist())

    sub_l = G.subgraph(labeled_ids)
    sub_u = G.subgraph(unlabeled_ids)

    comps_l = sorted([len(c) for c in nx.connected_components(sub_l)], reverse=True)
    comps_u = sorted([len(c) for c in nx.connected_components(sub_u)], reverse=True)

    print()
    print(f"labeled subgraph: {sub_l.number_of_edges()} internal edges")
    print(f"  connected components: {len(comps_l)}")
    print(f"  largest 10 component sizes: {comps_l[:10]}")
    print(f"  fraction of labeled in largest component: {comps_l[0] / len(labeled_ids):.4f}")
    print(f"  fraction of labeled in top-3 components:  {sum(comps_l[:3]) / len(labeled_ids):.4f}")
    print()
    print(f"unlabeled subgraph: {sub_u.number_of_edges()} internal edges")
    print(f"  connected components: {len(comps_u)}")
    print(f"  largest 10 component sizes: {comps_u[:10]}")
    print(f"  fraction of unlabeled in largest component: {comps_u[0] / len(unlabeled_ids):.4f}")
    print()

    # Cross-edges: how connected ARE labeled-to-unlabeled? This matters because
    # unlabeled agents only feel the LM directly; labeled feel LM via these
    # cross-edges (peer propagation from unlabeled-with-LM-prediction to labeled).
    cross_edges = sum(1 for u, v in G.edges()
                      if (u in labeled_ids and v in unlabeled_ids)
                      or (u in unlabeled_ids and v in labeled_ids))
    total_edges = G.number_of_edges()
    print(f"labeled-unlabeled cross-edges: {cross_edges}  ({cross_edges / total_edges:.4f} of all edges)")
    print()

    # Verdict.
    largest_frac = comps_l[0] / len(labeled_ids)
    if largest_frac >= 0.95:
        verdict = "OK: labeled is essentially one connected component."
    elif largest_frac >= 0.80:
        verdict = "MARGINAL: labeled has one dominant component but with significant fragmentation."
    else:
        verdict = "FRAGMENTED: labeled splits into many small components. Rerunning with a graph-aware split is advisable."
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
