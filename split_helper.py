"""Labeled/unlabeled split helper.

Controlled by env var LABELED_SPLIT. If unset, the default positional split
(first 80% of df rows = labeled) is used and everything is identity.

If LABELED_SPLIT points to a pickle of int64 row indices, those rows form the
labeled subset; the complement is unlabeled. The runners apply this once at
startup to reorder df, innate, peer_sus consistently so that labeled rows come
first (rows 0..n-1) and unlabeled follow (rows n..N-1). All downstream code
that uses df.iloc[:n] / arr[:n] / arr[n:] continues to work unchanged.
"""
from __future__ import annotations

import os
import pickle

import numpy as np

_NEW_ORDER_CACHE = None
_TOTAL_CACHE = None
_N_LABELED_CACHE = None


def _load() -> tuple[np.ndarray, int, int] | None:
    path = os.environ.get("LABELED_SPLIT", "").strip()
    if not path:
        return None
    with open(path, "rb") as f:
        labeled_rows = pickle.load(f)
    labeled_rows = np.asarray(labeled_rows, dtype=np.int64)
    return labeled_rows


def get_new_order(total: int) -> np.ndarray:
    """Permutation of length `total` putting labeled rows first.
    Identity if LABELED_SPLIT is unset. Cached after first call."""
    global _NEW_ORDER_CACHE, _TOTAL_CACHE, _N_LABELED_CACHE
    if _NEW_ORDER_CACHE is not None and _TOTAL_CACHE == total:
        return _NEW_ORDER_CACHE
    loaded = _load()
    if loaded is None:
        order = np.arange(total, dtype=np.int64)
        n = int(total * 0.8)
    else:
        labeled_rows = loaded
        labeled_set = set(int(x) for x in labeled_rows)
        unlabeled_rows = np.array([i for i in range(total) if i not in labeled_set],
                                  dtype=np.int64)
        order = np.concatenate([labeled_rows, unlabeled_rows])
        n = int(labeled_rows.size)
        print(f"[split_helper] LABELED_SPLIT active. n_labeled={n}, n_unlabeled={total - n}")
    _NEW_ORDER_CACHE = order
    _TOTAL_CACHE = total
    _N_LABELED_CACHE = n
    return order


def get_n_labeled(total: int) -> int:
    """Number of labeled agents under the active split."""
    if _N_LABELED_CACHE is None or _TOTAL_CACHE != total:
        get_new_order(total)
    return _N_LABELED_CACHE


def split_suffix() -> str:
    """Filename suffix to disambiguate caches between positional and connected splits."""
    if os.environ.get("LABELED_SPLIT", "").strip():
        return "_connected"
    return ""


def reorder_array(arr: np.ndarray) -> np.ndarray:
    """Apply the active permutation to a 1-D array of length total."""
    arr = np.asarray(arr)
    return arr[get_new_order(arr.shape[0])]
