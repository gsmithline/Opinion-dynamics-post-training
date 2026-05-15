"""Sliver-training: restrict SFT to a feature-defined subgroup S of the labeled set.

Class-based interface. A `Subgroup` encapsulates a feature filter and exposes:
  - mask computation
  - SFT training-set filter
  - sidecar persistence (saved next to the trajectory .pk)
  - per-subgroup trajectory means (for plot scripts)

Two ways to specify a subgroup via the SFT_SUBGROUP env var:

    SFT_SUBGROUP=age_young
        Look up a curated preset from PRESETS below.

    SFT_SUBGROUP=col:region;vals:zilinsky kraj, namestovo|zilinsky kraj, martin
        Inline ad-hoc spec. No code edit needed. Format:
            col:<column_name>;vals:<v1>|<v2>|<v3>
        The full spec string becomes the tag.

When SFT_SUBGROUP is unset, Subgroup.from_env() returns None and every code
path that consults it falls back to byte-identical pre-existing behavior.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd


# --- Subgroup ----------------------------------------------------------------

@dataclass
class Subgroup:
    tag: str
    column: Optional[str] = None
    values: Optional[set] = None
    custom_fn: Optional[Callable[[pd.DataFrame], np.ndarray]] = None

    # Lazy mask cache (keyed by len(df), so a re-ordered df invalidates).
    _mask_cache: Optional[np.ndarray] = field(default=None, repr=False)
    _mask_len: Optional[int] = field(default=None, repr=False)

    def compute_mask(self, df: pd.DataFrame) -> np.ndarray:
        """Bool mask of len(df). True = row is in this subgroup."""
        if self._mask_cache is not None and self._mask_len == len(df):
            return self._mask_cache
        if self.custom_fn is not None:
            mask = np.asarray(self.custom_fn(df)).astype(bool)
        elif self.column is not None and self.values is not None:
            mask = df[self.column].isin(self.values).to_numpy().astype(bool)
        else:
            raise ValueError(f"Subgroup {self.tag!r} has no filter (need custom_fn or column+values)")
        if mask.shape != (len(df),):
            raise RuntimeError(
                f"mask shape {mask.shape} != (len(df)={len(df)},) for tag={self.tag!r}"
            )
        self._mask_cache = mask
        self._mask_len = len(df)
        return mask

    def apply_to_labeled(self, df_labeled: pd.DataFrame, y_labeled):
        """Filter (df_labeled, y_labeled) to rows in S.
        Caller is responsible for passing the LABELED subset only."""
        mask = self.compute_mask(df_labeled)
        y_arr = np.asarray(y_labeled)
        if y_arr.shape[0] != len(df_labeled):
            raise RuntimeError(
                f"y_labeled len {y_arr.shape[0]} != df_labeled len {len(df_labeled)}"
            )
        return df_labeled.loc[mask].reset_index(drop=True), y_arr[mask]

    def save_sidecar(self, out_dir, run_tag: str, df_full: pd.DataFrame) -> Path:
        """Persist the FULL-population mask alongside the trajectory."""
        mask = self.compute_mask(df_full)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{run_tag}_subgroup_mask.pk"
        with open(out, "wb") as f:
            pickle.dump({
                "tag": self.tag,
                "mask": mask,
                "column": self.column,
                "values": sorted(self.values) if self.values else None,
            }, f)
        print(f"[subgroup] tag={self.tag}  |S|={int(mask.sum())} / {len(df_full)}  -> {out}")
        return out

    def split_means(self, traj: np.ndarray, mask: np.ndarray) -> dict:
        """Given (N, T+1) trajectory and length-N row mask, return
            {mu_S, mu_not_S, n_S, n_not_S, gap = mu_S - mu_not_S}.
        Row order of traj must match row order of mask."""
        if traj.shape[0] != mask.shape[0]:
            raise RuntimeError(f"traj rows {traj.shape[0]} != mask len {mask.shape[0]}")
        not_mask = ~mask
        m_s = traj[mask].mean(axis=0) if mask.any() else np.full(traj.shape[1], np.nan)
        m_ns = traj[not_mask].mean(axis=0) if not_mask.any() else np.full(traj.shape[1], np.nan)
        return {
            "mu_S": m_s,
            "mu_not_S": m_ns,
            "n_S": int(mask.sum()),
            "n_not_S": int(not_mask.sum()),
            "gap": m_s - m_ns,
        }

    # --- constructors --------------------------------------------------------

    @classmethod
    def from_env(cls, var: str = "SFT_SUBGROUP") -> Optional["Subgroup"]:
        """Read env var `var` (default SFT_SUBGROUP). Returns None when unset."""
        spec = os.environ.get(var, "").strip()
        if not spec:
            return None
        return cls.from_spec(spec)

    @classmethod
    def from_spec(cls, spec: str) -> "Subgroup":
        if spec in PRESETS:
            return PRESETS[spec]
        if "col:" in spec and "vals:" in spec:
            return cls._parse_inline(spec)
        raise ValueError(
            f"SFT_SUBGROUP={spec!r} not recognized. Either:\n"
            f"  - preset: one of {sorted(PRESETS)}\n"
            f"  - inline: col:<column>;vals:<v1>|<v2>|<v3>"
        )

    @staticmethod
    def _parse_inline(spec: str) -> "Subgroup":
        parts = {}
        for chunk in spec.split(";"):
            if ":" not in chunk:
                continue
            k, v = chunk.split(":", 1)
            parts[k.strip()] = v
        col = parts.get("col", "").strip()
        vals_raw = parts.get("vals", "")
        vals = {v.strip() for v in vals_raw.split("|") if v.strip()}
        if not col or not vals:
            raise ValueError(f"inline subgroup spec missing col or vals: {spec!r}")
        return Subgroup(tag=spec, column=col, values=vals)

    @classmethod
    def from_sidecar(cls, path) -> "Subgroup":
        """Reconstruct a Subgroup from a saved sidecar (cached mask + tag)."""
        with open(path, "rb") as f:
            d = pickle.load(f)
        sg = cls(
            tag=d["tag"],
            column=d.get("column"),
            values=set(d["values"]) if d.get("values") else None,
        )
        sg._mask_cache = np.asarray(d["mask"]).astype(bool)
        sg._mask_len = sg._mask_cache.shape[0]
        return sg


# --- preset registry ---------------------------------------------------------
#
# To add a new preset, append one entry below. No other file needs to change.

def _age_young_fn(df: pd.DataFrame) -> np.ndarray:
    a = pd.to_numeric(df["age"], errors="coerce")
    return ((a >= 14) & (a <= 17)).to_numpy()


def _age_older_fn(df: pd.DataFrame) -> np.ndarray:
    a = pd.to_numeric(df["age"], errors="coerce")
    return (a >= 18).to_numpy()


# Top-3 highest-mean age buckets (14, 15, 17). Excludes age 16 which has a
# below-average mean and dilutes the sliver signal.
def _age_high3_fn(df: pd.DataFrame) -> np.ndarray:
    a = pd.to_numeric(df["age"], errors="coerce")
    return a.isin([14, 15, 17]).to_numpy()


# Bottom-3 lowest-mean age buckets (16, 21, 23). Symmetric counterpart to
# age_high3 for testing pull-direction reversal (sliver on low-mean S should
# pull V\S DOWN, mirroring sliver on high-mean S pulling V\S UP).
def _age_low3_fn(df: pd.DataFrame) -> np.ndarray:
    a = pd.to_numeric(df["age"], errors="coerce")
    return a.isin([16, 21, 23]).to_numpy()


PRESETS: dict[str, Subgroup] = {
    "age_young": Subgroup(tag="age_young", custom_fn=_age_young_fn),
    "age_older": Subgroup(tag="age_older", custom_fn=_age_older_fn),
    "age_high3": Subgroup(tag="age_high3", custom_fn=_age_high3_fn),
    "age_low3": Subgroup(tag="age_low3", custom_fn=_age_low3_fn),
    "region_high3": Subgroup(
        tag="region_high3",
        column="region",
        values={
            "zilinsky kraj, namestovo",
            "trenciansky kraj, trencin",
            "zilinsky kraj, martin",
        },
    ),
    "region_low3": Subgroup(
        tag="region_low3",
        column="region",
        values={
            "banskobystricky kraj, banska bystrica",
            "kosicky kraj, michalovce",
            "nitriansky kraj, nitra",
        },
    ),
}


# --- thin module-level helpers ----------------------------------------------

def subgroup_tag() -> str:
    """Return the SFT_SUBGROUP env value verbatim, or '' when unset.
    Kept for callers that just want to detect activation cheaply."""
    return os.environ.get("SFT_SUBGROUP", "").strip()
