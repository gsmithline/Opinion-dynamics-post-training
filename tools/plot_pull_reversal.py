"""Symmetric pull-reversal test on the UNLABELED side.

For a given feature, compare sliver-on-S_high vs sliver-on-S_low:
  - If sliver_S_high pulls mu_V\\S UP and sliver_S_low pulls mu_V\\S DOWN,
    that is the directional population-steering signal.
  - If both move the same way, it is collapse-to-something, not pull.
  - If both flat, no detectable effect at this beta.

Plots mu_V\\S(t) - mu_V\\S(0) (centered at zero) so direction is visible.
Optionally overlays baseline-track lines for reference.

CLI:
  python tools/plot_pull_reversal.py \\
      --feature age \\
      --high age_high3 --low age_low3 \\
      --betas 0 1 10 \\
      --include-track \\
      --out plots/pull_reversal_age.png
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

# pandas 2.2 unpickling shim
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

from subgroup_helper import Subgroup

TARGET = "relation_to_smoking"
DEFAULT_PROFILES = f"pokec_dataset/lcc_profiles_{TARGET}.pk"
DEFAULT_YLAB = "pokec_dataset/parametric_params/y_label2163.pk"
DEFAULT_YUNL = "pokec_dataset/parametric_params/y_unlabel_label2163.pk"


def _beta_to_tag(direction: str, subgroup: str, beta: float, kind: str) -> str:
    """direction: 'sliver' or 'track'. kind matches the cluster naming."""
    if beta == 0.0:
        suffix = "sft_b0"
    else:
        b_int = int(beta) if beta == int(beta) else beta
        suffix = f"sftkl_b{b_int}"
    return f"fj_{direction}_{subgroup}_{suffix}_egta"


def _load_traj(results_dir: Path, tag: str) -> np.ndarray | None:
    p = results_dir / f"llm_{tag}_trajectory.pk"
    if not p.exists():
        return None
    return pickle.load(open(p, "rb"))


def _mu_vbackslash_s(traj: np.ndarray, mask_full: np.ndarray,
                     n_lab: int) -> np.ndarray:
    """mean opinion of (V\\S intersect V_unlabeled) over rounds."""
    not_s = ~mask_full
    unlabeled = np.zeros(traj.shape[0], dtype=bool)
    unlabeled[n_lab:] = True
    sel = not_s & unlabeled
    if not sel.any():
        raise RuntimeError("no unlabeled non-S rows; mask wrong?")
    return traj[sel].mean(axis=0)


def _mu_ref(traj: np.ndarray, mask_ref: np.ndarray, n_lab: int) -> np.ndarray:
    """Mean opinion of (R intersect V_unlabeled) over rounds.
    Same fixed R used for every trajectory so cross-run comparison is apples-to-apples."""
    unlabeled = np.zeros(traj.shape[0], dtype=bool)
    unlabeled[n_lab:] = True
    sel = mask_ref & unlabeled
    if not sel.any():
        raise RuntimeError("no unlabeled rows in reference set")
    return traj[sel].mean(axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature", required=True, help="label for the figure (e.g., age)")
    p.add_argument("--high", required=True, help="preset name for high-mean S (e.g., age_high3)")
    p.add_argument("--low", required=True, help="preset name for low-mean S (e.g., age_low3)")
    p.add_argument("--high-tag", default=None,
                   help="Override: the subgroup substring used in run tags for HIGH. "
                        "Defaults to --high. Use when condor tags differ from preset names, "
                        "e.g. --high alc_abstinent --high-tag alc_abs.")
    p.add_argument("--low-tag", default=None,
                   help="Override for LOW tag substring (defaults to --low).")
    p.add_argument("--betas", nargs="+", type=float, required=True,
                   help="Beta values to plot, one panel per beta")
    p.add_argument("--ref", default=None,
                   help="Optional fixed readout population (preset or inline spec, "
                        "e.g. age_mid). When set, plot mu_R(t)-mu_R(0) using this "
                        "fixed R for both sliver runs instead of each run's own V\\S. "
                        "Avoids the confound that V\\S_high and V\\S_low are different sets.")
    p.add_argument("--include-track", action="store_true",
                   help="Overlay baseline-track lines if their trajectories exist")
    p.add_argument("--results-dir", default="pokec_dataset/results")
    p.add_argument("--profiles", default=DEFAULT_PROFILES)
    p.add_argument("--y-lab", default=DEFAULT_YLAB)
    p.add_argument("--y-unl", default=DEFAULT_YUNL)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    df = pickle.load(open(args.profiles, "rb"))
    y_lab = pickle.load(open(args.y_lab, "rb"))
    y_unl = pickle.load(open(args.y_unl, "rb"))
    n_lab = len(y_lab)
    n_unl = len(y_unl)
    if len(df) != n_lab + n_unl:
        raise RuntimeError(f"df len {len(df)} != n_lab+n_unl ({n_lab}+{n_unl})")

    sg_high = Subgroup.from_spec(args.high)
    sg_low = Subgroup.from_spec(args.low)
    mask_high = sg_high.compute_mask(df)
    mask_low = sg_low.compute_mask(df)

    n_unl_not_high = int(((~mask_high)[n_lab:]).sum())
    n_unl_not_low = int(((~mask_low)[n_lab:]).sum())
    print(f"[pull_reversal] feature={args.feature}  unlabeled |V\\S_high|={n_unl_not_high}  |V\\S_low|={n_unl_not_low}")

    mask_ref = None
    n_unl_ref = None
    if args.ref is not None:
        sg_ref = Subgroup.from_spec(args.ref)
        mask_ref = sg_ref.compute_mask(df)
        n_unl_ref = int(mask_ref[n_lab:].sum())
        print(f"[pull_reversal] ref={args.ref}  unlabeled |R|={n_unl_ref}")

    results_dir = Path(args.results_dir)
    n_panels = len(args.betas)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.4), sharey=True)
    if n_panels == 1:
        axes = [axes]

    high_tag_str = args.high_tag if args.high_tag else args.high
    low_tag_str = args.low_tag if args.low_tag else args.low

    missing = []
    for ax, beta in zip(axes, args.betas):
        tag_sh = _beta_to_tag("sliver", high_tag_str, beta, "sliver")
        tag_sl = _beta_to_tag("sliver", low_tag_str, beta, "sliver")
        tag_th = _beta_to_tag("track", high_tag_str, beta, "track")
        tag_tl = _beta_to_tag("track", low_tag_str, beta, "track")

        traj_sh = _load_traj(results_dir, tag_sh)
        traj_sl = _load_traj(results_dir, tag_sl)
        for tag, t in [(tag_sh, traj_sh), (tag_sl, traj_sl)]:
            if t is None:
                missing.append(tag)
                continue

        def _readout(traj, run_mask):
            if mask_ref is not None:
                return _mu_ref(traj, mask_ref, n_lab)
            return _mu_vbackslash_s(traj, run_mask, n_lab)

        if traj_sh is not None:
            mu_sh = _readout(traj_sh, mask_high)
            lbl_h = f"sliver_{args.high}"
            lbl_h += f"  (n_R={n_unl_ref})" if mask_ref is not None else f"  (n_V\\S={n_unl_not_high})"
            ax.plot(np.arange(len(mu_sh)), mu_sh - mu_sh[0],
                    color="tab:blue", linewidth=2.0, label=lbl_h)
        if traj_sl is not None:
            mu_sl = _readout(traj_sl, mask_low)
            lbl_l = f"sliver_{args.low}"
            lbl_l += f"  (n_R={n_unl_ref})" if mask_ref is not None else f"  (n_V\\S={n_unl_not_low})"
            ax.plot(np.arange(len(mu_sl)), mu_sl - mu_sl[0],
                    color="tab:red", linewidth=2.0, label=lbl_l)

        if args.include_track:
            traj_th = _load_traj(results_dir, tag_th)
            traj_tl = _load_traj(results_dir, tag_tl)
            if traj_th is not None:
                mu_th = _readout(traj_th, mask_high)
                ax.plot(np.arange(len(mu_th)), mu_th - mu_th[0],
                        color="tab:blue", linewidth=1.2, linestyle="--", alpha=0.6,
                        label=f"track_{args.high}")
            if traj_tl is not None:
                mu_tl = _readout(traj_tl, mask_low)
                ax.plot(np.arange(len(mu_tl)), mu_tl - mu_tl[0],
                        color="tab:red", linewidth=1.2, linestyle="--", alpha=0.6,
                        label=f"track_{args.low}")

        ax.axhline(0, color="gray", linewidth=0.8, alpha=0.6)
        ax.set_title(f"beta = {beta}")
        ax.set_xlabel("round")
        ylab = "mu_R(t) - mu_R(0)" if mask_ref is not None else "mu_V\\S(t) - mu_V\\S(0)"
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)

    axes[0].legend(loc="best", fontsize=8)
    readout_str = f"fixed R = {args.ref}" if mask_ref is not None else "each run's own V\\S"
    fig.suptitle(f"Pull-reversal test on unlabeled  (readout: {readout_str}),  feature = {args.feature}",
                 y=1.02)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=140)
    print(f"wrote {out}")
    if missing:
        print(f"[pull_reversal] missing trajectories (skipped): {missing}")


if __name__ == "__main__":
    main()
