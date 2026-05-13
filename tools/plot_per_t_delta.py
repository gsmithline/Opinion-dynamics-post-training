"""Plot per-snapshot deviation gain Delta_t(t) for a subset of EGTA labels.

Reads <stem>_equilibria.npz produced by LLM_experiments.equilibrium and plots
each selected label's Delta_t trajectory on one axis. The whole point: tell
whether 'unstable at terminal' (Delta_T > tol) is a converged-yet-unstable
state vs trajectory that just hadn't plateaued by t=T.

Usage:
    python tools/plot_per_t_delta.py \
        $HOME/Opinion-dynamics-post-training/figs/eq_sftrlkl/egta_fj_bundle_equilibria.npz \
        --label-regex '^sft(_b0|kl_b[0-9.]+)$' \
        --out figs/delta_t_sftkl.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path, help="<stem>_equilibria.npz produced by equilibrium.py")
    ap.add_argument("--label-regex", type=str, default=".*",
                    help="Regex of labels to include. Default '.*' = all.")
    ap.add_argument("--tol", type=float, default=None,
                    help="Stability tolerance to draw as horizontal line. "
                         "Default: read from the npz.")
    ap.add_argument("--log-y", action="store_true",
                    help="Use log scale for Delta_t (useful when RL-KL rows blow up).")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    dev = d["deviation_gain"]    # (T, K)
    labels = [str(x) for x in d["labels"]]
    t_axis = d["t"]              # (T,)
    tol = float(args.tol if args.tol is not None else d["tol"])

    pat = re.compile(args.label_regex)
    sel_idx = [i for i, lbl in enumerate(labels) if pat.match(lbl)]
    if not sel_idx:
        print(f"no labels matched regex '{args.label_regex}'")
        print(f"available labels: {labels}")
        return
    sel_labels = [labels[i] for i in sel_idx]

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.95, len(sel_idx)))
    for color, i, lbl in zip(cmap, sel_idx, sel_labels):
        ax.plot(t_axis, dev[:, i], marker="o", linewidth=1.6, color=color, label=lbl)
    ax.axhline(tol, color="red", linestyle=":", linewidth=1.0, label=f"tol={tol:g}")
    ax.set_xlabel("snapshot time t")
    ax.set_ylabel(r"$\Delta_t = M[t,i,i] - \min_j M[t,i,j]$")
    if args.log_y:
        ax.set_yscale("symlog", linthresh=max(tol, 1e-4))
    ax.set_title(f"Per-t deviation gain  (regex: {args.label_regex})")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"selected {len(sel_idx)} labels: {sel_labels}")
    print(f"wrote {args.out}")
    # Per-label terminal Delta + crude plateau check (last 2 snapshots).
    print()
    print(f"{'label':<22} {'Delta_T':>10} {'last2_slope':>12} {'verdict':<18}")
    print("-" * 66)
    if dev.shape[0] >= 2:
        for i, lbl in zip(sel_idx, sel_labels):
            d_T = dev[-1, i]
            slope = dev[-1, i] - dev[-2, i]
            if abs(slope) < 0.005:
                verdict = "plateaued"
            elif slope > 0:
                verdict = "still rising"
            else:
                verdict = "still decreasing"
            print(f"{lbl:<22} {d_T:>10.4f} {slope:>+12.4f} {verdict:<18}")


if __name__ == "__main__":
    main()
