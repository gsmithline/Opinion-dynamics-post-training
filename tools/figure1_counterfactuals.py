"""Figure 1: counterfactual population-mean trajectories.

Overlays mean(x_t) over rounds t for:
    free  : platform-free FJ from innate (oracle coincides with this in our setup)
    ref   : frozen reference (untuned) LM
    beta  : SFT-KL LM at each beta in the family
"""
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("pokec_dataset/results")
N_LABELED = 1730


def load_tag(tag: str) -> np.ndarray:
    p1 = RESULTS_DIR / f"llm_{tag}_trajectory.pk"
    p2 = RESULTS_DIR / f"llm_llm_untuned_{tag}_trajectory.pk"
    if p1.exists():
        return np.asarray(pickle.load(open(p1, "rb")), dtype=np.float64)
    if p2.exists():
        return np.asarray(pickle.load(open(p2, "rb")), dtype=np.float64)
    raise FileNotFoundError(f"no trajectory for {tag}; tried {p1}, {p2}")


def discover(glob_pat: str) -> list[str]:
    rx = re.compile("^llm_" + glob_pat.replace("*", ".*") + r"_trajectory\.pk$")
    return [p.name[len("llm_"):-len("_trajectory.pk")]
            for p in sorted(RESULTS_DIR.glob("llm_*_trajectory.pk")) if rx.match(p.name)]


def parse_beta(tag: str) -> float | None:
    m = re.search(r"_b(\d+(?:p\d+)?)_", tag)
    return None if not m else float(m.group(1).replace("p", "."))


def slice_subset(t: np.ndarray, subset: str) -> np.ndarray:
    if subset == "whole":     return t
    if subset == "labeled":   return t[:N_LABELED, :]
    if subset == "unlabeled": return t[N_LABELED:, :]
    raise ValueError(subset)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family-glob", default="fj_sftkl_*_egta")
    ap.add_argument("--extra", action="append", default=["fj_sft_b0_egta"])
    ap.add_argument("--ref",  default="fj_untuned_egta")
    ap.add_argument("--free", default="fj_free_egta")
    ap.add_argument("--subset", choices=["whole", "labeled", "unlabeled"], default="whole")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tags = [t for t in discover(args.family_glob) + list(args.extra)
            if t not in (args.ref, args.free)]
    tags = sorted(tags, key=lambda t: parse_beta(t) if parse_beta(t) is not None else float("inf"))

    ref  = slice_subset(load_tag(args.ref),  args.subset)
    free = slice_subset(load_tag(args.free), args.subset)
    T_plus_1 = ref.shape[1]
    rounds = np.arange(T_plus_1)

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    # SFT-KL betas, colored by log-beta.
    betas = [parse_beta(t) for t in tags]
    valid = [(t, b) for t, b in zip(tags, betas) if b is not None]
    if valid:
        b_arr = np.array([b for _, b in valid], dtype=float)
        b_log = np.log10(np.clip(b_arr, 1e-3, 1e3))
        norm_b = (b_log - b_log.min()) / max(float(np.ptp(b_log)), 1e-9)
    cmap = plt.cm.viridis
    for i, (tag, beta) in enumerate(valid):
        traj = slice_subset(load_tag(tag), args.subset)
        color = cmap(0.1 + 0.85 * norm_b[i])
        ax.plot(rounds, traj.mean(axis=0), "o-", color=color, linewidth=1.7, markersize=4,
                label=fr"$\beta$={beta:g}")
    ax.plot(rounds, ref.mean(axis=0),  "k-",  linewidth=2.4, label=r"$x_t^{\mathrm{ref}}$ (untuned)")
    ax.plot(rounds, free.mean(axis=0), "k--", linewidth=2.0,
            label=r"$x_t^{\mathrm{free}}$ (FJ no platform; = oracle in our setup)")

    ax.set_xlabel("round t")
    ax.set_ylabel(fr"$\mathrm{{mean}}(x_t)$  (subset={args.subset})")
    ax.set_title(r"Figure 1: counterfactual population-mean trajectories")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best", ncol=2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
