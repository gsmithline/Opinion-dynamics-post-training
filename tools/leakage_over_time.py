"""Leakage-over-time: D(pop_t(beta), pop_t(reference)) vs t, one line per beta.

For each round t, computes the distance between the population distribution
induced by each beta-trained model and the population distribution induced by
the reference (untuned) model at the same round. Plots all betas on one axis.

If higher beta -> population leaks toward the base-model attractor, then
high-beta lines should converge to zero over t while low-beta lines stay
elevated.

Usage:
    python tools/leakage_over_time.py \
        --family-glob "fj_sftkl_*_egta" --extra fj_sft_b0_egta \
        --reference fj_untuned_egta \
        --out-fig figs/leakage_over_time_sftkl.png
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
from scipy.stats import wasserstein_distance

RESULTS_DIR = Path("pokec_dataset/results")
HIST_BINS = 51


def load_trajectory(tag: str) -> np.ndarray:
    p1 = RESULTS_DIR / f"llm_{tag}_trajectory.pk"
    p2 = RESULTS_DIR / f"llm_llm_untuned_{tag}_trajectory.pk"
    if p1.exists():
        with open(p1, "rb") as f:
            return np.asarray(pickle.load(f), dtype=np.float64)
    if p2.exists():
        with open(p2, "rb") as f:
            return np.asarray(pickle.load(f), dtype=np.float64)
    raise FileNotFoundError(f"no trajectory pickle for {tag}; tried {p1} and {p2}")


def discover(glob_pat: str) -> list[str]:
    rx = re.compile("^llm_" + glob_pat.replace("*", ".*") + r"_trajectory\.pk$")
    tags = []
    for p in sorted(RESULTS_DIR.glob("llm_*_trajectory.pk")):
        m = rx.match(p.name)
        if m:
            tags.append(p.name[len("llm_"):-len("_trajectory.pk")])
    return tags


def parse_beta(tag: str) -> float | None:
    m = re.search(r"_b(\d+(?:p\d+)?)_", tag)
    if not m:
        return None
    return float(m.group(1).replace("p", "."))


def kl_hist(p: np.ndarray, q: np.ndarray, n_bins: int = HIST_BINS) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[-1] = 1.0 + 1e-9
    pc, _ = np.histogram(p, bins=edges)
    qc, _ = np.histogram(q, bins=edges)
    P = (pc + 1e-3) / (pc.sum() + n_bins * 1e-3)
    Q = (qc + 1e-3) / (qc.sum() + n_bins * 1e-3)
    return float(np.sum(P * np.log(P / Q)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family-glob", type=str, required=True)
    ap.add_argument("--extra", type=str, action="append", default=[])
    ap.add_argument("--reference", type=str, required=True)
    ap.add_argument("--out-fig", type=Path, required=True)
    args = ap.parse_args()

    tags = discover(args.family_glob) + list(args.extra)
    tags = [t for t in tags if t != args.reference]
    if not tags:
        raise SystemExit(f"no tags matched '{args.family_glob}' + extras")

    ref = load_trajectory(args.reference)
    T_plus_1 = ref.shape[1]
    rounds = np.arange(T_plus_1)
    print(f"reference: {args.reference}, traj shape {ref.shape}")

    series: dict[str, dict] = {}
    for tag in tags:
        traj = load_trajectory(tag)
        if traj.shape != ref.shape:
            print(f"[skip] {tag}: shape {traj.shape} differs from ref {ref.shape}")
            continue
        beta = parse_beta(tag)
        ws = np.array([wasserstein_distance(traj[:, t], ref[:, t]) for t in rounds])
        kls = np.array([kl_hist(traj[:, t], ref[:, t]) for t in rounds])
        md = np.abs(traj.mean(axis=0) - ref.mean(axis=0))
        series[tag] = {"beta": beta, "wass": ws, "kl": kls, "mean_diff": md}
        print(f"  loaded {tag:<26} beta={beta}  W_T={ws[-1]:.4f}  KL_T={kls[-1]:.4f}  |Δμ_T|={md[-1]:.4f}")

    # Sort by beta for consistent color ordering.
    ordered = sorted(series.items(), key=lambda kv: (kv[1]["beta"] is None, kv[1]["beta"] or 0.0))
    betas_for_color = [kv[1]["beta"] for kv in ordered if kv[1]["beta"] is not None]
    if betas_for_color:
        b_arr = np.array(betas_for_color, dtype=float)
        b_log = np.log10(np.clip(b_arr, 1e-3, 1e3))
        norm_b = (b_log - b_log.min()) / max(float(np.ptp(b_log)), 1e-9)
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    ax_w, ax_k, ax_m = axes
    col_i = 0
    for tag, s in ordered:
        if s["beta"] is not None and betas_for_color:
            color = cmap(0.1 + 0.85 * norm_b[col_i])
            col_i += 1
        else:
            color = "gray"
        label = fr"$\beta$={s['beta']:g}" if s["beta"] is not None else tag
        ax_w.plot(rounds, s["wass"], "o-", color=color, linewidth=1.7, markersize=4, label=label)
        ax_k.plot(rounds, s["kl"],   "o-", color=color, linewidth=1.7, markersize=4, label=label)
        ax_m.plot(rounds, s["mean_diff"], "o-", color=color, linewidth=1.7, markersize=4, label=label)

    for ax in (ax_w, ax_k, ax_m):
        ax.set_xlabel("round t")
        ax.grid(alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)

    ax_w.set_ylabel(r"$W_1(D_t(\beta), D_t(\mathrm{untuned}))$")
    ax_w.set_title("Wasserstein-1")
    ax_w.legend(fontsize=8, loc="best")

    ax_k.set_ylabel(r"$\mathrm{KL}(D_t(\beta) \| D_t(\mathrm{untuned}))$")
    ax_k.set_title("KL on 51-bin histogram")

    ax_m.set_ylabel(r"$|\mathrm{mean}(D_t(\beta)) - \mathrm{mean}(D_t(\mathrm{untuned}))|$")
    ax_m.set_title("|mean difference|")

    fig.suptitle(f"Population leakage over time: family='{args.family_glob}' -> '{args.reference}'")
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_fig}")


if __name__ == "__main__":
    main()
