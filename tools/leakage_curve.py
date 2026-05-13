"""Leakage curve: D(D_T(beta), D_T(reference)) vs beta.

Tests the hypothesis 'higher beta -> trained-model-induced population D_T leaks
toward the reference (base-model-induced) D_T'. Plots the divergence between
each beta's terminal population distribution and a chosen reference run,
across multiple distance functionals (Wasserstein-1, KL on hist, |mean diff|).

For SFT-KL with the right (small-beta -> labels, large-beta -> untuned) limit,
the curves should monotone-decrease in beta as the trained policy stops fitting
the labels and starts deferring to the prior.

Usage:
    python tools/leakage_curve.py \
        --family-glob "fj_sftkl_*_egta" --extra fj_sft_b0_egta \
        --reference fj_untuned_egta \
        --out-fig figs/leakage_sftkl.png \
        --out-csv figs/leakage_sftkl.csv
"""
from __future__ import annotations

import argparse
import csv
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
    """tag may be a full label like 'fj_sftkl_b1_egta' or 'fj_untuned_egta'.
    Trajectory pickle paths:
        llm_<tag>_trajectory.pk             (training and CF runs)
        llm_llm_untuned_<tag>_trajectory.pk (untuned run)
    """
    p1 = RESULTS_DIR / f"llm_{tag}_trajectory.pk"
    p2 = RESULTS_DIR / f"llm_llm_untuned_{tag}_trajectory.pk"
    if p1.exists():
        with open(p1, "rb") as f:
            return np.asarray(pickle.load(f), dtype=np.float64)
    if p2.exists():
        with open(p2, "rb") as f:
            return np.asarray(pickle.load(f), dtype=np.float64)
    raise FileNotFoundError(f"no trajectory pickle for tag={tag}; tried {p1} and {p2}")


def discover_family(glob_pat: str) -> list[str]:
    """Return tags matching glob (without the 'llm_' prefix and '_trajectory.pk' suffix)."""
    rx = re.compile("^llm_" + glob_pat.replace("*", ".*") + r"_trajectory\.pk$")
    tags = []
    for p in sorted(RESULTS_DIR.glob("llm_*_trajectory.pk")):
        m = rx.match(p.name)
        if m:
            tag = p.name[len("llm_"):-len("_trajectory.pk")]
            tags.append(tag)
    return tags


def parse_beta(tag: str) -> float | None:
    m = re.search(r"_b(\d+(?:p\d+)?)_", tag)
    if not m:
        return None
    s = m.group(1).replace("p", ".")
    try:
        return float(s)
    except ValueError:
        return None


def kl_hist(p_samples: np.ndarray, q_samples: np.ndarray, n_bins: int = HIST_BINS) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[-1] = 1.0 + 1e-9
    pc, _ = np.histogram(p_samples, bins=edges)
    qc, _ = np.histogram(q_samples, bins=edges)
    p = (pc + 1e-3) / (pc.sum() + n_bins * 1e-3)
    q = (qc + 1e-3) / (qc.sum() + n_bins * 1e-3)
    return float(np.sum(p * np.log(p / q)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family-glob", type=str, required=True,
                    help="Glob over the <tag> portion. E.g. 'fj_sftkl_*_egta'.")
    ap.add_argument("--extra", type=str, action="append", default=[],
                    help="Additional tag to include (e.g. fj_sft_b0_egta). Repeatable.")
    ap.add_argument("--reference", type=str, required=True,
                    help="Tag whose D_T we measure leakage AGAINST. E.g. fj_untuned_egta.")
    ap.add_argument("--out-fig", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    tags = discover_family(args.family_glob) + list(args.extra)
    # Drop the reference if it shadows a family tag.
    tags = [t for t in tags if t != args.reference]
    if not tags:
        print(f"no tags matched family-glob='{args.family_glob}' + extras={args.extra}")
        return

    print(f"reference: {args.reference}")
    print(f"family members ({len(tags)}): {tags}")

    ref_traj = load_trajectory(args.reference)
    ref_T = ref_traj[:, -1]
    ref_mean_T = float(ref_T.mean())
    print(f"reference D_T: mean={ref_mean_T:.4f}  std={ref_T.std():.4f}  n={ref_T.size}")
    print()

    rows = []
    for tag in tags:
        traj = load_trajectory(tag)
        x_T = traj[:, -1]
        beta = parse_beta(tag)
        if beta is None:
            # Allow non-beta tags through (e.g. sft_b0 has no decimal in the name? actually fj_sft_b0_egta does).
            beta_str = "-"
        else:
            beta_str = f"{beta:g}"
        w = wasserstein_distance(x_T, ref_T)
        kl = kl_hist(x_T, ref_T)
        mean_diff = abs(float(x_T.mean()) - ref_mean_T)
        rows.append({
            "tag": tag, "beta": beta, "beta_str": beta_str,
            "mean_T": float(x_T.mean()), "std_T": float(x_T.std()),
            "wass_to_ref": w, "kl_to_ref": kl, "mean_diff_to_ref": mean_diff,
        })

    rows_with_beta = [r for r in rows if r["beta"] is not None]
    rows_with_beta.sort(key=lambda r: r["beta"])
    rows_no_beta = [r for r in rows if r["beta"] is None]

    print(f"{'tag':<26} {'beta':>7} {'mean_T':>8} {'mean_diff':>10} "
          f"{'KL_to_ref':>10} {'W_to_ref':>9}")
    print("-" * 76)
    for r in rows_with_beta + rows_no_beta:
        print(f"{r['tag']:<26} {r['beta_str']:>7} {r['mean_T']:>8.4f} "
              f"{r['mean_diff_to_ref']:>10.4f} {r['kl_to_ref']:>10.4f} "
              f"{r['wass_to_ref']:>9.4f}")
    print()
    print(f"reference mean_T = {ref_mean_T:.4f}")

    # CSV.
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "beta", "mean_T", "std_T",
                                          "wass_to_ref", "kl_to_ref", "mean_diff_to_ref"])
        w.writeheader()
        for r in rows:
            w.writerow({"tag": r["tag"], "beta": "" if r["beta"] is None else r["beta"],
                        "mean_T": r["mean_T"], "std_T": r["std_T"],
                        "wass_to_ref": r["wass_to_ref"], "kl_to_ref": r["kl_to_ref"],
                        "mean_diff_to_ref": r["mean_diff_to_ref"]})

    # Plot.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), constrained_layout=True)
    ax_w, ax_k, ax_m = axes
    if rows_with_beta:
        betas = [r["beta"] for r in rows_with_beta]
        ws = [r["wass_to_ref"] for r in rows_with_beta]
        ks = [r["kl_to_ref"] for r in rows_with_beta]
        ms = [r["mean_diff_to_ref"] for r in rows_with_beta]
        ax_w.plot(betas, ws, marker="o", color="tab:blue")
        ax_k.plot(betas, ks, marker="o", color="tab:green")
        ax_m.plot(betas, ms, marker="o", color="tab:red")
        for ax, ys in [(ax_w, ws), (ax_k, ks), (ax_m, ms)]:
            ax.set_xscale("symlog", linthresh=0.005)
            ax.set_xlabel(r"$\beta$")
            ax.grid(alpha=0.3)
            ax.axhline(0, color="black", linewidth=0.5)
    ax_w.set_ylabel(r"$W_1(D_T(\beta), D_T(\mathrm{ref}))$")
    ax_w.set_title("Wasserstein-1")
    ax_k.set_ylabel(r"$\mathrm{KL}(D_T(\beta) \| D_T(\mathrm{ref}))$")
    ax_k.set_title("KL on 51-bin histogram")
    ax_m.set_ylabel(r"$|\mathrm{mean}(D_T(\beta)) - \mathrm{mean}(D_T(\mathrm{ref}))|$")
    ax_m.set_title("|mean difference|")
    fig.suptitle(f"Leakage curve: family='{args.family_glob}' -> reference='{args.reference}'")
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_fig}")


if __name__ == "__main__":
    main()
