"""Figure 4: labeled vs unlabeled exposure gap.

D_O(beta) = RMSE(x_{T,labeled}^beta, x_{T,labeled}^ref)
D_U(beta) = RMSE(x_{T,unlabeled}^beta, x_{T,unlabeled}^ref)

Plus the ratio D_U / D_O. The 'population-structure' point: agents that receive
the LM's predictions directly (unlabeled) are more steerable than those that
only feel the LM via peer influence (labeled).
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
    raise FileNotFoundError(tag)


def discover(glob_pat: str) -> list[str]:
    rx = re.compile("^llm_" + glob_pat.replace("*", ".*") + r"_trajectory\.pk$")
    return [p.name[len("llm_"):-len("_trajectory.pk")]
            for p in sorted(RESULTS_DIR.glob("llm_*_trajectory.pk")) if rx.match(p.name)]


def parse_beta(tag: str) -> float | None:
    m = re.search(r"_b(\d+(?:p\d+)?)_", tag)
    return None if not m else float(m.group(1).replace("p", "."))


def paired_rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family-glob", default="fj_sftkl_*_egta")
    ap.add_argument("--extra", action="append", default=["fj_sft_b0_egta"])
    ap.add_argument("--ref", default="fj_untuned_egta")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    fam = [t for t in discover(args.family_glob) + list(args.extra) if t != args.ref]
    ref_full = load_tag(args.ref)
    ref_lab  = ref_full[:N_LABELED, -1]
    ref_unl  = ref_full[N_LABELED:, -1]

    rows = []
    for tag in fam:
        beta = parse_beta(tag)
        if beta is None:
            continue
        x_full = load_tag(tag)
        d_o = paired_rmse(x_full[:N_LABELED, -1], ref_lab)
        d_u = paired_rmse(x_full[N_LABELED:, -1], ref_unl)
        rows.append((beta, tag, d_o, d_u))
    rows.sort()
    betas = np.array([r[0] for r in rows])
    d_O   = np.array([r[2] for r in rows])
    d_U   = np.array([r[3] for r in rows])
    ratio = d_U / np.where(d_O > 1e-9, d_O, 1e-9)

    print(f"{'beta':>8} {'D_O (labeled)':>14} {'D_U (unlabeled)':>16} {'D_U/D_O':>9}")
    print("-" * 52)
    for b, do, du, r in zip(betas, d_O, d_U, ratio):
        print(f"{b:>8.3g} {do:>14.4f} {du:>16.4f} {r:>9.2f}")

    fig, (ax, ax_r) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    lin = max(betas[betas > 0].min() / 2, 1e-3) if any(betas > 0) else 1e-3

    ax.plot(betas, d_O, "o-", color="tab:blue",   linewidth=2.0, label=r"$D_O(\beta)$ labeled")
    ax.plot(betas, d_U, "s-", color="tab:orange", linewidth=2.0, label=r"$D_U(\beta)$ unlabeled")
    ax.set_xscale("symlog", linthresh=lin)
    ax.set_xlabel(r"$\beta_{\mathrm{KL}}$"); ax.set_ylabel("paired RMSE vs ref")
    ax.set_title("Subgroup distance to reference")
    ax.legend(); ax.grid(alpha=0.3)

    ax_r.plot(betas, ratio, "o-", color="tab:purple", linewidth=2.0)
    ax_r.axhline(1.0, color="black", linestyle=":", linewidth=0.8)
    ax_r.set_xscale("symlog", linthresh=lin)
    ax_r.set_xlabel(r"$\beta_{\mathrm{KL}}$"); ax_r.set_ylabel(r"$D_U(\beta) / D_O(\beta)$")
    ax_r.set_title("Exposure ratio (unlabeled / labeled)")
    ax_r.grid(alpha=0.3)

    fig.suptitle(r"Figure 4: labeled vs unlabeled exposure gap")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
