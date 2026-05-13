"""Figure 3: terminal distances to counterfactuals vs beta.

For each beta, plots:
    d(x_T^beta, x_T^ref)   -- distance to frozen reference (untuned)
    d(x_T^beta, x_T^free)  -- distance to platform-free FJ baseline

These tell you which counterfactual the LM-mediated population is steering
toward as beta varies. Three metrics: paired RMSE, Wasserstein-1, |mean diff|.
Oracle coincides with free in our setup so it's omitted (would draw the same
line).
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


def slice_subset(t: np.ndarray, subset: str) -> np.ndarray:
    if subset == "whole":     return t
    if subset == "labeled":   return t[:N_LABELED, :]
    if subset == "unlabeled": return t[N_LABELED:, :]
    raise ValueError(subset)


def paired_rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family-glob", default="fj_sftkl_*_egta")
    ap.add_argument("--extra", action="append", default=["fj_sft_b0_egta"])
    ap.add_argument("--ref",   default="fj_untuned_egta")
    ap.add_argument("--free",  default="fj_free_egta")
    ap.add_argument("--subset", choices=["whole", "labeled", "unlabeled"], default="whole")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    fam = [t for t in discover(args.family_glob) + list(args.extra)
           if t not in (args.ref, args.free)]
    rows = []
    for tag in fam:
        beta = parse_beta(tag)
        if beta is None:
            continue
        x_T = slice_subset(load_tag(tag), args.subset)[:, -1]
        rows.append((beta, tag, x_T))
    rows.sort()
    if not rows:
        raise SystemExit("no rows")

    ref_T  = slice_subset(load_tag(args.ref),  args.subset)[:, -1]
    free_T = slice_subset(load_tag(args.free), args.subset)[:, -1]

    betas, rmse_r, rmse_f, w_r, w_f, dmu_r, dmu_f = [], [], [], [], [], [], []
    for beta, _, x_T in rows:
        betas.append(beta)
        rmse_r.append(paired_rmse(x_T, ref_T))
        rmse_f.append(paired_rmse(x_T, free_T))
        w_r.append(float(wasserstein_distance(x_T, ref_T)))
        w_f.append(float(wasserstein_distance(x_T, free_T)))
        dmu_r.append(abs(float(x_T.mean()) - float(ref_T.mean())))
        dmu_f.append(abs(float(x_T.mean()) - float(free_T.mean())))
    betas = np.array(betas)

    print(f"{'beta':>8} {'RMSE_ref':>10} {'RMSE_free':>10} {'W_ref':>10} {'W_free':>10} "
          f"{'|dmu_ref|':>10} {'|dmu_free|':>11}")
    print("-" * 80)
    for b, rr, rf, wr, wf, dr, df in zip(betas, rmse_r, rmse_f, w_r, w_f, dmu_r, dmu_f):
        print(f"{b:>8.3g} {rr:>10.4f} {rf:>10.4f} {wr:>10.4f} {wf:>10.4f} {dr:>10.4f} {df:>11.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    ax_r, ax_w, ax_m = axes
    lin = max(betas[betas > 0].min() / 2, 1e-3) if any(betas > 0) else 1e-3

    ax_r.plot(betas, rmse_r, "o-", color="tab:blue",   linewidth=2.0, label="d(β, ref)")
    ax_r.plot(betas, rmse_f, "s-", color="tab:orange", linewidth=2.0, label="d(β, free)")
    ax_r.set_xscale("symlog", linthresh=lin)
    ax_r.set_xlabel(r"$\beta_{\mathrm{KL}}$"); ax_r.set_ylabel("paired RMSE"); ax_r.set_title("Paired RMSE")
    ax_r.legend(); ax_r.grid(alpha=0.3)

    ax_w.plot(betas, w_r, "o-", color="tab:blue",   linewidth=2.0, label="d(β, ref)")
    ax_w.plot(betas, w_f, "s-", color="tab:orange", linewidth=2.0, label="d(β, free)")
    ax_w.set_xscale("symlog", linthresh=lin)
    ax_w.set_xlabel(r"$\beta_{\mathrm{KL}}$"); ax_w.set_ylabel(r"Wasserstein-1"); ax_w.set_title("Wasserstein-1")
    ax_w.legend(); ax_w.grid(alpha=0.3)

    ax_m.plot(betas, dmu_r, "o-", color="tab:blue",   linewidth=2.0, label="d(β, ref)")
    ax_m.plot(betas, dmu_f, "s-", color="tab:orange", linewidth=2.0, label="d(β, free)")
    ax_m.set_xscale("symlog", linthresh=lin)
    ax_m.set_xlabel(r"$\beta_{\mathrm{KL}}$"); ax_m.set_ylabel(r"$|\mu^\beta_T - \mu^{*}_T|$"); ax_m.set_title("|mean diff|")
    ax_m.legend(); ax_m.grid(alpha=0.3)

    fig.suptitle(f"Figure 3: distances to counterfactuals  (subset={args.subset})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
