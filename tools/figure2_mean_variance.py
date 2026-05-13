"""Figure 2: terminal mean and variance vs beta.

Two panels:
    Left:  |mu_T(beta) - mu_T(ref)| vs beta (log-x). Horizontal at free's value.
    Right: Var(x_T(beta)) vs beta (log-x). Horizontal at free's and ref's variances.

Headline claim from the project description: increasing beta pulls mean toward
ref while VARIANCE INCREASES (not collapses). Panel 2 directly tests this.
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


def slice_subset(t: np.ndarray, subset: str) -> np.ndarray:
    if subset == "whole":     return t
    if subset == "labeled":   return t[:N_LABELED, :]
    if subset == "unlabeled": return t[N_LABELED:, :]
    raise ValueError(subset)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family-glob", default="fj_sftkl_*_egta")
    ap.add_argument("--extra", action="append", default=["fj_sft_b0_egta"])
    ap.add_argument("--ref",   default="fj_untuned_egta")
    ap.add_argument("--free",  default="fj_free_egta")
    ap.add_argument("--subset", choices=["whole", "labeled", "unlabeled"], default="whole")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    fam = discover(args.family_glob) + list(args.extra)
    fam = [t for t in fam if t not in (args.ref, args.free)]
    rows = []
    for tag in fam:
        beta = parse_beta(tag)
        if beta is None:
            continue
        x_T = slice_subset(load_tag(tag), args.subset)[:, -1]
        rows.append((beta, tag, float(x_T.mean()), float(x_T.var())))
    rows.sort()
    if not rows:
        raise SystemExit(f"no betas matched '{args.family_glob}'")

    ref_x_T  = slice_subset(load_tag(args.ref),  args.subset)[:, -1]
    free_x_T = slice_subset(load_tag(args.free), args.subset)[:, -1]
    mu_ref   = float(ref_x_T.mean());   var_ref  = float(ref_x_T.var())
    mu_free  = float(free_x_T.mean());  var_free = float(free_x_T.var())

    betas = np.array([r[0] for r in rows])
    mus   = np.array([r[2] for r in rows])
    vars_ = np.array([r[3] for r in rows])
    print(f"reference  (untuned): mu_T={mu_ref:.4f}  var_T={var_ref:.6f}")
    print(f"free (no platform):   mu_T={mu_free:.4f}  var_T={var_free:.6f}")
    print()
    print(f"{'beta':>8} {'mu_T':>10} {'|mu-mu_ref|':>13} {'var_T':>12}")
    print("-" * 50)
    for b, mu, v in zip(betas, mus, vars_):
        print(f"{b:>8.3g} {mu:>10.4f} {abs(mu - mu_ref):>13.4f} {v:>12.6f}")

    fig, (ax_mu, ax_var) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    ax_mu.plot(betas, np.abs(mus - mu_ref), "o-", color="tab:blue", linewidth=2.0)
    ax_mu.axhline(abs(mu_free - mu_ref), color="black", linestyle="--", linewidth=1.0,
                  label=fr"free vs ref: $|\mu^{{\mathrm{{free}}}}_T - \mu^{{\mathrm{{ref}}}}_T|$ = {abs(mu_free-mu_ref):.4f}")
    ax_mu.set_xscale("symlog", linthresh=max(betas[betas > 0].min() / 2, 1e-3) if any(betas > 0) else 1e-3)
    ax_mu.set_xlabel(r"$\beta_{\mathrm{KL}}$")
    ax_mu.set_ylabel(r"$|\mu^\beta_T - \mu^{\mathrm{ref}}_T|$")
    ax_mu.set_title("Distance to reference mean")
    ax_mu.legend(fontsize=9)
    ax_mu.grid(alpha=0.3)

    ax_var.plot(betas, vars_, "o-", color="tab:purple", linewidth=2.0, label=r"Var$(x_T^\beta)$")
    ax_var.axhline(var_ref,  color="black", linestyle="-",  linewidth=1.2, label=fr"ref: {var_ref:.4f}")
    ax_var.axhline(var_free, color="black", linestyle="--", linewidth=1.0, label=fr"free: {var_free:.4f}")
    ax_var.set_xscale("symlog", linthresh=max(betas[betas > 0].min() / 2, 1e-3) if any(betas > 0) else 1e-3)
    ax_var.set_xlabel(r"$\beta_{\mathrm{KL}}$")
    ax_var.set_ylabel(r"Var$(x_T^\beta)$")
    ax_var.set_title("Terminal population variance")
    ax_var.legend(fontsize=9)
    ax_var.grid(alpha=0.3)

    fig.suptitle(f"Figure 2: mean & variance vs $\\beta$  (subset={args.subset})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
