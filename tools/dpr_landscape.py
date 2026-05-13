"""Decoupled-performative-risk landscape in the style of the ICLR 2026 PP blog.

Panels:
    A. DPR heatmap M[T, beta_D, beta_M] (theta_D = world-inducing, theta_M = model).
       Diagonal marks coupled risk PR(beta). Stars at PS points (where Δ_T < tol).
       Triangle at PO (argmin diagonal). Arrows show BR steps: where each row's
       argmin_j M[T, i, j] sits relative to its own diagonal (the 'would-be RRM
       step in beta-space').
    B. PR(beta) and min_j DPR(beta, .) on one axis. Gap between them = PG.
    C. PG(beta) = PR - min_j DPR, with stability tolerance line.
    D. BR(beta): index of best-response model as a discrete step function.

This works for a SINGLE family at a time (filter via --family-regex). Cross-
family comparisons need separate runs because scales differ wildly between
e.g. SFT-KL (PR ~ 1) and RL-KL (PR ~ 50 due to saturation pathology).

Usage:
    python tools/dpr_landscape.py \
        $HOME/Opinion-dynamics-post-training/pokec_dataset/results/egta_fj_bundle.npz \
        --family-regex '^(sft_b0|sftkl_b[0-9.]+)$' \
        --t-index -1 \
        --out figs/dpr_landscape_sftkl.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm


def parse_beta(label: str) -> float | None:
    """Extract beta from labels like 'sftkl_b0.01', 'cf_b10', 'sft_b0'."""
    m = re.search(r"_b(\d+(?:\.\d+)?)$", label)
    if not m:
        return None
    return float(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path, help="Bundle .npz from egta_matrix.py / egta_fj.py")
    ap.add_argument("--family-regex", type=str, required=True,
                    help="Regex matching labels to include (e.g. '^sftkl_b' or '^cf_b').")
    ap.add_argument("--t-index", type=int, default=-1,
                    help="Snapshot index. Default -1 = terminal.")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="Stability tolerance (Δ < tol = PS).")
    ap.add_argument("--log-color", action="store_true",
                    help="Log color scale for the DPR heatmap (use if PR spans 10x+ range).")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    M_full = d["M"]               # (T, K, K)
    labels_full = [str(x) for x in d["labels"]]
    t_axis = d["t"] if "t" in d.files else np.arange(M_full.shape[0])
    t_idx = args.t_index if args.t_index >= 0 else M_full.shape[0] + args.t_index
    t_val = int(t_axis[t_idx])
    print(f"using snapshot index {t_idx} (t={t_val}); bundle has T={M_full.shape[0]}")

    pat = re.compile(args.family_regex)
    sel = []
    for i, lbl in enumerate(labels_full):
        if pat.match(lbl):
            beta = parse_beta(lbl)
            if beta is not None:
                sel.append((beta, i, lbl))
    sel.sort()
    if not sel:
        print(f"no labels matched '{args.family_regex}' or none had parseable beta")
        print(f"available labels: {labels_full}")
        return

    betas = np.array([s[0] for s in sel])
    idx = [s[1] for s in sel]
    labels = [s[2] for s in sel]
    K = len(sel)
    print(f"family: {labels} (K={K})")

    M = M_full[t_idx][np.ix_(idx, idx)]  # (K, K) submatrix in beta-order
    pr = np.einsum("ii->i", M)            # diagonal = PR(beta) at t
    br_idx = np.argmin(M, axis=1)         # argmin per row
    pr_min = M[np.arange(K), br_idx]       # min_j DPR(beta_i, beta_j)
    pg = pr - pr_min                       # performative gap
    is_ps = pg < args.tol                  # mask of PS points
    po_idx = int(np.argmin(pr))            # performative optimum (in this family)

    print(f"PR(beta) at t={t_val}: {dict(zip([f'{b:g}' for b in betas], pr.round(4).tolist()))}")
    print(f"PG(beta) at t={t_val}: {dict(zip([f'{b:g}' for b in betas], pg.round(4).tolist()))}")
    print(f"BR(beta):              {dict(zip(labels, [labels[j] for j in br_idx]))}")
    print(f"PS labels (Δ<{args.tol}): {[l for l, s in zip(labels, is_ps) if s]}")
    print(f"PO label (argmin PR):  {labels[po_idx]}")

    # ----- Figure -----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    ax_h, ax_curve, ax_pg, ax_br = axes.ravel()

    # Panel A: DPR heatmap.
    if args.log_color:
        norm = LogNorm(vmin=max(M.min(), 1e-3), vmax=M.max())
    else:
        norm = None
    im = ax_h.imshow(M, origin="lower", cmap="viridis", norm=norm, aspect="auto")
    cbar = fig.colorbar(im, ax=ax_h, label=r"$\mathrm{DPR}(\beta_D, \beta_M)$ at $t=$" + f"{t_val}")
    # Diagonal markers (PS = star, PO = triangle, else = circle).
    for i in range(K):
        marker = "*" if is_ps[i] else ("v" if i == po_idx else "o")
        size = 220 if is_ps[i] or i == po_idx else 70
        ax_h.scatter(i, i, marker=marker, s=size, edgecolor="red", facecolor="none",
                     linewidths=1.7, zorder=5)
    # BR-step arrows: from (i, i) along the row to (i, br_idx[i]).
    for i in range(K):
        if br_idx[i] != i:
            ax_h.annotate("", xy=(br_idx[i], i), xytext=(i, i),
                          arrowprops=dict(arrowstyle="->", color="white", lw=1.4, alpha=0.85))
    ax_h.set_xticks(range(K))
    ax_h.set_yticks(range(K))
    ax_h.set_xticklabels([f"{b:g}" for b in betas], rotation=0, fontsize=8)
    ax_h.set_yticklabels([f"{b:g}" for b in betas], fontsize=8)
    ax_h.set_xlabel(r"$\beta_M$ (model column)")
    ax_h.set_ylabel(r"$\beta_D$ (world row)")
    ax_h.set_title(f"Panel A: DPR landscape at t={t_val}\n"
                   f"red star = PS (Δ<{args.tol}), red triangle = PO, white arrow = BR step")

    # Panel B: PR and min_j DPR.
    ax_curve.plot(betas, pr,    "o-", color="tab:blue",   linewidth=2.0, label=r"$\mathrm{PR}(\beta) = M[\beta,\beta]$")
    ax_curve.plot(betas, pr_min, "s--", color="tab:orange", linewidth=1.6, label=r"$\min_j \mathrm{DPR}(\beta, j)$")
    ax_curve.set_xscale("symlog", linthresh=max(betas[betas > 0].min() / 2, 1e-3) if any(betas > 0) else 1e-3)
    ax_curve.set_xlabel(r"$\beta$")
    ax_curve.set_ylabel("loss")
    ax_curve.set_title(f"Panel B: PR(β) and min_j DPR(β, j) at t={t_val}")
    ax_curve.legend(fontsize=10, loc="best")
    ax_curve.grid(alpha=0.3)

    # Panel C: Performative gap.
    ax_pg.plot(betas, pg, "o-", color="tab:purple", linewidth=2.0)
    ax_pg.axhline(args.tol, color="red", linestyle=":", linewidth=1.0, label=f"tol={args.tol:g}")
    ax_pg.fill_between(betas, 0, args.tol, color="tab:green", alpha=0.10)
    ax_pg.set_xscale("symlog", linthresh=max(betas[betas > 0].min() / 2, 1e-3) if any(betas > 0) else 1e-3)
    ax_pg.set_xlabel(r"$\beta$")
    ax_pg.set_ylabel(r"$\mathrm{PG}(\beta) = \mathrm{PR}(\beta) - \min_j \mathrm{DPR}(\beta, j)$")
    ax_pg.set_title(f"Panel C: performative gap at t={t_val}\nstable region shaded green")
    ax_pg.legend(fontsize=10)
    ax_pg.grid(alpha=0.3)

    # Panel D: BR step in beta-space.
    br_betas = np.array([betas[j] for j in br_idx])
    ax_br.plot(betas, br_betas, "o-", color="tab:red", linewidth=2.0, label=r"$\beta \to \mathrm{BR}(\beta)$")
    ax_br.plot(betas, betas, "k--", linewidth=1.0, alpha=0.4, label=r"$\beta = \beta$ (identity)")
    ax_br.set_xscale("symlog", linthresh=max(betas[betas > 0].min() / 2, 1e-3) if any(betas > 0) else 1e-3)
    ax_br.set_yscale("symlog", linthresh=max(betas[betas > 0].min() / 2, 1e-3) if any(betas > 0) else 1e-3)
    ax_br.set_xlabel(r"current $\beta$")
    ax_br.set_ylabel(r"BR $\beta$")
    ax_br.set_title(f"Panel D: BR map (would-be RRM step) at t={t_val}")
    ax_br.legend(fontsize=10)
    ax_br.grid(alpha=0.3)

    fig.suptitle(f"DPR landscape: family='{args.family_regex}', npz={args.npz.name}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
