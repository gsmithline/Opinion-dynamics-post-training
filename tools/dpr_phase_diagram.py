"""DPR(beta_D, beta_M) phase diagram at one snapshot t.

The decoupled-risk surface across betas in a single family, drawn the way the
ICLR PP blog draws DPR landscapes: two beta axes, color = DPR.

Usage:
    python tools/dpr_phase_diagram.py \
        pokec_dataset/results/egta_fj_bundle.npz \
        --family-regex '^(sft_b0|sftkl_b[0-9.]+)$' \
        --t-index -1 \
        --out figs/dpr_phase_sftkl.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def parse_beta(label: str) -> float | None:
    m = re.search(r"_b(\d+(?:\.\d+)?)$", label)
    if not m:
        return None
    return float(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--family-regex", type=str, required=True)
    ap.add_argument("--t-index", type=int, default=-1)
    ap.add_argument("--log-color", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    M_full = d["M"]
    labels_full = [str(x) for x in d["labels"]]
    t_axis = d["t"] if "t" in d.files else np.arange(M_full.shape[0])
    t_idx = args.t_index if args.t_index >= 0 else M_full.shape[0] + args.t_index
    t_val = int(t_axis[t_idx])

    pat = re.compile(args.family_regex)
    sel = []
    for i, lbl in enumerate(labels_full):
        if pat.match(lbl):
            b = parse_beta(lbl)
            if b is not None:
                sel.append((b, i, lbl))
    sel.sort()
    if not sel:
        raise SystemExit(f"no betas matched '{args.family_regex}' in labels {labels_full}")

    betas = [s[0] for s in sel]
    idx = [s[1] for s in sel]
    K = len(sel)
    M = M_full[t_idx][np.ix_(idx, idx)]
    pr = np.einsum("ii->i", M)
    pg = pr - M.min(axis=1)
    po_idx = int(np.argmin(pr))

    print(f"t = {t_val}, K = {K}")
    print(f"betas: {betas}")
    print(f"DPR matrix:\n{M.round(3)}")
    print(f"PR  (diag): {pr.round(3).tolist()}")
    print(f"PG (Δ):     {pg.round(3).tolist()}")
    print(f"PO at β = {betas[po_idx]}")

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    norm = LogNorm(vmin=max(M.min(), 1e-3), vmax=M.max()) if args.log_color else None
    im = ax.imshow(M, origin="lower", cmap="viridis", norm=norm, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, label=fr"$\mathrm{{DPR}}_{{t={t_val}}}(\beta_D, \beta_M)$")

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f"{b:g}" for b in betas])
    ax.set_yticklabels([f"{b:g}" for b in betas])
    ax.set_xlabel(r"$\beta_M$  (model)")
    ax.set_ylabel(r"$\beta_D$  (world / deployed policy)")
    ax.set_title(f"Decoupled performative risk surface at t={t_val}")

    # Annotate each cell with its DPR value (small text).
    for i in range(K):
        for j in range(K):
            txt_color = "white" if (norm and norm(M[i, j]) > 0.5) or (not norm and M[i, j] > 0.5 * M.max()) else "black"
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color=txt_color, fontsize=8)

    # Mark diagonal with a thin black box around each (i, i) cell.
    for i in range(K):
        ax.add_patch(plt.Rectangle((i - 0.45, i - 0.45), 0.9, 0.9, fill=False,
                                    edgecolor="black", linewidth=0.8))
    # PO triangle on its diagonal cell.
    ax.scatter(po_idx, po_idx, marker="v", s=200, edgecolor="red", facecolor="none", linewidths=1.8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
