"""Track decoupled performative risk DPR(beta_D, beta_M) across rounds for a
chosen pair of betas. Mirrors the ICLR PP blog's Figure 6/7 style: x-axis is
round (iteration), y-axis is loss, one line per (D, M) combination.

For two betas (a, b) you get four lines:
    M[t, a, a]  -- coupled risk of model_a on its own world
    M[t, a, b]  -- model_b on world_a (off-diagonal)
    M[t, b, a]  -- model_a on world_b (off-diagonal)
    M[t, b, b]  -- coupled risk of model_b on its own world

Plus a panel for the performative gap PG(t) = M[t, *, *] - min_j M[t, *, j]
restricted to the two-model submatrix, for each row.

Usage:
    python tools/dpr_across_rounds.py \
        pokec_dataset/results/egta_fj_bundle.npz \
        --beta-a 0 --beta-b 10 \
        --family-regex '^(sft_b0|sftkl_b[0-9.]+)$' \
        --out figs/dpr_rounds_sftkl_b0_b10.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_beta(label: str) -> float | None:
    m = re.search(r"_b(\d+(?:\.\d+)?)$", label)
    if not m:
        return None
    return float(m.group(1))


def find_beta_index(labels: list[str], beta: float, family_regex: str) -> int:
    pat = re.compile(family_regex)
    candidates = []
    for i, lbl in enumerate(labels):
        if not pat.match(lbl):
            continue
        b = parse_beta(lbl)
        if b is None:
            continue
        if abs(b - beta) < 1e-9:
            candidates.append((i, lbl))
    if not candidates:
        raise SystemExit(f"no label in family '{family_regex}' has beta={beta}; "
                         f"available: {[(i, l) for i, l in enumerate(labels) if pat.match(l)]}")
    if len(candidates) > 1:
        print(f"warning: multiple matches for beta={beta}: {candidates}; using first")
    return candidates[0][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path, help="Bundle .npz")
    ap.add_argument("--beta-a", type=float, required=True)
    ap.add_argument("--beta-b", type=float, required=True)
    ap.add_argument("--family-regex", type=str, default=r"^(sft_b0|sftkl_b[0-9.]+)$",
                    help="Regex for label disambiguation. Default = SFT-KL family.")
    ap.add_argument("--log-y", action="store_true",
                    help="Log y-axis for the DPR panel (useful when scales differ a lot).")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    M = d["M"]                              # (T, K, K)
    labels = [str(x) for x in d["labels"]]
    t_axis = d["t"] if "t" in d.files else np.arange(M.shape[0])
    T = M.shape[0]

    ia = find_beta_index(labels, args.beta_a, args.family_regex)
    ib = find_beta_index(labels, args.beta_b, args.family_regex)
    la, lb = labels[ia], labels[ib]
    print(f"beta_a = {args.beta_a}  -> label '{la}'  (index {ia})")
    print(f"beta_b = {args.beta_b}  -> label '{lb}'  (index {ib})")

    # 4 series, all of shape (T,).
    dpr_aa = M[:, ia, ia]
    dpr_ab = M[:, ia, ib]
    dpr_ba = M[:, ib, ia]
    dpr_bb = M[:, ib, ib]

    # Performative gap restricted to the 2x2 sub-matrix.
    pr_a   = dpr_aa
    pr_b   = dpr_bb
    minr_a = np.minimum(dpr_aa, dpr_ab)
    minr_b = np.minimum(dpr_ba, dpr_bb)
    pg_a   = pr_a - minr_a
    pg_b   = pr_b - minr_b

    print()
    print(f"{'t':>5} {'DPR(a,a)':>10} {'DPR(a,b)':>10} {'DPR(b,a)':>10} {'DPR(b,b)':>10} "
          f"{'PG_a':>8} {'PG_b':>8}")
    print("-" * 70)
    for k in range(T):
        print(f"{t_axis[k]:>5} {dpr_aa[k]:>10.4f} {dpr_ab[k]:>10.4f} "
              f"{dpr_ba[k]:>10.4f} {dpr_bb[k]:>10.4f} {pg_a[k]:>8.4f} {pg_b[k]:>8.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    ax_dpr, ax_pg = axes

    # Color/style convention: model A = blue, model B = orange. Coupled = solid, off-diag = dashed.
    ax_dpr.plot(t_axis, dpr_aa, "o-",  color="tab:blue",   linewidth=2.0,
                label=fr"DPR($\beta$={args.beta_a:g}, $\beta$={args.beta_a:g})  coupled")
    ax_dpr.plot(t_axis, dpr_ab, "s--", color="tab:blue",   linewidth=1.6, alpha=0.75,
                label=fr"DPR($\beta$={args.beta_a:g}, $\beta$={args.beta_b:g})  off-diag")
    ax_dpr.plot(t_axis, dpr_ba, "s--", color="tab:orange", linewidth=1.6, alpha=0.75,
                label=fr"DPR($\beta$={args.beta_b:g}, $\beta$={args.beta_a:g})  off-diag")
    ax_dpr.plot(t_axis, dpr_bb, "o-",  color="tab:orange", linewidth=2.0,
                label=fr"DPR($\beta$={args.beta_b:g}, $\beta$={args.beta_b:g})  coupled")
    ax_dpr.set_xlabel("snapshot t")
    ax_dpr.set_ylabel(r"DPR$_t(\beta_D, \beta_M)$")
    if args.log_y:
        ax_dpr.set_yscale("log")
    ax_dpr.set_title(f"Two-model DPR vs round  ({la} vs {lb})")
    ax_dpr.legend(fontsize=8, loc="best")
    ax_dpr.grid(alpha=0.3)

    ax_pg.plot(t_axis, pg_a, "o-", color="tab:blue",   linewidth=2.0,
               label=fr"PG($\beta$={args.beta_a:g}) on 2x2 submatrix")
    ax_pg.plot(t_axis, pg_b, "o-", color="tab:orange", linewidth=2.0,
               label=fr"PG($\beta$={args.beta_b:g}) on 2x2 submatrix")
    ax_pg.axhline(0, color="black", linewidth=0.5)
    ax_pg.set_xlabel("snapshot t")
    ax_pg.set_ylabel(r"PG_t = PR - min$_j$ DPR  (restricted to {a, b})")
    ax_pg.set_title("Performative gap vs round (restricted)")
    ax_pg.legend(fontsize=9, loc="best")
    ax_pg.grid(alpha=0.3)

    fig.suptitle(f"Two-model decoupled risk over rounds: family='{args.family_regex}'")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
