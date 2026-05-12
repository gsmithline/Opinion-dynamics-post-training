"""Capacity x beta grid from masked-SFT sweep.
Rows: LoRA rank {r1, r2, r8, r32, full-FT}.  Columns: mean-opinion, variance.
All beta overlaid per capacity, baselines drawn on every row for reference.

Data:
  baselines -> pokec_dataset/results/
  masked LLMs -> /Users/gabesmithline/Desktop/results_mask/
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

BASE = "pokec_dataset/results/"
LLM = "/Users/gabesmithline/Desktop/results_mask/"
OUT = "figs"
os.makedirs(OUT, exist_ok=True)

# Innate reference
y_lab = pickle.load(open("pokec_dataset/parametric_params/y_label2163.pk", "rb"))
y_unlab = pickle.load(open("pokec_dataset/parametric_params/y_unlabel_label2163.pk", "rb"))
innate = np.concatenate([y_lab, y_unlab])
innate_mean = float(innate.mean())
innate_var = float(innate.var())

# capacity rows (label, stem_suffix)
capacities = [
    ("r=1",      "r1"),
    ("r=2",      "r2"),
    ("r=8",      "r8"),
    ("r=32",     ""),     # default rank, no suffix
    ("full-FT",  "ff"),
]

# beta columns (label, stem_for_beta, color)
betas = [
    ("beta=0",   "sft_mask",             "#1f77b4"),
    ("beta=0.3", "sftkl_mask_b0p3",      "#2ca02c"),
    ("beta=1",   "sftkl_mask_b1",        "#ff7f0e"),
    ("beta=3",   "sftkl_mask_b3",        "#d62728"),
    ("beta=10",  "sftkl_mask_b10",       "#9467bd"),
]

# baselines (label, stem, color)
baselines = [
    ("perfect", "perfect",        "#7f7f7f"),
    ("MLP",     "neural_net_mlp", "#8c564b"),
    ("ridge",   "ridge",          "#17becf"),
    ("mean",    "mean",           "#bcbd22"),
]

# untuned Qwen (beta -> infinity) — same file on every row
UNTUNED_PATH = os.path.join(LLM, "llm_llm_untuned_trajectory.pk")


def llm_stem(beta_stem: str, cap_suffix: str) -> str:
    # convention: llm_{beta_stem_with_rank_inserted}
    # examples:
    #   sft_mask + ''    -> llm_sft_mask
    #   sft_mask + 'r1'  -> llm_sft_mask_r1
    #   sft_mask + 'ff'  -> llm_sft_mask_ff
    #   sftkl_mask_b1 + ''   -> llm_sftkl_mask_b1
    #   sftkl_mask_b1 + 'r1' -> llm_sftkl_mask_r1_b1
    #   sftkl_mask_b1 + 'ff' -> llm_sftkl_mask_ff_b1
    if cap_suffix == "":
        return f"llm_llm_{beta_stem}"
    if beta_stem == "sft_mask":
        return f"llm_llm_sft_mask_{cap_suffix}"
    # split sftkl_mask_b{X} into sftkl_mask and b{X}
    head = "sftkl_mask"
    bpart = beta_stem.replace("sftkl_mask_", "")  # e.g. "b1"
    return f"llm_llm_{head}_{cap_suffix}_{bpart}"


def load_traj(path):
    if not os.path.exists(path):
        return None
    return pickle.load(open(path, "rb"))


fig, axes = plt.subplots(len(capacities), 2, figsize=(13, 2.6 * len(capacities)),
                         sharex=True)

final_rows = []  # for summary table

for i, (cap_label, cap_suffix) in enumerate(capacities):
    ax_m, ax_v = axes[i, 0], axes[i, 1]

    # baselines on every row (dashed). Legend only on top row.
    for lab, stem, col in baselines:
        tr = load_traj(os.path.join(BASE, f"{stem}_trajectory.pk"))
        if tr is None:
            continue
        r = np.arange(tr.shape[1])
        ax_m.plot(r, tr.mean(axis=0), ls="--", color=col, lw=1.1, alpha=0.8,
                  label=lab if i == 0 else None)
        ax_v.plot(r, tr.var(axis=0),  ls="--", color=col, lw=1.1, alpha=0.8,
                  label=lab if i == 0 else None)
        if i == 0:
            final_rows.append((f"baseline:{lab}", tr[:, -1].mean(), tr[:, -1].var()))

    # LLMs for this capacity
    for b_lab, b_stem, b_col in betas:
        path = os.path.join(LLM, llm_stem(b_stem, cap_suffix) + "_trajectory.pk")
        tr = load_traj(path)
        if tr is None:
            print(f"MISSING {path}")
            continue
        r = np.arange(tr.shape[1])
        ax_m.plot(r, tr.mean(axis=0), marker="o", ms=2.5, color=b_col,
                  label=b_lab if i == 0 else None)
        ax_v.plot(r, tr.var(axis=0),  marker="o", ms=2.5, color=b_col,
                  label=b_lab if i == 0 else None)
        final_rows.append((f"{cap_label}/{b_lab}", tr[:, -1].mean(), tr[:, -1].var()))

    # untuned Qwen (beta -> infinity): single trajectory shared across rows
    tr_u = load_traj(UNTUNED_PATH)
    if tr_u is not None:
        r = np.arange(tr_u.shape[1])
        ax_m.plot(r, tr_u.mean(axis=0), "-", color="black", lw=1.4,
                  label="untuned (beta=inf)" if i == 0 else None)
        ax_v.plot(r, tr_u.var(axis=0), "-", color="black", lw=1.4,
                  label="untuned (beta=inf)" if i == 0 else None)
        if i == 0:
            final_rows.append(("untuned", tr_u[:, -1].mean(), tr_u[:, -1].var()))

    ax_m.axhline(innate_mean, color="black", ls=":", alpha=0.5)
    ax_v.axhline(innate_var,  color="black", ls=":", alpha=0.5)
    ax_m.set_ylabel(cap_label, fontsize=11)
    ax_m.grid(alpha=0.3)
    ax_v.grid(alpha=0.3)

axes[0, 0].set_title("mean expressed opinion")
axes[0, 1].set_title("variance")
axes[-1, 0].set_xlabel("retraining step t")
axes[-1, 1].set_xlabel("retraining step t")
axes[0, 0].legend(fontsize=7, loc="lower right", ncol=2)

fig.suptitle("masked-SFT sweep: capacity (rows) x beta (colors). Dotted=innate; dashed grey=perfect",
             y=1.00)
fig.tight_layout()
out_path = os.path.join(OUT, "fig_mask_grid_capacity_beta.png")
fig.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"wrote {out_path}")

# Print summary table
print("\nFinal-round stats (T=30):")
print(f"  innate: mean={innate_mean:.4f}  var={innate_var:.5f}")
print(f"  {'config':28s}  mean     Δinnate    var")
for name, m, v in final_rows:
    print(f"  {name:28s}  {m:.4f}  {m-innate_mean:+.4f}   {v:.5f}")
