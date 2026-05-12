"""beta=0 ONLY across capacities -- hump vs rise comparison for meeting."""
import os, pickle
import numpy as np
import matplotlib.pyplot as plt

LLM = "/Users/gabesmithline/Desktop/results_mask/"
BASE = "pokec_dataset/results/"
OUT = "figs"
os.makedirs(OUT, exist_ok=True)

y_lab = pickle.load(open("pokec_dataset/parametric_params/y_label2163.pk", "rb"))
y_unlab = pickle.load(open("pokec_dataset/parametric_params/y_unlabel_label2163.pk", "rb"))
innate = np.concatenate([y_lab, y_unlab])
innate_mean = float(innate.mean())
innate_var = float(innate.var())

runs = [
    ("r=1",      f"{LLM}llm_llm_sft_mask_r1_trajectory.pk",  "#8da0cb"),
    ("r=2",      f"{LLM}llm_llm_sft_mask_r2_trajectory.pk",  "#66c2a5"),
    ("r=8",      f"{LLM}llm_llm_sft_mask_r8_trajectory.pk",  "#fc8d62"),
    ("r=32",     f"{LLM}llm_llm_sft_mask_trajectory.pk",      "#e78ac3"),
    ("full-FT",  f"{LLM}llm_llm_sft_mask_ff_trajectory.pk",   "#a6d854"),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
for lab, path, col in runs:
    tr = pickle.load(open(path, "rb"))
    r = np.arange(tr.shape[1])
    axes[0].plot(r, tr.mean(axis=0), "-o", ms=3, color=col, label=lab)
    axes[1].plot(r, tr.var(axis=0),  "-o", ms=3, color=col, label=lab)

# baselines for context
for lab, stem, col in [("perfect", "perfect", "#7f7f7f"), ("MLP", "neural_net_mlp", "#8c564b")]:
    tr = pickle.load(open(f"{BASE}{stem}_trajectory.pk", "rb"))
    r = np.arange(tr.shape[1])
    axes[0].plot(r, tr.mean(axis=0), "--", color=col, lw=1.3, alpha=0.7, label=lab)
    axes[1].plot(r, tr.var(axis=0), "--", color=col, lw=1.3, alpha=0.7, label=lab)

axes[0].axhline(innate_mean, color="black", ls=":", alpha=0.6, label=f"innate mean {innate_mean:.3f}")
axes[1].axhline(innate_var, color="black", ls=":", alpha=0.6, label=f"innate var {innate_var:.4f}")
axes[0].set_title("beta=0 (masked SFT, no KL anchor): mean expressed opinion")
axes[1].set_title("beta=0 (masked SFT, no KL anchor): variance")
for ax in axes:
    ax.set_xlabel("retraining step t")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("mean")
axes[1].set_ylabel("variance")
fig.suptitle("Capacity effect at beta=0: hump at r=32, runaway at full-FT, slow rise at r=1/r=2")
fig.tight_layout()
out = os.path.join(OUT, "fig_mask_beta0_capacity.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")
