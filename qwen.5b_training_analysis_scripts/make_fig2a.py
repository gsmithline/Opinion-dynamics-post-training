"""Fig 2a replica: post-peer-interaction mean opinion across rounds.
Loads trajectory pickles (shape [n_agents, T+1]) and plots per-round population mean.
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

RES = "pokec_dataset/results/"
OUT = "figs"
os.makedirs(OUT, exist_ok=True)

runs = [
    ("perfect",             "perfect",             "perfect",            "#7f7f7f", "--"),
    ("mean",                "mean",                "mean",               "#bcbd22", "--"),
    ("ridge",               "ridge",               "ridge",              "#17becf", "--"),
    ("neural_net_mlp",      "neural_net_mlp",      "MLP",                "#8c564b", "--"),
    ("llm_llm_sft",         "LLM SFT (β=0)",       "LLM SFT (β=0)",      "#1f77b4", "-"),
    ("llm_llm_sftkl_b0p3",  "LLM SFT+KL β=0.3",    "LLM SFT+KL β=0.3",   "#2ca02c", "-"),
    ("llm_llm_sftkl_b1",    "LLM SFT+KL β=1",      "LLM SFT+KL β=1",     "#ff7f0e", "-"),
    ("llm_llm_sftkl_b3",    "LLM SFT+KL β=3",      "LLM SFT+KL β=3",     "#d62728", "-"),
    ("llm_llm_sftkl_b10",   "LLM SFT+KL β=10",     "LLM SFT+KL β=10",    "#9467bd", "-"),
]

# innate reference
y_lab   = pickle.load(open("pokec_dataset/parametric_params/y_label2163.pk", "rb"))
y_unlab = pickle.load(open("pokec_dataset/parametric_params/y_unlabel_label2163.pk", "rb"))
innate  = np.concatenate([y_lab, y_unlab])
innate_mean = float(innate.mean())

innate_std = float(innate.std())

innate_var = float(innate.var())

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
for fname, _, label, color, ls in runs:
    path = os.path.join(RES, fname + "_trajectory.pk")
    if not os.path.exists(path):
        print(f"  skip (missing): {path}")
        continue
    traj = pickle.load(open(path, "rb"))              # (n_agents, T+1)
    per_round_mean = traj.mean(axis=0)                # (T+1,)
    per_round_var  = traj.var(axis=0)                 # (T+1,)
    rounds = np.arange(per_round_mean.shape[0])
    axes[0].plot(rounds, per_round_mean, marker="o", markersize=3,
                 linestyle=ls, color=color, label=label)
    axes[1].plot(rounds, per_round_var, marker="o", markersize=3,
                 linestyle=ls, color=color, label=label)

axes[0].axhline(innate_mean, color="black", linestyle=":", alpha=0.6,
                label=f"innate mean = {innate_mean:.4f}")
axes[0].set_xlabel("retraining step t")
axes[0].set_ylabel("mean expressed opinion")
axes[0].set_title("mean opinion across rounds")
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=8, loc="best", ncol=2)

axes[1].axhline(innate_var, color="black", linestyle=":", alpha=0.6,
                label=f"innate var = {innate_var:.4f}")
axes[1].set_xlabel("retraining step t")
axes[1].set_ylabel("variance of expressed opinion")
axes[1].set_title("variance across rounds (homogenization)")
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=8, loc="best", ncol=2)

fig.suptitle("Fig 2a (replica): opinion after peer interaction — mean and variance")
fig.tight_layout()
out_path = os.path.join(OUT, "fig2a_mean_variance.png")
fig.savefig(out_path, dpi=130)
print(f"wrote {out_path}")

# diagnostic: final-round means + stds
print("\nFinal round population stats:")
print(f"  {'method':24s}  mean     Δinnate     std      Δinnate_std")
for fname, _, label, _, _ in runs:
    path = os.path.join(RES, fname + "_trajectory.pk")
    if os.path.exists(path):
        traj = pickle.load(open(path, "rb"))
        m = traj[:, -1].mean()
        s = traj[:, -1].std()
        print(f"  {label:24s}  {m:.4f}  {m-innate_mean:+.4f}    {s:.4f}  {s-innate_std:+.4f}")
