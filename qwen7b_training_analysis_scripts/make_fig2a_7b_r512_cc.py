"""Fig 2a replica for the 7B LoRA r=512 chat-template beta sweep.

Reads trajectories from /Users/gabesmithline/Desktop/results_7b_512mask/ and
plots per-round population mean and variance across 9 beta values, plus the
untuned 7B base model and the original-paper baselines (perfect/mean/ridge/MLP)
from pokec_dataset/results/.
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

RES = "/Users/gabesmithline/Desktop/results_7b_512mask"
BASELINE_RES = "pokec_dataset/results/"
OUT = "/Users/gabesmithline/Desktop/ellis_work/Opinion-dynamics-post-training/figs"
os.makedirs(OUT, exist_ok=True)

runs = [
    ("llm_llm_7b_r512_cc_b0",     0.0,  "beta=0 (SFT)"),
    ("llm_llm_7b_r512_cc_b0p01",  0.01, "beta=0.01"),
    ("llm_llm_7b_r512_cc_b0p05",  0.05, "beta=0.05"),
    ("llm_llm_7b_r512_cc_b0p1",   0.1,  "beta=0.1"),
    ("llm_llm_7b_r512_cc_b0p3",   0.3,  "beta=0.3"),
    ("llm_llm_7b_r512_cc_b0p5",   0.5,  "beta=0.5"),
    ("llm_llm_7b_r512_cc_b1",     1.0,  "beta=1"),
    ("llm_llm_7b_r512_cc_b3",     3.0,  "beta=3"),
    ("llm_llm_7b_r512_cc_b10",    10.0, "beta=10"),
]

# (filename stem, source dir, label, color, linestyle)
baselines = [
    ("llm_llm_untuned_untuned_7b", RES,          "untuned 7B (no post-training)", "black",   "--"),
    ("perfect",                    BASELINE_RES, "perfect",                       "#d62728", "--"),
    ("mean",                       BASELINE_RES, "mean",                          "#ff7f0e", "--"),
    ("ridge",                      BASELINE_RES, "ridge",                         "#8c564b", "--"),
    ("neural_net_mlp",             BASELINE_RES, "MLP",                           "#e377c2", "--"),
]

cmap = plt.get_cmap("viridis")
colors = [cmap(i / (len(runs) - 1)) for i in range(len(runs))]

y_lab   = pickle.load(open("pokec_dataset/parametric_params/y_label2163.pk", "rb"))
y_unlab = pickle.load(open("pokec_dataset/parametric_params/y_unlabel_label2163.pk", "rb"))
innate  = np.concatenate([y_lab, y_unlab])
innate_mean = float(innate.mean())
innate_std  = float(innate.std())
innate_var  = float(innate.var())

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

for (fname, beta, label), color in zip(runs, colors):
    path = os.path.join(RES, fname + "_trajectory.pk")
    if not os.path.exists(path):
        print(f"  skip (missing): {path}")
        continue
    traj = pickle.load(open(path, "rb"))
    per_round_mean = traj.mean(axis=0)
    per_round_var  = traj.var(axis=0)
    rounds = np.arange(per_round_mean.shape[0])
    axes[0].plot(rounds, per_round_mean, marker="o", markersize=3,
                 linestyle="-", color=color, label=label)
    axes[1].plot(rounds, per_round_var, marker="o", markersize=3,
                 linestyle="-", color=color, label=label)

for fname, src, label, color, ls in baselines:
    path = os.path.join(src, fname + "_trajectory.pk")
    if not os.path.exists(path):
        print(f"  skip (missing baseline): {path}")
        continue
    traj = pickle.load(open(path, "rb"))
    per_round_mean = traj.mean(axis=0)
    per_round_var  = traj.var(axis=0)
    rounds = np.arange(per_round_mean.shape[0])
    axes[0].plot(rounds, per_round_mean, linestyle=ls, color=color,
                 alpha=0.85, label=label)
    axes[1].plot(rounds, per_round_var, linestyle=ls, color=color,
                 alpha=0.85, label=label)

axes[0].axhline(innate_mean, color="black", linestyle=":", alpha=0.6,
                label=f"innate mean = {innate_mean:.4f}")
axes[0].set_xlabel("retraining step t")
axes[0].set_ylabel("mean expressed opinion")
axes[0].set_title("7B LoRA r=512 (chat template): mean opinion across rounds")
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=8, loc="best", ncol=2)

axes[1].axhline(innate_var, color="black", linestyle=":", alpha=0.6,
                label=f"innate var = {innate_var:.4f}")
axes[1].set_xlabel("retraining step t")
axes[1].set_ylabel("variance of expressed opinion")
axes[1].set_title("7B LoRA r=512 (chat template): variance across rounds")
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=8, loc="best", ncol=2)

fig.suptitle("Fig 2a replica (7B LoRA r=512, chat-templated prompts): opinion family vs beta")
fig.tight_layout()
out_path = os.path.join(OUT, "fig2a_7b_r512_cc.png")
fig.savefig(out_path, dpi=130)
print(f"wrote {out_path}")

print("\nFinal-round population stats:")
print(f"  {'method':>30s}  {'mean':>8s}  {'d_innate':>9s}  {'std':>8s}  {'var':>9s}")
for fname, beta, label in runs:
    path = os.path.join(RES, fname + "_trajectory.pk")
    if not os.path.exists(path):
        continue
    traj = pickle.load(open(path, "rb"))
    m = float(traj[:, -1].mean())
    s = float(traj[:, -1].std())
    v = float(traj[:, -1].var())
    print(f"  {label:>30s}  {m:>8.4f}  {m-innate_mean:+.4f}    {s:>8.4f}  {v:>9.5f}")

for fname, src, label, _color, _ls in baselines:
    path = os.path.join(src, fname + "_trajectory.pk")
    if not os.path.exists(path):
        continue
    traj = pickle.load(open(path, "rb"))
    m = float(traj[:, -1].mean())
    s = float(traj[:, -1].std())
    v = float(traj[:, -1].var())
    print(f"  {label:>30s}  {m:>8.4f}  {m-innate_mean:+.4f}    {s:>8.4f}  {v:>9.5f}")

print(f"\ninnate reference: mean={innate_mean:.4f}  std={innate_std:.4f}  var={innate_var:.5f}")
