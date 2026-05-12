"""Build the key comparison figures from wandb history.
Saves PNGs under ./figs/.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

os.makedirs("figs", exist_ok=True)

api = wandb.Api()
runs = list(api.runs("opinion-dynamics-llm", per_page=50))
rows = []
for r in runs:
    h = r.history(pandas=True)
    if h is None or h.empty:
        continue
    h["run"] = r.name
    rows.append(h)
df = pd.concat(rows, ignore_index=True)

# Collapse to one row per (run, round): bfill/ffill across fragmented _step rows
metrics = ["target_mean","target_std","pred_mean","pred_std",
           "opinion_mean","opinion_std","mean_drift_from_innate",
           "pred_bias_vs_target","std_ratio_to_innate"]
keep = [c for c in metrics if c in df.columns]
collapsed = (
    df.dropna(subset=["round"])
      .sort_values(["run","round","_step"])
      .groupby(["run","round"])[keep]
      .apply(lambda s: s.bfill().ffill().iloc[0])
      .reset_index()
)
collapsed.to_csv("wandb_history_collapsed.csv", index=False)

# Order and color runs
baseline_order = ["perfect","mean","ridge","mlp"]
llm_order = ["llm_sft","llm_sftkl_b0p3","llm_sftkl_b1","llm_sftkl_b3","llm_sftkl_b10"]
all_runs = baseline_order + llm_order
palette = {
    "perfect": "#888888", "mean": "#aaaaaa", "ridge": "#666666", "mlp": "#000000",
    "llm_sft": "#1f77b4", "llm_sftkl_b0p3": "#2ca02c",
    "llm_sftkl_b1": "#ff7f0e", "llm_sftkl_b3": "#d62728", "llm_sftkl_b10": "#9467bd",
}


def lineplot(metric, ylabel, title, outfile, ref_val=None, ref_label=None):
    fig, ax = plt.subplots(figsize=(8,5))
    for name in all_runs:
        sub = collapsed[collapsed["run"] == name].dropna(subset=[metric])
        if sub.empty: continue
        style = "-" if name.startswith("llm") else "--"
        ax.plot(sub["round"], sub[metric],
                marker="o", linestyle=style, color=palette.get(name,"gray"), label=name)
    if ref_val is not None:
        ax.axhline(ref_val, color="black", linestyle=":", alpha=0.5, label=ref_label or f"ref={ref_val:.3f}")
    ax.set_xlabel("round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(f"figs/{outfile}", dpi=130)
    plt.close(fig)
    print(f"  wrote figs/{outfile}")


# Reference: innate baseline
import pickle
y_lab = pickle.load(open("pokec_dataset/parametric_params/y_label2163.pk","rb"))
y_unlab = pickle.load(open("pokec_dataset/parametric_params/y_unlabel_label2163.pk","rb"))
innate = np.concatenate([y_lab, y_unlab])
innate_mean = float(innate.mean())

print("Building figures...")
lineplot("target_mean",  "target_mean (SFT label mean)",
         "Training-target drift over rounds  (LLMs ↑, baselines ↓)",
         "fig_target_mean.png", ref_val=innate_mean, ref_label=f"innate mean={innate_mean:.4f}")
lineplot("opinion_mean", "population opinion_mean",
         "Population mean over rounds",
         "fig_opinion_mean.png", ref_val=innate_mean, ref_label=f"innate mean={innate_mean:.4f}")
lineplot("opinion_std",  "population opinion_std",
         "Population spread over rounds",
         "fig_opinion_std.png")
lineplot("mean_drift_from_innate", "mean drift (population − innate)",
         "Population mean drift from innate",
         "fig_drift.png", ref_val=0.0, ref_label="no drift")
lineplot("target_std", "target_std", "Training-target spread over rounds", "fig_target_std.png")
lineplot("pred_mean", "pred_mean (LLM output)",
         "LLM prediction mean per round", "fig_pred_mean.png",
         ref_val=innate_mean, ref_label=f"innate mean={innate_mean:.4f}")
lineplot("pred_std",  "pred_std (LLM output spread)",
         "LLM prediction spread per round", "fig_pred_std.png")

# Final-round summary by β
llm_tag_to_beta = {
    "llm_sft": 0.0, "llm_sftkl_b0p3": 0.3, "llm_sftkl_b1": 1.0,
    "llm_sftkl_b3": 3.0, "llm_sftkl_b10": 10.0,
}
final_rows = []
for name, beta in llm_tag_to_beta.items():
    sub = collapsed[(collapsed["run"] == name) & (collapsed["round"] == collapsed["round"].max())]
    if sub.empty: continue
    r = sub.iloc[0].to_dict()
    r["beta"] = beta; r["run"] = name
    final_rows.append(r)
final_df = pd.DataFrame(final_rows).sort_values("beta")
print("\nFinal round, by beta:")
print(final_df[["beta","target_mean","opinion_mean","mean_drift_from_innate"]].to_string(index=False))

if not final_df.empty:
    fig, ax = plt.subplots(figsize=(7,5))
    for metric, color in [("target_mean","tab:blue"),
                          ("opinion_mean","tab:orange"),
                          ("mean_drift_from_innate","tab:green")]:
        if metric in final_df.columns and final_df[metric].notna().any():
            ax.plot(final_df["beta"], final_df[metric], "o-", color=color, label=metric)
    ax.axhline(innate_mean, color="black", linestyle=":", alpha=0.5, label=f"innate mean={innate_mean:.4f}")
    ax.set_xscale("symlog", linthresh=0.3)
    ax.set_xlabel("KL β (0 → SFT only, large → near base model)")
    ax.set_ylabel("value at final round")
    ax.set_title("β-sweep at final round: does higher β reduce population pull?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("figs/fig_beta_sweep.png", dpi=130)
    plt.close(fig)
    print("  wrote figs/fig_beta_sweep.png")

print("\nDone. figs/ has the plots.")
