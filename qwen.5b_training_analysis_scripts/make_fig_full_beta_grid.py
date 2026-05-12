"""Full beta sweep grid: capacity (rows) x [mean, variance] (columns).
Low-beta (0 to 10) from results_mask/, high-beta (100 to 1000) from results_large_beta/.
Baselines + untuned overlaid on every row."""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

BASE = "pokec_dataset/results/"
OLD  = "/Users/gabesmithline/Desktop/results_mask/"
BIG  = "/Users/gabesmithline/Desktop/results_large_beta/"
OUT  = "figs"
os.makedirs(OUT, exist_ok=True)

y_lab = pickle.load(open("pokec_dataset/parametric_params/y_label2163.pk", "rb"))
y_unl = pickle.load(open("pokec_dataset/parametric_params/y_unlabel_label2163.pk", "rb"))
innate = np.concatenate([y_lab, y_unl])
innate_mean = float(innate.mean())
innate_var = float(innate.var())

capacities = [
    ("r=1",     "r1"),
    ("r=2",     "r2"),
    ("r=8",     "r8"),
    ("r=32",    ""),
    ("full-FT", "ff"),
]

betas = [0.0, 0.3, 1.0, 3.0, 10.0, 100.0, 200.0, 300.0, 400.0, 500.0,
         700.0, 800.0, 900.0, 1000.0]

# betas below 100 -> OLD, 100+ -> BIG (except r=1 which has no high-beta)
baselines = [
    ("perfect", "perfect",        "#7f7f7f"),
    ("MLP",     "neural_net_mlp", "#8c564b"),
    ("ridge",   "ridge",          "#17becf"),
    ("mean",    "mean",           "#bcbd22"),
]
UNTUNED = os.path.join(OLD, "llm_llm_untuned_trajectory.pk")


def old_stem(beta, cs):
    if beta == 0.0:
        bs = "sft_mask"
    else:
        b_tag = {0.3: "b0p3", 1.0: "b1", 3.0: "b3", 10.0: "b10"}[beta]
        bs = f"sftkl_mask_{b_tag}"
    if cs == "":
        return f"llm_llm_{bs}"
    if bs == "sft_mask":
        return f"llm_llm_sft_mask_{cs}"
    b_part = bs.replace("sftkl_mask_", "")
    return f"llm_llm_sftkl_mask_{cs}_{b_part}"


def new_stem(beta, cs):
    tag = f"b{int(beta)}"
    if cs == "":
        return f"llm_llm_sftkl_mask_{tag}"
    return f"llm_llm_sftkl_mask_{cs}_{tag}"


def beta_path(beta, cs):
    if beta < 100:
        return os.path.join(OLD, old_stem(beta, cs) + "_trajectory.pk")
    return os.path.join(BIG, new_stem(beta, cs) + "_trajectory.pk")


def load_traj(path):
    if not os.path.exists(path):
        return None
    return pickle.load(open(path, "rb"))


# color map: linear viridis for betas; untuned = black
norm = Normalize(vmin=0.0, vmax=1000.0)
cmap = plt.cm.viridis

def color_for_beta(b):
    return cmap(norm(b))


fig, axes = plt.subplots(len(capacities), 2, figsize=(14, 2.7 * len(capacities)),
                         sharex=True)

final_rows = []
for i, (cap_label, cs) in enumerate(capacities):
    ax_m, ax_v = axes[i, 0], axes[i, 1]

    # baselines on every row; legend only on top row
    for lab, stem, col in baselines:
        tr = load_traj(os.path.join(BASE, f"{stem}_trajectory.pk"))
        if tr is None:
            continue
        r = np.arange(tr.shape[1])
        ax_m.plot(r, tr.mean(axis=0), ls="--", color=col, lw=1.0, alpha=0.8,
                  label=lab if i == 0 else None)
        ax_v.plot(r, tr.var(axis=0),  ls="--", color=col, lw=1.0, alpha=0.8,
                  label=lab if i == 0 else None)
        if i == 0:
            final_rows.append((f"baseline:{lab}", tr[:, -1].mean(), tr[:, -1].var()))

    # LLM beta sweep for this capacity
    for b in betas:
        path = beta_path(b, cs)
        tr = load_traj(path)
        if tr is None:
            continue
        r = np.arange(tr.shape[1])
        col = color_for_beta(b)
        ax_m.plot(r, tr.mean(axis=0), "-", color=col, lw=1.2, alpha=0.9)
        ax_v.plot(r, tr.var(axis=0),  "-", color=col, lw=1.2, alpha=0.9)
        final_rows.append((f"{cap_label}/b={b}", tr[:, -1].mean(), tr[:, -1].var()))

    # untuned
    tr_u = load_traj(UNTUNED)
    if tr_u is not None:
        r = np.arange(tr_u.shape[1])
        ax_m.plot(r, tr_u.mean(axis=0), "-", color="black", lw=1.8,
                  label="untuned (b=inf)" if i == 0 else None)
        ax_v.plot(r, tr_u.var(axis=0), "-", color="black", lw=1.8,
                  label="untuned (b=inf)" if i == 0 else None)
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

# shared colorbar for beta
sm = ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02,
                    location="right", label="beta  (b=inf shown as black line)")

fig.suptitle("masked-SFT sweep (full): capacity x beta. Dotted=innate; dashed=classical baselines",
             y=0.995)

out_path = os.path.join(OUT, "fig_full_beta_grid.png")
fig.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"wrote {out_path}")

print(f"\nFinal-round stats (T=30), innate mean={innate_mean:.4f}, var={innate_var:.5f}")
print(f"  {'config':28s}  final     drift     var")
for name, m, v in final_rows:
    print(f"  {name:28s}  {m:.4f}  {m-innate_mean:+.4f}   {v:.5f}")
