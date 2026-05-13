"""Paired-vector leakage over time: D(x_t(beta), x_t(reference)) vs t.

Agent-level paired distances (NOT distributional). For the same agent at the
same round, compare its opinion under the beta-trained policy vs its opinion
under the reference (untuned) policy. Summary norms:
    RMSE_t = sqrt(mean((x_beta - x_ref)^2))
    MAE_t  = mean(|x_beta - x_ref|)
    |dmu_t| = |mean(x_beta) - mean(x_ref)|   (scalar mean diff for reference)

If population steering by the trained model converges to the reference, RMSE
and MAE go to zero agent-by-agent (not just on average). |dmu| going to zero
without RMSE doing so means distributions matched but individuals shuffled.

Usage:
    python tools/leakage_over_time.py \
        --family-glob "fj_sftkl_*_egta" --extra fj_sft_b0_egta \
        --reference fj_untuned_egta \
        --subset whole \
        --out-fig figs/leakage_over_time_sftkl_whole.png
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
N_LABELED = 1730  # matches int(2163 * 0.8); first n rows are labeled.


def load_trajectory(tag: str) -> np.ndarray:
    p1 = RESULTS_DIR / f"llm_{tag}_trajectory.pk"
    p2 = RESULTS_DIR / f"llm_llm_untuned_{tag}_trajectory.pk"
    if p1.exists():
        with open(p1, "rb") as f:
            return np.asarray(pickle.load(f), dtype=np.float64)
    if p2.exists():
        with open(p2, "rb") as f:
            return np.asarray(pickle.load(f), dtype=np.float64)
    raise FileNotFoundError(f"no trajectory pickle for {tag}; tried {p1} and {p2}")


def discover(glob_pat: str) -> list[str]:
    rx = re.compile("^llm_" + glob_pat.replace("*", ".*") + r"_trajectory\.pk$")
    tags = []
    for p in sorted(RESULTS_DIR.glob("llm_*_trajectory.pk")):
        m = rx.match(p.name)
        if m:
            tags.append(p.name[len("llm_"):-len("_trajectory.pk")])
    return tags


def parse_beta(tag: str) -> float | None:
    m = re.search(r"_b(\d+(?:p\d+)?)_", tag)
    if not m:
        return None
    return float(m.group(1).replace("p", "."))


def slice_subset(traj: np.ndarray, subset: str) -> np.ndarray:
    if subset == "whole":
        return traj
    if subset == "labeled":
        return traj[:N_LABELED, :]
    if subset == "unlabeled":
        return traj[N_LABELED:, :]
    raise ValueError(f"unknown subset {subset}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family-glob", type=str, required=True)
    ap.add_argument("--extra", type=str, action="append", default=[])
    ap.add_argument("--reference", type=str, required=True)
    ap.add_argument("--subset", choices=["whole", "labeled", "unlabeled"], default="whole",
                    help="Agent subset to compare. 'labeled' matches wandb's target_mean source.")
    ap.add_argument("--out-fig", type=Path, required=True)
    args = ap.parse_args()

    tags = discover(args.family_glob) + list(args.extra)
    tags = [t for t in tags if t != args.reference]
    if not tags:
        raise SystemExit(f"no tags matched '{args.family_glob}' + extras")

    ref_full = load_trajectory(args.reference)
    ref = slice_subset(ref_full, args.subset)
    T_plus_1 = ref.shape[1]
    rounds = np.arange(T_plus_1)
    n_agents = ref.shape[0]
    print(f"reference={args.reference}  subset={args.subset}  shape={ref.shape}")

    series: dict[str, dict] = {}
    for tag in tags:
        traj_full = load_trajectory(tag)
        if traj_full.shape != ref_full.shape:
            print(f"[skip] {tag}: shape {traj_full.shape} differs from ref {ref_full.shape}")
            continue
        traj = slice_subset(traj_full, args.subset)
        beta = parse_beta(tag)
        diff = traj - ref                                       # (n_agents, T+1)
        rmse = np.sqrt(np.mean(diff ** 2, axis=0))              # (T+1,)
        mae  = np.mean(np.abs(diff), axis=0)
        dmu  = np.abs(traj.mean(axis=0) - ref.mean(axis=0))
        series[tag] = {"beta": beta, "rmse": rmse, "mae": mae, "dmu": dmu}
        print(f"  loaded {tag:<26} beta={beta}  RMSE_T={rmse[-1]:.4f}  "
              f"MAE_T={mae[-1]:.4f}  |Δμ_T|={dmu[-1]:.4f}")

    ordered = sorted(series.items(),
                     key=lambda kv: (kv[1]["beta"] is None, kv[1]["beta"] or 0.0))
    betas_for_color = [kv[1]["beta"] for kv in ordered if kv[1]["beta"] is not None]
    if betas_for_color:
        b_arr = np.array(betas_for_color, dtype=float)
        b_log = np.log10(np.clip(b_arr, 1e-3, 1e3))
        norm_b = (b_log - b_log.min()) / max(float(np.ptp(b_log)), 1e-9)
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    ax_rmse, ax_mae, ax_dmu = axes
    col_i = 0
    for tag, s in ordered:
        if s["beta"] is not None and betas_for_color:
            color = cmap(0.1 + 0.85 * norm_b[col_i])
            col_i += 1
        else:
            color = "gray"
        label = fr"$\beta$={s['beta']:g}" if s["beta"] is not None else tag
        ax_rmse.plot(rounds, s["rmse"], "o-", color=color, linewidth=1.7, markersize=4, label=label)
        ax_mae.plot(rounds,  s["mae"],  "o-", color=color, linewidth=1.7, markersize=4)
        ax_dmu.plot(rounds,  s["dmu"],  "o-", color=color, linewidth=1.7, markersize=4)

    for ax in (ax_rmse, ax_mae, ax_dmu):
        ax.set_xlabel("round t")
        ax.grid(alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)

    ax_rmse.set_ylabel(r"$\sqrt{\mathrm{mean}((x_t(\beta) - x_t(\mathrm{ref}))^2)}$")
    ax_rmse.set_title(f"RMSE per agent  (subset={args.subset}, n={n_agents})")
    ax_rmse.legend(fontsize=8, loc="best")

    ax_mae.set_ylabel(r"$\mathrm{mean}(|x_t(\beta) - x_t(\mathrm{ref})|)$")
    ax_mae.set_title("MAE per agent")

    ax_dmu.set_ylabel(r"$|\mathrm{mean}(x_t(\beta)) - \mathrm{mean}(x_t(\mathrm{ref}))|$")
    ax_dmu.set_title("|mean diff|  (scalar, for reference)")

    fig.suptitle(f"Paired-vector leakage to '{args.reference}'  (family='{args.family_glob}', subset={args.subset})")
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_fig}")


if __name__ == "__main__":
    main()
