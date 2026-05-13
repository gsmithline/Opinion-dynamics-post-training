"""Compare population-mean opinion across rounds for every intervention.

Loads the FJ trajectory pickles produced by the EGTA sweep (one per condor tag)
and prints a table of mean(opinions[:, t]) for each round t. Used to answer
the question 'is RL-KL pulling the opinion mean higher than untuned / SFT-KL?'

Trajectory pickle format (from pokec_simulations.py:686):
    heatmap_res1 = (agent_num, retrain_T + 1) array
    columns are rounds 0, 1, ..., T
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("pokec_dataset/results")

# (display_label, trajectory_filename) ordered for readable side-by-side table.
RUNS = [
    ("untuned",      "llm_llm_untuned_fj_untuned_egta_trajectory.pk"),
    ("sft_b0",       "llm_fj_sft_b0_egta_trajectory.pk"),
    ("sftkl_b0.01",  "llm_fj_sftkl_b0p01_egta_trajectory.pk"),
    ("sftkl_b0.1",   "llm_fj_sftkl_b0p1_egta_trajectory.pk"),
    ("sftkl_b1",     "llm_fj_sftkl_b1_egta_trajectory.pk"),
    ("sftkl_b10",    "llm_fj_sftkl_b10_egta_trajectory.pk"),
    ("rlkl_b0.01",   "llm_fj_rlkl_b0p01_egta_trajectory.pk"),
    ("rlkl_b0.1",    "llm_fj_rlkl_b0p1_egta_trajectory.pk"),
    ("rlkl_b1",      "llm_fj_rlkl_b1_egta_trajectory.pk"),
    ("rlkl_b10",     "llm_fj_rlkl_b10_egta_trajectory.pk"),
]


def load_trajectory(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        x = pickle.load(f)
    return np.asarray(x, dtype=np.float64)


def main() -> None:
    means: dict[str, np.ndarray] = {}
    T_plus_1: int | None = None
    n_agents: int | None = None
    for label, fname in RUNS:
        p = RESULTS_DIR / fname
        if not p.exists():
            print(f"[skip] missing {p}")
            continue
        traj = load_trajectory(p)
        if traj.ndim != 2:
            print(f"[skip] {label}: unexpected shape {traj.shape}")
            continue
        if T_plus_1 is None:
            T_plus_1 = traj.shape[1]
            n_agents = traj.shape[0]
        elif traj.shape[1] != T_plus_1:
            print(f"[warn] {label}: T+1={traj.shape[1]} differs from {T_plus_1}")
        means[label] = traj.mean(axis=0)
        print(f"  loaded {label:<14} shape={traj.shape}")

    if not means:
        print("no trajectories loaded; aborting.")
        return

    rounds = list(range(T_plus_1))
    labels = list(means.keys())

    # Header row.
    print()
    print("Population-mean opinion per round")
    print("=" * 80)
    header = "round | " + " | ".join(f"{lbl:>11}" for lbl in labels)
    print(header)
    print("-" * len(header))
    for t in rounds:
        row = f"{t:>5} | " + " | ".join(f"{means[lbl][t]:>11.4f}" for lbl in labels)
        print(row)

    # Deltas vs untuned at terminal.
    if "untuned" in means:
        print()
        print("Terminal (t=T) means and delta vs untuned")
        print("=" * 60)
        u_T = means["untuned"][-1]
        print(f"  untuned                   = {u_T:.4f}")
        for lbl in labels:
            if lbl == "untuned":
                continue
            v_T = means[lbl][-1]
            print(f"  {lbl:<22}    = {v_T:.4f}  (Δ vs untuned = {v_T - u_T:+.4f})")

    # Initial vs terminal shift for each.
    print()
    print("Per-method drift (terminal - initial)")
    print("=" * 60)
    for lbl in labels:
        v = means[lbl]
        print(f"  {lbl:<22}    initial={v[0]:.4f}  terminal={v[-1]:.4f}  drift={v[-1] - v[0]:+.4f}")


if __name__ == "__main__":
    main()
