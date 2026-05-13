"""Population-steering analysis across interventions over time.

Answers (in part) the questions:
    1. Does beta change drift *magnitude* or *qualitative trajectory*?
    2. Which betas preserve minority modes vs collapse to unimodal?
    3. Do distinct betas converge to distinct D* at terminal?

For each intervention's trajectory pickle (N, T+1), computes:
    * per-round (mean, std, skew, kurtosis, bimodality_coef)
    * per-round KL(D_t || D_0) using a binned histogram on [0, 1]
    * per-round Wasserstein-1(D_t, D_0)

Produces:
    * 2x3 multi-panel figure (terminal histograms + 5 trajectory panels)
    * CSV with all per-round numbers for downstream plotting
    * Stdout summary table

CLI:
    python tools/analyze_population_steering.py \
        --tag-glob "fj_cf_*_egta" \
        --include-untuned \
        --out-fig figs/pop_steering_cf.png \
        --out-csv figs/pop_steering_cf.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, wasserstein_distance

RESULTS_DIR = Path("pokec_dataset/results")
HIST_BINS = 51

DEFAULT_HIST_OVERLAY_BETAS = [0.01, 0.1, 1.0, 10.0]


def trajectory_paths(tag_glob: str, include_untuned: bool) -> dict[str, Path]:
    """Return {label: path} for trajectory pickles matching `tag_glob`.

    Trajectory filename pattern from pokec_simulations.py / run_cf_fj.py:
        llm_<tag>_trajectory.pk
    Untuned pattern from run_untuned_llm.py:
        llm_llm_untuned_<RUN_TAG>_trajectory.pk
    """
    out: dict[str, Path] = {}
    pat = re.compile(rf"^llm_({tag_glob.replace('*', '.*')})_trajectory\.pk$")
    for p in sorted(RESULTS_DIR.glob("llm_*_trajectory.pk")):
        m = pat.match(p.name)
        if m:
            out[m.group(1)] = p
    if include_untuned:
        for p in sorted(RESULTS_DIR.glob("llm_llm_untuned_*_trajectory.pk")):
            label = "untuned_" + p.name[len("llm_llm_untuned_"):-len("_trajectory.pk")]
            out[label] = p
    return out


def parse_beta(label: str) -> float | None:
    """Extract beta from a label like 'fj_cf_b0p01_egta' or 'fj_sftkl_b1_egta'.
    Returns None for static rows (e.g. 'untuned_...')."""
    m = re.search(r"_b(\d+(?:p\d+)?)_", label)
    if not m:
        return None
    s = m.group(1).replace("p", ".")
    try:
        return float(s)
    except ValueError:
        return None


def family(label: str) -> str:
    """Group label by training-style family for color coding."""
    if "sftkl" in label or "sft_b" in label:
        return "sft"
    if "rlkl" in label:
        return "rlkl"
    if "cf" in label:
        return "cf"
    if "untuned" in label:
        return "untuned"
    return "other"


def bimodality_coefficient(x: np.ndarray) -> float:
    """SAS bimodality coefficient. > 0.555 suggests bimodal/multimodal.

    BC = (skew^2 + 1) / (kurt + 3*(n-1)^2 / ((n-2)*(n-3)))
    """
    n = len(x)
    if n < 4:
        return float("nan")
    s = skew(x)
    k = kurtosis(x, fisher=True)
    denom = k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if denom <= 0:
        return float("nan")
    return float((s * s + 1.0) / denom)


def kl_histogram(p: np.ndarray, q: np.ndarray, n_bins: int = HIST_BINS) -> float:
    """KL(P||Q) computed on a binned histogram over [0, 1] with Laplace smoothing."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[-1] = 1.0 + 1e-9
    p_counts, _ = np.histogram(p, bins=edges)
    q_counts, _ = np.histogram(q, bins=edges)
    p_smoothed = (p_counts + 1e-3) / (p_counts.sum() + n_bins * 1e-3)
    q_smoothed = (q_counts + 1e-3) / (q_counts.sum() + n_bins * 1e-3)
    return float(np.sum(p_smoothed * np.log(p_smoothed / q_smoothed)))


def analyze_one(traj: np.ndarray) -> dict[str, np.ndarray]:
    """Per-round summary stats. traj is (N, T+1)."""
    N, T_plus_1 = traj.shape
    x0 = traj[:, 0]
    means = traj.mean(axis=0)
    stds = traj.std(axis=0)
    skews = np.array([skew(traj[:, t]) for t in range(T_plus_1)])
    kurts = np.array([kurtosis(traj[:, t], fisher=True) for t in range(T_plus_1)])
    bims = np.array([bimodality_coefficient(traj[:, t]) for t in range(T_plus_1)])
    kls = np.array([kl_histogram(traj[:, t], x0) for t in range(T_plus_1)])
    ws = np.array([wasserstein_distance(traj[:, t], x0) for t in range(T_plus_1)])
    return {
        "mean": means, "std": stds, "skew": skews, "kurt": kurts,
        "bim": bims, "kl_to_x0": kls, "wass_to_x0": ws,
    }


FAMILY_COLORS = {
    "sft": "tab:blue",
    "rlkl": "tab:red",
    "cf": "tab:green",
    "untuned": "black",
    "other": "tab:gray",
}


def beta_alpha(beta: float | None) -> float:
    """Lower beta -> lower alpha, so high-beta lines stand out as nearly opaque."""
    if beta is None:
        return 1.0
    b = float(np.clip(beta, 1e-3, 100.0))
    return 0.35 + 0.55 * (np.log10(b) + 2) / 4  # 0.35 at b=0.01, 0.9 at b=100


def make_figure(stats: dict[str, dict[str, np.ndarray]],
                trajs: dict[str, np.ndarray],
                out_fig: Path,
                hist_overlay_betas: list[float]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.ravel()
    ax_hist, ax_mean, ax_std, ax_bim, ax_kl, ax_wass = axes

    rounds = None
    for label, s in stats.items():
        if rounds is None:
            rounds = np.arange(len(s["mean"]))
        fam = family(label)
        beta = parse_beta(label)
        color = FAMILY_COLORS.get(fam, "tab:gray")
        alpha = beta_alpha(beta)
        linestyle = "-" if fam in ("cf", "sft") else ("--" if fam == "rlkl" else ":")
        ax_mean.plot(rounds, s["mean"], color=color, alpha=alpha, linestyle=linestyle, label=label)
        ax_std.plot(rounds,  s["std"],  color=color, alpha=alpha, linestyle=linestyle)
        ax_bim.plot(rounds,  s["bim"],  color=color, alpha=alpha, linestyle=linestyle)
        ax_kl.plot(rounds,   s["kl_to_x0"],   color=color, alpha=alpha, linestyle=linestyle)
        ax_wass.plot(rounds, s["wass_to_x0"], color=color, alpha=alpha, linestyle=linestyle)

    # Panel A: terminal histogram overlay for CF betas only (otherwise unreadable).
    overlay_set = set(hist_overlay_betas)
    for label, traj in trajs.items():
        fam = family(label)
        beta = parse_beta(label)
        if fam != "cf" or beta not in overlay_set:
            if fam != "untuned":
                continue
        x_T = traj[:, -1]
        color = FAMILY_COLORS.get(fam, "tab:gray")
        alpha = 0.55
        ax_hist.hist(x_T, bins=HIST_BINS, range=(0, 1), histtype="step",
                     color=color, alpha=alpha, linewidth=1.6,
                     label=f"{label} (terminal)")
    # Initial distribution as reference.
    any_traj = next(iter(trajs.values()))
    ax_hist.hist(any_traj[:, 0], bins=HIST_BINS, range=(0, 1), histtype="step",
                 color="black", linestyle="--", linewidth=1.3, label="initial x_0")

    ax_hist.set_title("Panel A: terminal opinion distribution D_T (CF + untuned)")
    ax_hist.set_xlabel("opinion value")
    ax_hist.set_ylabel("agent count")
    ax_hist.legend(fontsize=7, loc="upper right")
    ax_hist.axhline(0, color="gray", lw=0.5)

    ax_mean.axhline(0.555, alpha=0)  # spacer
    ax_mean.set_title("Panel B: population mean(t)")
    ax_mean.set_xlabel("round t"); ax_mean.set_ylabel("mean opinion")
    ax_mean.legend(fontsize=6, loc="best", ncol=2)
    ax_mean.grid(alpha=0.3)

    ax_std.set_title("Panel C: population std(t)")
    ax_std.set_xlabel("round t"); ax_std.set_ylabel("std opinion")
    ax_std.grid(alpha=0.3)

    ax_bim.set_title("Panel D: bimodality coefficient(t)")
    ax_bim.axhline(0.555, color="red", linestyle=":", alpha=0.7, label="BC=0.555")
    ax_bim.set_xlabel("round t"); ax_bim.set_ylabel("bimodality coef")
    ax_bim.legend(fontsize=7)
    ax_bim.grid(alpha=0.3)

    ax_kl.set_title("Panel E: KL(D_t || D_0)")
    ax_kl.set_xlabel("round t"); ax_kl.set_ylabel("KL on 51-bin hist")
    ax_kl.grid(alpha=0.3)

    ax_wass.set_title("Panel F: Wasserstein(D_t, D_0)")
    ax_wass.set_xlabel("round t"); ax_wass.set_ylabel("W_1")
    ax_wass.grid(alpha=0.3)

    fig.suptitle("Population-steering analysis: per-beta distributional trajectories")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_csv(stats: dict[str, dict[str, np.ndarray]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label", "beta", "family", "t", "mean", "std", "skew", "kurt", "bim", "kl_to_x0", "wass_to_x0"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for label, s in stats.items():
            beta = parse_beta(label)
            fam = family(label)
            for t in range(len(s["mean"])):
                w.writerow({
                    "label": label,
                    "beta": "" if beta is None else f"{beta}",
                    "family": fam,
                    "t": t,
                    "mean":       f"{s['mean'][t]:.6f}",
                    "std":        f"{s['std'][t]:.6f}",
                    "skew":       f"{s['skew'][t]:.6f}",
                    "kurt":       f"{s['kurt'][t]:.6f}",
                    "bim":        f"{s['bim'][t]:.6f}",
                    "kl_to_x0":   f"{s['kl_to_x0'][t]:.6f}",
                    "wass_to_x0": f"{s['wass_to_x0'][t]:.6f}",
                })


def print_summary(stats: dict[str, dict[str, np.ndarray]]) -> None:
    print()
    print(f"{'label':<26} {'beta':>7} {'fam':<8} {'mean_T':>8} {'std_T':>8} "
          f"{'bim_T':>8} {'KL_T':>8} {'W_T':>8}")
    print("-" * 88)
    rows = []
    for label, s in stats.items():
        beta = parse_beta(label)
        rows.append((parse_beta(label) if parse_beta(label) is not None else -1.0,
                     label, beta, family(label), s))
    rows.sort()
    for _, label, beta, fam, s in rows:
        beta_str = f"{beta:.3g}" if beta is not None else "-"
        print(f"{label:<26} {beta_str:>7} {fam:<8} "
              f"{s['mean'][-1]:>8.4f} {s['std'][-1]:>8.4f} "
              f"{s['bim'][-1]:>8.4f} {s['kl_to_x0'][-1]:>8.4f} {s['wass_to_x0'][-1]:>8.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-glob", type=str, default="fj_*_egta",
                    help="Glob over the <tag> portion of llm_<tag>_trajectory.pk")
    ap.add_argument("--include-untuned", action="store_true",
                    help="Also include llm_llm_untuned_*_trajectory.pk files.")
    ap.add_argument("--out-fig", type=Path, default=Path("figs/pop_steering.png"))
    ap.add_argument("--out-csv", type=Path, default=Path("figs/pop_steering.csv"))
    ap.add_argument("--hist-overlay-betas", type=float, nargs="+",
                    default=DEFAULT_HIST_OVERLAY_BETAS,
                    help=("Betas to overlay in the terminal histogram panel. "
                          "Must match parse_beta output exactly (e.g. 0.01 not 0.010). "
                          f"Default: {DEFAULT_HIST_OVERLAY_BETAS}"))
    args = ap.parse_args()

    paths = trajectory_paths(args.tag_glob, args.include_untuned)
    if not paths:
        print(f"no trajectory files matched tag-glob='{args.tag_glob}' "
              f"(include_untuned={args.include_untuned}) in {RESULTS_DIR}")
        return

    trajs: dict[str, np.ndarray] = {}
    stats: dict[str, dict[str, np.ndarray]] = {}
    for label, p in paths.items():
        with open(p, "rb") as f:
            traj = np.asarray(pickle.load(f), dtype=np.float64)
        if traj.ndim != 2:
            print(f"[skip] {label}: shape={traj.shape}")
            continue
        trajs[label] = traj
        stats[label] = analyze_one(traj)
        print(f"  loaded {label:<26} shape={traj.shape}")

    print_summary(stats)
    write_csv(stats, args.out_csv)
    make_figure(stats, trajs, args.out_fig, args.hist_overlay_betas)
    print()
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_fig}")


if __name__ == "__main__":
    main()
