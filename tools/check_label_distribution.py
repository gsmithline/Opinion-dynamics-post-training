"""Diagnostic: compare labeled vs unlabeled opinion distributions.

Tests mechanism #1 from the RL-KL vs SFT-KL discussion:
    is h_p biased upward because the labeled training subset has a higher
    opinion mean than the unlabeled population?

Outputs per-subset (mean, std, median, quantiles) and a difference, plus the
empirical bin distribution at K=11 (the RLKL_N_BINS default).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

DATA_DIR = Path("pokec_dataset/parametric_params")
K = 11


def load(name: str) -> np.ndarray:
    with open(DATA_DIR / name, "rb") as f:
        x = pickle.load(f)
    return np.asarray(x, dtype=np.float64).ravel()


def summarize(name: str, y: np.ndarray) -> dict:
    return {
        "name": name,
        "n": int(y.size),
        "mean": float(y.mean()),
        "std": float(y.std(ddof=0)),
        "median": float(np.median(y)),
        "q10": float(np.quantile(y, 0.10)),
        "q90": float(np.quantile(y, 0.90)),
        "min": float(y.min()),
        "max": float(y.max()),
    }


def bin_histogram(y: np.ndarray, lo: float, hi: float, k: int) -> np.ndarray:
    edges = np.linspace(lo, hi, k + 1)
    counts, _ = np.histogram(y, bins=edges)
    return counts / max(counts.sum(), 1)


def main() -> None:
    y_lab = load("y_label2163.pk")
    y_unl = load("y_unlabel_label2163.pk")

    s_lab = summarize("labeled (SFT/h_p training subset)", y_lab)
    s_unl = summarize("unlabeled (held-out population)", y_unl)

    print("=" * 72)
    print("Opinion distribution summary")
    print("=" * 72)
    for s in (s_lab, s_unl):
        print(f"  {s['name']}  (n={s['n']})")
        print(f"    mean={s['mean']:.4f}  std={s['std']:.4f}  median={s['median']:.4f}")
        print(f"    [q10, q90] = [{s['q10']:.4f}, {s['q90']:.4f}]  range=[{s['min']:.4f}, {s['max']:.4f}]")
    print()
    print(f"  Δ mean (labeled - unlabeled) = {s_lab['mean'] - s_unl['mean']:+.4f}")
    print(f"  Δ median                      = {s_lab['median'] - s_unl['median']:+.4f}")

    # Empirical bin distribution at K=11 over the joint range (so both share edges).
    lo = float(min(y_lab.min(), y_unl.min()))
    hi = float(max(y_lab.max(), y_unl.max()))
    p_lab = bin_histogram(y_lab, lo, hi, K)
    p_unl = bin_histogram(y_unl, lo, hi, K)
    print()
    print(f"Empirical bin distribution, K={K}, range=[{lo:.3f}, {hi:.3f}]")
    print(f"  bin |  labeled  | unlabeled |  Δ (labeled - unlabeled)")
    print(f"  ----+-----------+-----------+--------------------------")
    for k, (pl, pu) in enumerate(zip(p_lab, p_unl)):
        bar_l = "#" * int(pl * 40)
        bar_u = "." * int(pu * 40)
        print(f"  {k:>3} |  {pl:.4f}  |  {pu:.4f}  |  {pl - pu:+.4f}   {bar_l}{bar_u}")

    # Conclusion the user cares about.
    print()
    delta = s_lab["mean"] - s_unl["mean"]
    upper_mass_lab = p_lab[K // 2 + 1:].sum()
    upper_mass_unl = p_unl[K // 2 + 1:].sum()
    print(f"Upper-half (bin > K/2) mass: labeled={upper_mass_lab:.4f}  unlabeled={upper_mass_unl:.4f}  "
          f"Δ={upper_mass_lab - upper_mass_unl:+.4f}")
    if abs(delta) < 0.01:
        verdict = "Negligible label-subset mean bias. Mechanism #1 unlikely."
    elif delta > 0:
        verdict = "Labeled subset skews UPWARD vs unlabeled. h_p will tilt RL-KL toward higher bins. Consistent with the observed RL-KL upward pull."
    else:
        verdict = "Labeled subset skews DOWNWARD vs unlabeled. Mechanism #1 would predict a downward pull, opposite of what you observed."
    print()
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
