"""Plot sliver-training spillover: does training on subgroup S pull V\\S toward mu_S?

For each beta, produces a panel comparing:
  - baseline run (SFT-KL trained on full V')
  - sliver run  (SFT-KL trained on V' intersect S)

Each panel shows mu_S(t) and mu_V\\S(t) over rounds, for both runs.
If the sliver curves bend toward each other and the baseline curves stay
parallel, that's evidence of demographic-targeted spillover.

The subgroup mask for the BASELINE is recomputed on-the-fly from the same
preset/spec, since the baseline run has no sidecar. The sliver runs are
required to have a sidecar (saved automatically by pokec_simulations.py).

CLI:
  python tools/plot_sliver_spillover.py \\
      --subgroup age_high3 \\
      --baseline-tags fj_sft_b0_egta fj_sftkl_b1_egta fj_sftkl_b10_egta \\
      --sliver-tags   fj_sftkl_age_high3_b0_egta fj_sftkl_age_high3_b1_egta fj_sftkl_age_high3_b10_egta \\
      --betas 0 1 10 \\
      --results-dir pokec_dataset/results \\
      --out plots/sliver_age_high3.png
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running from repo root: tools/ -> sys.path
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

# pandas 2.2 unpickling shim (same pattern as other scripts).
_orig_sd_init = pd.StringDtype.__init__
def _sd_init(self, storage=None, na_value=None, *args, **kwargs):
    try:
        _orig_sd_init(self, storage=storage)
    except TypeError:
        _orig_sd_init(self)
pd.StringDtype.__init__ = _sd_init
from pandas.core.arrays.string_ import StringArray
_orig_ss = getattr(StringArray, "__setstate__", None)
def _ss(self, state):
    try:
        if _orig_ss:
            return _orig_ss(self, state)
    except Exception:
        pass
    def _find(x):
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, (tuple, list)):
            for e in x:
                r = _find(e)
                if r is not None:
                    return r
        return None
    arr = _find(state)
    if arr is None:
        raise ValueError("no ndarray in state")
    StringArray.__init__(self, arr.astype(object))
StringArray.__setstate__ = _ss

from subgroup_helper import Subgroup


# --- repo paths (default to env-mirrored layout) ----------------------------
TARGET = "relation_to_smoking"
DEFAULT_PROFILES = f"pokec_dataset/lcc_profiles_{TARGET}.pk"
DEFAULT_YLAB = "pokec_dataset/parametric_params/y_label2163.pk"
DEFAULT_YUNL = "pokec_dataset/parametric_params/y_unlabel_label2163.pk"


def _trajectory_path(results_dir: Path, run_tag: str) -> Path:
    return results_dir / f"llm_{run_tag}_trajectory.pk"


def _sidecar_path(results_dir: Path, run_tag: str) -> Path:
    return results_dir / f"llm_{run_tag}_subgroup_mask.pk"


def _load_full_df(profiles_path: str, y_lab_path: str, y_unl_path: str) -> pd.DataFrame:
    """Reconstruct the full df in (labeled, then unlabeled) row order."""
    df = pickle.load(open(profiles_path, "rb"))
    y_lab = pickle.load(open(y_lab_path, "rb"))
    y_unl = pickle.load(open(y_unl_path, "rb"))
    n_lab = len(y_lab)
    n_unl = len(y_unl)
    if len(df) != n_lab + n_unl:
        raise RuntimeError(
            f"df len {len(df)} != n_lab+n_unl ({n_lab}+{n_unl})"
        )
    return df


def _load_traj_and_mask(results_dir: Path, run_tag: str,
                       fallback_subgroup: Subgroup | None,
                       df_full: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray, str]:
    """Returns (trajectory, mask, mask_source).
    mask_source is 'sidecar' or 'recomputed'."""
    traj_path = _trajectory_path(results_dir, run_tag)
    if not traj_path.exists():
        raise FileNotFoundError(f"trajectory missing: {traj_path}")
    traj = pickle.load(open(traj_path, "rb"))

    sidecar = _sidecar_path(results_dir, run_tag)
    if sidecar.exists():
        sg = Subgroup.from_sidecar(sidecar)
        mask = sg.compute_mask(df_full) if df_full is not None else sg._mask_cache
        if mask is None:
            raise RuntimeError(f"sidecar at {sidecar} has no cached mask and df_full not supplied")
        return traj, mask, "sidecar"

    if fallback_subgroup is None or df_full is None:
        raise RuntimeError(
            f"no sidecar at {sidecar} and no fallback subgroup/df provided for {run_tag}"
        )
    mask = fallback_subgroup.compute_mask(df_full)
    if mask.shape[0] != traj.shape[0]:
        raise RuntimeError(
            f"mask len {mask.shape[0]} != traj rows {traj.shape[0]} for {run_tag}"
        )
    return traj, mask, "recomputed"


def _plot_panel(ax, traj_base, mask_base, traj_sliver, mask_sliver, beta: float):
    rounds = np.arange(traj_base.shape[1])
    out_base = Subgroup(tag="_dummy", custom_fn=lambda _: mask_base).split_means(traj_base, mask_base)
    out_sliv = Subgroup(tag="_dummy", custom_fn=lambda _: mask_sliver).split_means(traj_sliver, mask_sliver)

    ax.plot(rounds, out_base["mu_S"], color="tab:blue", linestyle="--",
            label=f"baseline mu_S (n={out_base['n_S']})")
    ax.plot(rounds, out_base["mu_not_S"], color="tab:orange", linestyle="--",
            label=f"baseline mu_V\\S (n={out_base['n_not_S']})")
    ax.plot(rounds, out_sliv["mu_S"], color="tab:blue", linestyle="-",
            label=f"sliver mu_S")
    ax.plot(rounds, out_sliv["mu_not_S"], color="tab:orange", linestyle="-",
            label=f"sliver mu_V\\S")

    ax.set_title(f"beta = {beta}")
    ax.set_xlabel("round")
    ax.set_ylabel("mean opinion")
    ax.grid(alpha=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subgroup", required=True,
                   help="Preset name or inline spec (e.g. age_high3 or 'col:age;vals:14|15|17')")
    p.add_argument("--baseline-tags", nargs="+", required=True,
                   help="Run tags for baseline (full V' trained) runs, one per beta")
    p.add_argument("--sliver-tags", nargs="+", required=True,
                   help="Run tags for sliver runs, one per beta")
    p.add_argument("--betas", nargs="+", type=float, required=True,
                   help="Beta values, same order as the tag lists")
    p.add_argument("--results-dir", default="pokec_dataset/results")
    p.add_argument("--profiles", default=DEFAULT_PROFILES)
    p.add_argument("--y-lab", default=DEFAULT_YLAB)
    p.add_argument("--y-unl", default=DEFAULT_YUNL)
    p.add_argument("--out", required=True, help="Output figure path")
    args = p.parse_args()

    if not (len(args.baseline_tags) == len(args.sliver_tags) == len(args.betas)):
        p.error("--baseline-tags, --sliver-tags, --betas must have equal length")

    sg = Subgroup.from_spec(args.subgroup)
    df_full = _load_full_df(args.profiles, args.y_lab, args.y_unl)
    print(f"[plot_sliver] subgroup={sg.tag}  |df_full|={len(df_full)}  "
          f"|S in df_full|={int(sg.compute_mask(df_full).sum())}")

    results_dir = Path(args.results_dir)
    n_panels = len(args.betas)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.0 * n_panels, 4.2), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, beta, btag, stag in zip(axes, args.betas, args.baseline_tags, args.sliver_tags):
        traj_b, mask_b, src_b = _load_traj_and_mask(results_dir, btag, sg, df_full)
        traj_s, mask_s, src_s = _load_traj_and_mask(results_dir, stag, sg, df_full)
        print(f"  beta={beta}  baseline={btag} (mask:{src_b}, T={traj_b.shape[1]-1})  "
              f"sliver={stag} (mask:{src_s}, T={traj_s.shape[1]-1})")
        _plot_panel(ax, traj_b, mask_b, traj_s, mask_s, beta)

    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Sliver spillover: subgroup = {sg.tag}", y=1.02)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
