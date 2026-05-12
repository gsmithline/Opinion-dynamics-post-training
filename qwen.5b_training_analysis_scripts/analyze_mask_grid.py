"""Summarize masked-SFT sweep: peak detection, hump vs runaway, end-slope, and var.
Produces a capacity x beta table. Classifies each trajectory as:
  rise   : final within 2% of peak (still rising or plateau)
  hump   : peak at least 0.25 pp above final (turned and came down)
  flat   : tiny motion from innate
"""
import os
import pickle
import numpy as np

LLM = "/Users/gabesmithline/Desktop/results_mask/"

y_lab = pickle.load(open("pokec_dataset/parametric_params/y_label2163.pk", "rb"))
y_unlab = pickle.load(open("pokec_dataset/parametric_params/y_unlabel_label2163.pk", "rb"))
innate = np.concatenate([y_lab, y_unlab])
innate_mean = float(innate.mean())
innate_var = float(innate.var())

caps = [("r1", "r=1"), ("r2", "r=2"), ("r8", "r=8"), ("", "r=32"), ("ff", "full-FT")]
bets = [("sft_mask", "beta=0"),
        ("sftkl_mask_b0p3", "beta=0.3"),
        ("sftkl_mask_b1",   "beta=1"),
        ("sftkl_mask_b3",   "beta=3"),
        ("sftkl_mask_b10",  "beta=10")]


def stem(beta_stem, cap_suffix):
    if cap_suffix == "":
        return f"llm_llm_{beta_stem}"
    if beta_stem == "sft_mask":
        return f"llm_llm_sft_mask_{cap_suffix}"
    bpart = beta_stem.replace("sftkl_mask_", "")
    return f"llm_llm_sftkl_mask_{cap_suffix}_{bpart}"


def classify(mean_curve, innate):
    peak = mean_curve.max()
    peak_t = int(mean_curve.argmax())
    final = mean_curve[-1]
    drift = final - innate
    peak_drop = peak - final
    end_slope = mean_curve[-1] - mean_curve[-5]  # last 5 rounds
    if abs(drift) < 0.002:
        cls = "flat"
    elif peak_drop > 0.0025 and peak_t < len(mean_curve) - 2:
        cls = "hump"
    else:
        cls = "rise"
    return peak, peak_t, final, drift, end_slope, cls


print(f"innate_mean = {innate_mean:.4f}   innate_var = {innate_var:.5f}\n")
hdr = f"{'config':18s}  {'peak':>7s}  {'t*':>3s}  {'final':>7s}  {'drift':>7s}  {'end5':>7s}  {'var_T':>7s}  class"
print(hdr)
print("-" * len(hdr))

for cap_suffix, cap_label in caps:
    for beta_stem, beta_label in bets:
        path = os.path.join(LLM, stem(beta_stem, cap_suffix) + "_trajectory.pk")
        if not os.path.exists(path):
            print(f"MISSING {path}")
            continue
        tr = pickle.load(open(path, "rb"))
        m = tr.mean(axis=0)
        v_final = float(tr[:, -1].var())
        peak, peak_t, final, drift, end_slope, cls = classify(m, innate_mean)
        name = f"{cap_label}/{beta_label}"
        print(f"{name:18s}  {peak:.4f}  {peak_t:3d}  {final:.4f}  {drift:+.4f}  {end_slope:+.4f}  {v_final:.5f}  {cls}")
    print()
