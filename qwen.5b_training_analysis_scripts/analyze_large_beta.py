"""Summarize high-beta sweep (β=100..1000) across capacities."""
import os, pickle
import numpy as np

BIG = "/Users/gabesmithline/Desktop/results_large_beta/"
OLD = "/Users/gabesmithline/Desktop/results_mask/"

y_lab = pickle.load(open("pokec_dataset/parametric_params/y_label2163.pk", "rb"))
y_unl = pickle.load(open("pokec_dataset/parametric_params/y_unlabel_label2163.pk", "rb"))
innate = np.concatenate([y_lab, y_unl])
inn_mean = float(innate.mean())
inn_var  = float(innate.var())

# Include β=0, 0.3, 1, 3, 10 from OLD dir (already there); β=100-1000 from BIG.
# Tag mapping: (label, cap_suffix) per capacity
caps = [("r=2", "r2"), ("r=8", "r8"), ("r=32", ""), ("full-FT", "ff")]
betas_old = [(0.0, "sft_mask"),  (0.3, "sftkl_mask_b0p3"),
             (1.0, "sftkl_mask_b1"), (3.0, "sftkl_mask_b3"),
             (10.0, "sftkl_mask_b10")]
betas_new = [(100.0, "b100"), (200.0, "b200"), (300.0, "b300"),
             (400.0, "b400"), (500.0, "b500"), (700.0, "b700"),
             (800.0, "b800"), (900.0, "b900"), (1000.0, "b1000")]

def old_stem(bs, cs):
    if cs == "":  return f"llm_llm_{bs}"
    if bs == "sft_mask": return f"llm_llm_sft_mask_{cs}"
    bp = bs.replace("sftkl_mask_", "")
    return f"llm_llm_sftkl_mask_{cs}_{bp}"

def new_stem(b_tag, cs):
    if cs == "": return f"llm_llm_sftkl_mask_{b_tag}"
    return f"llm_llm_sftkl_mask_{cs}_{b_tag}"

# untuned
try:
    tr_u = pickle.load(open(OLD + "llm_llm_untuned_trajectory.pk", "rb"))
    u_drift = tr_u[:, -1].mean() - inn_mean
    u_var   = tr_u[:, -1].var()
    print(f"untuned (β=∞)       drift={u_drift:+.4f}  var={u_var:.5f}")
except FileNotFoundError:
    pass
print(f"innate mean={inn_mean:.4f}  var={inn_var:.5f}\n")

for cap_label, cs in caps:
    print(f"--- {cap_label} ---")
    for b, bs in betas_old:
        p = os.path.join(OLD, old_stem(bs, cs) + "_trajectory.pk")
        if not os.path.exists(p): continue
        tr = pickle.load(open(p, "rb"))
        print(f"  β={b:<6}  final={tr[:,-1].mean():.4f}  drift={tr[:,-1].mean()-inn_mean:+.4f}  var={tr[:,-1].var():.5f}")
    for b, tag in betas_new:
        p = os.path.join(BIG, new_stem(tag, cs) + "_trajectory.pk")
        if not os.path.exists(p): continue
        tr = pickle.load(open(p, "rb"))
        print(f"  β={b:<6}  final={tr[:,-1].mean():.4f}  drift={tr[:,-1].mean()-inn_mean:+.4f}  var={tr[:,-1].var():.5f}")
    print()
