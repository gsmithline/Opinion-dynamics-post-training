"""Check for readout artifacts in stored trajectories.
Looks for: mass at/near 1.0 (clip-to-1), mass at exactly 0.5 (parse-fail fallback),
and raw predictor output distribution vs expressed-opinion distribution.
"""
import os, pickle, numpy as np

RES = "pokec_dataset/results/"

def summarize(name, a, tol=1e-6):
    a = np.asarray(a).ravel()
    n = a.size
    at_1   = (np.abs(a - 1.0) < tol).sum()
    at_0   = (np.abs(a - 0.0) < tol).sum()
    at_0p5 = (np.abs(a - 0.5) < tol).sum()
    above_95 = (a > 0.95).sum()
    below_05 = (a < 0.05).sum()
    print(f"  {name:32s} n={n:>5}  mean={a.mean():.4f}  std={a.std():.4f}  "
          f"exact1={at_1:>4}({100*at_1/n:4.1f}%)  exact0.5={at_0p5:>4}({100*at_0p5/n:4.1f}%)  "
          f"exact0={at_0:>4}  >0.95={above_95:>4}({100*above_95/n:4.1f}%)")

for tag in ["llm_llm_sft", "llm_llm_sftkl_b10", "mean", "ridge", "neural_net_mlp"]:
    print(f"\n=== {tag} ===")
    for suf in ["_trajectory.pk", "_equilibrium.pk", "_FJequilibrium.pk"]:
        p = RES + tag + suf
        if not os.path.exists(p): continue
        obj = pickle.load(open(p, "rb"))
        if isinstance(obj, np.ndarray):
            if obj.ndim == 2:
                print(f" {suf}  shape={obj.shape}")
                summarize("round 0", obj[:, 0])
                summarize("round mid (T/2)", obj[:, obj.shape[1]//2])
                summarize("round final", obj[:, -1])
            else:
                print(f" {suf}  shape={obj.shape}")
                summarize("flat", obj)
        else:
            print(f" {suf}  type={type(obj).__name__}  (not ndarray)")

# Also check innate
print("\n=== innate opinions ===")
y_lab = pickle.load(open("pokec_dataset/parametric_params/y_label2163.pk", "rb"))
y_unlab = pickle.load(open("pokec_dataset/parametric_params/y_unlabel_label2163.pk", "rb"))
innate = np.concatenate([y_lab, y_unlab])
summarize("innate full", innate)
summarize("innate labeled", y_lab)
summarize("innate unlabeled", y_unlab)
