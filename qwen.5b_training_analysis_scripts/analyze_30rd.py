"""Extract hypothesis-relevant stats from 30-round trajectories."""
import os, pickle, numpy as np

RES = "pokec_dataset/results/"
runs = [
    ("perfect",            "perfect"),
    ("mean",               "mean"),
    ("ridge",              "ridge"),
    ("neural_net_mlp",     "MLP"),
    ("llm_llm_sft",        "LLM b=0"),
    ("llm_llm_sftkl_b0p3", "LLM b=0.3"),
    ("llm_llm_sftkl_b1",   "LLM b=1"),
    ("llm_llm_sftkl_b3",   "LLM b=3"),
    ("llm_llm_sftkl_b10",  "LLM b=10"),
]

y_lab   = pickle.load(open("pokec_dataset/parametric_params/y_label2163.pk","rb"))
y_unlab = pickle.load(open("pokec_dataset/parametric_params/y_unlabel_label2163.pk","rb"))
innate  = np.concatenate([y_lab, y_unlab])
im, iv, istd = float(innate.mean()), float(innate.var()), float(innate.std())
print(f"innate: mean={im:.4f}  std={istd:.4f}  var={iv:.6f}\n")

rows = []
for f,lbl in runs:
    p = RES + f + "_trajectory.pk"
    if not os.path.exists(p): continue
    t = pickle.load(open(p,"rb"))          # (N, T+1)
    N, T1 = t.shape
    m  = t.mean(0); v = t.var(0); s = t.std(0)
    last10_slope = np.polyfit(np.arange(10), m[-10:], 1)[0]
    # per-agent drift vs innate (final - round0)
    drift = t[:,-1] - t[:,0]
    rows.append(dict(label=lbl, N=N, T=T1-1,
                     m_final=m[-1], d_mean=m[-1]-im,
                     v_final=v[-1], s_final=s[-1], d_std=s[-1]-istd,
                     slope10=last10_slope,
                     drift_mean=drift.mean(), drift_std=drift.std(),
                     drift_abs=np.abs(drift).mean(),
                     m0=m[0], m5=m[5] if T1>5 else np.nan, m15=m[15] if T1>15 else np.nan))

print(f"{'method':12s} {'final_mean':>10s} {'Δinnate':>8s} {'final_std':>9s} {'Δstd':>8s} {'slope_last10':>13s} {'|drift|avg':>10s}")
for r in rows:
    print(f"{r['label']:12s} {r['m_final']:10.4f} {r['d_mean']:+8.4f} {r['s_final']:9.4f} {r['d_std']:+8.4f} {r['slope10']:+13.6f} {r['drift_abs']:10.4f}")

print("\nearly-vs-late trajectory (mean opinion):")
print(f"{'method':12s} {'t=0':>8s} {'t=5':>8s} {'t=15':>8s} {'t=30':>8s}")
for r in rows:
    print(f"{r['label']:12s} {r['m0']:8.4f} {r['m5']:8.4f} {r['m15']:8.4f} {r['m_final']:8.4f}")

# β-ordering check
llm = [r for r in rows if r['label'].startswith("LLM")]
betas = [0, 0.3, 1, 3, 10]
print("\nβ-ordering check (final mean):")
for b,r in zip(betas, llm):
    print(f"  β={b:<4} mean={r['m_final']:.4f}  Δinnate={r['d_mean']:+.4f}")
monotone = all(llm[i]['m_final'] >= llm[i+1]['m_final'] for i in range(len(llm)-1))
print(f"  strictly decreasing with β? {monotone}")

# gap: LLM min vs baseline max
base_max = max(r['m_final'] for r in rows if not r['label'].startswith("LLM"))
llm_min  = min(r['m_final'] for r in rows if r['label'].startswith("LLM"))
print(f"\nLLM-vs-baseline gap: min LLM final = {llm_min:.4f}, max baseline final = {base_max:.4f}  (gap = {llm_min-base_max:+.4f})")

# subpop drift: split by innate quantile (using full traj round 0 as proxy)
print("\nper-config drift by innate-opinion quartile (final − innate):")
for f,lbl in runs:
    p = RES + f + "_trajectory.pk"
    if not os.path.exists(p): continue
    t = pickle.load(open(p,"rb"))
    d = t[:,-1] - t[:,0]
    q = np.quantile(t[:,0], [0.25,0.5,0.75])
    bins = np.digitize(t[:,0], q)
    means = [d[bins==i].mean() for i in range(4)]
    print(f"  {lbl:12s}  Q1={means[0]:+.4f}  Q2={means[1]:+.4f}  Q3={means[2]:+.4f}  Q4={means[3]:+.4f}")
