"""Analyze LoRA rank sweep + full FT results."""
import os, pickle, numpy as np

RES = "/Users/gabesmithline/Desktop/results_full_small_loras/"

# (filename_tag, rank_label, beta)
configs = [
    ("sft_ff",         "full-FT", 0.0),
    ("sftkl_ff_b0p3",  "full-FT", 0.3),
    ("sftkl_ff_b1",    "full-FT", 1.0),
    ("sftkl_ff_b3",    "full-FT", 3.0),
    ("sftkl_ff_b10",   "full-FT", 10.0),
    ("sft_r8",         "r=8",     0.0),
    ("sftkl_r8_b0p3",  "r=8",     0.3),
    ("sftkl_r8_b1",    "r=8",     1.0),
    ("sftkl_r8_b3",    "r=8",     3.0),
    ("sftkl_r8_b10",   "r=8",     10.0),
    ("sft_r2",         "r=2",     0.0),
    ("sftkl_r2_b0p3",  "r=2",     0.3),
    ("sftkl_r2_b1",    "r=2",     1.0),
    ("sftkl_r2_b3",    "r=2",     3.0),
    ("sftkl_r2_b10",   "r=2",     10.0),
    ("sft_r1",         "r=1",     0.0),
    ("sftkl_r1_b1",    "r=1",     1.0),
    ("sftkl_r1_b3",    "r=1",     3.0),
    ("sftkl_r1_b10",   "r=1",     10.0),
]

y_lab   = pickle.load(open("pokec_dataset/parametric_params/y_label2163.pk","rb"))
y_unlab = pickle.load(open("pokec_dataset/parametric_params/y_unlabel_label2163.pk","rb"))
innate  = np.concatenate([y_lab, y_unlab])
im, istd = float(innate.mean()), float(innate.std())
print(f"innate: mean={im:.4f}  std={istd:.4f}\n")

rows = []
for tag, rank, beta in configs:
    p = RES + f"llm_llm_{tag}_trajectory.pk"
    if not os.path.exists(p):
        print(f"MISSING: {p}")
        continue
    t = pickle.load(open(p,"rb"))
    m_final = t[:,-1].mean()
    s_final = t[:,-1].std()
    slope = np.polyfit(np.arange(10), t.mean(0)[-10:], 1)[0]
    rows.append((rank, beta, m_final, m_final-im, s_final, slope, t))

# group by rank
print(f"{'rank':8s} {'beta':>6s} {'final_mean':>10s} {'Δinnate':>9s} {'std':>7s} {'slope10':>11s}")
for r in rows:
    rank, beta, m, d, s, sl, _ = r
    print(f"{rank:8s} {beta:6.1f} {m:10.4f} {d:+9.4f} {s:7.4f} {sl:+11.6f}")

# β-ordering per rank
print("\nβ-ordering check (strictly decreasing final_mean in β?):")
for rank in ["full-FT", "r=8", "r=2", "r=1"]:
    sub = sorted([r for r in rows if r[0]==rank], key=lambda x: x[1])
    means = [x[2] for x in sub]
    betas = [x[1] for x in sub]
    mono = all(means[i] >= means[i+1] for i in range(len(means)-1))
    print(f"  {rank:8s}  betas={betas}  means={[f'{m:.4f}' for m in means]}  mono={mono}")

# spread (β=0 − β=10) per rank — the β-family gap
print("\nβ-family spread (β=0 mean − β=10 mean):")
for rank in ["full-FT", "r=8", "r=2", "r=1"]:
    sub = {b: m for r2,b,m,*_ in rows if r2==rank}
    if 0.0 in sub and 10.0 in sub:
        print(f"  {rank:8s}  β=0: {sub[0.0]:.4f}   β=10: {sub[10.0]:.4f}   spread: {sub[0.0]-sub[10.0]:+.4f}")

# subpop asymmetry
print("\nQ1/Q4 drift by rank (β=0 only, to isolate capacity effect):")
for rank in ["full-FT","r=8","r=2","r=1"]:
    for r2,b,*_,t in rows:
        if r2==rank and b==0.0:
            d = t[:,-1] - t[:,0]
            q = np.quantile(t[:,0],[0.25,0.75])
            q1 = d[t[:,0]<=q[0]].mean()
            q4 = d[t[:,0]>=q[1]].mean()
            print(f"  {rank:8s}  Q1={q1:+.4f}  Q4={q4:+.4f}  asymmetry={q1-q4:+.4f}")
            break
