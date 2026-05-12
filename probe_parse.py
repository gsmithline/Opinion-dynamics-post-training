"""Probe: does the base LLM produce parseable floats on real Pokec prompts?
Loads cached profile df, samples N rows, runs readout() on base model (no training).
Prints parse_fail_rate + a handful of raw outputs + distribution stats.
"""
import os
import pickle
import numpy as np
import pandas as pd

from llm_predictor import _load_model, build_prompt, readout, parse_prob, _NUM_RE

TARGET = "relation_to_smoking"
N_SAMPLE = int(os.environ.get("PROBE_N", 80))

profiles_path = f"pokec_dataset/lcc_profiles_{TARGET}.pk"
with open(profiles_path, "rb") as f:
    df = pickle.load(f)

rng = np.random.default_rng(0)
idx = rng.choice(len(df), size=min(N_SAMPLE, len(df)), replace=False)
sample = df.iloc[idx].reset_index(drop=True)

prompts = [build_prompt(r) for _, r in sample.iterrows()]
print(f"[probe] built {len(prompts)} prompts")
print(f"[probe] example prompt:\n{prompts[0]}\n---")

model, tok = _load_model()
preds = readout(model, tok, prompts)

print(f"[probe] pred stats: mean={preds.mean():.3f} std={preds.std():.3f} "
      f"min={preds.min():.3f} max={preds.max():.3f} uniq={len(np.unique(preds))}")
# show a few raw decodes too
import torch
with torch.no_grad():
    enc = tok(prompts[:8], return_tensors="pt", padding=True, truncation=True).to(model.device)
    out = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    for i, ids in enumerate(gen):
        txt = tok.decode(ids, skip_special_tokens=True)
        parsed = parse_prob(txt)
        has_num = _NUM_RE.search(txt) is not None
        print(f"[probe] sample {i}: parsed={parsed:.3f} has_num={has_num} raw={txt!r}")
