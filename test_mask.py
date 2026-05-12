"""Standalone test: verify SFT loss masking works correctly.
No full simulation; just loads Qwen, fakes a small labeled batch, checks labels tensor.
"""
import os
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("SFT_SANITY", "1")
os.environ.setdefault("USE_LORA", "1")
os.environ.setdefault("LORA_R", "8")
os.environ.setdefault("LORA_ALPHA", "16")
os.environ.setdefault("SFT_EPOCHS", "1")
os.environ.setdefault("BATCH_SIZE", "4")
os.environ.setdefault("TRAINING_STYLE", "sft_kl")
os.environ.setdefault("KL_BETA", "1.0")

import pandas as pd
import numpy as np
from llm_predictor import _load_model, build_prompt, sft_on_round, _STATE

# fake 8-agent labeled subset
rows = [
    {"age": 23, "gender": 0.0, "relation_to_alcohol": "pijem prilezitostne"},
    {"age": 31, "gender": 1.0, "relation_to_alcohol": "nepijem"},
    {"age": 45, "gender": 0.0, "relation_to_alcohol": "abstinent"},
    {"age": 19, "gender": 1.0, "relation_to_alcohol": "pijem pravidelne"},
    {"age": 28, "gender": 0.0, "relation_to_alcohol": "prilezitostne"},
    {"age": 52, "gender": 1.0, "relation_to_alcohol": "nepijem"},
    {"age": 35, "gender": 0.0, "relation_to_alcohol": "pijem"},
    {"age": 22, "gender": 1.0, "relation_to_alcohol": "abstinent"},
]
df = pd.DataFrame(rows)
prompts = [build_prompt(r) for _, r in df.iterrows()]
y = np.array([0.57, 0.33, 0.12, 0.78, 0.50, 0.21, 0.64, 0.09])

print("="*60)
print("Loading model (Qwen2.5-0.5B-Instruct) ...")
print("="*60)
model, tok = _load_model()
print(f"model device: {next(model.parameters()).device}")

print("\n" + "="*60)
print("Running 1-round SFT with SFT_SANITY=1 to verify masking ...")
print("="*60)
_STATE["round"] = 1
sft_on_round(model, tok, prompts, y)

print("\n[test_mask] If labels show ~80-95% masked with tail tokens unmasked, fix is working.")
