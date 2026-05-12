"""Compare pi_theta (loaded from a saved LoRA adapter) to pi_ref (base model)
at the first-digit position across unlabeled profiles.

Three metrics together separate (A) genuine distributional shift from
(B) greedy-argmax-flip over near-equal distributions:

  TV per profile        0.5 * sum_d |P_theta(d) - P_ref(d)|
  E[d | policy]         sum_d d * P(d)
  argmax disagreement   fraction of profiles where argmax(P_theta) != argmax(P_ref)

Case (A) signature: TV mean > ~0.15, E[d] differs materially between policies.
Case (B) signature: TV mean < ~0.05, E[d] nearly equal, argmax disagreement > 30%.

Usage:
  python compare_policies.py --adapter_path ./adapters/llm_7b_r512_b0p5
  python compare_policies.py --adapter_path ./adapters/llm_7b_r512_b1 \
      --base_model Qwen/Qwen2.5-7B-Instruct --n_probe 400
"""
import argparse
import os
import pickle
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_predictor import build_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter_path", required=True,
                    help="Directory with adapter_config.json and adapter_model.safetensors")
    ap.add_argument("--profiles_pk", default="pokec_dataset/lcc_profiles_relation_to_smoking.pk")
    ap.add_argument("--n_labeled_frac", type=float, default=0.8)
    ap.add_argument("--n_probe", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"device={device} dtype={dtype}")

    print(f"loading tokenizer+base: {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype).to(device)
    base.config.pad_token_id = tok.pad_token_id

    print(f"loading adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base, args.adapter_path).to(device)
    model.eval()

    print(f"loading profiles: {args.profiles_pk}")
    df = pickle.load(open(args.profiles_pk, "rb"))
    n_total = len(df)
    n_lab = int(n_total * args.n_labeled_frac)
    df_unl = df.iloc[n_lab:].copy()
    df_unl["age"] = pd.to_numeric(df_unl["age"], errors="coerce")
    df_unl["gender"] = pd.to_numeric(df_unl["gender"], errors="coerce")
    prompts = [build_prompt(r, tokenizer=tok) for _, r in df_unl.iterrows()][: args.n_probe]
    print(f"n probe prompts: {len(prompts)}")

    digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    digit_ids_t = torch.tensor(digit_ids, device=device)

    @torch.no_grad()
    def first_digit_probs(use_adapter: bool):
        P = np.zeros((len(prompts), 10), dtype=float)
        ctx = nullcontext() if use_adapter else model.disable_adapter()
        with ctx:
            for i in range(0, len(prompts), args.batch_size):
                batch = [p + "0." for p in prompts[i : i + args.batch_size]]
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
                out = model(**enc)
                logits = out.logits[:, -1, :].float()
                probs = torch.softmax(logits, dim=-1)
                p_d = probs[:, digit_ids_t]
                p_d = (p_d / p_d.sum(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
                P[i : i + len(batch)] = p_d
        return P

    print("forward pass pi_ref (adapter disabled)")
    P_ref = first_digit_probs(use_adapter=False)
    print("forward pass pi_theta (adapter enabled)")
    P_theta = first_digit_probs(use_adapter=True)

    digits = np.arange(10)
    tv = 0.5 * np.abs(P_theta - P_ref).sum(axis=1)
    E_ref = (P_ref * digits).sum(axis=1)
    E_theta = (P_theta * digits).sum(axis=1)
    argmax_ref = P_ref.argmax(axis=1)
    argmax_theta = P_theta.argmax(axis=1)
    disagree = (argmax_theta != argmax_ref).mean()

    print()
    print("=== first-digit distribution: pi_theta vs pi_ref ===")
    print(f"TV per profile        mean={tv.mean():.4f}  median={np.median(tv):.4f}  max={tv.max():.4f}")
    print(f"E[d | pi_ref]         mean={E_ref.mean():.4f}  std={E_ref.std():.4f}")
    print(f"E[d | pi_theta]       mean={E_theta.mean():.4f}  std={E_theta.std():.4f}")
    print(f"|E_theta - E_ref|     mean={np.abs(E_theta - E_ref).mean():.4f}")
    print(f"argmax disagreement   {disagree:.1%}")
    print()
    print("per-digit mean probability (averaged across profiles):")
    hdr = "        " + "".join(f"{d:>7d}" for d in range(10))
    ref_row = "pi_ref  " + "".join(f"{v:>7.3f}" for v in P_ref.mean(axis=0))
    theta_row = "pi_theta" + "".join(f"{v:>7.3f}" for v in P_theta.mean(axis=0))
    print(hdr)
    print(ref_row)
    print(theta_row)
    print()

    if tv.mean() < 0.05 and disagree > 0.30:
        verdict = "CASE (B): near-equal distribution, argmax flipped. Greedy-decoding artifact."
    elif tv.mean() > 0.15:
        verdict = "CASE (A): genuine distributional shift."
    else:
        verdict = "INTERMEDIATE: mixed signal, inspect per-digit mass above."
    print(f"verdict: {verdict}")

    if args.out_json:
        import json
        payload = {
            "adapter_path": args.adapter_path,
            "base_model": args.base_model,
            "n_probe": int(len(prompts)),
            "tv_mean": float(tv.mean()),
            "tv_median": float(np.median(tv)),
            "tv_max": float(tv.max()),
            "E_ref_mean": float(E_ref.mean()),
            "E_theta_mean": float(E_theta.mean()),
            "E_diff_mean": float(np.abs(E_theta - E_ref).mean()),
            "argmax_disagree": float(disagree),
            "digit_prob_ref": P_ref.mean(axis=0).tolist(),
            "digit_prob_theta": P_theta.mean(axis=0).tolist(),
            "verdict": verdict,
        }
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
