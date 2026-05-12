"""Check whether untuned Qwen-7B puts nontrivial soft mass on low digit tokens
at the first-digit position after '0.'. Tests the mode-collapse hypothesis
that LoRA + SFT+KL amplify pre-existing low-token mass in the prior."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda"
print(f"loading {MODEL} on {DEVICE}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEVICE).eval()

profiles = [
    "- age: 18\n- gender: female\n- relation_to_alcohol: does not drink",
    "- age: 25\n- gender: male\n- relation_to_alcohol: drinks occasionally",
    "- age: 40\n- gender: male\n- relation_to_alcohol: drinks regularly",
    "- age: 22\n- gender: female\n- relation_to_alcohol: drinks",
    "- age: 30\n- gender: male\n- relation_to_alcohol: unknown",
]

def build_msgs(profile):
    return [{"role": "user", "content":
        "Estimate this user's attitude toward smoking based on their profile.\n"
        f"Profile:\n{profile}\n\n"
        "Output a single number in [0, 1] (1 = very positive, 0 = very negative). "
        "Respond with only the number, e.g. 0.42."}]

digit_ids = {d: tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)}
print(f"digit token ids: {digit_ids}")

print(f"\n{'profile':>44s} | " + " ".join(f"d{d}" for d in range(10)) + " | top-3 tokens")
low_all, mid_all, hi_all = [], [], []
for p in profiles:
    msgs = build_msgs(p)
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "0."
    enc = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**enc)
    probs = torch.softmax(out.logits[0, -1].float(), dim=-1)
    dp = {d: float(probs[tid]) for d, tid in digit_ids.items()}
    top3 = torch.topk(probs, 3)
    top3_str = ", ".join(f"{tok.decode([int(t)])!r}:{float(p):.3f}" for p, t in zip(top3.values, top3.indices))
    low = sum(dp[d] for d in [0, 1, 2, 3])
    mid = sum(dp[d] for d in [4, 5, 6])
    hi  = sum(dp[d] for d in [7, 8, 9])
    low_all.append(low); mid_all.append(mid); hi_all.append(hi)
    short = p.replace("\n", " ")[:40]
    print(f"{short:>44s} | " + " ".join(f"{dp[d]:.2f}" for d in range(10)) + f" | {top3_str}")

import statistics as s
print(f"\nMean over {len(profiles)} profiles:")
print(f"  P(digit in 0,1,2,3) = {s.mean(low_all):.4f}  (mass on 0.0-0.3X outputs)")
print(f"  P(digit in 4,5,6)   = {s.mean(mid_all):.4f}  (mass on 0.4-0.6X outputs)")
print(f"  P(digit in 7,8,9)   = {s.mean(hi_all):.4f}  (mass on 0.7-0.9X outputs)")
