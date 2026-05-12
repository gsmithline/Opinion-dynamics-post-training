"""Untuned Qwen2.5-0.5B-Instruct as performative predictor (beta -> infinity limit).

Model never trains. build_prompt depends only on static profile, so the LLM's
predictions for unlabeled agents are a fixed vector across all rounds. We run
the exact same FJ loop (platform_sus=1, steer=0) as pokec_simulations.run_simulation
and save a (2163, 31) trajectory.

Output: /Users/gabesmithline/Desktop/results_mask/llm_llm_untuned_trajectory.pk
"""
import os
import re
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

# --- shim for pandas 2.2 unpickling a pandas-1.x profiles pkl ---------------
_orig_sd_init = pd.StringDtype.__init__
def _sd_init(self, storage=None, na_value=None, *args, **kwargs):
    try: _orig_sd_init(self, storage=storage)
    except TypeError: _orig_sd_init(self)
pd.StringDtype.__init__ = _sd_init

from pandas.core.arrays.string_ import StringArray
_orig_ss = getattr(StringArray, "__setstate__", None)
def _ss(self, state):
    try:
        if _orig_ss: return _orig_ss(self, state)
    except Exception: pass
    def _find(x):
        if isinstance(x, np.ndarray): return x
        if isinstance(x, (tuple, list)):
            for e in x:
                r = _find(e)
                if r is not None: return r
        return None
    arr = _find(state)
    if arr is None: raise ValueError("no ndarray in state")
    StringArray.__init__(self, arr.astype(object))
StringArray.__setstate__ = _ss
# ---------------------------------------------------------------------------

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))
# untuned Qwen often prefaces with prose before emitting a number;
# SFT runs used 6 because SFT taught "0.42\n" directly.  Bump for untuned.
MAX_NEW_TOKENS = int(os.environ.get("UNTUNED_MAX_NEW_TOKENS", 32))
# READOUT_TEMPERATURE>0 switches readout from greedy to stochastic sampling at
# that temperature. Used to measure distributional width of pi_ref, to compare
# against training-run pi_theta at matched temperature.
READOUT_TEMPERATURE = float(os.environ.get("READOUT_TEMPERATURE", "0"))

# mirror llm_predictor.build_prompt / parse_prob / readout exactly
PROMPT_COLS = ["age", "gender", "relation_to_alcohol"]
SK_ALCOHOL_EXACT = {"nepije alkohol": "does not drink",
                    "pije pravidelne": "drinks regularly",
                    "pije prilezitostne": "drinks occasionally",
                    "pije pravidelne alkohol": "drinks regularly",
                    "abstinent": "does not drink",
                    "alkoholik": "alcoholic"}

def translate_alcohol(val):
    s = str(val).strip().lower()
    if s in SK_ALCOHOL_EXACT:
        return SK_ALCOHOL_EXACT[s]
    if "nepij" in s or "abstin" in s or "apstin" in s:
        return "does not drink"
    if "pravidel" in s:
        return "drinks regularly"
    if "prilezitost" in s or "prilezitos" in s:
        return "drinks occasionally"
    if "pij" in s:
        return "drinks"
    return "unknown"

def build_prompt_raw(row):
    """Same raw string as the SFT runs (reference / fallback)."""
    lines = []
    for col in PROMPT_COLS:
        val = row.get(col, "")
        if pd.isna(val) or val == "" or str(val) == "nan":
            continue
        if col == "age":
            if float(val) == 0.0:
                continue
            val = int(val)
        elif col == "gender":
            val = {0.0: "female", 1.0: "male"}.get(float(val), "unknown")
        elif col == "relation_to_alcohol":
            val = translate_alcohol(val)
        lines.append(f"- {col}: {val}")
    profile_str = "\n".join(lines) if lines else "- (no profile info)"
    return (
        "You will estimate a user's attitude toward smoking based on their profile.\n"
        "Profile:\n"
        f"{profile_str}\n\n"
        "Task: Output a single number in [0, 1] representing the probability that this "
        "user has a POSITIVE attitude toward smoking (1 = very positive, 0 = very negative).\n"
        "Output format: a single number, e.g. 0.42. No other text.\n"
        "Answer: "
    )


def build_messages(row):
    """Chat-template variant for the untuned probe.  Same content, wrapped as a
    user turn so Qwen-Instruct's native format gives clean numeric outputs."""
    lines = []
    for col in PROMPT_COLS:
        val = row.get(col, "")
        if pd.isna(val) or val == "" or str(val) == "nan":
            continue
        if col == "age":
            if float(val) == 0.0:
                continue
            val = int(val)
        elif col == "gender":
            val = {0.0: "female", 1.0: "male"}.get(float(val), "unknown")
        elif col == "relation_to_alcohol":
            val = translate_alcohol(val)
        lines.append(f"- {col}: {val}")
    profile_str = "\n".join(lines) if lines else "- (no profile info)"
    user_msg = (
        "Estimate this user's attitude toward smoking based on their profile.\n"
        "Profile:\n"
        f"{profile_str}\n\n"
        "Output a single number in [0, 1] (1 = very positive, 0 = very negative). "
        "Respond with only the number, e.g. 0.42."
    )
    return [{"role": "user", "content": user_msg}]

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")

def parse_prob(text, default=0.5):
    m = _NUM_RE.search(text)
    if m is None:
        return default
    try:
        v = float(m.group(0))
    except ValueError:
        return default
    return float(np.clip(v, 0.0, 1.0))

@torch.no_grad()
def readout(model, tok, prompts, batch_size=BATCH_SIZE, debug_fail_samples=5):
    model.eval()
    preds = np.zeros(len(prompts), dtype=float)
    n_fail = 0
    fail_samples = []
    ok_samples = []
    gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tok.pad_token_id)
    if READOUT_TEMPERATURE > 0:
        gen_kwargs.update(do_sample=True, temperature=READOUT_TEMPERATURE, top_p=1.0, top_k=0)
    else:
        gen_kwargs["do_sample"] = False
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        out = model.generate(**enc, **gen_kwargs)
        gen = out[:, enc["input_ids"].shape[1]:]
        for j, ids in enumerate(gen):
            txt = tok.decode(ids, skip_special_tokens=True)
            if _NUM_RE.search(txt) is None:
                n_fail += 1
                if len(fail_samples) < debug_fail_samples:
                    fail_samples.append(txt)
            else:
                if len(ok_samples) < debug_fail_samples:
                    ok_samples.append(txt)
            preds[i+j] = parse_prob(txt)
    print(f"[readout] parse_fail={n_fail}/{len(prompts)} ({n_fail/max(len(prompts),1):.1%})")
    print(f"[readout] first failed outputs: {[repr(s) for s in fail_samples]}")
    print(f"[readout] first   ok   outputs: {[repr(s) for s in ok_samples]}")
    return preds, n_fail


@torch.no_grad()
def entropy_probe(model, tok, prompts, batch_size=BATCH_SIZE, n_probe=100):
    """Teacher-forced entropy probe at the first-digit position. Matches
    llm_predictor.entropy_probe so π_ref (this script) and π_θ (training
    script) are measured with identical methodology."""
    model.eval()
    probe_prompts = list(prompts)[:n_probe]
    digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    digit_ids_t = torch.tensor(digit_ids, device=DEVICE)
    H_full_chunks, H_digits_chunks = [], []
    digit_prob_sum = torch.zeros(10, device=DEVICE)
    n_seen = 0
    for i in range(0, len(probe_prompts), batch_size):
        batch = [p + "0." for p in probe_prompts[i : i + batch_size]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        out = model(**enc)
        logits = out.logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        H_full = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
        p_d = probs[:, digit_ids_t]
        p_d_norm = p_d / p_d.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        H_digits = -(p_d_norm * torch.log(p_d_norm + 1e-12)).sum(dim=-1)
        H_full_chunks.append(H_full)
        H_digits_chunks.append(H_digits)
        digit_prob_sum += p_d.sum(dim=0)
        n_seen += logits.shape[0]
    result = {
        "entropy_full":   float(torch.cat(H_full_chunks).mean()),
        "entropy_digits": float(torch.cat(H_digits_chunks).mean()),
    }
    for d in range(10):
        result[f"digit_prob_{d}"] = float(digit_prob_sum[d] / max(n_seen, 1))
    return result


def load_base_model():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    # MPS + bf16 produces "!!!" garbage on Qwen2.5; use fp32 on MPS/CPU
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype).to(DEVICE)
    model.config.pad_token_id = tok.pad_token_id
    return model, tok

TARGET = "relation_to_smoking"
PROFILES = "pokec_dataset/lcc_profiles_" + TARGET + ".pk"
GRAPH    = "pokec_dataset/lcc_graph_" + TARGET + ".pk"
YLAB     = "pokec_dataset/parametric_params/y_label2163.pk"
YUNL     = "pokec_dataset/parametric_params/y_unlabel_label2163.pk"
PEER_PK  = "pokec_dataset/parametric_params/hetero_peer_sus2163.pkl"
OUT_DIR  = os.environ.get("OUT_DIR", "/Users/gabesmithline/Desktop/results_mask/")
RUN_TAG  = os.environ.get("RUN_TAG", "base")
RETRAIN_T = 30
FJ_K      = 100


def main():
    df        = pickle.load(open(PROFILES, "rb"))
    network   = pickle.load(open(GRAPH,    "rb"))
    y_lab     = pickle.load(open(YLAB,     "rb"))
    y_unl     = pickle.load(open(YUNL,     "rb"))
    peer_sus  = pickle.load(open(PEER_PK,  "rb"))

    innate = np.array(y_lab + y_unl)
    agent_num = len(innate)
    n = int(agent_num * 0.8)
    print(f"agent_num={agent_num}, n_labeled={n}, n_unlabeled={agent_num - n}")

    df_unlabeled = df.iloc[n:].copy()
    # dtype normalisation to match pokec_simulations.main
    df_unlabeled["age"]    = pd.to_numeric(df_unlabeled["age"],    errors="coerce")
    df_unlabeled["gender"] = pd.to_numeric(df_unlabeled["gender"], errors="coerce")

    # adjacency + degree-norm (matches pokec_simulations.run_simulation)
    nodelist = df["user_id"].values
    adj_mat = nx.to_numpy_array(network, nodelist=nodelist)
    weight_mat = adj_mat.copy()
    degs_inv = 1.0 / np.sum(adj_mat, axis=0)
    degs_inv[np.isinf(degs_inv)] = 0.0
    degs_inv[degs_inv > 1.1]     = 0.0
    W_norm = weight_mat * degs_inv[:, None]

    # strong-performativity params (same as run_opinion_dynamics)
    platform_sus = np.ones(agent_num)

    x_star_mean = float(innate.mean())
    x_star_std  = float(innate.std())

    # ---- wandb init -----------------------------------------------
    if _HAS_WANDB:
        try:
            wandb.init(
                project="opinion-dynamics-llm",
                name=RUN_TAG if RUN_TAG else f"untuned_{BASE_MODEL.split('/')[-1]}",
                config={
                    "model_name": "llm_untuned",
                    "run_tag": RUN_TAG,
                    "base_model": BASE_MODEL,
                    "training_style": "none",
                    "kl_beta": None,
                    "retrain_T": RETRAIN_T,
                    "fj_K": FJ_K,
                    "agent_num": agent_num,
                    "n_labeled": n,
                    "n_unlabeled": agent_num - n,
                    "platform_sus_mean": float(np.mean(platform_sus)),
                    "platform_sus_std":  float(np.std(platform_sus)),
                    "peer_sus_mean": float(np.mean(peer_sus)),
                    "peer_sus_std":  float(np.std(peer_sus)),
                    "x_star_mean": x_star_mean,
                    "x_star_std":  x_star_std,
                },
                reinit=False,
            )
            wandb.define_metric("round")
            for pat in ("opinion_*", "pred_*", "dist_to_innate", "opinion_range",
                        "mean_drift_from_innate", "abs_mean_drift_from_innate",
                        "std_ratio_to_innate", "target_*", "parse_fail_*",
                        "entropy_*", "digit_prob_*", "base_entropy_*", "base_digit_prob_*"):
                wandb.define_metric(pat, step_metric="round")
        except Exception as e:
            print(f"[run_untuned] wandb init skipped: {e}")

    # ---- one-shot LLM readout on pretrained Qwen -------------------
    print(f"loading pretrained model on {DEVICE}...")
    model, tok = load_base_model()
    print("building chat-template prompts for unlabeled set...")
    messages_unlabeled = [build_messages(r) for _, r in df_unlabeled.iterrows()]
    prompts_unlabeled = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_unlabeled
    ]
    print(f"example rendered prompt:\n{prompts_unlabeled[0]!r}\n---")
    print(f"running readout ({len(prompts_unlabeled)} prompts)...")
    base_preds, n_fail = readout(model, tok, prompts_unlabeled)
    print(f"base preds: mean={base_preds.mean():.4f} std={base_preds.std():.4f} "
          f"min={base_preds.min():.4f} max={base_preds.max():.4f}")

    fail_rate  = n_fail / max(len(base_preds), 1)
    pred_mean  = float(base_preds.mean())
    pred_std   = float(base_preds.std())
    pred_min   = float(base_preds.min())
    pred_max   = float(base_preds.max())

    print("running entropy probe on pretrained model...")
    base_ent = entropy_probe(model, tok, prompts_unlabeled, n_probe=100)
    print(f"base entropy: H_full={base_ent['entropy_full']:.3f} "
          f"H_digits={base_ent['entropy_digits']:.3f}  "
          "digit_P: " + " ".join(f"{d}:{base_ent[f'digit_prob_{d}']:.3f}" for d in range(10)))

    # ---- FJ simulation loop ---------------------------------------
    traj = np.zeros((agent_num, RETRAIN_T + 1))
    traj[:, 0] = innate.copy()
    x_labeled_prior = innate[:n].copy()

    platform_predictions = np.zeros(agent_num)
    platform_predictions[n:] = base_preds   # fixed forever

    for t in range(RETRAIN_T):
        round_num = t + 1
        target_mean  = float(np.mean(x_labeled_prior))
        target_std   = float(np.std(x_labeled_prior))
        target_range = float(np.max(x_labeled_prior) - np.min(x_labeled_prior))

        # Log target stats (mirrors pokec_simulations.py line 510)
        if _HAS_WANDB and wandb.run is not None:
            try:
                wandb.log({
                    "round": round_num,
                    "target_mean":  target_mean,
                    "target_std":   target_std,
                    "target_range": target_range,
                })
            except Exception:
                pass

        # Log LLM pred stats (mirrors llm_predictor.py line 419).
        # Untuned preds are fixed, but we log per round so panels look identical to training.
        if _HAS_WANDB and wandb.run is not None:
            try:
                wandb.log({
                    "round": round_num,
                    "pred_mean": pred_mean,
                    "pred_std":  pred_std,
                    "pred_min":  pred_min,
                    "pred_max":  pred_max,
                    "pred_bias_vs_target":       float(pred_mean - target_mean),
                    "pred_std_ratio_vs_target":  float(pred_std / max(target_std, 1e-9)),
                })
            except Exception:
                pass

        # Log parse_fail stats per round (constant for untuned, mirrors readout log)
        if _HAS_WANDB and wandb.run is not None:
            try:
                wandb.log({
                    "round": round_num,
                    "parse_fail_count": n_fail,
                    "parse_fail_rate":  fail_rate,
                })
            except Exception:
                pass

        # Log entropy stats per round (constant for untuned; gives a flat
        # reference line for the beta=infinity policy in training panel plots).
        if _HAS_WANDB and wandb.run is not None:
            try:
                wandb.log({"round": round_num, **base_ent})
            except Exception:
                pass

        # labeled get their own current expressed opinion as platform prediction
        platform_predictions[:n] = x_labeled_prior
        x_zero = (1.0 - platform_sus) * innate + platform_sus * platform_predictions
        x_temp = x_zero.copy()
        for _ in range(FJ_K):
            x_temp = peer_sus * x_zero + (1.0 - peer_sus) * (W_norm @ x_temp)
        traj[:, t + 1] = x_temp
        x_labeled_prior = x_temp[:n].copy()

        # Log opinion stats (mirrors pokec_simulations.py line 553)
        if _HAS_WANDB and wandb.run is not None:
            try:
                wandb.log({
                    "round": round_num,
                    "opinion_mean": float(np.mean(x_temp)),
                    "opinion_std":  float(np.std(x_temp)),
                    "opinion_min":  float(np.min(x_temp)),
                    "opinion_max":  float(np.max(x_temp)),
                    "opinion_q25":  float(np.quantile(x_temp, 0.25)),
                    "opinion_q75":  float(np.quantile(x_temp, 0.75)),
                    "opinion_range": float(np.max(x_temp) - np.min(x_temp)),
                    "opinion_hist": wandb.Histogram(x_temp, num_bins=40),
                    "dist_to_innate": float(np.linalg.norm(x_temp - innate) / np.sqrt(agent_num)),
                    "mean_drift_from_innate":     float(np.mean(x_temp) - x_star_mean),
                    "abs_mean_drift_from_innate": float(abs(np.mean(x_temp) - x_star_mean)),
                    "std_ratio_to_innate":        float(np.std(x_temp) / max(x_star_std, 1e-9)),
                })
            except Exception as e:
                print(f"[run_untuned] wandb log skipped: {e}")

        if round_num % 5 == 0 or t == 0:
            print(f"  t={round_num:2d}  mean={x_temp.mean():.4f}  var={x_temp.var():.5f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"llm_llm_untuned_{RUN_TAG}_trajectory.pk")
    pickle.dump(traj, open(out_path, "wb"))
    print(f"wrote {out_path}")

    # also cache the base preds for later reuse / inspection
    pickle.dump(base_preds, open(os.path.join(OUT_DIR, f"llm_llm_untuned_{RUN_TAG}_base_preds.pk"), "wb"))

    if _HAS_WANDB and wandb.run is not None:
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
