"""Closed-form q* runner for the FJ opinion dynamics simulator.

For each round t:
    1. Refit h_p(v_k | x) via HistGBM on current labeled (features, bin(opinion)).
    2. Score pi_ref(v_k | x_i) under the frozen base LM over the K candidate bin
       strings (no LM training, ever).
    3. Compute q*(v_k | x_i) propto pi_ref(v_k | x_i) * h_p(v_k | x_i)^(1/beta)
       and normalize over k.
    4. Use the expected value E_{q*}[v] = sum_k v_k * q*(v_k | x_i) as the
       prediction for each unlabeled agent i.
    5. Run the same FJ dynamics step as run_untuned_llm.py.

The LM weights never change, so this run shows up in the EGTA cross-eval as
another --static-run (model column = base LM at every snapshot, world column =
the trajectory written by this script).

Env vars:
    KL_BETA           required, beta for q* (smaller beta = more concentrated on argmax h_p).
    BASE_MODEL        HF id, default Qwen/Qwen2.5-0.5B-Instruct.
    RUN_TAG           output filename component.
    RETRAIN_T         FJ rounds, default 30.
    CF_N_BINS         K, default 11 (matches RLKL_N_BINS).
    BATCH_SIZE        prompt batch size for the pi_ref scoring pass, default 16.
    OUT_DIR           default pokec_dataset/results.
"""
from __future__ import annotations

import os
import pickle
import re

import numpy as np
import pandas as pd
import networkx as nx
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

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

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
KL_BETA = float(os.environ["KL_BETA"])
RUN_TAG = os.environ.get("RUN_TAG", "cf")
RETRAIN_T = int(os.environ.get("RETRAIN_T", 30))
N_BINS = int(os.environ.get("CF_N_BINS", 11))
OUT_DIR = os.environ.get("OUT_DIR", "pokec_dataset/results")

PROMPT_COLS = ["age", "gender", "relation_to_alcohol"]
TARGET = "relation_to_smoking"
PROFILES = "pokec_dataset/lcc_profiles_" + TARGET + ".pk"
GRAPH    = "pokec_dataset/lcc_graph_" + TARGET + ".pk"
YLAB     = "pokec_dataset/parametric_params/y_label2163.pk"
YUNL     = "pokec_dataset/parametric_params/y_unlabel_label2163.pk"
PEER_PK  = "pokec_dataset/parametric_params/hetero_peer_sus2163.pkl"
FJ_K     = 100

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


def build_messages(row):
    """Mirrors run_untuned_llm.build_messages so pi_ref here matches the
    untuned reference used by the trained runs."""
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


def opdyn_features(df: pd.DataFrame) -> np.ndarray:
    sub = df[PROMPT_COLS].copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X = enc.fit_transform(sub.astype(object).values)
    return X.astype(np.float32)


def bin_centers(n_bins: int) -> np.ndarray:
    """v_k = k/(n_bins-1). Matches llm_predictor._opdyn_bin_strings."""
    return np.array([k / (n_bins - 1) for k in range(n_bins)], dtype=np.float64)


def bin_strings(n_bins: int) -> list[str]:
    return [f"{v:.2f}" for v in bin_centers(n_bins)]


def fit_h_p(df_labeled: pd.DataFrame, y_labeled: np.ndarray, n_bins: int):
    """Returns a callable mapping df_query -> (n_query, K) probability array.
    Same logic as llm_predictor._fit_opdyn_reward_model."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bins[-1] = 1.0 + 1e-9
    y_class = np.clip(np.digitize(y_labeled, bins) - 1, 0, n_bins - 1).astype(np.int64)
    X = opdyn_features(df_labeled)
    if len(np.unique(y_class)) < 2:
        const = np.zeros(n_bins, dtype=np.float64)
        const[int(y_class[0])] = 1.0
        smoothed = (const + 1e-3) / (1.0 + n_bins * 1e-3)
        def predict(df_query):
            return np.tile(smoothed, (len(df_query), 1))
        return predict
    clf = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.1)
    clf.fit(X, y_class)
    classes = clf.classes_
    def predict(df_query):
        Xq = opdyn_features(df_query)
        proba_observed = clf.predict_proba(Xq)
        full = np.full((Xq.shape[0], n_bins), 1e-3, dtype=np.float64)
        for col_i, cls in enumerate(classes):
            full[:, int(cls)] = proba_observed[:, col_i] + 1e-3
        full /= full.sum(axis=1, keepdims=True)
        return full
    return predict


@torch.no_grad()
def score_pi_ref(model, tok, prompts: list[str], cand_token_ids: list[list[int]],
                 batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Return log_pi_ref of shape (n_prompts, K).

    For each prompt p and candidate c, builds [p tokens | c tokens] and reads
    out the sum of token log-probs for the c positions. K forward passes per
    prompt batch (one per candidate), which is cheaper than (n_prompts*K)
    individual passes.
    """
    model.eval()
    K = len(cand_token_ids)
    n = len(prompts)
    log_pi = np.zeros((n, K), dtype=np.float64)

    # Pre-tokenize each prompt once (variable length, left-padded per batch below).
    prompt_ids_list = [tok.encode(p, add_special_tokens=False) for p in prompts]

    for start in range(0, n, batch_size):
        batch_ids = prompt_ids_list[start:start + batch_size]
        B = len(batch_ids)
        max_p = max(len(x) for x in batch_ids)
        # Left-pad prompts so the prompt's last token sits at a known column.
        pad_id = tok.pad_token_id
        prompt_pad = torch.full((B, max_p), pad_id, dtype=torch.long, device=DEVICE)
        prompt_attn = torch.zeros((B, max_p), dtype=torch.long, device=DEVICE)
        for i, ids in enumerate(batch_ids):
            prompt_pad[i, -len(ids):] = torch.tensor(ids, dtype=torch.long, device=DEVICE)
            prompt_attn[i, -len(ids):] = 1

        for k, cand_ids in enumerate(cand_token_ids):
            T_c = len(cand_ids)
            cand = torch.tensor(cand_ids, dtype=torch.long, device=DEVICE)
            full_ids = torch.cat([prompt_pad, cand.unsqueeze(0).expand(B, T_c)], dim=1)
            full_attn = torch.cat([prompt_attn, torch.ones((B, T_c), dtype=torch.long, device=DEVICE)], dim=1)
            out = model(input_ids=full_ids, attention_mask=full_attn)
            logits = out.logits  # (B, max_p + T_c, V)
            slice_logits = logits[:, max_p - 1: max_p + T_c - 1, :]  # (B, T_c, V)
            log_probs = torch.log_softmax(slice_logits.float(), dim=-1)
            gather_idx = cand.unsqueeze(0).unsqueeze(-1).expand(B, T_c, 1)
            log_p_per_pos = log_probs.gather(2, gather_idx).squeeze(-1)  # (B, T_c)
            log_pi[start:start + B, k] = log_p_per_pos.sum(dim=1).cpu().numpy()
    return log_pi


def closed_form_q_star(log_pi_ref: np.ndarray, h_p: np.ndarray, beta: float) -> np.ndarray:
    """q*(v_k | x) ∝ pi_ref(v_k | x) * h_p(v_k | x)^(1/beta).

    Computed in log-space then softmaxed for stability. Returns (n, K).
    """
    log_h = np.log(np.clip(h_p, 1e-12, None))
    log_unnorm = log_pi_ref + (1.0 / max(beta, 1e-9)) * log_h
    log_unnorm -= log_unnorm.max(axis=1, keepdims=True)
    q = np.exp(log_unnorm)
    q /= q.sum(axis=1, keepdims=True)
    return q


def expected_value(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return q @ v


def main():
    df       = pickle.load(open(PROFILES, "rb"))
    network  = pickle.load(open(GRAPH,    "rb"))
    y_lab    = pickle.load(open(YLAB,     "rb"))
    y_unl    = pickle.load(open(YUNL,     "rb"))
    peer_sus = pickle.load(open(PEER_PK,  "rb"))

    innate = np.array(y_lab + y_unl)
    agent_num = len(innate)
    n = int(agent_num * 0.8)
    print(f"[cf] agent_num={agent_num}, n_labeled={n}, n_unlabeled={agent_num - n}, "
          f"K={N_BINS}, beta={KL_BETA}, T={RETRAIN_T}")

    df_labeled   = df.iloc[:n].copy()
    df_unlabeled = df.iloc[n:].copy()
    for d in (df_labeled, df_unlabeled):
        d["age"]    = pd.to_numeric(d["age"],    errors="coerce")
        d["gender"] = pd.to_numeric(d["gender"], errors="coerce")

    nodelist = df["user_id"].values
    adj_mat = nx.to_numpy_array(network, nodelist=nodelist)
    weight_mat = adj_mat.copy()
    degs_inv = 1.0 / np.sum(adj_mat, axis=0)
    degs_inv[np.isinf(degs_inv)] = 0.0
    degs_inv[degs_inv > 1.1] = 0.0
    W_norm = weight_mat * degs_inv[:, None]

    platform_sus = np.ones(agent_num)

    print(f"[cf] loading base model {BASE_MODEL} on {DEVICE}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype).to(DEVICE)
    model.config.pad_token_id = tok.pad_token_id
    model.eval()

    # pi_ref is a frozen function of the prompts. Prompts depend only on static
    # profile features (not on the round), so we score it ONCE.
    messages_unlabeled = [build_messages(r) for _, r in df_unlabeled.iterrows()]
    prompts_unlabeled = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_unlabeled
    ]
    print(f"[cf] example prompt:\n{prompts_unlabeled[0]!r}\n---")

    cand_strings = bin_strings(N_BINS)
    cand_token_ids = [tok.encode(" " + s, add_special_tokens=False) for s in cand_strings]
    print(f"[cf] bin candidates: {cand_strings}")
    print(f"[cf] candidate token lengths: {[len(ids) for ids in cand_token_ids]}")

    print(f"[cf] scoring pi_ref over {len(prompts_unlabeled)} prompts x {N_BINS} bins (one-shot)")
    log_pi_ref = score_pi_ref(model, tok, prompts_unlabeled, cand_token_ids)
    pi_ref = np.exp(log_pi_ref - log_pi_ref.max(axis=1, keepdims=True))
    pi_ref /= pi_ref.sum(axis=1, keepdims=True)
    print(f"[cf] pi_ref bin marginals (mean over agents): "
          f"{pi_ref.mean(axis=0).round(3).tolist()}")

    v = bin_centers(N_BINS)
    traj = np.zeros((agent_num, RETRAIN_T + 1))
    traj[:, 0] = innate.copy()
    x_labeled_prior = innate[:n].copy()
    platform_predictions = np.zeros(agent_num)

    if _HAS_WANDB:
        try:
            wandb.init(
                project="opinion-dynamics-llm",
                name=RUN_TAG,
                config={
                    "model_name": "llm_cf",
                    "run_tag": RUN_TAG,
                    "base_model": BASE_MODEL,
                    "training_style": "cf",
                    "kl_beta": KL_BETA,
                    "n_bins": N_BINS,
                    "retrain_T": RETRAIN_T,
                    "agent_num": agent_num,
                    "n_labeled": n,
                    "n_unlabeled": agent_num - n,
                    "bin_strings": cand_strings,
                },
                reinit=False,
            )
            wandb.define_metric("round")
            for pat in ("opinion_*", "pred_*", "target_*", "h_p_*", "q_star_*", "pi_ref_*"):
                wandb.define_metric(pat, step_metric="round")
        except Exception as e:
            print(f"[cf] wandb init skipped: {e}")

    for t in range(RETRAIN_T):
        round_num = t + 1
        # Refit h_p on (df_labeled features, bin of CURRENT labeled opinions).
        # The closed-form q* uses this round-t reward.
        predict_h_p = fit_h_p(df_labeled, x_labeled_prior, N_BINS)
        h_p_unl = predict_h_p(df_unlabeled)  # (n_unlabeled, K)

        q_star = closed_form_q_star(log_pi_ref, h_p_unl, KL_BETA)  # (n_unlabeled, K)
        pred_unl = expected_value(q_star, v)
        platform_predictions[n:] = pred_unl
        platform_predictions[:n] = x_labeled_prior

        target_mean  = float(np.mean(x_labeled_prior))
        target_std   = float(np.std(x_labeled_prior))
        pred_mean    = float(pred_unl.mean())
        pred_std     = float(pred_unl.std())

        if _HAS_WANDB and wandb.run is not None:
            try:
                wandb.log({
                    "round": round_num,
                    "target_mean": target_mean,
                    "target_std":  target_std,
                    "pred_mean": pred_mean,
                    "pred_std":  pred_std,
                    "pred_bias_vs_target": pred_mean - target_mean,
                    "h_p_argmax_mean": float(np.mean(h_p_unl.argmax(axis=1))),
                    "q_star_entropy_mean": float(np.mean(
                        -(q_star * np.log(np.clip(q_star, 1e-12, None))).sum(axis=1)
                    )),
                    "q_star_argmax_mean": float(np.mean(q_star.argmax(axis=1))),
                })
            except Exception:
                pass

        x_zero = (1.0 - platform_sus) * innate + platform_sus * platform_predictions
        x_temp = x_zero.copy()
        for _ in range(FJ_K):
            x_temp = peer_sus * x_zero + (1.0 - peer_sus) * (W_norm @ x_temp)
        traj[:, t + 1] = x_temp
        x_labeled_prior = x_temp[:n].copy()

        if _HAS_WANDB and wandb.run is not None:
            try:
                wandb.log({
                    "round": round_num,
                    "opinion_mean": float(np.mean(x_temp)),
                    "opinion_std":  float(np.std(x_temp)),
                    "opinion_min":  float(np.min(x_temp)),
                    "opinion_max":  float(np.max(x_temp)),
                })
            except Exception:
                pass

        if round_num % 5 == 0 or t == 0:
            print(f"  t={round_num:2d}  pred_mean={pred_mean:.4f}  "
                  f"opinion_mean={x_temp.mean():.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    cache_key = f"llm_{RUN_TAG}"
    traj_path = os.path.join(OUT_DIR, f"{cache_key}_trajectory.pk")
    equ_path  = os.path.join(OUT_DIR, f"{cache_key}_equilibrium.pk")
    fjeq_path = os.path.join(OUT_DIR, f"{cache_key}_FJequilibrium.pk")
    pickle.dump(traj, open(traj_path, "wb"))
    pickle.dump(traj[:, -1].copy(), open(equ_path, "wb"))
    pickle.dump(traj[:, 1].copy(), open(fjeq_path, "wb"))
    print(f"[cf] wrote {traj_path}")
    print(f"[cf] wrote {equ_path}")
    print(f"[cf] wrote {fjeq_path}")

    if _HAS_WANDB and wandb.run is not None:
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
