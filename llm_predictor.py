"""
LLM-as-Algo predictor: replaces Jiduan's SigmoidMLP in predicting() with
a small LM that is warm-started across retrain rounds.

Probe mode: "parse" — model emits a single number in [0,1], we regex it.
SFT target: f"{y:.2f}".

Module-level cache persists (model, tokenizer, ref_state) across rounds so
theta^(t) warm-starts from theta^(t-1). First call loads the base model and
wraps with LoRA. Reference policy theta_0 is a frozen copy for KL (v1).
"""

import json
import os
import re
import copy
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
USE_LORA = os.environ.get("USE_LORA", "1") not in ("0", "false", "False", "")
LORA_R = int(os.environ.get("LORA_R", 32))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", 64))
SFT_EPOCHS_PER_ROUND = int(os.environ.get("SFT_EPOCHS", 1))
SFT_LR = float(os.environ.get("SFT_LR", 5e-5))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))
MAX_NEW_TOKENS = 6
# Training style:
#   "sft"    — vanilla cross-entropy SFT, no KL anchor.
#   "sft_kl" — form (I): cross-entropy + β·KL on next-token distributions.
#   "rl_kl"  — form (II) Korbak-Williams: -E_{y~q}[log h_p(y|x)] + β·KL(q||π_ref)
#              with K-class restricted softmax over a coarse opinion grid. The
#              optimum of this objective matches the analytical β-family closed
#              form `q* ∝ π_ref · h_p^(1/β)`.
TRAINING_STYLE = os.environ.get("TRAINING_STYLE", "sft_kl")
KL_BETA = float(os.environ.get("KL_BETA", 1.0))
# Coarse-grid bin count for rl_kl. y∈[0,1] is binned to K = RLKL_N_BINS values
# at v_k = k/(K-1) for k=0..K-1. Strings are f"{v_k:.2f}" so they tokenize the
# same way as the existing sft / sft_kl targets.
RLKL_N_BINS = int(os.environ.get("RLKL_N_BINS", 11))
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Post-hoc analysis knobs.
# SAVE_ADAPTER=1 writes the LoRA adapter after every sft_on_round call, overwriting the
# same directory so final-round weights persist. ADAPTER_SAVE_DIR overrides the default
# ./adapters/$RUN_TAG path. READOUT_TEMPERATURE>0 switches readout from greedy to
# stochastic sampling at that temperature (for measuring distributional width at the
# plateau rather than just the greedy mode).
#
# SAVE_ADAPTER_PER_ROUND=1 (only honored when SAVE_ADAPTER=1) writes to
# {save_dir}/round_{round} every SAVE_ADAPTER_EVERY_N_ROUNDS rounds, leaving prior
# rounds in place. SAVE_PROMPTS=1 dumps the round's labeled prompts as JSON in the
# save_dir root so cross-evaluation pipelines (e.g. EGTA FJ adapter) can re-tokenize
# the same way the run trained.
SAVE_ADAPTER = os.environ.get("SAVE_ADAPTER", "0") == "1"
SAVE_ADAPTER_PER_ROUND = os.environ.get("SAVE_ADAPTER_PER_ROUND", "0") == "1"
SAVE_ADAPTER_EVERY_N_ROUNDS = int(os.environ.get("SAVE_ADAPTER_EVERY_N_ROUNDS", 5))
SAVE_PROMPTS = os.environ.get("SAVE_PROMPTS", "0") == "1"
ADAPTER_SAVE_DIR = os.environ.get("ADAPTER_SAVE_DIR", "")
READOUT_TEMPERATURE = float(os.environ.get("READOUT_TEMPERATURE", "0"))

#profiles to run
PROMPT_COLS = [ #keep same as original paper
    "age",
    "gender",
    "relation_to_alcohol",
]

TARGET_NAME = "relation_to_smoking"   # same as from training


_STATE = {
    "model": None,
    "tokenizer": None,
    "ref_model": None,   # frozen pi_base for full-FT KL (None when USE_LORA=True)
    "round": 0,
    "wandb_run": None,
}


def _init_wandb():
    if not _HAS_WANDB or _STATE["wandb_run"] is not None:
        return
    _run_tag = os.environ.get("RUN_TAG", "")
    run = wandb.init(
        project="opinion-dynamics-llm",
        name=f"{TRAINING_STYLE}-{BASE_MODEL.split('/')[-1]}-b{KL_BETA}"
             + (f"-r{LORA_R}" if USE_LORA else "-ff")
             + (f"-{_run_tag}" if _run_tag else ""),
        config={
            "base_model": BASE_MODEL,
            "training_style": TRAINING_STYLE,
            "kl_beta": KL_BETA,
            "use_lora": USE_LORA,
            "lora_r": LORA_R if USE_LORA else None,
            "lora_alpha": LORA_ALPHA if USE_LORA else None,
            "sft_epochs_per_round": SFT_EPOCHS_PER_ROUND,
            "lr": SFT_LR,
            "batch_size": BATCH_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "device": DEVICE,
            "prompt_cols": PROMPT_COLS,
            "target": TARGET_NAME,
            "probe_mode": "parse_float",
        },
        reinit=False,
    )
    _STATE["wandb_run"] = run
    _STATE["wandb_run_id"] = run.id   


def _load_model():
    """First-time load: base + tokenizer + LoRA wrap. Subsequent rounds reuse."""
    if _STATE["model"] is not None:
        return _STATE["model"], _STATE["tokenizer"]

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # required for batched causal-LM generation

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32 if DEVICE == "cpu" else torch.bfloat16,
    ).to(DEVICE)
    model.config.pad_token_id = tok.pad_token_id

    if USE_LORA:
        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "v_proj"],   # TODO: widen if underfitting
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
    elif KL_BETA > 0 and TRAINING_STYLE in ("sft_kl", "rl_kl"):
        # Full FT + KL anchor: hold a frozen copy of pi_base for the KL reference.
        ref = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32 if DEVICE == "cpu" else torch.bfloat16,
        ).to(DEVICE)
        ref.config.pad_token_id = tok.pad_token_id
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        _STATE["ref_model"] = ref

    _STATE["model"] = model
    _STATE["tokenizer"] = tok
    return model, tok



SK_ALCOHOL_EXACT = {
    "pijem prilezitostne": "I drink occasionally",
    "abstinent": "I abstain from alcohol",
    "uz nepijem": "I no longer drink",
    "nepijem": "I don't drink",
    "pijem pravidelne": "I drink regularly",
    "prilezitostne": "occasionally",
    "pijem": "I drink",
    "nikdy": "never",
    "alkoholik": "alcoholic",
}

def translate_alcohol(val: str) -> str:
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


def build_prompt(row: pd.Series, tokenizer=None) -> str:
    """One prompt per user.

    Default: raw 'Answer: ' format (legacy behavior). Qwen-Instruct on this
    format mode-collapses to 3 low digits (pi_ref pred_mean ~0.30). Kept as
    default so already-queued jobs reproduce exactly.

    If env USE_CHAT_TEMPLATE=1, apply the tokenizer's chat template instead,
    matching run_untuned_llm.build_messages. Chat-templated pi_ref has much
    more spread (pred_mean ~0.71) and is the format the Instruct model was
    post-trained for. Pass tokenizer= explicitly, or rely on _STATE['tokenizer']
    populated by _load_model().
    """
    profile_lines = []
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
        profile_lines.append(f"- {col}: {val}")
    profile_str = "\n".join(profile_lines) if profile_lines else "- (no profile info)"

    if os.environ.get("USE_CHAT_TEMPLATE", "0") == "1":
        user_msg = (
            "Estimate this user's attitude toward smoking based on their profile.\n"
            "Profile:\n"
            f"{profile_str}\n\n"
            "Output a single number in [0, 1] (1 = very positive, 0 = very negative). "
            "Respond with only the number, e.g. 0.42."
        )
        messages = [{"role": "user", "content": user_msg}]
        tok = tokenizer if tokenizer is not None else _STATE.get("tokenizer")
        if tok is None:
            raise RuntimeError(
                "build_prompt(USE_CHAT_TEMPLATE=1) needs a tokenizer: pass tokenizer= or call _load_model() first"
            )
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return (
        "You will estimate a user's attitude toward smoking based on their profile.\n"
        "Profile:\n"
        f"{profile_str}\n\n"
        "Task: Output a single number in [0, 1] representing the probability that this "
        "user has a POSITIVE attitude toward smoking (1 = very positive, 0 = very negative).\n"
        "Output format: a single number, e.g. 0.42. No other text.\n"
        "Answer: "
    )


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")

def parse_prob(text: str, default: float = 0.5) -> float:
    m = _NUM_RE.search(text)
    if m is None:
        return default
    try:
        v = float(m.group(0))
    except ValueError:
        return default
    return float(np.clip(v, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Readout (batched greedy decode)
# ---------------------------------------------------------------------------
@torch.no_grad()
def readout(model, tok, prompts, batch_size=BATCH_SIZE):
    model.eval()
    preds = np.zeros(len(prompts), dtype=float)
    n_parse_fail = 0
    bad_samples = []
    gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tok.pad_token_id)
    if READOUT_TEMPERATURE > 0:
        gen_kwargs.update(do_sample=True, temperature=READOUT_TEMPERATURE, top_p=1.0, top_k=0)
    else:
        gen_kwargs["do_sample"] = False
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        out = model.generate(**enc, **gen_kwargs)
        gen = out[:, enc["input_ids"].shape[1]:]
        for j, ids in enumerate(gen):
            txt = tok.decode(ids, skip_special_tokens=True)
            if _NUM_RE.search(txt) is None:
                n_parse_fail += 1
                if len(bad_samples) < 5:
                    bad_samples.append(repr(txt))
            preds[i + j] = parse_prob(txt)
    fail_rate = n_parse_fail / max(len(prompts), 1)
    print(f"[readout] parse_fail={n_parse_fail}/{len(prompts)} ({fail_rate:.1%})")
    if bad_samples:
        print(f"[readout] first unparseable outputs: {bad_samples}")
    if _HAS_WANDB and wandb.run is not None:
        wandb.log({
            "parse_fail_rate": fail_rate,
            "parse_fail_count": n_parse_fail,
            "round": _STATE.get("round", 0),
        })
    return preds


@torch.no_grad()
def entropy_probe(model, tok, prompts, batch_size=BATCH_SIZE, n_probe=100):
    """Teacher-forced entropy probe at the first-digit position.

    Appends '0.' to each prompt, runs a forward pass, and reports mean
    full-vocab entropy, mean digit-restricted entropy (over tokens 0..9
    renormalized), and mean per-digit probability. Use n_probe=100 for cheap
    per-round logging. Assumes left-padded tokenization (model.generate path).
    """
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


class WandbForwardCallback(TrainerCallback):
    """Forward HF Trainer metrics to our persistent wandb run. Does not close the run."""
    def __init__(self, round_num):
        self.round_num = round_num
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not _HAS_WANDB or wandb.run is None or logs is None:
            return
        prefixed = {f"train/{k}": v for k, v in logs.items() if isinstance(v, (int, float))}
        prefixed["train/round"] = self.round_num
        wandb.log(prefixed)


class KLSFTTrainer(SFTTrainer):
    """SFT + beta * KL(pi_theta || pi_ref). pi_ref = base model (LoRA disabled)."""

    def __init__(self, *args, kl_beta=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_beta = kl_beta

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Standard SFT loss (CE on targets)
        outputs = model(**inputs)
        ce_loss = outputs.loss

        if self.kl_beta <= 0:
            return (ce_loss, outputs) if return_outputs else ce_loss

        # Policy logits (LoRA on)
        policy_logits = outputs.logits

        # Reference logits: LoRA-off when using PEFT; otherwise a frozen copy.
        with torch.no_grad():
            if USE_LORA:
                with model.disable_adapter():
                    ref_logits = model(**inputs).logits
            else:
                ref = _STATE["ref_model"]
                assert ref is not None, "ref_model not initialized for full-FT + KL"
                ref_logits = ref(**inputs).logits

        # KL on next-token dist at each position where a target label exists
        labels = inputs.get("labels")
        mask = (labels != -100).float() if labels is not None else torch.ones_like(policy_logits[..., 0])

        #shift for next-token prediction
        policy_logp = torch.log_softmax(policy_logits[:, :-1, :], dim=-1)
        ref_logp = torch.log_softmax(ref_logits[:, :-1, :], dim=-1)
        mask_shift = mask[:, 1:]

        # KL(pi_theta || pi_ref) = sum pi_theta * (logp_theta - logp_ref)
        policy_p = policy_logp.exp()
        kl = (policy_p * (policy_logp - ref_logp)).sum(dim=-1)   # [B, T-1]
        kl = (kl * mask_shift).sum() / mask_shift.sum().clamp_min(1.0)

        loss = ce_loss + self.kl_beta * kl
        self.log({"ce_loss": ce_loss.detach().item(), "kl": kl.detach().item()})
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Form (II) Korbak-Williams KL-RL with K-class restricted softmax over a
# coarse opinion grid.
# ---------------------------------------------------------------------------

def _opdyn_bin_strings(n_bins: int) -> list[str]:
    """Bin centers v_k = k/(n_bins-1) for k=0..n_bins-1, formatted to 2dp.

    Strings are e.g. ["0.00", "0.10", ..., "1.00"] for n_bins=11. They tokenize
    the same way as the existing sft target format (f"{y:.2f}"), so the
    restricted softmax is computed under the same tokenization the LM was
    pretrained / SFT'd on.
    """
    return [f"{k / (n_bins - 1):.2f}" for k in range(n_bins)]


def _opdyn_features(df: pd.DataFrame) -> np.ndarray:
    """Encode the prompt feature columns to a numeric matrix for the reward
    model. Uses ordinal encoding for categoricals; HistGBM handles the result
    natively. Mirrors the columns used in build_prompt (PROMPT_COLS).
    """
    cols = ["age", "gender", "relation_to_alcohol"]
    sub = df[cols].copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X = enc.fit_transform(sub.astype(object).values)
    return X.astype(np.float32)


def _fit_opdyn_reward_model(df_labeled: pd.DataFrame, y_labeled: np.ndarray, n_bins: int):
    """K-class HistGBM on (features, bin(y)). Returns a callable that takes a
    DataFrame and produces a (n, K) array of class probabilities.

    For the form (II) training loss, we treat these probabilities as the
    reward model h_p(v_k|x) over the K bin centers v_k.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bins[-1] = 1.0 + 1e-9  # right-open at 1.0 to keep y=1.0 in last bin
    y_class = np.clip(np.digitize(y_labeled, bins) - 1, 0, n_bins - 1).astype(np.int64)
    X = _opdyn_features(df_labeled)
    if len(np.unique(y_class)) < 2:
        # Degenerate: all examples fell into one bin. Return a constant
        # near-uniform smoothed distribution centered on that bin.
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
        Xq = _opdyn_features(df_query)
        proba_observed = clf.predict_proba(Xq)
        # HistGBM only reports columns for classes seen at fit time. Spread
        # back into the full K-bin grid; missing classes get smoothed mass.
        full = np.full((Xq.shape[0], n_bins), 1e-3, dtype=np.float64)
        for col_i, cls in enumerate(classes):
            full[:, int(cls)] = proba_observed[:, col_i] + 1e-3
        full /= full.sum(axis=1, keepdims=True)
        return full
    return predict


def _tokenize_opdyn_candidates(tokenizer, n_bins: int) -> list[list[int]]:
    """Pre-tokenize each bin string with a leading space for consistency with
    how the existing chat template appends completions. Returns a list of K
    token-id lists.
    """
    out = []
    for s in _opdyn_bin_strings(n_bins):
        ids = tokenizer.encode(" " + s, add_special_tokens=False)
        out.append(ids)
    return out


class OpdynRLKLCollator:
    """Pad pre-tokenized prompts and stack per-row log_rewards (K-vector)."""

    def __init__(self, pad_token_id: int):
        if pad_token_id is None:
            raise ValueError("pad_token_id is required.")
        self.pad_token_id = int(pad_token_id)

    def __call__(self, examples):
        max_len = max(len(ex["prompt_ids"]) for ex in examples)
        # Left-pad: causal-LM convention. Keeps the prompt's last token at the
        # rightmost position, so when we append candidate tokens they land at
        # well-defined relative offsets.
        ids, attn, log_r = [], [], []
        for ex in examples:
            n = len(ex["prompt_ids"])
            pad = max_len - n
            ids.append([self.pad_token_id] * pad + list(ex["prompt_ids"]))
            attn.append([0] * pad + [1] * n)
            log_r.append(ex["log_rewards"])
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "log_rewards": torch.tensor(log_r, dtype=torch.float32),
        }


class RLKLOpdynTrainer(SFTTrainer):
    """Form (II) Korbak-Williams KL-RL with K-class restricted softmax.

    Loss (mean over the batch, restricted to K candidate completions):

        L = -E_{k~q'_θ(·|x)}[ log h_p(v_k|x) ]
            + β · KL_K( q'_θ(·|x) || π_ref'(·|x) )

    where q'_θ(k|x) ∝ q_θ(candidate_token_ids[k] | prompt_x), and π_ref' is
    the same restricted distribution under the frozen reference (or the
    LoRA-disabled adapter-off branch when USE_LORA=True).

    Each batch carries `log_rewards`: shape (B, K), `log h_p(v_k|x_b)`.
    """

    def __init__(
        self,
        *args,
        kl_beta: float = 0.0,
        ref_model=None,
        candidate_token_ids: list[list[int]],
        use_lora: bool,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.kl_beta = float(kl_beta)
        self.ref_model = ref_model
        self.use_lora = bool(use_lora)
        if not self.use_lora and self.kl_beta > 0 and self.ref_model is None:
            raise ValueError("Full-FT + KL needs ref_model for the KL anchor.")
        self.candidate_token_ids = [list(ids) for ids in candidate_token_ids]
        self.K = len(self.candidate_token_ids)

    def _score_candidate(self, model, prompt_ids, attention_mask, cand_ids):
        """log q_model(cand_ids | prompt_ids) per row in the batch.

        Builds [prompt | candidate] for every row in the batch, runs one
        forward pass, and sums the log-probabilities of the candidate tokens
        at their respective positions. Returns shape (B,).
        """
        device = prompt_ids.device
        B, T_prompt = prompt_ids.shape
        cand = torch.tensor(cand_ids, dtype=torch.long, device=device)
        T_cand = cand.shape[0]

        full_ids = torch.cat([prompt_ids, cand.unsqueeze(0).expand(B, T_cand)], dim=1)
        full_attn = torch.cat([attention_mask, attention_mask.new_ones(B, T_cand)], dim=1)

        outputs = model(input_ids=full_ids, attention_mask=full_attn)
        logits = outputs.logits  # (B, T_prompt + T_cand, V)
        # logits[:, t, :] predicts token at position t+1.
        # We want predictions for candidate tokens at positions T_prompt .. T_prompt+T_cand-1,
        # which come from logits at positions T_prompt-1 .. T_prompt+T_cand-2.
        slice_logits = logits[:, T_prompt - 1:T_prompt + T_cand - 1, :]  # (B, T_cand, V)
        log_probs = torch.log_softmax(slice_logits, dim=-1)
        # Gather the candidate token ids.
        gather_idx = cand.unsqueeze(0).unsqueeze(-1).expand(B, T_cand, 1)
        log_p_per_pos = log_probs.gather(2, gather_idx).squeeze(-1)  # (B, T_cand)
        return log_p_per_pos.sum(dim=1)  # (B,)

    def _ref_logp_candidate(self, model, prompt_ids, attention_mask, cand_ids):
        """Like _score_candidate but under π_ref. For LoRA, disable the
        adapter; for full-FT, use the held-out frozen model."""
        if self.use_lora:
            with torch.no_grad():
                with model.disable_adapter():
                    return self._score_candidate(model, prompt_ids, attention_mask, cand_ids)
        with torch.no_grad():
            return self._score_candidate(self.ref_model, prompt_ids, attention_mask, cand_ids)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        log_rewards = inputs["log_rewards"]  # (B, K)

        B = prompt_ids.shape[0]
        device = prompt_ids.device

        log_q_policy = torch.zeros(B, self.K, device=device)
        log_q_ref = torch.zeros(B, self.K, device=device)
        for k, cand_ids in enumerate(self.candidate_token_ids):
            log_q_policy[:, k] = self._score_candidate(model, prompt_ids, attention_mask, cand_ids)
            log_q_ref[:, k] = self._ref_logp_candidate(model, prompt_ids, attention_mask, cand_ids)

        # Restricted softmax over the K candidates.
        log_q_policy_norm = log_q_policy - torch.logsumexp(log_q_policy, dim=1, keepdim=True)
        log_q_ref_norm = log_q_ref - torch.logsumexp(log_q_ref, dim=1, keepdim=True)
        q_policy = log_q_policy_norm.exp()  # (B, K)

        # Negative reward (= -E_{q'}[log h_p]) per row.
        neg_reward = -(q_policy * log_rewards.to(q_policy.dtype)).sum(dim=1)  # (B,)
        # K-class restricted KL(q'_θ || π_ref') per row.
        kl_per_row = (q_policy * (log_q_policy_norm - log_q_ref_norm)).sum(dim=1)  # (B,)

        loss = neg_reward.mean() + self.kl_beta * kl_per_row.mean()
        self.log({
            "neg_reward": neg_reward.detach().mean().item(),
            "kl_K": kl_per_row.detach().mean().item(),
        })
        # For TRL plumbing, return a dummy outputs object if requested.
        if return_outputs:
            return loss, {"loss": loss.detach()}
        return loss


def sft_on_round(model, tok, prompts_labeled, y_labeled, prompts_unlabeled=None):
    targets = [f"{float(y):.2f}" for y in y_labeled]

    ds = Dataset.from_dict({"prompt": list(prompts_labeled), "completion": targets})

    out_dir = f"./llm_ckpt/round_{_STATE['round']}"
    os.makedirs(out_dir, exist_ok=True)

    cfg = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", 1)),
        gradient_checkpointing=(os.environ.get("GRAD_CKPT", "0") == "1"),
        num_train_epochs=SFT_EPOCHS_PER_ROUND,
        learning_rate=SFT_LR,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        completion_only_loss=True,
        max_length=200,
        bf16=(DEVICE == "cuda"),
        optim=os.environ.get("OPTIMIZER", "adamw_torch"),
    )

    callbacks = [WandbForwardCallback(round_num=_STATE["round"])]
    if TRAINING_STYLE == "sft":
        trainer = SFTTrainer(
            model=model, processing_class=tok, args=cfg, train_dataset=ds,
            callbacks=callbacks,
        )
    elif TRAINING_STYLE == "sft_kl":
        trainer = KLSFTTrainer(
            model=model, processing_class=tok, args=cfg, train_dataset=ds,
            kl_beta=KL_BETA, callbacks=callbacks,
        )
    elif TRAINING_STYLE == "rl_kl":
        # rl_kl uses a separate code path (rl_kl_on_round) because it needs the
        # original DataFrame to build the reward model. predicting_llm dispatches
        # on TRAINING_STYLE before reaching sft_on_round.
        raise RuntimeError(
            "TRAINING_STYLE=rl_kl should be dispatched via rl_kl_on_round, not sft_on_round."
        )
    else:
        raise NotImplementedError(f"TRAINING_STYLE={TRAINING_STYLE} not yet supported")

    if os.environ.get("SFT_SANITY", "0") == "1" and _STATE["round"] == 1:
        try:
            batch = next(iter(trainer.get_train_dataloader()))
            lb = batch["labels"][0]
            n_masked = (lb == -100).sum().item()
            n_total = lb.numel()
            print(f"[sanity] round {_STATE['round']}: labels shape={tuple(lb.shape)}  "
                  f"masked={n_masked}/{n_total} ({100*n_masked/n_total:.0f}%)")
            print(f"[sanity] last 12 input_ids: {batch['input_ids'][0][-12:].tolist()}")
            print(f"[sanity] last 12 labels   : {lb[-12:].tolist()}")
        except Exception as e:
            print(f"[sanity] check failed: {e}")

    trainer.train()

    if SAVE_ADAPTER:
        if ADAPTER_SAVE_DIR:
            save_root = ADAPTER_SAVE_DIR
        else:
            tag = os.environ.get("RUN_TAG", "")
            if not tag:
                rank_str = str(LORA_R) if USE_LORA else "ff"
                tag = f"{TRAINING_STYLE}_b{KL_BETA}_r{rank_str}"
            save_root = f"./adapters/{tag}"
        round_num = _STATE["round"]
        if SAVE_ADAPTER_PER_ROUND:
            if round_num % SAVE_ADAPTER_EVERY_N_ROUNDS != 0:
                save_dir = None
            else:
                save_dir = os.path.join(save_root, f"round_{round_num}")
        else:
            save_dir = save_root
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            try:
                model.save_pretrained(save_dir)
                print(f"[save_adapter] round {round_num}: wrote adapter to {save_dir}")
            except Exception as e:
                print(f"[save_adapter] round {round_num}: save failed: {e}")
        if SAVE_PROMPTS:
            os.makedirs(save_root, exist_ok=True)
            try:
                with open(os.path.join(save_root, "prompts_labeled.json"), "w") as f:
                    json.dump(list(prompts_labeled), f)
                if prompts_unlabeled is not None:
                    with open(os.path.join(save_root, "prompts_unlabeled.json"), "w") as f:
                        json.dump(list(prompts_unlabeled), f)
            except Exception as e:
                print(f"[save_adapter] round {round_num}: prompts save failed: {e}")

    return model



def rl_kl_on_round(model, tok, prompts_labeled, y_labeled, df_labeled, prompts_unlabeled=None):
    """Form (II) Korbak-Williams KL-RL training round.

    1. Fit a K-class reward model h_p on (df_labeled features, bin(y_labeled)).
    2. Pre-tokenize the K candidate strings ("0.00", "0.10", ..., "1.00").
    3. Build a dataset of (prompt_ids, log_rewards) where
           log_rewards[k] = log h_p(v_k | x_i).
    4. Train the LLM via `RLKLOpdynTrainer` minimizing
           -E_{k~q'_θ(·|x)}[log h_p(v_k|x)] + β · KL_K(q'_θ || π_ref').
    """
    n_bins = RLKL_N_BINS
    reward_predict = _fit_opdyn_reward_model(df_labeled, np.asarray(y_labeled), n_bins)
    h_p = reward_predict(df_labeled)  # (n, K)
    h_p = np.clip(h_p, 1e-9, 1.0)
    log_h_p = np.log(h_p)  # (n, K)

    cand_ids = _tokenize_opdyn_candidates(tok, n_bins)
    if any(len(ids) == 0 for ids in cand_ids):
        raise RuntimeError("Empty candidate token ids; tokenizer encoded a candidate to nothing.")

    # Pre-tokenize prompts. Truncate from the front so the prompt ends at the
    # last natural token; we want the LM's next-token prediction to start at
    # the candidate.
    max_prompt_len = 200 - max(len(ids) for ids in cand_ids)
    rows = []
    for prompt, log_r in zip(prompts_labeled, log_h_p):
        ids = tok.encode(prompt, add_special_tokens=False)
        if len(ids) > max_prompt_len:
            ids = ids[-max_prompt_len:]
        rows.append({"prompt_ids": ids, "log_rewards": log_r.tolist()})

    ds = Dataset.from_list(rows)

    out_dir = f"./llm_ckpt/rl_kl_round_{_STATE['round']}"
    os.makedirs(out_dir, exist_ok=True)

    cfg = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", 1)),
        gradient_checkpointing=(os.environ.get("GRAD_CKPT", "0") == "1"),
        num_train_epochs=SFT_EPOCHS_PER_ROUND,
        learning_rate=SFT_LR,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        completion_only_loss=False,  # we never use TRL's CE; loss is custom.
        max_length=200,
        bf16=(DEVICE == "cuda"),
        optim=os.environ.get("OPTIMIZER", "adamw_torch"),
        remove_unused_columns=False,
    )

    callbacks = [WandbForwardCallback(round_num=_STATE["round"])]
    collator = OpdynRLKLCollator(pad_token_id=tok.pad_token_id)
    trainer = RLKLOpdynTrainer(
        model=model,
        processing_class=tok,
        args=cfg,
        train_dataset=ds,
        data_collator=collator,
        kl_beta=KL_BETA,
        ref_model=_STATE.get("ref_model"),
        candidate_token_ids=cand_ids,
        use_lora=USE_LORA,
        callbacks=callbacks,
    )

    if os.environ.get("SFT_SANITY", "0") == "1" and _STATE["round"] == 1:
        try:
            batch = next(iter(trainer.get_train_dataloader()))
            print(f"[rl_kl-sanity] round {_STATE['round']}: input_ids shape={tuple(batch['input_ids'].shape)} "
                  f"log_rewards shape={tuple(batch['log_rewards'].shape)} "
                  f"K={trainer.K} cand_ids[0]={cand_ids[0]}")
            print(f"[rl_kl-sanity] log_rewards[0,:3]={batch['log_rewards'][0,:3].tolist()} "
                  f"min={float(batch['log_rewards'].min()):.3f} max={float(batch['log_rewards'].max()):.3f}")
        except Exception as e:
            print(f"[rl_kl-sanity] check failed: {e}")

    trainer.train()

    if SAVE_ADAPTER:
        if ADAPTER_SAVE_DIR:
            save_root = ADAPTER_SAVE_DIR
        else:
            tag = os.environ.get("RUN_TAG", "")
            if not tag:
                rank_str = str(LORA_R) if USE_LORA else "ff"
                tag = f"{TRAINING_STYLE}_b{KL_BETA}_r{rank_str}"
            save_root = f"./adapters/{tag}"
        round_num = _STATE["round"]
        if SAVE_ADAPTER_PER_ROUND:
            if round_num % SAVE_ADAPTER_EVERY_N_ROUNDS != 0:
                save_dir = None
            else:
                save_dir = os.path.join(save_root, f"round_{round_num}")
        else:
            save_dir = save_root
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            try:
                model.save_pretrained(save_dir)
                print(f"[save_adapter] round {round_num}: wrote adapter to {save_dir}")
            except Exception as e:
                print(f"[save_adapter] round {round_num}: save failed: {e}")
        if SAVE_PROMPTS:
            os.makedirs(save_root, exist_ok=True)
            try:
                with open(os.path.join(save_root, "prompts_labeled.json"), "w") as f:
                    json.dump(list(prompts_labeled), f)
                if prompts_unlabeled is not None:
                    with open(os.path.join(save_root, "prompts_unlabeled.json"), "w") as f:
                        json.dump(list(prompts_unlabeled), f)
            except Exception as e:
                print(f"[save_adapter] round {round_num}: prompts save failed: {e}")

    return model


def predicting_llm(df_labeled, y_labeled, df_unlabeled, sim_params=None):
    """
    df_labeled : pd.DataFrame (n rows), profile columns
    y_labeled  : array-like length n, current x_labeled_prior (scalars in [0,1])
    df_unlabeled : pd.DataFrame (m rows)
    sim_params : dict of simulation-level config to log once on round 1

    returns: np.ndarray length m of unlabeled predictions in [0,1]
    """
    _init_wandb()
    model, tok = _load_model()
    _STATE["round"] += 1

    
    if _STATE["round"] == 1 and _HAS_WANDB and wandb.run is not None and sim_params:
        wandb.config.update({f"sim_{k}": v for k, v in sim_params.items()}, allow_val_change=True)
    print(f"\n===== [llm_predictor] round {_STATE['round']} | n_labeled={len(df_labeled)} n_unlabeled={len(df_unlabeled)} =====")

    prompts_labeled = [build_prompt(r) for _, r in df_labeled.iterrows()]
    prompts_unlabeled = [build_prompt(r) for _, r in df_unlabeled.iterrows()]

    # --- diagnostics: run once on round 1 -------------------------------
    if _STATE["round"] == 1:
        y_arr = np.asarray(y_labeled, dtype=float)
        print(f"[diag] y_labeled stats: mean={y_arr.mean():.3f} std={y_arr.std():.3f} "
              f"min={y_arr.min():.3f} max={y_arr.max():.3f} n_unique={len(np.unique(np.round(y_arr, 2)))}")
        uniq_prompts = len(set(prompts_labeled))
        print(f"[diag] unique prompts among labeled: {uniq_prompts}/{len(prompts_labeled)}")
        print(f"[diag] unique relation_to_alcohol values:")
        print(df_labeled["relation_to_alcohol"].value_counts().head(20))
    # --------------------------------------------------------------------

    # Pre-SFT baseline probe on round 1 only — does the base model discriminate?
    if _STATE["round"] == 1:
        base_preds = readout(model, tok, prompts_unlabeled[:100])
        print(f"[diag] BASE (pre-SFT) preds on 100 unlabeled: "
              f"mean={base_preds.mean():.3f} std={base_preds.std():.3f} "
              f"min={base_preds.min():.3f} max={base_preds.max():.3f} "
              f"n_unique={len(np.unique(np.round(base_preds, 2)))}")
        base_ent = entropy_probe(model, tok, prompts_unlabeled, n_probe=100)
        print(f"[diag] BASE entropy: H_full={base_ent['entropy_full']:.3f} "
              f"H_digits={base_ent['entropy_digits']:.3f} "
              f"top digits P: "
              + " ".join(f"{d}:{base_ent[f'digit_prob_{d}']:.3f}" for d in range(10)))
        if _HAS_WANDB and wandb.run is not None:
            wandb.log({"round": 0, **{f"base_{k}": v for k, v in base_ent.items()}})

    # Train on D^(t-1) = (prompts_labeled, y_labeled)
    if TRAINING_STYLE == "rl_kl":
        rl_kl_on_round(model, tok, prompts_labeled, np.asarray(y_labeled), df_labeled,
                       prompts_unlabeled=prompts_unlabeled)
    else:
        sft_on_round(model, tok, prompts_labeled, np.asarray(y_labeled),
                     prompts_unlabeled=prompts_unlabeled)

    # Read out on unlabeled
    preds = readout(model, tok, prompts_unlabeled)
    print(f"[llm_predictor] round {_STATE['round']} preds: mean={preds.mean():.3f} std={preds.std():.3f} min={preds.min():.3f} max={preds.max():.3f}")

    # Entropy probe on the trained policy (same 100 prompts each round for comparability).
    ent = entropy_probe(model, tok, prompts_unlabeled, n_probe=100)
    print(f"[llm_predictor] round {_STATE['round']} entropy: H_full={ent['entropy_full']:.3f} "
          f"H_digits={ent['entropy_digits']:.3f}  "
          "digit_P: " + " ".join(f"{d}:{ent[f'digit_prob_{d}']:.3f}" for d in range(10)))

    if _HAS_WANDB:
        # If HF Trainer finished our run, resume it so round-level logs land together
        if wandb.run is None:
            wandb.init(
                project="opinion-dynamics-llm",
                id=_STATE.get("wandb_run_id"),
                resume="allow",
            )
        print(f"[wandb] active run: {wandb.run.id if wandb.run else None}")
        target_mean = float(np.mean(y_labeled))
        target_std = float(np.std(y_labeled))
        wandb.log({
            "round": _STATE["round"],
            "pred_mean": float(preds.mean()),
            "pred_std": float(preds.std()),
            "pred_min": float(preds.min()),
            "pred_max": float(preds.max()),
            # Bias of LLM outputs vs the targets it was trained on this round
            "pred_bias_vs_target": float(preds.mean() - target_mean),
            "pred_std_ratio_vs_target": float(preds.std() / max(target_std, 1e-9)),
            **ent,
        })

    return preds
