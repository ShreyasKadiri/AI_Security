"""
Shared model utilities for all post-training notebooks.
Handles CPU-safe loading of Qwen2.5-0.5B and common eval helpers.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# Force CPU — no CUDA assumption
os.environ["CUDA_VISIBLE_DEVICES"] = ""

MODEL_ID   = "Qwen/Qwen2.5-0.5B-Instruct"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# CPU-safe training config — keeps memory under ~4GB
CPU_TRAIN_CONFIG = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,   # effective batch = 8
    "max_seq_length": 128,
    "num_train_epochs": 1,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "logging_steps": 10,
    "save_steps": 50,
    "fp16": False,                       # no mixed precision on CPU
    "bf16": False,
    "dataloader_num_workers": 0,         # Windows-safe
    "optim": "adamw_torch",
    "report_to": "none",                 # no wandb
}


# ---------------------------------------------------------------------------
# Model + tokenizer loader
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_id: str = MODEL_ID, for_reward_model: bool = False):
    """
    Load Qwen2.5-0.5B-Instruct for CPU training.
    for_reward_model=True: replaces LM head with scalar regression head.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",   # SFT needs right-padding
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model (CPU, fp32)...")
    if for_reward_model:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=1,           # scalar reward
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )

    model = model.to("cpu")
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model loaded: {param_count:.0f}M parameters, device=cpu")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Chat template helper
# ---------------------------------------------------------------------------
def format_chat_prompt(system: str, user: str, tokenizer) -> str:
    """Format using Qwen's chat template."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback if template not available
        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.1,
) -> str:
    """Run greedy/near-greedy generation on CPU."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=(temperature > 0),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # Decode only the generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Quick evaluation: accuracy on a list of {"prompt", "response"} pairs
# ---------------------------------------------------------------------------
def evaluate_sft(model, tokenizer, val_data: List[Dict], n_samples: int = 20) -> Dict:
    """
    Rough eval: compare first token of generated response to first token of gold response.
    Not rigorous — for visual inspection during training.
    """
    model.eval()
    correct, total = 0, 0
    samples = val_data[:n_samples]
    for item in samples:
        pred = generate_response(model, tokenizer, item["prompt"], max_new_tokens=30)
        gold = item["response"]
        # Simple overlap: does the prediction contain key words from gold?
        gold_words = set(gold.lower().split()[:5])
        pred_words = set(pred.lower().split())
        overlap = len(gold_words & pred_words)
        if overlap >= 2:
            correct += 1
        total += 1
    return {"word_overlap_accuracy": round(correct / max(total, 1), 3), "n_eval": total}


# ---------------------------------------------------------------------------
# Loss tracker for plotting
# ---------------------------------------------------------------------------
class LossTracker:
    def __init__(self):
        self.steps = []
        self.losses = []

    def record(self, step: int, loss: float):
        self.steps.append(step)
        self.losses.append(loss)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"steps": self.steps, "losses": self.losses}, f)
        print(f"Loss history saved to {path}")

    def plot_ascii(self, title: str = "Training loss"):
        """Simple ASCII plot for environments without matplotlib."""
        if not self.losses:
            print("No losses recorded.")
            return
        min_l, max_l = min(self.losses), max(self.losses)
        height = 10
        print(f"\n{title}")
        print(f"  {max_l:.4f} |", end="")
        for l in self.losses:
            bar = int((l - min_l) / max(max_l - min_l, 1e-9) * height)
            print("█" if bar >= height // 2 else "░", end="")
        print()
        print(f"  {min_l:.4f} |{'─' * len(self.losses)}")
        print(f"          step 0 → {self.steps[-1] if self.steps else 0}\n")


# ---------------------------------------------------------------------------
# Save / load checkpoint helpers
# ---------------------------------------------------------------------------
def save_model(model, tokenizer, name: str):
    path = MODELS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    print(f"Model saved to {path}")
    return str(path)


def load_saved_model(name: str, for_reward_model: bool = False):
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
    path = MODELS_DIR / name
    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    if for_reward_model:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(path), num_labels=1, trust_remote_code=True, torch_dtype=torch.float32
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(path), trust_remote_code=True, torch_dtype=torch.float32
        )
    return model.to("cpu"), tokenizer


if __name__ == "__main__":
    print("Model utils loaded. Testing format_chat_prompt...")
    # Dry run without downloading model
    dummy_tok = type("T", (), {"apply_chat_template": None, "pad_token": None, "eos_token": "[EOS]"})()
    prompt = format_chat_prompt("You are helpful.", "What is XSS?", dummy_tok)
    print(prompt)
