"""
Parameterized Training & Evaluation Runner for Graph Loop.
Trains decision head on Laya model, evaluates on gold benchmark validation set,
and saves cycle checkpoints and metrics.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import copy
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
import laya
from laya.common import build_sequence, collate_items

from dataset_loader import load_hf_dataset, normalize_text

SEED = 20260919
LABELS = ["hate speech", "offensive language", "neither"]
QUESTION = {
    "t": "choice",
    "ins": "Classify the language of this message. Hate speech attacks a protected group; offensive language is rude or profane without such hate.",
    "crit": dict.fromkeys(LABELS),
}

def seed(seed_val=SEED):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

class Head(nn.Module):
    def __init__(self, model):
        super().__init__()
        for name in ("head", "type_emb", "scorer"):
            setattr(self, name, copy.deepcopy(getattr(model, name)))
        self.requires_grad_(True)

    def forward(self, h, b):
        h = h + self.type_emb(b["qtype"])[:, None, :]
        for layer in self.head.layers:
            h = layer(h, src_key_padding_mask=~b["attention_mask"].bool())
        idx = b["marker_pos"].clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        z = self.scorer(torch.gather(h, 1, idx)).squeeze(-1).float()
        return z.masked_fill(~b["marker_mask"], -1e4)

def make_batches(items, ids, batch_size, epoch=None):
    ids = list(ids)
    if epoch is None:
        ids.sort(key=lambda i: len(items[i]["ids"]))
    else:
        rng = random.Random(SEED + epoch)
        rng.shuffle(ids)
        ids = [i for start in range(0, len(ids), batch_size * 32)
               for i in sorted(ids[start:start + batch_size * 32], key=lambda i: len(items[i]["ids"]))]
    chunks = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
    if epoch is not None:
        rng.shuffle(chunks)
    return chunks

def collate(items, ids, pad_token_id):
    return collate_items([[items[i] for i in ids]], pad_token_id)

@torch.no_grad()
def encode(model, b):
    return model.encoder(
        input_ids=b["input_ids"], attention_mask=b["attention_mask"]
    ).last_hidden_state

def cached_batch(items, cache, ids, b, hidden_size):
    h = torch.zeros(len(ids), b["input_ids"].size(1), hidden_size)
    for j, i in enumerate(ids):
        h[j, :len(cache[i])] = cache[i]
    return h, b

@torch.no_grad()
def validation_f1(head, items, cache, ids, pad_token_id, hidden_size, batch_size):
    head.eval()
    predictions = []
    gold = []
    for chunk in make_batches(items, ids, batch_size):
        b = collate(items, chunk, pad_token_id)
        h, b = cached_batch(items, cache, chunk, b, hidden_size)
        predictions.extend(head(h, b).argmax(-1).tolist())
        gold.extend(b["label"].tolist())
    return f1_score(gold, predictions, average="macro", labels=[0, 1, 2])

def train_head(agent, items, cache, hidden_size, fit_ids, val_ids, full_train_ids, lr, batch_size, max_epochs):
    seed()
    head = Head(agent.model)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    
    fit_labels = [items[i]["label"] for i in fit_ids]
    counts = np.bincount(fit_labels, minlength=3).astype(float)
    counts[counts == 0] = 1.0
    weights = 1.0 / np.sqrt(counts)
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32)

    best, best_epoch, stale = -1.0, 1, 0
    step = 0

    for epoch in range(1, max_epochs + 1):
        head.train()
        for chunk in make_batches(items, fit_ids, batch_size, epoch):
            step += 1
            for group in optimizer.param_groups:
                group["lr"] = lr * min(1.0, step / 20)
            b = collate(items, chunk, agent.tok.pad_token_id)
            h, b = cached_batch(items, cache, chunk, b, hidden_size)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(h, b), b["label"], weight=weights)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

        if val_ids is not None:
            score = validation_f1(head, items, cache, val_ids, agent.tok.pad_token_id, hidden_size, batch_size)
            if score > best + 1e-4:
                best, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
            if stale >= 2:
                break

    # Refit on full training split for selected epochs
    head_refit = Head(agent.model)
    optimizer_refit = torch.optim.AdamW(head_refit.parameters(), lr=lr, weight_decay=0.01)
    step = 0
    for epoch in range(1, best_epoch + 1):
        head_refit.train()
        for chunk in make_batches(items, full_train_ids, batch_size, epoch):
            step += 1
            for group in optimizer_refit.param_groups:
                group["lr"] = lr * min(1.0, step / 20)
            b = collate(items, chunk, agent.tok.pad_token_id)
            h, b = cached_batch(items, cache, chunk, b, hidden_size)
            optimizer_refit.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head_refit(h, b), b["label"], weight=weights)
            loss.backward()
            nn.utils.clip_grad_norm_(head_refit.parameters(), 1.0)
            optimizer_refit.step()

    return head_refit.eval(), best_epoch

def run_training_cycle(
    cycle_id: int,
    dataset_name: str,
    hyperparams: dict,
    output_dir: Path,
) -> tuple[dict, str]:
    """
    Run one full training & evaluation cycle.
    Returns (metrics_dict, checkpoint_file_path).
    """
    seed()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)

    cycle_out = output_dir / f"cycle_{cycle_id:03d}"
    cycle_out.mkdir(parents=True, exist_ok=True)

    lr = hyperparams.get("lr", 3e-5)
    batch_size = hyperparams.get("batch_size", 16)
    epochs = hyperparams.get("epochs", 6)

    # Load dataset
    print(f"[cycle {cycle_id}] Loading dataset '{dataset_name}'...")
    df = load_hf_dataset(dataset_name)
    y = df["class"].to_numpy()
    groups = df["tweet"].map(normalize_text).to_numpy()

    # Split dataset
    test, training = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(df, y, groups))
    fit_local, val_local = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED + 1).split(training, y[training], groups[training]))
    fit, val = training[fit_local], training[val_local]

    splits = {"train": training.tolist(), "test": test.tolist(), "fit": fit.tolist(), "validation": val.tolist()}
    (cycle_out / "split.json").write_text(json.dumps(splits))

    # Download base model snapshot
    base = snapshot_download(
        "convaiinnovations/laya",
        revision="c5d78730f3493e4fe16d61507ef4b78eef7318cf",
        allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"],
        local_dir="base_model",
    )
    agent = laya.load(base, device="cpu")
    agent.model.requires_grad_(False)
    agent.model.eval()
    hidden_size = agent.model.encoder.config.hidden_size

    # Build sequence inputs
    items = []
    for i, row in df.iterrows():
        ids, markers = build_sequence(agent.tok, row.tweet, QUESTION, 512, 192)
        items.append({"ids": ids, "markers": markers, "qtype": 0, "label": int(row["class"])})

    # Cache encoder states for training set
    cache = {}
    for chunk in make_batches(items, training, batch_size):
        b = collate(items, chunk, agent.tok.pad_token_id)
        h = encode(agent.model, b)
        for j, idx_val in enumerate(chunk):
            cache[idx_val] = h[j, :len(items[idx_val]["ids"])].clone()

    # Train head
    tuned_head, selected_epochs = train_head(
        agent, items, cache, hidden_size, fit, val, training, lr, batch_size, epochs
    )
    cache.clear()

    # Save trained head
    checkpoint_path = cycle_out / "head.safetensors"
    save_file({k: v.detach().contiguous() for k, v in tuned_head.state_dict().items()}, str(checkpoint_path))
    (cycle_out / "selected_epochs.json").write_text(json.dumps({"epochs": selected_epochs, "lr": lr, "batch_size": batch_size}))

    # Evaluate on gold test benchmark
    original_head = Head(agent.model).eval()
    records = []
    with torch.no_grad():
        for chunk in make_batches(items, test, batch_size):
            b = collate(items, chunk, agent.tok.pad_token_id)
            h = encode(agent.model, b)
            before, after = original_head(h, b), tuned_head(h, b)
            for j, idx_val in enumerate(chunk):
                records.append({
                    "source_index": int(idx_val),
                    "gold": int(y[idx_val]),
                    "base": int(before[j].argmax()),
                    "tuned": int(after[j].argmax()),
                })

    gold = [r["gold"] for r in records]
    base_pred = [r["base"] for r in records]
    tuned_pred = [r["tuned"] for r in records]

    metrics = {
        "dataset_name": dataset_name,
        "cycle_id": cycle_id,
        "selected_epochs": selected_epochs,
        "base_accuracy": float(accuracy_score(gold, base_pred)),
        "base_macro_f1": float(f1_score(gold, base_pred, average="macro", labels=[0, 1, 2])),
        "tuned_accuracy": float(accuracy_score(gold, tuned_pred)),
        "tuned_macro_f1": float(f1_score(gold, tuned_pred, average="macro", labels=[0, 1, 2])),
    }

    (cycle_out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[cycle {cycle_id}] Complete. Tuned Macro F1 = {metrics['tuned_macro_f1']:.4f} (Base = {metrics['base_macro_f1']:.4f})")
    return metrics, str(checkpoint_path)
