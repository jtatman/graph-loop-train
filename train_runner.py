"""
Parameterized Training & Evaluation Runner for Graph Loop.
Features:
- Micro-batch processing (200-500 samples)
- Granular CPU progress logging with tqdm
- Soft target KL divergence distillation + Cross-Entropy loss
- Multi-faceted evaluation: Fixed Prompt Benchmark Suite + Holdout Dataset Eval
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import copy
import json
import random
from pathlib import Path
from typing import Set, Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_fscore_support
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

# Fixed Benchmark Prompt Suite (Standardized test questions across cycles)
FIXED_BENCHMARK_PROMPTS = [
    {"text": "I really appreciate the help you gave me on this project today.", "label": 2},
    {"text": "You are stupid and should go jump off a bridge you scum.", "label": 0},
    {"text": "Damn that movie was fucking hilarious man!", "label": 1},
    {"text": "All members of that ethnic group are subhuman trash and must be destroyed.", "label": 0},
    {"text": "Can you please send me the documentation for the API endpoint?", "label": 2},
    {"text": "What the hell are you talking about you idiot?", "label": 1},
    {"text": "We should treat everyone with respect regardless of their background.", "label": 2},
    {"text": "Go back to your country you parasitic filth.", "label": 0},
    {"text": "Shut up and sit down before I kick your ass.", "label": 1},
    {"text": "The weather forecast predicts light rain in the afternoon.", "label": 2},
    {"text": "Those people are a cancer on our society and need to be eliminated.", "label": 0},
    {"text": "Holy shit that was an insane basketball play!", "label": 1},
    {"text": "The conference call is scheduled for 3 PM EST tomorrow.", "label": 2},
    {"text": "Get out of here you miserable piece of shit.", "label": 1},
    {"text": "We will eliminate every last one of your kind from this earth.", "label": 0},
    {"text": "Thank you for sharing your thoughts on the dataset architecture.", "label": 2},
    {"text": "Bitch please, you don't know anything about coding.", "label": 1},
    {"text": "Subhuman animals like you don't deserve rights.", "label": 0},
    {"text": "The quick brown fox jumps over the lazy dog.", "label": 2},
    {"text": "Damn it, I forgot my keys on the counter again.", "label": 1},
]

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

def compute_loss(logits, labels, soft_targets=None, class_weights=None, temperature=1.5):
    """
    Computes loss: KL divergence for soft targets, or Cross-Entropy for hard labels.
    """
    if soft_targets is not None:
        # Soft-target KL divergence distillation loss (JEV method)
        log_probs = F.log_softmax(logits[:, :3] / temperature, dim=-1)
        targets = soft_targets[:, :3]
        targets = targets / (targets.sum(dim=-1, keepdim=True) + 1e-8)
        loss = F.kl_div(log_probs, targets, reduction="batchmean") * (temperature ** 2)
        return loss
    else:
        # Hard label Cross-Entropy loss
        return F.cross_entropy(logits[:, :3], labels, weight=class_weights)

@torch.no_grad()
def validation_f1(head, items, cache, ids, pad_token_id, hidden_size, batch_size):
    head.eval()
    predictions, gold = [], []
    for chunk in make_batches(items, ids, batch_size):
        b = collate(items, chunk, pad_token_id)
        h, b = cached_batch(items, cache, chunk, b, hidden_size)
        predictions.extend(head(h, b).argmax(-1).tolist())
        gold.extend(b["label"].tolist())
    return f1_score(gold, predictions, average="macro", labels=[0, 1, 2], zero_division=0)

def train_head_micro_batch(
    agent, items, cache, hidden_size, fit_ids, val_ids, full_train_ids,
    lr, batch_size, max_epochs, has_soft_targets=False
):
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

    pbar_epoch = tqdm(range(1, max_epochs + 1), desc="[CPU Head Tuning]", leave=False, unit="epoch")
    for epoch in pbar_epoch:
        head.train()
        chunks = make_batches(items, fit_ids, batch_size, epoch)
        for chunk in chunks:
            step += 1
            for group in optimizer.param_groups:
                group["lr"] = lr * min(1.0, step / 10)
            b = collate(items, chunk, agent.tok.pad_token_id)
            h, b = cached_batch(items, cache, chunk, b, hidden_size)

            soft_batch = None
            if has_soft_targets:
                soft_list = [items[i]["soft_target"] for i in chunk if items[i]["soft_target"] is not None]
                if len(soft_list) == len(chunk):
                    soft_batch = torch.tensor(np.array(soft_list), dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(head(h, b), b["label"], soft_targets=soft_batch, class_weights=weights)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

        if val_ids is not None and len(val_ids) > 0:
            score = validation_f1(head, items, cache, val_ids, agent.tok.pad_token_id, hidden_size, batch_size)
            pbar_epoch.set_postfix({"val_f1": f"{score:.4f}", "best_f1": f"{best:.4f}"})
            if score > best + 1e-4:
                best, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
            if stale >= 2:
                break
        else:
            best_epoch = max_epochs

    # Refit pass on full micro-batch training set
    head_refit = Head(agent.model)
    optimizer_refit = torch.optim.AdamW(head_refit.parameters(), lr=lr, weight_decay=0.01)
    step = 0
    for epoch in range(1, best_epoch + 1):
        head_refit.train()
        for chunk in make_batches(items, full_train_ids, batch_size, epoch):
            step += 1
            for group in optimizer_refit.param_groups:
                group["lr"] = lr * min(1.0, step / 10)
            b = collate(items, chunk, agent.tok.pad_token_id)
            h, b = cached_batch(items, cache, chunk, b, hidden_size)
            soft_batch = None
            if has_soft_targets:
                soft_list = [items[i]["soft_target"] for i in chunk if items[i]["soft_target"] is not None]
                if len(soft_list) == len(chunk):
                    soft_batch = torch.tensor(np.array(soft_list), dtype=torch.float32)

            optimizer_refit.zero_grad(set_to_none=True)
            loss = compute_loss(head_refit(h, b), b["label"], soft_targets=soft_batch, class_weights=weights)
            loss.backward()
            nn.utils.clip_grad_norm_(head_refit.parameters(), 1.0)
            optimizer_refit.step()

    return head_refit.eval(), best_epoch

def evaluate_fixed_benchmark(agent, tuned_head) -> dict:
    """Evaluates tuned head on static Fixed Benchmark Suite."""
    original_head = Head(agent.model).eval()
    bench_items = []
    for item in FIXED_BENCHMARK_PROMPTS:
        ids, markers = build_sequence(agent.tok, item["text"], QUESTION, 512, 192)
        bench_items.append({"ids": ids, "markers": markers, "qtype": 0, "label": item["label"]})

    bench_ids = list(range(len(bench_items)))
    b = collate(bench_items, bench_ids, agent.tok.pad_token_id)
    with torch.no_grad():
        h = encode(agent.model, b)
        base_logits = original_head(h, b)
        tuned_logits = tuned_head(h, b)
        base_preds = base_logits.argmax(-1).tolist()
        tuned_preds = tuned_logits.argmax(-1).tolist()

    gold = [item["label"] for item in FIXED_BENCHMARK_PROMPTS]
    base_f1 = float(f1_score(gold, base_preds, average="macro", labels=[0, 1, 2], zero_division=0))
    tuned_f1 = float(f1_score(gold, tuned_preds, average="macro", labels=[0, 1, 2], zero_division=0))
    base_acc = float(accuracy_score(gold, base_preds))
    tuned_acc = float(accuracy_score(gold, tuned_preds))

    return {
        "fixed_bench_base_f1": base_f1,
        "fixed_bench_tuned_f1": tuned_f1,
        "fixed_bench_base_acc": base_acc,
        "fixed_bench_tuned_acc": tuned_acc,
        "fixed_bench_f1_delta": tuned_f1 - base_f1,
    }

def run_training_cycle(
    cycle_id: int,
    dataset_name: str,
    hyperparams: dict,
    output_dir: Path,
    micro_batch_size: int = 300,
    seen_ids: Optional[Set[str]] = None,
) -> tuple[dict, str, List[str]]:
    """
    Run one full micro-batch training & multi-faceted evaluation cycle.
    Returns (metrics_dict, checkpoint_file_path, used_sample_ids).
    """
    seed()
    num_threads = min(4, os.cpu_count() or 2)
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)


    cycle_out = output_dir / f"cycle_{cycle_id:03d}"
    cycle_out.mkdir(parents=True, exist_ok=True)

    lr = hyperparams.get("lr", 3e-5)
    batch_size = hyperparams.get("batch_size", 16)
    epochs = hyperparams.get("epochs", 6)

    # Load micro-batch dataset
    print(f"\n[cycle {cycle_id:03d}] Loading micro-batch (max {micro_batch_size} samples) from dataset '{dataset_name}'...")
    df, used_sample_ids = load_hf_dataset(dataset_name, micro_batch_size=micro_batch_size, seen_ids=seen_ids)
    print(f"[cycle {cycle_id:03d}] Loaded {len(df)} unseen rows. First sample ID: {used_sample_ids[0] if used_sample_ids else 'None'}")

    y = df["class"].to_numpy()
    groups = df["tweet"].map(normalize_text).to_numpy()
    has_soft_targets = df["target"].iloc[0] is not None if "target" in df.columns else False

    # Split micro-batch dataset (Stratified KFold)
    if len(df) >= 10 and len(np.unique(y)) > 1:
        test, training = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(df, y, groups))
        fit_local, val_local = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED + 1).split(training, y[training], groups[training]))
        fit, val = training[fit_local], training[val_local]
    else:
        # Simple split for small micro-batches
        n = len(df)
        indices = list(range(n))
        test = indices[::5]
        training = [i for i in indices if i not in test]
        val = training[::4]
        fit = [i for i in training if i not in val]

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

    # Build sequence inputs with tqdm progress
    print(f"[cycle {cycle_id:03d}] Tokenizing & building sequence inputs...")
    items = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="[Tokenizing]", unit="row"):
        ids, markers = build_sequence(agent.tok, row.tweet, QUESTION, 512, 192)
        items.append({
            "ids": ids,
            "markers": markers,
            "qtype": 0,
            "label": int(row["class"]),
            "soft_target": row["target"] if has_soft_targets else None,
        })

    # Cache encoder states for micro-batch training set with tqdm progress
    cache = {}
    train_chunks = make_batches(items, training, batch_size)
    print(f"[cycle {cycle_id:03d}] Pre-caching encoder states ({len(training)} training rows)...")
    for chunk in tqdm(train_chunks, desc="[Pre-caching]", unit="batch"):
        b = collate(items, chunk, agent.tok.pad_token_id)
        h = encode(agent.model, b)
        for j, idx_val in enumerate(chunk):
            cache[idx_val] = h[j, :len(items[idx_val]["ids"])].clone()

    # Train head
    print(f"[cycle {cycle_id:03d}] Fine-tuning head on CPU (epochs={epochs}, lr={lr}, soft_targets={has_soft_targets})...")
    tuned_head, selected_epochs = train_head_micro_batch(
        agent, items, cache, hidden_size, fit, val, training, lr, batch_size, epochs, has_soft_targets=has_soft_targets
    )
    cache.clear()

    # Save trained head
    checkpoint_path = cycle_out / "head.safetensors"
    save_file({k: v.detach().contiguous() for k, v in tuned_head.state_dict().items()}, str(checkpoint_path))
    (cycle_out / "selected_epochs.json").write_text(json.dumps({"epochs": selected_epochs, "lr": lr, "batch_size": batch_size}))

    # Evaluate on Holdout Micro-Batch Test set
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

    # Calculate granular classification metrics
    precision, recall, f1, support = precision_recall_fscore_support(gold, tuned_pred, labels=[0, 1, 2], zero_division=0)
    per_class_metrics = {}
    for idx_lbl, name_lbl in enumerate(LABELS):
        per_class_metrics[name_lbl] = {
            "precision": float(precision[idx_lbl]),
            "recall": float(recall[idx_lbl]),
            "f1-score": float(f1[idx_lbl]),
            "support": int(support[idx_lbl]),
        }

    # Evaluate Fixed Benchmark Suite
    bench_results = evaluate_fixed_benchmark(agent, tuned_head)

    metrics = {
        "dataset_name": dataset_name,
        "cycle_id": cycle_id,
        "selected_epochs": selected_epochs,
        "micro_batch_samples": len(df),
        "holdout_test_samples": len(test),
        "base_accuracy": float(accuracy_score(gold, base_pred)),
        "base_macro_f1": float(f1_score(gold, base_pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "tuned_accuracy": float(accuracy_score(gold, tuned_pred)),
        "tuned_macro_f1": float(f1_score(gold, tuned_pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "tuned_micro_f1": float(f1_score(gold, tuned_pred, average="micro", labels=[0, 1, 2], zero_division=0)),
        "tuned_weighted_f1": float(f1_score(gold, tuned_pred, average="weighted", labels=[0, 1, 2], zero_division=0)),
        "per_class_metrics": per_class_metrics,
        "fixed_benchmark": bench_results,
    }

    (cycle_out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Print Formatted Classification Report Console Summary
    print("\n" + "=" * 60)
    print(f"CYCLE {cycle_id:03d} EVALUATION REPORT | Dataset: '{dataset_name}'")
    print("=" * 60)
    print(f"Micro-Batch Size: {len(df)} samples | Holdout Test: {len(test)} samples")
    print(f"Holdout Macro F1: Base = {metrics['base_macro_f1']:.4f} -> Tuned = {metrics['tuned_macro_f1']:.4f} (Delta: {metrics['tuned_macro_f1'] - metrics['base_macro_f1']:+.4f})")
    print(f"Fixed Benchmark F1: Base = {bench_results['fixed_bench_base_f1']:.4f} -> Tuned = {bench_results['fixed_bench_tuned_f1']:.4f} (Delta: {bench_results['fixed_bench_f1_delta']:+.4f})")
    print("\nPer-Class Classification Breakdown:")
    for lbl, m in per_class_metrics.items():
        print(f"  - {lbl:<20}: Precision={m['precision']:.3f}, Recall={m['recall']:.3f}, F1={m['f1-score']:.3f} (Support: {m['support']})")
    print("=" * 60 + "\n")

    return metrics, str(checkpoint_path), used_sample_ids
