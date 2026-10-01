import os
os.environ["USE_TF"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import copy, hashlib, html, json, random, re, unicodedata
from pathlib import Path
from urllib.request import urlopen
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

SEED, BATCH = 20260919, 16
LABELS = ["hate speech", "offensive language", "neither"]
QUESTION = {
    "t": "choice",
    "ins": "Classify the language of this message. Hate speech attacks a protected group; offensive language is rude or profane without such hate.",
    "crit": dict.fromkeys(LABELS),
}

def seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def verify(path, expected):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Unexpected file contents: {path}")

def normalize(text):
    text = unicodedata.normalize("NFKC", html.unescape(text))
    return re.sub(r"\s+", " ", text).strip().casefold()

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

def batches(ids, epoch=None):
    ids = list(ids)
    if epoch is None:
        ids.sort(key=lambda i: len(items[i]["ids"]))
    else:
        rng = random.Random(SEED + epoch)
        rng.shuffle(ids)
        ids = [i for start in range(0, len(ids), BATCH * 32)
               for i in sorted(ids[start:start + BATCH * 32],
                               key=lambda i: len(items[i]["ids"]))]
    chunks = [ids[i:i + BATCH] for i in range(0, len(ids), BATCH)]
    if epoch is not None:
        rng.shuffle(chunks)
    return chunks

def collate(ids):
    return collate_items([[items[i] for i in ids]], agent.tok.pad_token_id)

@torch.no_grad()
def encode(b):
    return agent.model.encoder(
        input_ids=b["input_ids"], attention_mask=b["attention_mask"]
    ).last_hidden_state

def cached_batch(ids):
    b = collate(ids)
    h = torch.zeros(len(ids), b["input_ids"].size(1), hidden_size)
    for j, i in enumerate(ids):
        h[j, :len(cache[i])] = cache[i]
    return h, b

@torch.no_grad()
def validation_f1(head, ids):
    head.eval()
    predictions = []
    gold = []
    for chunk in batches(ids):
        h, b = cached_batch(chunk)
        predictions.extend(head(h, b).argmax(-1).tolist())
        gold.extend(b["label"].tolist())
    return f1_score(gold, predictions, average="macro", labels=[0, 1, 2])

def train(ids, epochs, validation=None):
    seed()
    head = Head(agent.model)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-5, weight_decay=0.01)
    weights = 1 / np.sqrt(np.bincount(y[ids], minlength=3).astype(float))
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32)
    step, best_epoch, stale, best = 0, 0, 0, -1.0
    for epoch in range(1, epochs + 1):
        head.train()
        for chunk in batches(ids, epoch):
            step += 1
            for group in optimizer.param_groups:
                group["lr"] = 3e-5 * min(1.0, step / 20)
            h, b = cached_batch(chunk)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(h, b), b["label"], weight=weights)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        if validation is not None:
            score = validation_f1(head, validation)
            print(f"Epoch {epoch}: validation macro F1 = {score:.4f}", flush=True)
            if score > best + 1e-4:
                best, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
            if stale >= 2:
                break
        else:
            print(f"Refit epoch {epoch}/{epochs}", flush=True)
    return head.eval(), best_epoch

torch.set_num_threads(8)
torch.set_num_interop_threads(1)
seed()
out = Path("output")
out.mkdir(exist_ok=False)  # Use a fresh directory for each run.
source = Path("source.parquet")
if not source.exists():
    url = "https://huggingface.co/datasets/tdavidson/hate_speech_offensive/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    with urlopen(url, timeout=120) as response:
        source.write_bytes(response.read())
verify(source, "dd9fa2324dd645e46615b52cff924058933eb2ceff6699c7b02fcbdd5167a7b3")
df = pd.read_parquet(source)
y = df["class"].to_numpy()
groups = df["tweet"].map(normalize).to_numpy()
test, training = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(df, y, groups))
fit_local, val_local = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED + 1).split(training, y[training], groups[training]))
fit, val = training[fit_local], training[val_local]
assert not set(groups[training]) & set(groups[test])
assert not set(groups[fit]) & set(groups[val])
splits = {"train": training.tolist(), "test": test.tolist(), "fit": fit.tolist(), "validation": val.tolist()}
(out / "split.json").write_text(json.dumps(splits))
print({name: len(ids) for name, ids in splits.items()}, flush=True)

base = snapshot_download(
    "convaiinnovations/laya",
    revision="c5d78730f3493e4fe16d61507ef4b78eef7318cf",
    allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"],
    local_dir="base_model",
)
verify(Path(base) / "model.safetensors", "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c")
agent = laya.load(base, device="cpu")
agent.model.requires_grad_(False)
agent.model.eval()
hidden_size = agent.model.encoder.config.hidden_size
items = []
for i, row in df.iterrows():
    ids, markers = build_sequence(agent.tok, row.tweet, QUESTION, 512, 192)
    assert len(markers) == 3
    items.append({"ids": ids, "markers": markers, "qtype": 0, "label": int(row["class"])})

# Cache only the training pool. The frozen encoder stays in eval mode.
cache = {}
for n, chunk in enumerate(batches(training), 1):
    h = encode(collate(chunk))
    for j, i in enumerate(chunk):
        cache[i] = h[j, :len(items[i]["ids"])].clone()
    if n % 25 == 0:
        print(f"Cached {len(cache)}/{len(training)} training rows", flush=True)
selected, epochs = train(fit, 6, validation=val)
del selected
print(f"Selected {epochs} epochs; restarting from the original head", flush=True)
tuned, _ = train(training, epochs)
save_file({k: v.detach().contiguous() for k, v in tuned.state_dict().items()}, str(out / "head.safetensors"))
(out / "selected_epochs.json").write_text(json.dumps({"epochs": epochs}))
cache.clear()

# Both heads use the same encoder pass on every test batch.
original = Head(agent.model).eval()
records = []
with torch.no_grad(), (out / "test_predictions.jsonl").open("w") as f:
    for n, chunk in enumerate(batches(test), 1):
        b = collate(chunk)
        h = encode(b)
        before, after = original(h, b), tuned(h, b)
        for j, i in enumerate(chunk):
            row = {"source_index": int(i), "gold": int(y[i]),
                   "base": int(before[j].argmax()), "tuned": int(after[j].argmax()),
                   "base_logits": before[j].tolist(), "tuned_logits": after[j].tolist()}
            f.write(json.dumps(row) + "\n")
            records.append(row)
        if n % 100 == 0:
            print(f"Tested {len(records)}/{len(test)} rows", flush=True)
gold = [r["gold"] for r in records]
majority = int(np.bincount(y[training]).argmax())
predictions = {name: [r[name] for r in records] for name in ("base", "tuned")}
predictions["majority"] = [majority] * len(gold)
metrics = {}
for name, pred in predictions.items():
    metrics[name] = {
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, average="macro", labels=[0, 1, 2])),
        "per_class": classification_report(gold, pred, labels=[0, 1, 2], target_names=LABELS, output_dict=True, zero_division=0),
    }
    print(f"{name}: accuracy={metrics[name]['accuracy']:.4f}, macro_f1={metrics[name]['macro_f1']:.4f}")
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))
