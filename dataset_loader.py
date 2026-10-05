"""
Dataset Loader & Micro-Batch Sample Ledger for Graph Loop Training.
Supports micro-batch sharding, non-overlapping sample tracking, soft target distributions
for JEV distillation (SargeDev/jev-distill-corpus-v3), and domain datasets (dnagpt/laya-bio).
"""

import os
import html
import re
import unicodedata
from pathlib import Path
from typing import Set, Tuple, List, Optional
from urllib.request import urlopen
import pandas as pd
import numpy as np

DEFAULT_SOURCE_URL = "https://huggingface.co/datasets/tdavidson/hate_speech_offensive/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"

def normalize_text(text: str) -> str:
    """Normalize input text (NFKC, HTML unescape, strip extra spaces, casefold)."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = unicodedata.normalize("NFKC", html.unescape(text))
    return re.sub(r"\s+", " ", text).strip().casefold()

EXCLUDED_PATTERNS = [
    re.compile(r"snake-v\d*", re.IGNORECASE),
    re.compile(r"grid_world", re.IGNORECASE),
    re.compile(r"game_action", re.IGNORECASE),
    re.compile(r"action:\s*(move|turn|step|up|down|left|right|forward|backward)", re.IGNORECASE),
    re.compile(r"x_y_axis", re.IGNORECASE),
    re.compile(r"joystick", re.IGNORECASE),
    re.compile(r"game_state", re.IGNORECASE),
]

def is_excluded_sample(sample_id: str, text: str) -> bool:
    """Filter out video game movement/control vectors or off-domain non-language samples."""
    id_str = str(sample_id) if sample_id is not None else ""
    text_str = str(text) if text is not None else ""
    for pat in EXCLUDED_PATTERNS:
        if pat.search(id_str) or pat.search(text_str):
            return True
    return False

def load_baseline_dataset(
    source_path: str = "source.parquet",
    micro_batch_size: int = 2500,
    seen_ids: Optional[Set[str]] = None,
    replay_ratio: float = 0.30,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load micro-batch partition from baseline dataset, filtering out seen sample IDs with experience replay."""
    source = Path(source_path)
    if not source.exists():
        print(f"[dataset_loader] Downloading baseline parquet from {DEFAULT_SOURCE_URL}...")
        with urlopen(DEFAULT_SOURCE_URL, timeout=120) as response:
            source.write_bytes(response.read())
    df_raw = pd.read_parquet(source)
    if "tweet" not in df_raw.columns or "class" not in df_raw.columns:
        raise ValueError(f"Unexpected columns in baseline dataset: {df_raw.columns}")

    df_raw["sample_id"] = [f"base_{i}" for i in range(len(df_raw))]
    df_raw["kind"] = "choice"
    df_raw["target"] = None

    if seen_ids and len(seen_ids) > 0 and replay_ratio > 0.0:
        df_seen = df_raw[df_raw["sample_id"].isin(seen_ids)]
        df_unseen = df_raw[~df_raw["sample_id"].isin(seen_ids)].reset_index(drop=True)
        replay_count = min(len(df_seen), int(micro_batch_size * replay_ratio))
        unseen_target = micro_batch_size - replay_count
        df_unseen_sampled = df_unseen.sample(n=min(len(df_unseen), unseen_target), random_state=42).reset_index(drop=True) if len(df_unseen) > 0 else pd.DataFrame()
        used_sample_ids = df_unseen_sampled["sample_id"].tolist() if not df_unseen_sampled.empty else []

        if len(df_seen) > 0 and replay_count > 0:
            df_replay_sampled = df_seen.sample(n=replay_count, random_state=42).reset_index(drop=True)
            df = pd.concat([df_unseen_sampled, df_replay_sampled], ignore_index=True)
            print(f"[dataset_loader] Experience Replay Active: {len(df_unseen_sampled)} new rows + {len(df_replay_sampled)} replay rows from ledger (Total: {len(df)} rows)")
        else:
            df = df_unseen_sampled
    else:
        df = df_raw[~df_raw["sample_id"].isin(seen_ids)].reset_index(drop=True) if seen_ids else df_raw
        if micro_batch_size and len(df) > micro_batch_size:
            df = df.sample(n=micro_batch_size, random_state=42).reset_index(drop=True)
        used_sample_ids = df["sample_id"].tolist()

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df[["sample_id", "tweet", "class", "target", "kind"]], used_sample_ids

def load_hf_dataset(
    dataset_name: str,
    micro_batch_size: int = 2500,
    seen_ids: Optional[Set[str]] = None,
    replay_ratio: float = 0.30,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load micro-batch partition from HuggingFace Hub dataset or baseline.
    Supports Experience Replay sampling (30% historical ledger, 70% new unseen).
    Parses soft target distributions for JEV distillation datasets.
    """
    if dataset_name.lower() in ("baseline", "default", "tdavidson/hate_speech_offensive"):
        return load_baseline_dataset(micro_batch_size=micro_batch_size, seen_ids=seen_ids, replay_ratio=replay_ratio)

    from datasets import load_dataset, get_dataset_config_names
    print(f"[dataset_loader] Loading HF dataset: '{dataset_name}'...")

    # Auto-resolve dataset config if missing
    config_name = None
    try:
        cfgs = get_dataset_config_names(dataset_name)
        if cfgs and len(cfgs) > 0:
            config_name = cfgs[0]
            print(f"[dataset_loader] Auto-selected dataset config '{config_name}' for '{dataset_name}'.")
    except Exception:
        pass

    try:
        if config_name:
            ds = load_dataset(dataset_name, config_name, split="train")
        else:
            ds = load_dataset(dataset_name, split="train")
        df_raw = ds.to_pandas()
    except Exception as e:
        print(f"[dataset_loader] ERROR: Failed to load HF dataset '{dataset_name}' ({e}).")
        raise RuntimeError(f"Dataset '{dataset_name}' failed to load: {e}")

    # Generate or extract persistent sample_ids
    if "id" in df_raw.columns:
        df_raw["sample_id"] = df_raw["id"].astype(str)
    else:
        df_raw["sample_id"] = [f"{dataset_name.replace('/', '_')}_{i}" for i in range(len(df_raw))]

    # Identify text column
    text_col = None
    for col in ["state", "tweet", "text", "sequence", "sentence", "input", "content", "prompt", "instruction", "messages"]:
        if col in df_raw.columns:
            text_col = col
            break
    if text_col is None:
        for col in df_raw.columns:
            if df_raw[col].dtype == object or isinstance(df_raw[col].iloc[0], str):
                text_col = col
                break
    if text_col is None:
        raise ValueError(f"No suitable text column in '{dataset_name}' with columns {df_raw.columns.tolist()}")

    # Check for JEV soft target probability distribution ('target' column with floats/lists)
    has_soft_target = False
    if "target" in df_raw.columns and isinstance(df_raw["target"].iloc[0], (list, np.ndarray)):
        has_soft_target = True
        df_raw["kind"] = df_raw.get("kind", "score")
        # Create hard class argmax as backup label
        df_raw["class"] = [int(np.argmax(t)) % 3 for t in df_raw["target"]]
    else:
        df_raw["target"] = None
        df_raw["kind"] = "choice"

        label_col = None
        for col in ["class", "label", "target", "labels", "category", "choice", "action", "tool"]:
            if col in df_raw.columns and col != text_col:
                label_col = col
                break
        if label_col is None:
            df_raw["class"] = 0
        else:
            if not pd.api.types.is_numeric_dtype(df_raw[label_col]):
                df_raw["class"] = pd.Categorical(df_raw[label_col]).codes % 3
            else:
                df_raw["class"] = df_raw[label_col].astype(int) % 3

    df_raw["tweet"] = df_raw[text_col].astype(str)

    # Apply Off-Domain Subset Exclusion Filter (e.g. video game control vectors / snake-v1 actions)
    excluded_mask = [is_excluded_sample(sid, txt) for sid, txt in zip(df_raw["sample_id"], df_raw["tweet"])]
    excluded_count = sum(excluded_mask)
    if excluded_count > 0:
        df_raw = df_raw[~pd.Series(excluded_mask, index=df_raw.index)].reset_index(drop=True)
        print(f"[dataset_loader] Off-Domain Subset Filter: Filtered out {excluded_count} game-control / off-domain rows from '{dataset_name}' (Remaining: {len(df_raw)} rows).")

    # Experience Replay Sampling (30% ledger replay + 70% new unseen)
    if seen_ids and len(seen_ids) > 0 and replay_ratio > 0.0:
        df_seen = df_raw[df_raw["sample_id"].isin(seen_ids)]
        df_unseen = df_raw[~df_raw["sample_id"].isin(seen_ids)].reset_index(drop=True)
        if len(df_unseen) == 0:
            raise RuntimeError(f"All samples in '{dataset_name}' have already been processed in the sample ledger.")

        replay_count = min(len(df_seen), int(micro_batch_size * replay_ratio))
        unseen_target = micro_batch_size - replay_count
        df_unseen_sampled = df_unseen.sample(n=min(len(df_unseen), unseen_target), random_state=42).reset_index(drop=True)
        used_sample_ids = df_unseen_sampled["sample_id"].tolist()

        if len(df_seen) > 0 and replay_count > 0:
            df_replay_sampled = df_seen.sample(n=replay_count, random_state=42).reset_index(drop=True)
            df = pd.concat([df_unseen_sampled, df_replay_sampled], ignore_index=True)
            print(f"[dataset_loader] Experience Replay Active: {len(df_unseen_sampled)} new rows + {len(df_replay_sampled)} replay rows from ledger (Total: {len(df)} rows)")
        else:
            df = df_unseen_sampled
    else:
        df = df_raw[~df_raw["sample_id"].isin(seen_ids)].reset_index(drop=True) if seen_ids else df_raw
        if len(df) == 0:
            raise RuntimeError(f"All samples in '{dataset_name}' have already been processed in the sample ledger.")
        if micro_batch_size and len(df) > micro_batch_size:
            df = df.sample(n=micro_batch_size, random_state=42).reset_index(drop=True)
        used_sample_ids = df["sample_id"].tolist()

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df[["sample_id", "tweet", "class", "target", "kind"]], used_sample_ids

if __name__ == "__main__":
    df_batch, ids = load_baseline_dataset(micro_batch_size=100)
    print(f"Loaded micro-batch: {len(df_batch)} rows, {len(ids)} IDs.")
    print(df_batch.head(2))
