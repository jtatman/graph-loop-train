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

def load_baseline_dataset(
    source_path: str = "source.parquet",
    micro_batch_size: int = 300,
    seen_ids: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load micro-batch partition from baseline dataset, filtering out seen sample IDs."""
    source = Path(source_path)
    if not source.exists():
        print(f"[dataset_loader] Downloading baseline parquet from {DEFAULT_SOURCE_URL}...")
        with urlopen(DEFAULT_SOURCE_URL, timeout=120) as response:
            source.write_bytes(response.read())
    df = pd.read_parquet(source)
    if "tweet" not in df.columns or "class" not in df.columns:
        raise ValueError(f"Unexpected columns in baseline dataset: {df.columns}")

    df["sample_id"] = [f"base_{i}" for i in range(len(df))]
    if seen_ids:
        df = df[~df["sample_id"].isin(seen_ids)].reset_index(drop=True)

    if micro_batch_size and len(df) > micro_batch_size:
        # Sample micro-batch cleanly while preserving columns
        df = df.sample(n=micro_batch_size, random_state=42).reset_index(drop=True)


    df["kind"] = "choice"
    df["target"] = None  # Hard target class
    batch_ids = df["sample_id"].tolist()
    return df[["sample_id", "tweet", "class", "target", "kind"]], batch_ids

def load_hf_dataset(
    dataset_name: str,
    micro_batch_size: int = 300,
    seen_ids: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load micro-batch partition from HuggingFace Hub dataset or baseline.
    Parses soft target distributions for JEV distillation datasets.
    """
    if dataset_name.lower() in ("baseline", "default", "tdavidson/hate_speech_offensive"):
        return load_baseline_dataset(micro_batch_size=micro_batch_size, seen_ids=seen_ids)

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
        df = ds.to_pandas()
    except Exception as e:
        print(f"[dataset_loader] ERROR: Failed to load HF dataset '{dataset_name}' ({e}).")
        raise RuntimeError(f"Dataset '{dataset_name}' failed to load: {e}")

    # Generate or extract persistent sample_ids
    if "id" in df.columns:
        df["sample_id"] = df["id"].astype(str)
    else:
        df["sample_id"] = [f"{dataset_name.replace('/', '_')}_{i}" for i in range(len(df))]

    if seen_ids:
        unseen_mask = ~df["sample_id"].isin(seen_ids)
        df = df[unseen_mask].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError(f"All samples in '{dataset_name}' have already been processed in the sample ledger.")

    # Identify text column
    text_col = None
    for col in ["state", "tweet", "text", "sequence", "sentence", "input", "content", "prompt", "instruction", "messages"]:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        for col in df.columns:
            if df[col].dtype == object or isinstance(df[col].iloc[0], str):
                text_col = col
                break
    if text_col is None:
        raise ValueError(f"No suitable text column in '{dataset_name}' with columns {df.columns.tolist()}")

    # Check for JEV soft target probability distribution ('target' column with floats/lists)
    has_soft_target = False
    if "target" in df.columns and isinstance(df["target"].iloc[0], (list, np.ndarray)):
        has_soft_target = True
        df["kind"] = df.get("kind", "score")
        # Create hard class argmax as backup label
        df["class"] = [int(np.argmax(t)) % 3 for t in df["target"]]
    else:
        df["target"] = None
        df["kind"] = "choice"

        label_col = None
        for col in ["class", "label", "target", "labels", "category", "choice", "action", "tool"]:
            if col in df.columns and col != text_col:
                label_col = col
                break
        if label_col is None:
            df["class"] = 0
        else:
            if not pd.api.types.is_numeric_dtype(df[label_col]):
                df["class"] = pd.Categorical(df[label_col]).codes % 3
            else:
                df["class"] = df[label_col].astype(int) % 3

    # Micro-batch sampling
    if micro_batch_size and len(df) > micro_batch_size:
        df = df.sample(n=micro_batch_size, random_state=42).reset_index(drop=True)

    df["tweet"] = df[text_col].astype(str)
    batch_ids = df["sample_id"].tolist()
    return df[["sample_id", "tweet", "class", "target", "kind"]], batch_ids

if __name__ == "__main__":
    df_batch, ids = load_baseline_dataset(micro_batch_size=100)
    print(f"Loaded micro-batch: {len(df_batch)} rows, {len(ids)} IDs.")
    print(df_batch.head(2))
