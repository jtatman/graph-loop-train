"""
Dataset Loader & Schema Normalizer for Graph Loop Training.
Supports local source.parquet baseline, target datasets (dnagpt/laya-bio, SargeDev/jev-distill-corpus-v3),
and generic HuggingFace classification datasets.
"""

import os
import html
import re
import unicodedata
from pathlib import Path
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

def load_baseline_dataset(source_path: str = "source.parquet") -> pd.DataFrame:
    """Load the default benchmark baseline dataset (tdavidson/hate_speech_offensive)."""
    source = Path(source_path)
    if not source.exists():
        print(f"[dataset_loader] Downloading baseline parquet from {DEFAULT_SOURCE_URL}...")
        with urlopen(DEFAULT_SOURCE_URL, timeout=120) as response:
            source.write_bytes(response.read())
    df = pd.read_parquet(source)
    if "tweet" not in df.columns or "class" not in df.columns:
        raise ValueError(f"Unexpected columns in baseline dataset: {df.columns}")
    return df

def load_hf_dataset(dataset_name: str, max_samples: int = 20000) -> pd.DataFrame:
    """
    Load a dataset from HuggingFace Hub or local parquet and normalize
    columns to ('tweet', 'class').
    """
    if dataset_name.lower() in ("baseline", "default", "tdavidson/hate_speech_offensive"):
        return load_baseline_dataset()

    try:
        from datasets import load_dataset
        print(f"[dataset_loader] Loading HF dataset: '{dataset_name}'...")
        ds = load_dataset(dataset_name, split="train")
        df = ds.to_pandas()
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        # Identify text column
        text_col = None
        for col in ["tweet", "text", "sentence", "input", "content", "document", "message"]:
            if col in df.columns:
                text_col = col
                break
        if text_col is None:
            # Pick first string column
            for col in df.columns:
                if df[col].dtype == object or isinstance(df[col].iloc[0], str):
                    text_col = col
                    break
        if text_col is None:
            raise ValueError(f"Could not find suitable text column in dataset '{dataset_name}' with columns {df.columns.tolist()}")

        # Identify label column
        label_col = None
        for col in ["class", "label", "target", "labels", "category", "sentiment"]:
            if col in df.columns:
                label_col = col
                break
        if label_col is None:
            for col in df.columns:
                if col != text_col and (pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col])):
                    label_col = col
                    break

        if label_col is None:
            # Fallback: synthesize label 0
            df["class"] = 0
        else:
            # Map labels to 0, 1, 2 modulo 3 if numeric, or categorical code
            if not pd.api.types.is_numeric_dtype(df[label_col]):
                df["class"] = pd.Categorical(df[label_col]).codes
            else:
                df["class"] = df[label_col].astype(int) % 3

        df["tweet"] = df[text_col].astype(str)
        return df[["tweet", "class"]]

    except Exception as e:
        print(f"[dataset_loader] Failed to load '{dataset_name}': {e}. Falling back to baseline dataset.")
        return load_baseline_dataset()

if __name__ == "__main__":
    df_base = load_baseline_dataset()
    print(f"Loaded baseline dataset: {len(df_base)} rows.")
    print(df_base.head(2))
