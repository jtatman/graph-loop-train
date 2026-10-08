"""
Specialty Dataset Curation & Explicit Label Normalization Engine.
Parses, normalizes, and formats HF datasets for specialty domains:
1. Sentiment Domain:
   - Target mapping: 0 = Positive / Bullish, 1 = Negative / Bearish, 2 = Neutral
   - Standardizes text inputs into decision prompts.
   - Saves to 'data/curated_sentiment_dataset.parquet'.

2. Agent Domain (Tool Selection & Fallback Routing):
   - Target mapping: 0 = Pass / Tool Selected, 1 = Fail / Tool Rejected, 2 = Fallback / Default
   - Saves to 'data/curated_agent_dataset.parquet'.
"""

import os
import re
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import load_dataset

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Standardized Decision Prompt Formatting
SENTIMENT_PROMPT_TEMPLATE = (
    "Evaluate the financial sentiment of the following text:\n"
    "Text: \"{text}\"\n"
    "Choose the correct sentiment class:\n"
    "[0] Positive / Bullish\n"
    "[1] Negative / Bearish\n"
    "[2] Neutral"
)

AGENT_PROMPT_TEMPLATE = (
    "Evaluate the tool routing request:\n"
    "Query / Input: \"{text}\"\n"
    "Choose the correct execution outcome:\n"
    "[0] Pass / Tool Selected\n"
    "[1] Fail / Tool Rejected\n"
    "[2] Neutral / Fallback"
)

def curate_sentiment_datasets() -> pd.DataFrame:
    print("[curate_specialty] Processing Sentiment Datasets...")
    records = []

    # 1. zeroshot/twitter-financial-news-sentiment
    try:
        print("  - Fetching 'zeroshot/twitter-financial-news-sentiment'...")
        ds = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
        df_zero = ds.to_pandas()
        for idx, row in df_zero.iterrows():
            text = str(row.get("text", "")).strip()
            raw_label = row.get("label", None)
            if not text or raw_label is None:
                continue
            if raw_label == 0:    # Bearish
                norm_label = 1
            elif raw_label == 1:  # Bullish
                norm_label = 0
            elif raw_label == 2:  # Neutral
                norm_label = 2
            else:
                continue

            records.append({
                "sample_id": f"sentiment_zeroshot_{idx}",
                "tweet": SENTIMENT_PROMPT_TEMPLATE.format(text=text),
                "class": norm_label,
                "target": None,
                "kind": "choice",
                "source": "zeroshot/twitter-financial-news-sentiment"
            })
        print(f"    Loaded {len(df_zero)} records from zeroshot.")
    except Exception as e:
        print(f"    WARNING: Failed zeroshot dataset: {e}")

    # 2. FinGPT/fingpt-sentiment-train
    try:
        print("  - Fetching 'FinGPT/fingpt-sentiment-train'...")
        ds = load_dataset("FinGPT/fingpt-sentiment-train", split="train")
        df_fingpt = ds.to_pandas()
        for idx, row in df_fingpt.iterrows():
            instruction = str(row.get("instruction", "")).strip()
            inp = str(row.get("input", "")).strip()
            text = f"{instruction} {inp}".strip() if instruction and inp else (inp or instruction)
            output = str(row.get("output", "")).strip().lower()
            if not text or not output:
                continue

            if "positive" in output or "bullish" in output:
                norm_label = 0
            elif "negative" in output or "bearish" in output:
                norm_label = 1
            elif "neutral" in output:
                norm_label = 2
            else:
                continue

            records.append({
                "sample_id": f"sentiment_fingpt_{idx}",
                "tweet": SENTIMENT_PROMPT_TEMPLATE.format(text=text),
                "class": norm_label,
                "target": None,
                "kind": "choice",
                "source": "FinGPT/fingpt-sentiment-train"
            })
        print(f"    Loaded {len(df_fingpt)} records from FinGPT.")
    except Exception as e:
        print(f"    WARNING: Failed FinGPT dataset: {e}")

    # 3. Jean-Baptiste/financial_news_sentiment
    try:
        print("  - Fetching 'Jean-Baptiste/financial_news_sentiment'...")
        ds = load_dataset("Jean-Baptiste/financial_news_sentiment", split="train")
        df_jb = ds.to_pandas()
        for idx, row in df_jb.iterrows():
            text = str(row.get("text", row.get("sentence", ""))).strip()
            raw_label = row.get("label", row.get("labels", None))
            if not text or raw_label is None:
                continue

            if isinstance(raw_label, str):
                raw_label_str = raw_label.lower()
                if "positive" in raw_label_str or "bullish" in raw_label_str:
                    norm_label = 0
                elif "negative" in raw_label_str or "bearish" in raw_label_str:
                    norm_label = 1
                elif "neutral" in raw_label_str:
                    norm_label = 2
                else:
                    continue
            else:
                norm_label = int(raw_label) % 3

            records.append({
                "sample_id": f"sentiment_jb_{idx}",
                "tweet": SENTIMENT_PROMPT_TEMPLATE.format(text=text),
                "class": norm_label,
                "target": None,
                "kind": "choice",
                "source": "Jean-Baptiste/financial_news_sentiment"
            })
        print(f"    Loaded {len(df_jb)} records from Jean-Baptiste.")
    except Exception as e:
        print(f"    WARNING: Failed Jean-Baptiste dataset: {e}")

    df_sentiment = pd.DataFrame(records)
    if not df_sentiment.empty:
        df_sentiment = df_sentiment.drop_duplicates(subset=["tweet"]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        out_path = DATA_DIR / "curated_sentiment_dataset.parquet"
        df_sentiment.to_parquet(out_path, index=False)
        print(f"[curate_specialty] Saved {len(df_sentiment)} normalized sentiment records to '{out_path}'.")
        print("  - Class Distribution:", df_sentiment["class"].value_counts().to_dict())

    return df_sentiment

def extract_text_and_label_from_conversation(convs):
    """Helper to parse conversational turns for tool calling / user queries."""
    if not isinstance(convs, (list, np.ndarray)):
        return "", 2
    
    text = ""
    has_tool_call = False
    has_error = False

    for turn in convs:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from", turn.get("role", ""))).lower()
        val = str(turn.get("value", turn.get("content", "")))

        if role in ["human", "user"] and not text:
            text = val
        if "function_call" in role or "tool_call" in role or "tool_call" in val or "function_call" in val:
            has_tool_call = True
        if "error" in val.lower() or "rejected" in val.lower():
            has_error = True

    if has_tool_call:
        label = 0
    elif has_error:
        label = 1
    else:
        label = 2

    return text.strip(), label

def curate_agent_datasets() -> pd.DataFrame:
    print("[curate_specialty] Processing Agent & Tool Decision Datasets...")
    records = []

    # 1. Team-ACE/ToolACE
    try:
        print("  - Fetching 'Team-ACE/ToolACE'...")
        ds = load_dataset("Team-ACE/ToolACE", split="train")
        df_ace = ds.to_pandas()
        for idx, row in df_ace.iterrows():
            convs = row.get("conversations", row.get("messages", []))
            text, norm_label = extract_text_and_label_from_conversation(convs)
            if not text:
                # Fallback to string representation of row
                text = str(row.get("query", row.get("instruction", ""))).strip()
                norm_label = 0 if "tool" in str(row).lower() else 2
            if not text:
                continue

            records.append({
                "sample_id": f"agent_toolace_{idx}",
                "tweet": AGENT_PROMPT_TEMPLATE.format(text=text[:1000]),
                "class": norm_label,
                "target": None,
                "kind": "choice",
                "source": "Team-ACE/ToolACE"
            })
        print(f"    Loaded {len(df_ace)} records from ToolACE.")
    except Exception as e:
        print(f"    WARNING: Failed ToolACE dataset: {e}")

    # 2. lockon/glaive_toolcall_en
    try:
        print("  - Fetching 'lockon/glaive_toolcall_en'...")
        ds = load_dataset("lockon/glaive_toolcall_en", split="train")
        df_glaive = ds.to_pandas()
        for idx, row in df_glaive.iterrows():
            convs = row.get("conversations", [])
            text, norm_label = extract_text_and_label_from_conversation(convs)
            if not text:
                continue

            records.append({
                "sample_id": f"agent_glaive_{idx}",
                "tweet": AGENT_PROMPT_TEMPLATE.format(text=text[:1000]),
                "class": norm_label,
                "target": None,
                "kind": "choice",
                "source": "lockon/glaive_toolcall_en"
            })
        print(f"    Loaded {len(df_glaive)} records from glaive_toolcall_en.")
    except Exception as e:
        print(f"    WARNING: Failed glaive_toolcall_en dataset: {e}")

    df_agent = pd.DataFrame(records)
    if not df_agent.empty:
        df_agent = df_agent.drop_duplicates(subset=["tweet"]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        out_path = DATA_DIR / "curated_agent_dataset.parquet"
        df_agent.to_parquet(out_path, index=False)
        print(f"[curate_specialty] Saved {len(df_agent)} normalized agent records to '{out_path}'.")
        print("  - Class Distribution:", df_agent["class"].value_counts().to_dict())
    else:
        print("[curate_specialty] ERROR: No agent records were generated!")

    return df_agent

if __name__ == "__main__":
    curate_sentiment_datasets()
    curate_agent_datasets()
