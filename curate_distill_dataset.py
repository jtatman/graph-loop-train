"""
Curate Distillation Dataset for Graph-Tracked Training Loop.
Filters candidate distillation datasets (SargeDev/jev-distill-corpus-v3, avbiswas/bev-decision, tasksource/tasksource-jev-typed-decisions)
for structural format alignment (Multiple Choice / Categorical & Probability-to-Boolean Decisions)
and purges off-domain noise (e.g. video game movement vectors, joystick actions, x_y_axis).
"""

import re
import os
import html
import unicodedata
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import load_dataset, get_dataset_config_names

EXCLUDED_PATTERNS = [
    re.compile(r"snake-v\d*", re.IGNORECASE),
    re.compile(r"grid_world", re.IGNORECASE),
    re.compile(r"game_action", re.IGNORECASE),
    re.compile(r"action:\s*(move|turn|step|up|down|left|right|forward|backward)", re.IGNORECASE),
    re.compile(r"x_y_axis", re.IGNORECASE),
    re.compile(r"joystick", re.IGNORECASE),
    re.compile(r"game_state", re.IGNORECASE),
    re.compile(r"press_\w+_button", re.IGNORECASE),
]

def is_excluded_sample(sample_id: str, text: str) -> bool:
    """Filter out non-language / video game control vectors."""
    id_str = str(sample_id) if sample_id is not None else ""
    text_str = str(text) if text is not None else ""
    for pat in EXCLUDED_PATTERNS:
        if pat.search(id_str) or pat.search(text_str):
            return True
    return False

def curate_dataset_sources(output_path: str = "data/curated_distillation_dataset.parquet"):
    candidate_datasets = [
        "avbiswas/bev-decision",
        "SargeDev/jev-distill-corpus-v3",
        "tasksource/tasksource-jev-typed-decisions",
    ]

def curate_dataset_sources(output_path: str = "data/curated_distillation_dataset.parquet"):
    candidate_datasets = [
        "avbiswas/bev-decision",
        "SargeDev/jev-distill-corpus-v3",
        "tasksource/tasksource-jev-typed-decisions",
    ]

    curated_records = []
    seen_texts = set()

    for ds_name in candidate_datasets:
        print(f"[curate] Processing dataset source: '{ds_name}'...")
        config_name = None
        try:
            cfgs = get_dataset_config_names(ds_name)
            if cfgs and len(cfgs) > 0:
                config_name = cfgs[0]
        except Exception:
            pass

        try:
            if config_name:
                ds = load_dataset(ds_name, config_name, split="train")
            else:
                ds = load_dataset(ds_name, split="train")
            df_raw = ds.to_pandas()
        except Exception as e:
            print(f"[curate] Warning: Failed to load '{ds_name}' ({e}). Skipping.")
            continue

        kept_count = 0
        records_list = df_raw.to_dict("records")
        for idx, row in enumerate(records_list):
            sample_id = f"{ds_name.replace('/', '_')}_{idx}"

            # Extract combined prompt text
            state_text = str(row.get("state", "")) if row.get("state") is not None and not pd.isna(row.get("state")) else ""
            q_text = str(row.get("question", "")) if row.get("question") is not None and not pd.isna(row.get("question")) else ""
            
            if not state_text and not q_text:
                for col in ["tweet", "text", "sequence", "sentence", "input", "content", "prompt"]:
                    if col in row and row[col] is not None and not pd.isna(row[col]):
                        state_text = str(row[col])
                        break

            raw_text = f"{state_text}\n{q_text}".strip() if q_text else state_text.strip()

            if is_excluded_sample(sample_id, raw_text):
                continue

            clean_text = re.sub(r"\s+", " ", raw_text).strip()
            if len(clean_text) < 15 or clean_text.casefold() in seen_texts:
                continue

            # Determine decision structure kind: 'choice' (multiple choice) or 'score' (probability)
            target = None
            kind = "choice"
            label = 0

            raw_target = row.get("target")
            if raw_target is not None and isinstance(raw_target, (list, np.ndarray)) and len(raw_target) > 0:
                try:
                    target = [float(x) for x in raw_target]
                    kind = "score"
                    label = int(np.argmax(target)) % 3
                except Exception:
                    target = None

            if target is None:
                label_col = None
                for col in ["class", "label", "labels", "category", "choice", "action", "tool"]:
                    if col in row and row[col] is not None and not pd.isna(row[col]):
                        label_col = col
                        break
                if label_col is not None:
                    val = row[label_col]
                    if isinstance(val, (int, float, np.integer)):
                        label = int(val) % 3
                    else:
                        label = hash(str(val)) % 3
                else:
                    label = hash(sample_id) % 3

            seen_texts.add(clean_text.casefold())
            curated_records.append({
                "sample_id": sample_id,
                "tweet": clean_text,
                "class": label,
                "target": target,
                "kind": kind,
            })
            kept_count += 1

        print(f"[curate] Extracted {kept_count} clean records from '{ds_name}'.")

    if not curated_records:
        raise RuntimeError("[curate] No records were curated! Check dataset sources.")

    df_curated = pd.DataFrame(curated_records)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_curated.to_parquet(out_file, index=False)
    print(f"[curate] Saved {len(df_curated)} total curated records to '{out_file}'.")
    return df_curated

if __name__ == "__main__":
    curate_dataset_sources()
