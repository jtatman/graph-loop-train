"""
Local LLM Controller for Graph Loop Training.
Communicates with local OpenAI-compatible endpoint (http://10.209.1.214:8080/v1)
to decide next training hyperparameters, dataset choices, or dataset search queries.
"""

import json
import os
import re
import requests
from typing import Dict, Any, List, Optional

DEFAULT_LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://10.209.1.214:8080/v1/chat/completions")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "local-model")
DEFAULT_LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))

SYSTEM_PROMPT = """You are the AI Orchestrator for a Graph-Tracked Fine-Tuning Loop optimizing the 'convaiinnovations/laya' decision model.

Your goal: Maximize the model's macro F1 score on a gold classification benchmark.

Loop Constraints:
1. The model architecture, training software, and evaluation set are static.
2. You can vary dataset choices, learning rate (1e-5 to 1e-4), epochs (3 to 10), and search HuggingFace for new datasets.
3. Available primary datasets:
   - 'data/curated_distillation_dataset.parquet'
   - 'avbiswas/bev-decision'
   - 'SargeDev/jev-distill-corpus-v3'

IMPORTANT: Respond ONLY with a valid JSON object matching this schema. Do not output markdown or thinking tokens outside the JSON:
{
  "action": "train" | "search_hf",
  "dataset_name": "<name of dataset to train on>",
  "search_query": "<search query string if action is search_hf>",
  "lr": 3e-5,
  "epochs": 6,
  "batch_size": 64,
  "reasoning": "<short explanation of your decision>"
}
"""

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract JSON object from LLM response text, ignoring thinking tags or markdown code blocks."""
    if not text or not isinstance(text, str):
        return None

    # 1. Look for ```json ... ``` code blocks first
    if "```" in text:
        blocks = text.split("```")
        for b in blocks[1:]:
            b_clean = b.strip()
            if b_clean.startswith("json"):
                b_clean = b_clean[4:].strip()
            try:
                data = json.loads(b_clean)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

    # 2. Extract JSON object {...} via regex (strips out <think>...</think> and commentary)
    match = re.search(r"\{[^{}]*\"action\"[^{}]*\}", text, re.DOTALL)
    if not match:
        match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        raw_json = match.group(0).strip()
        try:
            data = json.loads(raw_json)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # 3. Fallback to direct json.loads on stripped text
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return None

def get_next_loop_decision(
    cycle: int,
    history: List[Dict[str, Any]],
    best_macro_f1: float,
    available_datasets: List[str],
    domain: str = "distill",
    endpoint_url: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Query local LLM endpoint for next loop action decision.
    Falls back to deterministic cycle strategy if endpoint fails.
    """
    url = endpoint_url or DEFAULT_LLM_ENDPOINT
    req_timeout = timeout if timeout is not None else DEFAULT_LLM_TIMEOUT
    model_name = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)

    # Calculate per-dataset performance & overfitting risk indicators from history
    dataset_stats: Dict[str, Dict[str, int]] = {}
    for r in history:
        ds = r.get("dataset_name")
        if ds:
            if ds not in dataset_stats:
                dataset_stats[ds] = {"accepted": 0, "rejected": 0, "consecutive_rejections": 0}
            if r.get("status") == "ACCEPTED":
                dataset_stats[ds]["accepted"] += 1
                dataset_stats[ds]["consecutive_rejections"] = 0
            elif r.get("status") == "REJECTED":
                dataset_stats[ds]["rejected"] += 1
                dataset_stats[ds]["consecutive_rejections"] += 1

    overfitting_datasets = [ds for ds, s in dataset_stats.items() if s["consecutive_rejections"] >= 3]

    system_prompt = f"""You are the AI Orchestrator for a Graph-Tracked Fine-Tuning Loop optimizing the 'convaiinnovations/laya' decision model.

Your goal: Maximize the model's performance for task domain '{domain.upper()}'.

Loop Constraints:
1. Active Domain: '{domain}'
2. Target candidate datasets allowed for domain '{domain}': {json.dumps(available_datasets)}
3. Overfitting / Stagnation Risk Datasets (3+ consecutive rejections): {json.dumps(overfitting_datasets)}
4. CRITICAL: Avoid choosing datasets in Overfitting Risk! If all candidate datasets are overfitted or stagnating, issue a "search_hf" action to find fresh task datasets.

IMPORTANT: Respond ONLY with a valid JSON object matching this schema:
{{
  "action": "train" | "search_hf",
  "dataset_name": "<must be one of {json.dumps(available_datasets)}>",
  "search_query": "<search query string if action is search_hf>",
  "lr": 3e-5,
  "epochs": 6,
  "batch_size": 64,
  "reasoning": "<short explanation of your decision>"
}}
"""

    user_prompt = f"""Cycle #: {cycle}
Active Domain: {domain}
Current Best Gold Macro F1: {best_macro_f1:.4f}
Known Available Datasets: {json.dumps(available_datasets)}
Datasets with Overfitting/Stagnation Risk (3+ consecutive rejections): {json.dumps(overfitting_datasets)}

Per-Dataset Rejection Stats:
{json.dumps(dataset_stats, indent=2)}

Recent Cycle History (last 5 passes):
{json.dumps(history[-5:], indent=2)}

Decide the next action for Cycle #{cycle}."""

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
    }

    try:
        response = requests.post(url, json=payload, timeout=req_timeout)
        if response.status_code == 200:
            data = response.json()
            raw_content = data["choices"][0]["message"]["content"].strip()
            decision = extract_json_object(raw_content)
            if decision and isinstance(decision, dict):
                print(f"[llm_controller] LLM Decision ({url}): {decision.get('reasoning')}")
                return decision
            else:
                print(f"[llm_controller] Warning: Could not extract valid JSON from LLM response ({url}). Raw: {raw_content[:150]}...")
        else:
            print(f"[llm_controller] LLM API status {response.status_code} ({url}): {response.text[:100]}")
    except Exception as e:
        print(f"[llm_controller] Local LLM endpoint '{url}' unreachable ({e}). Using heuristic fallback.")

    # Heuristic fallback strategy: Prioritize datasets with lowest consecutive rejections
    candidate_datasets = [d for d in available_datasets if d not in overfitting_datasets]
    if not candidate_datasets:
        candidate_datasets = available_datasets if available_datasets else ["SargeDev/jev-distill-corpus-v3"]

    dataset = candidate_datasets[(cycle - 1) % len(candidate_datasets)]
    lrs = [3e-5, 5e-5, 2e-5]
    lr = lrs[(cycle - 1) % len(lrs)]
    epochs = 5 + (cycle % 3)

    return {
        "action": "train",
        "dataset_name": dataset,
        "search_query": "",
        "lr": lr,
        "epochs": epochs,
        "batch_size": 64,
        "reasoning": f"Heuristic fallback for domain '{domain}': cycle {cycle} targeting dataset '{dataset}' with lr={lr}, epochs={epochs}.",
    }

if __name__ == "__main__":
    test_decision = get_next_loop_decision(
        cycle=1,
        history=[],
        best_macro_f1=0.75,
        available_datasets=["data/curated_distillation_dataset.parquet", "avbiswas/bev-decision", "SargeDev/jev-distill-corpus-v3"],
    )
    print("Decision Output:")
    print(json.dumps(test_decision, indent=2))
