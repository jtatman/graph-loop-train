"""
Local LLM Controller for Graph Loop Training.
Communicates with local OpenAI-compatible endpoint (http://10.209.1.214:8080/v1)
to decide next training hyperparameters, dataset choices, or dataset search queries.
"""

import json
import os
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
   - 'tdavidson/hate_speech_offensive' (baseline benchmark)
   - 'dnagpt/laya-bio'
   - 'SargeDev/jev-distill-corpus-v3'

You must respond ONLY with a valid JSON object matching this schema:
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

    system_prompt = f"""You are the AI Orchestrator for a Graph-Tracked Fine-Tuning Loop optimizing the 'convaiinnovations/laya' decision model.

Your goal: Maximize the model's performance for task domain '{domain.upper()}'.

Loop Constraints:
1. Active Domain: '{domain}'
2. Target candidate datasets allowed for domain '{domain}': {json.dumps(available_datasets)}
3. You can vary dataset choice (MUST be one of the allowed target candidate datasets for '{domain}'), learning rate (1e-5 to 1e-4), epochs (3 to 10), or search HuggingFace for new datasets.

You must respond ONLY with a valid JSON object matching this schema:
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
Known Available Datasets for '{domain}': {json.dumps(available_datasets)}

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
        "max_tokens": 200,
    }

    try:
        response = requests.post(url, json=payload, timeout=req_timeout)
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            # Extract JSON if surrounded by markdown code blocks
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:].strip()
            decision = json.loads(content)
            print(f"[llm_controller] LLM Decision ({url}): {decision.get('reasoning')}")
            return decision
        else:
            print(f"[llm_controller] LLM API status {response.status_code} ({url}): {response.text[:100]}")
    except Exception as e:
        print(f"[llm_controller] Local LLM endpoint '{url}' unreachable ({e}). Using heuristic fallback.")

    # Heuristic fallback strategy if local LLM server is offline/unresponsive
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
        available_datasets=["tdavidson/hate_speech_offensive", "dnagpt/laya-bio", "SargeDev/jev-distill-corpus-v3"],
    )
    print("Decision Output:")
    print(json.dumps(test_decision, indent=2))
