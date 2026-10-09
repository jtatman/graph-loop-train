"""
HuggingFace Dataset Search Helper for Graph Loop Training.
Uses local /usr/bin/hf binary and huggingface_hub Python library to discover
relevant classification datasets on HuggingFace Hub.
"""

import os
import subprocess
import json
from typing import List, Dict, Any
from huggingface_hub import HfApi

HF_CLI_PATH = "/usr/bin/hf"

def search_datasets_cli(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search HuggingFace Hub using local hf CLI binary if available.
    """
    if os.path.exists(HF_CLI_PATH):
        try:
            env = os.environ.copy()
            cmd = [HF_CLI_PATH, "datasets", "search", query, "--limit", str(limit)]
            res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=30)
            if res.returncode == 0 and res.stdout.strip():
                results = []
                for line in res.stdout.strip().splitlines():
                    if line.strip():
                        results.append({"dataset_id": line.strip(), "source": "hf_cli"})
                if results:
                    return results
        except Exception as e:
            print(f"[hf_search] CLI search notice: {e}")

    # Fallback to Python API
    return search_datasets_api(query, limit)

NON_ENGLISH_KEYWORDS = ["persian", "chinese", "arabic", "russian", "spanish", "french", "german", "japanese", "korean", "italian", "portuguese", "turkish", "vietnamese", "farsi", "hindi"]

def search_datasets_api(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search HuggingFace Hub using huggingface_hub Python API with strict English filtering.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    try:
        datasets = list(api.list_datasets(search=query, limit=limit * 3, full=False))
        results = []
        for ds in datasets:
            ds_id_lower = ds.id.lower()
            if any(lang in ds_id_lower for lang in NON_ENGLISH_KEYWORDS):
                continue
            tags = getattr(ds, "tags", []) or []
            # Skip if tagged explicitly with non-en language tag
            if any(t.startswith("language:") and not t.endswith(":en") for t in tags):
                continue

            results.append({
                "dataset_id": ds.id,
                "author": getattr(ds, "author", ""),
                "last_modified": str(getattr(ds, "lastModified", "")),
                "tags": tags,
                "downloads": getattr(ds, "downloads", 0),
                "likes": getattr(ds, "likes", 0),
            })
            if len(results) >= limit:
                break
        return results
    except Exception as e:
        print(f"[hf_search] API search error: {e}")
        return []

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "classification"
    print(f"Searching datasets for query: '{q}'...")
    res = search_datasets_cli(q, limit=5)
    print(json.dumps(res, indent=2))
