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

def search_datasets_api(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search HuggingFace Hub using huggingface_hub Python API.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    try:
        datasets = list(api.list_datasets(search=query, limit=limit, full=False))
        results = []
        for ds in datasets:
            results.append({
                "dataset_id": ds.id,
                "author": getattr(ds, "author", ""),
                "last_modified": str(getattr(ds, "lastModified", "")),
                "tags": getattr(ds, "tags", []),
                "downloads": getattr(ds, "downloads", 0),
                "likes": getattr(ds, "likes", 0),
            })
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
