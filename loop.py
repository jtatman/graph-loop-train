"""
Graph-Tracked Training Loop Orchestrator for Laya Model.
Executes iterative training passes, queries local LLM controller, evaluates gold benchmark F1,
and manages checkpoint promotion and rollback.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Any

from hf_search import search_datasets_cli
from llm_controller import get_next_loop_decision
from train_runner import run_training_cycle

MAX_TOTAL_CYCLES = 20
MAX_STAGNATION_CYCLES = 5

DEFAULT_DATASETS = [
    "tdavidson/hate_speech_offensive",
    "dnagpt/laya-bio",
    "SargeDev/jev-distill-corpus-v3",
]

def run_graph_loop(
    max_cycles: int = MAX_TOTAL_CYCLES,
    max_stagnation: int = MAX_STAGNATION_CYCLES,
    output_dir_path: str = "output",
):
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    cycle_history: List[Dict[str, Any]] = []
    available_datasets = list(DEFAULT_DATASETS)

    best_macro_f1 = -1.0
    best_checkpoint_path = checkpoints_dir / "best_head.safetensors"
    consecutive_stagnation = 0

    print("=" * 60)
    print("STARTING GRAPH-TRACKED TRAINING LOOP FOR LAYA")
    print(f"Max Cycles: {max_cycles} | Max Stagnation: {max_stagnation}")
    print(f"Initial Candidate Datasets: {available_datasets}")
    print("=" * 60)

    for cycle in range(1, max_cycles + 1):
        print(f"\n>>> CYCLE #{cycle}/{max_cycles} (Stagnation Count: {consecutive_stagnation}/{max_stagnation})")

        # Query Local LLM Controller for next action decision
        decision = get_next_loop_decision(
            cycle=cycle,
            history=cycle_history,
            best_macro_f1=max(0.0, best_macro_f1),
            available_datasets=available_datasets,
        )

        action = decision.get("action", "train")

        # Handle HF search action if requested by LLM controller
        if action == "search_hf" and decision.get("search_query"):
            query = decision["search_query"]
            print(f"[loop] LLM requested HF dataset search for query: '{query}'...")
            search_results = search_datasets_cli(query, limit=5)
            new_ids = [r["dataset_id"] for r in search_results if r.get("dataset_id")]
            print(f"[loop] HF Search returned: {new_ids}")
            for nid in new_ids:
                if nid not in available_datasets:
                    available_datasets.append(nid)
            # Proceed to train on selected dataset
            dataset_name = decision.get("dataset_name") or (new_ids[0] if new_ids else available_datasets[0])
        else:
            dataset_name = decision.get("dataset_name", available_datasets[(cycle - 1) % len(available_datasets)])

        hyperparams = {
            "lr": decision.get("lr", 3e-5),
            "batch_size": decision.get("batch_size", 16),
            "epochs": decision.get("epochs", 6),
        }

        print(f"[loop] Cycle #{cycle} Plan -> Dataset: '{dataset_name}', Hyperparams: {hyperparams}")

        # Execute training pass
        try:
            metrics, cycle_ckpt_path = run_training_cycle(
                cycle_id=cycle,
                dataset_name=dataset_name,
                hyperparams=hyperparams,
                output_dir=output_dir,
            )
        except Exception as e:
            print(f"[loop] ERROR: Cycle #{cycle} training failed: {e}")
            cycle_history.append({
                "cycle": cycle,
                "dataset_name": dataset_name,
                "hyperparams": hyperparams,
                "status": "FAILED",
                "error": str(e),
            })
            consecutive_stagnation += 1
            if consecutive_stagnation >= max_stagnation:
                print(f"[loop] Exit condition met: {consecutive_stagnation} consecutive failures/stagnations.")
                break
            continue

        tuned_f1 = metrics["tuned_macro_f1"]
        delta_f1 = tuned_f1 - best_macro_f1 if best_macro_f1 > 0 else tuned_f1

        # Checkpoint Promotion & Rollback Decision Gate
        if best_macro_f1 < 0 or tuned_f1 > best_macro_f1 + 1e-4:
            # Net Positive Change -> Promote Checkpoint
            status = "ACCEPTED"
            old_best = best_macro_f1
            best_macro_f1 = tuned_f1
            consecutive_stagnation = 0
            shutil.copy2(cycle_ckpt_path, best_checkpoint_path)
            print(f"✅ [NET POSITIVE] Macro F1 improved from {old_best:.4f} to {tuned_f1:.4f} (+{delta_f1:.4f}). Promoted checkpoint!")
        else:
            # Net Negative / Neutral Change -> Rollback Checkpoint
            status = "REJECTED"
            consecutive_stagnation += 1
            if best_checkpoint_path.exists():
                shutil.copy2(best_checkpoint_path, cycle_ckpt_path)
                print(f"❌ [NET NEGATIVE/NEUTRAL] Macro F1 ({tuned_f1:.4f}) <= Best ({best_macro_f1:.4f}). Rolled back to best checkpoint.")

        record = {
            "cycle": cycle,
            "dataset_name": dataset_name,
            "hyperparams": hyperparams,
            "tuned_macro_f1": tuned_f1,
            "tuned_accuracy": metrics["tuned_accuracy"],
            "best_macro_f1_so_far": best_macro_f1,
            "delta_f1": delta_f1,
            "status": status,
            "stagnation_count": consecutive_stagnation,
            "reasoning": decision.get("reasoning", ""),
        }
        cycle_history.append(record)

        # Save loop summary state
        (output_dir / "loop_summary.json").write_text(json.dumps(cycle_history, indent=2))

        # Check exit condition
        if consecutive_stagnation >= max_stagnation:
            print(f"\n[loop] EXIT CONDITION MET: Non-improvement over {max_stagnation} consecutive cycles.")
            break

    print("\n" + "=" * 60)
    print("GRAPH LOOP COMPLETED")
    print(f"Total Cycles Run: {len(cycle_history)}")
    print(f"Final Best Gold Macro F1: {best_macro_f1:.4f}")
    print(f"Best Checkpoint Location: {best_checkpoint_path}")
    print("=" * 60)
    return cycle_history

if __name__ == "__main__":
    max_c = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_TOTAL_CYCLES
    run_graph_loop(max_cycles=max_c)
