"""
Graph-Tracked Training Loop Orchestrator for Laya Model.
Executes iterative micro-batch training passes, maintains non-overlapping sample ledger,
queries local LLM controller, evaluates Gold Benchmark + Fixed Prompt Suite,
and manages checkpoint promotion and rollback.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Any, Set

from hf_search import search_datasets_cli
from llm_controller import get_next_loop_decision
from train_runner import run_training_cycle

MAX_TOTAL_CYCLES = 20
MAX_STAGNATION_CYCLES = 5
DEFAULT_MICRO_BATCH_SIZE = 300

DEFAULT_DATASETS = [
    "tdavidson/hate_speech_offensive",
    "dnagpt/laya-bio",
    "SargeDev/jev-distill-corpus-v3",
]

def load_seen_ledger(ledger_path: Path) -> Set[str]:
    if ledger_path.exists():
        try:
            data = json.loads(ledger_path.read_text())
            return set(data.get("seen_sample_ids", []))
        except Exception:
            return set()
    return set()

def save_seen_ledger(ledger_path: Path, seen_ids: Set[str]):
    ledger_path.write_text(json.dumps({"seen_sample_ids": list(seen_ids), "count": len(seen_ids)}, indent=2))

def run_graph_loop(
    max_cycles: int = MAX_TOTAL_CYCLES,
    max_stagnation: int = MAX_STAGNATION_CYCLES,
    micro_batch_size: int = DEFAULT_MICRO_BATCH_SIZE,
    output_dir_path: str = "output",
):
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = checkpoints_dir / "seen_samples.json"

    seen_ids = load_seen_ledger(ledger_path)
    cycle_history: List[Dict[str, Any]] = []
    available_datasets = list(DEFAULT_DATASETS)

    best_macro_f1 = -1.0
    best_bench_f1 = -1.0
    best_checkpoint_path = checkpoints_dir / "best_head.safetensors"
    consecutive_stagnation = 0

    summary_path = output_dir / "loop_summary.json"
    if summary_path.exists() and best_checkpoint_path.exists():
        try:
            past_history = json.loads(summary_path.read_text())
            if isinstance(past_history, list) and len(past_history) > 0:
                cycle_history = past_history
                for rec in past_history:
                    if rec.get("status") == "ACCEPTED":
                        best_macro_f1 = max(best_macro_f1, rec.get("tuned_macro_f1", -1.0))
                        best_bench_f1 = max(best_bench_f1, rec.get("fixed_bench_f1", -1.0))
                print(f"[loop] Resuming session: loaded {len(past_history)} prior cycle records from {summary_path}.")
                print(f"[loop] Restored all-time best thresholds: Holdout F1 = {best_macro_f1:.4f} | Fixed Bench F1 = {best_bench_f1:.4f}")
        except Exception as e:
            print(f"[loop] Warning: Could not restore previous loop summary: {e}")

    start_cycle_index = len(cycle_history) + 1

    print("=" * 70)
    print("STARTING GRAPH-TRACKED TRAINING LOOP FOR LAYA")
    print(f"Max Cycles For This Run: {max_cycles} | Max Stagnation: {max_stagnation} | Micro-Batch Size: {micro_batch_size}")
    print(f"Initial Candidate Datasets: {available_datasets}")
    print(f"Sample Ledger: {len(seen_ids)} previously learned row IDs loaded from {ledger_path}")
    if best_checkpoint_path.exists():
        print(f"Promoted Checkpoint: Found '{best_checkpoint_path}' (Loaded as initial baseline)")
    print("=" * 70)

    for cycle_offset in range(max_cycles):
        cycle = start_cycle_index + cycle_offset
        print(f"\n>>> CYCLE #{cycle} (Run Pass {cycle_offset + 1}/{max_cycles}) | (Stagnation Count: {consecutive_stagnation}/{max_stagnation}) | Seen Ledger: {len(seen_ids)} rows")

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
            dataset_name = decision.get("dataset_name") or (new_ids[0] if new_ids else available_datasets[0])
        else:
            dataset_name = decision.get("dataset_name", available_datasets[(cycle - 1) % len(available_datasets)])

        hyperparams = {
            "lr": decision.get("lr", 3e-5),
            "batch_size": decision.get("batch_size", 16),
            "epochs": decision.get("epochs", 6),
        }

        print(f"[loop] Cycle #{cycle} Plan -> Dataset: '{dataset_name}', Micro-Batch: {micro_batch_size}, Hyperparams: {hyperparams}")

        # Execute micro-batch training pass
        try:
            metrics, cycle_ckpt_path, used_sample_ids = run_training_cycle(
                cycle_id=cycle,
                dataset_name=dataset_name,
                hyperparams=hyperparams,
                output_dir=output_dir,
                micro_batch_size=micro_batch_size,
                seen_ids=seen_ids,
            )
        except Exception as e:
            print(f"[loop] ERROR: Cycle #{cycle} micro-batch pass failed: {e}")
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
        bench_f1 = metrics["fixed_benchmark"]["fixed_bench_tuned_f1"]
        delta_holdout_f1 = tuned_f1 - best_macro_f1 if best_macro_f1 > 0 else tuned_f1
        delta_bench_f1 = bench_f1 - best_bench_f1 if best_bench_f1 > 0 else bench_f1

        # Checkpoint Promotion & Ledger Gate
        # Promotes if tuned holdout F1 improves AND fixed benchmark performance does not degrade
        is_net_positive = (
            best_macro_f1 < 0
            or (tuned_f1 > best_macro_f1 + 1e-4 and bench_f1 >= best_bench_f1 - 1e-4)
            or (bench_f1 > best_bench_f1 + 1e-4 and tuned_f1 >= best_macro_f1 - 1e-4)
        )

        if is_net_positive:
            status = "ACCEPTED"
            old_best_holdout = best_macro_f1
            old_best_bench = best_bench_f1
            best_macro_f1 = tuned_f1
            best_bench_f1 = bench_f1
            consecutive_stagnation = 0

            # Promote model weights
            shutil.copy2(cycle_ckpt_path, best_checkpoint_path)

            # Record learned sample IDs into persistent ledger
            seen_ids.update(used_sample_ids)
            save_seen_ledger(ledger_path, seen_ids)

            print(f"✅ [NET POSITIVE] Cycle #{cycle} Accepted!")
            print(f"   Holdout F1: {old_best_holdout:.4f} -> {tuned_f1:.4f} ({delta_holdout_f1:+.4f})")
            print(f"   Fixed Benchmark F1: {old_best_bench:.4f} -> {bench_f1:.4f} ({delta_bench_f1:+.4f})")
            print(f"   Sample Ledger: Recorded {len(used_sample_ids)} new learned rows (Total: {len(seen_ids)})")
        else:
            status = "REJECTED"
            consecutive_stagnation += 1
            if best_checkpoint_path.exists():
                shutil.copy2(best_checkpoint_path, cycle_ckpt_path)
            print(f"❌ [NET NEGATIVE/NEUTRAL] Cycle #{cycle} Rejected.")
            print(f"   Holdout F1: {tuned_f1:.4f} (Best: {best_macro_f1:.4f}) | Benchmark F1: {bench_f1:.4f} (Best: {best_bench_f1:.4f})")
            print(f"   Rolled back weights to best checkpoint. {len(used_sample_ids)} row IDs remain unlearned.")

        record = {
            "cycle": cycle,
            "dataset_name": dataset_name,
            "hyperparams": hyperparams,
            "micro_batch_size": len(used_sample_ids),
            "tuned_macro_f1": tuned_f1,
            "tuned_accuracy": metrics["tuned_accuracy"],
            "fixed_bench_f1": bench_f1,
            "best_macro_f1_so_far": best_macro_f1,
            "best_bench_f1_so_far": best_bench_f1,
            "status": status,
            "stagnation_count": consecutive_stagnation,
            "total_seen_samples": len(seen_ids),
            "reasoning": decision.get("reasoning", ""),
        }
        cycle_history.append(record)
        (output_dir / "loop_summary.json").write_text(json.dumps(cycle_history, indent=2))

        # Check exit condition
        if consecutive_stagnation >= max_stagnation:
            print(f"\n[loop] EXIT CONDITION MET: Non-improvement over {max_stagnation} consecutive cycles.")
            break

    print("\n" + "=" * 70)
    print("GRAPH LOOP COMPLETED")
    print(f"Total Cycles Run: {len(cycle_history)}")
    print(f"Final Best Holdout Macro F1: {best_macro_f1:.4f}")
    print(f"Final Best Fixed Benchmark F1: {best_bench_f1:.4f}")
    print(f"Total Learned Row IDs in Ledger: {len(seen_ids)}")
    print(f"Best Checkpoint Location: {best_checkpoint_path}")
    print("=" * 70)
    return cycle_history

if __name__ == "__main__":
    max_c = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_TOTAL_CYCLES
    mb_s = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_MICRO_BATCH_SIZE
    run_graph_loop(max_cycles=max_c, micro_batch_size=mb_s)
