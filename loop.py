"""
Graph-Tracked Training Loop Orchestrator for Laya Model.
Supports Task-Isolated Domain Pipelines (distill, bio, sentiment, agent),
Distillation Foundation Model Seeding, Domain Checkpoint Isolation,
and Dynamic Adaptive Batch/Threshold Scaling.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Any, Set

from hf_search import search_datasets_cli
from llm_controller import get_next_loop_decision
from train_runner import run_training_cycle

DEFAULT_TARGET_CYCLES = 25
DEFAULT_MAX_STAGNATION = 5
DEFAULT_MICRO_BATCH_SIZE = 1000

DOMAIN_TAXONOMY: Dict[str, List[str]] = {
    "distill": [
        "SargeDev/jev-distill-corpus-v3",
        "tasksource/tasksource-jev-typed-decisions",
        "ZefanCai/Open-Jev",
    ],
    "bio": [
        "dnagpt/laya-bio",
        "camel-ai/biology",
        "just-dna-seq/annotators",
    ],
    "sentiment": [
        "zeroshot/twitter-financial-news-sentiment",
        "FinGPT/fingpt-sentiment-train",
        "Jean-Baptiste/financial_news_sentiment",
    ],
    "agent": [
        "MaziyarPanahi/AgentToolDecisions-180K",
        "Team-ACE/ToolACE",
        "lockon/glaive_toolcall_en",
    ],
}

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
    domain: str = "distill",
    max_cycles: int = DEFAULT_TARGET_CYCLES,
    max_stagnation: int = DEFAULT_MAX_STAGNATION,
    micro_batch_size: int = DEFAULT_MICRO_BATCH_SIZE,
    output_dir_path: str = "output",
):
    domain = domain.lower()
    if domain not in DOMAIN_TAXONOMY:
        print(f"[loop] Warning: Unknown domain '{domain}'. Defaulting to 'distill'.")
        domain = "distill"

    output_dir = Path(output_dir_path) / domain
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    ledger_path = checkpoints_dir / f"seen_samples_{domain}.json"
    seen_ids = load_seen_ledger(ledger_path)
    cycle_history: List[Dict[str, Any]] = []
    available_datasets = list(DOMAIN_TAXONOMY[domain])

    best_macro_f1 = -1.0
    best_bench_f1 = -1.0
    best_checkpoint_path = checkpoints_dir / f"best_{domain}_head.safetensors"
    distill_checkpoint_path = checkpoints_dir / "best_distill_head.safetensors"

    # Stage 2 Domain Specialty Seeding: Load Distillation Base Model if domain checkpoint doesn't exist yet
    if domain != "distill" and not best_checkpoint_path.exists() and distill_checkpoint_path.exists():
        try:
            shutil.copy2(distill_checkpoint_path, best_checkpoint_path)
            print(f"[loop] Domain '{domain}' initial baseline seeded from promoted distillation base model: '{distill_checkpoint_path}'")
        except Exception as e:
            print(f"[loop] Warning: Could not seed from distillation base model: {e}")

    consecutive_stagnation = 0

    initial_base_holdout_f1 = 0.0
    initial_base_bench_f1 = 0.0

    # Restore session history for this domain if present
    summary_path = output_dir / "loop_summary.json"
    if summary_path.exists() and best_checkpoint_path.exists():
        try:
            past_history = json.loads(summary_path.read_text())
            if isinstance(past_history, list) and len(past_history) > 0:
                cycle_history = past_history
                initial_base_holdout_f1 = past_history[0].get("base_macro_f1", 0.0)
                initial_base_bench_f1 = past_history[0].get("base_bench_f1", 0.0)
                for rec in past_history:
                    if rec.get("status") == "ACCEPTED":
                        best_macro_f1 = max(best_macro_f1, rec.get("tuned_macro_f1", -1.0))
                        best_bench_f1 = max(best_bench_f1, rec.get("fixed_bench_f1", -1.0))
                print(f"[loop] Resuming domain '{domain}': loaded {len(past_history)} prior cycle records.")
                print(f"[loop] Restored best thresholds: Holdout F1 = {best_macro_f1:.4f} | Fixed Bench F1 = {best_bench_f1:.4f}")
        except Exception as e:
            print(f"[loop] Warning: Could not restore previous loop summary: {e}")

    # Success streak adaptive scaling: Expand iterations & stagnation tolerance if streak is strong
    accepted_count = sum(1 for r in cycle_history if r.get("status") == "ACCEPTED")
    if accepted_count >= 15 and len(cycle_history) >= 20:
        max_cycles = max(max_cycles, 35)
        max_stagnation = max(max_stagnation, 7)
        print(f"[loop] High Acceptance Streak Detected! Auto-scaling target cycles -> {max_cycles}, max stagnation -> {max_stagnation}")

    start_cycle_index = len(cycle_history) + 1

    print("=" * 70)
    print(f"STARTING GRAPH-TRACKED TRAINING LOOP | DOMAIN: '{domain.upper()}'")
    print(f"Max Cycles For This Run: {max_cycles} | Max Stagnation: {max_stagnation} | Micro-Batch Size: {micro_batch_size}")
    print(f"Candidate Datasets for {domain.upper()}: {available_datasets}")
    print(f"Sample Ledger: {len(seen_ids)} previously learned row IDs loaded from {ledger_path}")
    if best_checkpoint_path.exists():
        print(f"Promoted Checkpoint: Found '{best_checkpoint_path}' (Loaded as starting baseline)")
    print("=" * 70)

    for cycle_offset in range(max_cycles):
        cycle = start_cycle_index + cycle_offset
        print(f"\n>>> CYCLE #{cycle} (Pass {cycle_offset + 1}/{max_cycles}) | Domain: '{domain}' | Stagnation: {consecutive_stagnation}/{max_stagnation} | Ledger: {len(seen_ids)} rows")

        # Query Local LLM Controller for next action decision
        decision = get_next_loop_decision(
            cycle=cycle,
            history=cycle_history,
            best_macro_f1=max(0.0, best_macro_f1),
            available_datasets=available_datasets,
            domain=domain,
        )

        action = decision.get("action", "train")

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
            dataset_name = decision.get("dataset_name")
            if not dataset_name or dataset_name not in available_datasets:
                dataset_name = available_datasets[(cycle - 1) % len(available_datasets)]

        hyperparams = {
            "lr": decision.get("lr", 3e-5),
            "batch_size": decision.get("batch_size", 16),
            "epochs": decision.get("epochs", 6),
        }

        print(f"[loop] Cycle #{cycle} Plan -> Domain: '{domain}', Dataset: '{dataset_name}', Micro-Batch: {micro_batch_size}, Hyperparams: {hyperparams}")

        # Execute micro-batch training pass
        try:
            metrics, cycle_ckpt_path, used_sample_ids = run_training_cycle(
                cycle_id=cycle,
                dataset_name=dataset_name,
                hyperparams=hyperparams,
                output_dir=output_dir,
                micro_batch_size=micro_batch_size,
                seen_ids=seen_ids,
                domain=domain,
            )
        except Exception as e:
            print(f"[loop] ERROR: Cycle #{cycle} micro-batch pass failed: {e}")
            cycle_history.append({
                "cycle": cycle,
                "domain": domain,
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

        if initial_base_holdout_f1 == 0.0:
            initial_base_holdout_f1 = metrics.get("base_macro_f1", 0.0)
            initial_base_bench_f1 = metrics.get("fixed_benchmark", {}).get("fixed_bench_base_f1", 0.0)

        tuned_f1 = metrics["tuned_macro_f1"]
        bench_f1 = metrics["fixed_benchmark"]["fixed_bench_tuned_f1"]
        delta_holdout_f1 = tuned_f1 - best_macro_f1 if best_macro_f1 > 0 else tuned_f1
        delta_bench_f1 = bench_f1 - best_bench_f1 if best_bench_f1 > 0 else bench_f1

        # Checkpoint Promotion Gate
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

            # Promote model weights to domain checkpoint
            shutil.copy2(cycle_ckpt_path, best_checkpoint_path)
            # Maintain best_head.safetensors pointer
            shutil.copy2(cycle_ckpt_path, checkpoints_dir / "best_head.safetensors")

            # Record learned sample IDs into domain ledger
            seen_ids.update(used_sample_ids)
            save_seen_ledger(ledger_path, seen_ids)

            print(f"✅ [NET POSITIVE] Cycle #{cycle} (Domain '{domain}') Accepted!")
            print(f"   Holdout F1: {old_best_holdout:.4f} -> {tuned_f1:.4f} ({delta_holdout_f1:+.4f})")
            print(f"   Fixed Benchmark F1: {old_best_bench:.4f} -> {bench_f1:.4f} ({delta_bench_f1:+.4f})")
            print(f"   Sample Ledger: Recorded {len(used_sample_ids)} new learned rows (Total: {len(seen_ids)})")
            print(f"   Promoted Checkpoint: Saved to '{best_checkpoint_path}'")
        else:
            status = "REJECTED"
            consecutive_stagnation += 1
            if best_checkpoint_path.exists():
                shutil.copy2(best_checkpoint_path, cycle_ckpt_path)
            print(f"❌ [NET NEGATIVE/NEUTRAL] Cycle #{cycle} (Domain '{domain}') Rejected.")
            print(f"   Holdout F1: {tuned_f1:.4f} (Best: {best_macro_f1:.4f}) | Benchmark F1: {bench_f1:.4f} (Best: {best_bench_f1:.4f})")
            print(f"   Rolled back weights to best domain checkpoint. {len(used_sample_ids)} row IDs remain unlearned.")

        record = {
            "cycle": cycle,
            "domain": domain,
            "dataset_name": dataset_name,
            "hyperparams": hyperparams,
            "micro_batch_size": len(used_sample_ids),
            "base_macro_f1": metrics.get("base_macro_f1", 0.0),
            "base_bench_f1": metrics.get("fixed_benchmark", {}).get("fixed_bench_base_f1", 0.0),
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

        # Check early exit condition inside the loop
        if consecutive_stagnation >= max_stagnation:
            print(f"\n[loop] EXIT CONDITION MET: Stagnated over {consecutive_stagnation}/{max_stagnation} consecutive cycles.")
            break

    exit_reason = "TARGET_REACHED"
    if consecutive_stagnation >= max_stagnation:
        exit_reason = "STAGNATED_EARLY"

    accepted_cycles = sum(1 for r in cycle_history if r.get("status") == "ACCEPTED")

    run_summary = {
        "domain": domain,
        "exit_reason": exit_reason,
        "completed_cycles": len(cycle_history),
        "accepted_cycles": accepted_cycles,
        "initial_base_holdout_f1": initial_base_holdout_f1,
        "initial_base_bench_f1": initial_base_bench_f1,
        "best_macro_f1": best_macro_f1,
        "best_bench_f1": best_bench_f1,
        "holdout_f1_delta": best_macro_f1 - initial_base_holdout_f1 if initial_base_holdout_f1 > 0 else 0.0,
        "bench_f1_delta": best_bench_f1 - initial_base_bench_f1 if initial_base_bench_f1 > 0 else 0.0,
        "stagnation_count": consecutive_stagnation,
        "micro_batch_size": micro_batch_size,
        "checkpoint_path": str(best_checkpoint_path),
        "cycle_history": cycle_history,
    }

    print("\n" + "=" * 70)
    print(f"GRAPH LOOP COMPLETED | DOMAIN: '{domain.upper()}' | EXIT REASON: '{exit_reason}'")
    print(f"Total Cycles Run: {len(cycle_history)} | Accepted: {accepted_cycles}")
    print(f"Initial Base Model Metrics  -> Holdout F1: {initial_base_holdout_f1:.4f} | Fixed Bench F1: {initial_base_bench_f1:.4f}")
    print(f"Final Best Promoted Metrics -> Holdout F1: {best_macro_f1:.4f} | Fixed Bench F1: {best_bench_f1:.4f}")
    print(f"Net Gain Over Base Model    -> Holdout Δ: {best_macro_f1 - initial_base_holdout_f1:+.4f} | Fixed Bench Δ: {best_bench_f1 - initial_base_bench_f1:+.4f}")
    print(f"Total Learned Row IDs in Ledger: {len(seen_ids)}")
    print(f"Best Domain Checkpoint: {best_checkpoint_path}")
    print("=" * 70)
    return run_summary
    return run_summary

def run_sequential_pipeline(
    max_cycles: int = DEFAULT_TARGET_CYCLES,
    micro_batch_size: int = DEFAULT_MICRO_BATCH_SIZE,
):
    """Executes full sequential pipeline: Distillation -> Bio -> Sentiment -> Agent."""
    pipeline = ["distill", "bio", "sentiment", "agent"]
    print("=" * 70)
    print("LAUNCHING SEQUENTIAL TASK-ISOLATED TRAINING PIPELINE")
    print(f"Pipeline Order: {' -> '.join(pipeline)}")
    print("=" * 70)

    for dom in pipeline:
        print(f"\n>>> PIPELINE STAGE: Training Domain '{dom.upper()}'...")
        run_graph_loop(domain=dom, max_cycles=max_cycles, micro_batch_size=micro_batch_size)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in DOMAIN_TAXONOMY:
        dom = sys.argv[1]
        max_c = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TARGET_CYCLES
        mb_s = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_MICRO_BATCH_SIZE
        run_graph_loop(domain=dom, max_cycles=max_c, micro_batch_size=mb_s)
    elif len(sys.argv) > 1 and sys.argv[1] in ("pipeline", "sequential", "--pipeline"):
        max_c = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TARGET_CYCLES
        mb_s = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_MICRO_BATCH_SIZE
        run_sequential_pipeline(max_cycles=max_c, micro_batch_size=mb_s)
    else:
        max_c = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else DEFAULT_TARGET_CYCLES
        mb_s = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else DEFAULT_MICRO_BATCH_SIZE
        run_graph_loop(domain="distill", max_cycles=max_c, micro_batch_size=mb_s)
