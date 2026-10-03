"""
Meta-Loop Governor & Outer Auto-Restarter for Graph-Tracked Training Loop.
Manages outer loop passes, dynamic micro-batch scaling (1,500 -> 5,000 ceiling),
cardinal exit limits, and sequential domain pipelines.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from loop import run_graph_loop, DOMAIN_TAXONOMY, DEFAULT_TARGET_CYCLES, DEFAULT_MAX_STAGNATION, DEFAULT_MICRO_BATCH_SIZE

MAX_MICRO_BATCH_CEILING = 5000
MAX_META_PASSES = 5

def run_meta_loop(
    domain: str = "distill",
    initial_micro_batch: int = DEFAULT_MICRO_BATCH_SIZE,
    initial_max_cycles: int = DEFAULT_TARGET_CYCLES,
    max_meta_passes: int = MAX_META_PASSES,
    output_dir_path: str = "output",
) -> Dict[str, Any]:
    domain = domain.lower()
    meta_output_dir = Path(output_dir_path) / domain
    meta_output_dir.mkdir(parents=True, exist_ok=True)
    meta_summary_path = meta_output_dir / "meta_summary.json"

    meta_history: List[Dict[str, Any]] = []
    if meta_summary_path.exists():
        try:
            data = json.loads(meta_summary_path.read_text())
            if isinstance(data, list):
                meta_history = data
        except Exception:
            meta_history = []

    current_micro_batch = initial_micro_batch
    current_max_cycles = initial_max_cycles
    current_max_stagnation = DEFAULT_MAX_STAGNATION

    # Restore parameter evolution state if prior meta-pass history exists
    if meta_history:
        last_pass = meta_history[-1]
        current_micro_batch = last_pass.get("next_micro_batch", initial_micro_batch)
        current_max_cycles = last_pass.get("next_max_cycles", initial_max_cycles)
        current_max_stagnation = last_pass.get("next_max_stagnation", DEFAULT_MAX_STAGNATION)

    consecutive_ceiling_stagnations = 0

    print("=" * 80)
    print(f"LAUNCHING META-LOOP GOVERNOR | DOMAIN: '{domain.upper()}'")
    print(f"Starting Micro-Batch: {current_micro_batch} | Ceiling: {MAX_MICRO_BATCH_CEILING}")
    print(f"Starting Target Cycles: {current_max_cycles} | Max Meta-Passes: {max_meta_passes}")
    print(f"Previous Meta-Passes Completed: {len(meta_history)}")
    print("=" * 80)

    start_pass_idx = len(meta_history) + 1

    for pass_offset in range(max_meta_passes):
        meta_pass = start_pass_idx + pass_offset
        print(f"\n================================================================================")
        print(f"META-PASS #{meta_pass} (Pass {pass_offset + 1}/{max_meta_passes}) | DOMAIN: '{domain.upper()}' | Micro-Batch: {current_micro_batch} | Max Cycles: {current_max_cycles}")
        print(f"================================================================================\n")

        run_summary = run_graph_loop(
            domain=domain,
            max_cycles=current_max_cycles,
            max_stagnation=current_max_stagnation,
            micro_batch_size=current_micro_batch,
            output_dir_path=output_dir_path,
        )

        exit_reason = run_summary.get("exit_reason", "UNKNOWN")
        best_holdout_f1 = run_summary.get("best_macro_f1", 0.0)
        best_bench_f1 = run_summary.get("best_bench_f1", 0.0)
        completed_cycles = run_summary.get("completed_cycles", 0)
        accepted_cycles = run_summary.get("accepted_cycles", 0)

        # Parameter Evolution Rules for Next Meta-Pass
        next_micro_batch = current_micro_batch
        next_max_cycles = current_max_cycles
        next_max_stagnation = current_max_stagnation
        pass_action = "CONTINUE"

        if exit_reason == "STAGNATED_EARLY":
            if current_micro_batch < MAX_MICRO_BATCH_CEILING:
                next_micro_batch = min(MAX_MICRO_BATCH_CEILING, current_micro_batch + 1000)
                print(f"[meta_governor] Early Stagnation in Meta-Pass #{meta_pass}. Auto-scaling micro-batch: {current_micro_batch} -> {next_micro_batch}")
            else:
                consecutive_ceiling_stagnations += 1
                print(f"[meta_governor] Stagnated at micro-batch ceiling ({MAX_MICRO_BATCH_CEILING} samples). Ceiling Stagnation Count: {consecutive_ceiling_stagnations}/2")
                if consecutive_ceiling_stagnations >= 2:
                    print(f"[meta_governor] CARDINAL EXIT: Reached micro-batch ceiling ({MAX_MICRO_BATCH_CEILING}) and stagnated twice. Finalizing domain '{domain}'.")
                    pass_action = "TERMINATE_CEILING_STAGNATED"

        elif exit_reason == "TARGET_REACHED":
            consecutive_ceiling_stagnations = 0
            if accepted_cycles >= 15:
                next_max_cycles = min(50, current_max_cycles + 10)
                next_max_stagnation = min(7, current_max_stagnation + 2)
                print(f"[meta_governor] High Acceptance Streak! Expanding target cycles: {current_max_cycles} -> {next_max_cycles}")

        elif exit_reason == "DATASET_EXHAUSTED":
            print(f"[meta_governor] CARDINAL EXIT: Dataset pool for domain '{domain}' is fully exhausted.")
            pass_action = "TERMINATE_DATASET_EXHAUSTED"

        meta_record = {
            "meta_pass": meta_pass,
            "domain": domain,
            "micro_batch_size": current_micro_batch,
            "max_cycles": current_max_cycles,
            "max_stagnation": current_max_stagnation,
            "completed_cycles": completed_cycles,
            "accepted_cycles": accepted_cycles,
            "best_macro_f1": best_holdout_f1,
            "best_bench_f1": best_bench_f1,
            "exit_reason": exit_reason,
            "pass_action": pass_action,
            "next_micro_batch": next_micro_batch,
            "next_max_cycles": next_max_cycles,
            "next_max_stagnation": next_max_stagnation,
        }
        meta_history.append(meta_record)
        meta_summary_path.write_text(json.dumps(meta_history, indent=2))

        if pass_action.startswith("TERMINATE"):
            break

        current_micro_batch = next_micro_batch
        current_max_cycles = next_max_cycles
        current_max_stagnation = next_max_stagnation

    print("\n" + "=" * 80)
    print(f"META-LOOP GOVERNOR FINALIZED | DOMAIN: '{domain.upper()}'")
    print(f"Total Meta-Passes Executed: {len(meta_history)}")
    print(f"Final Best Holdout F1: {best_holdout_f1:.4f} | Best Bench F1: {best_bench_f1:.4f}")
    print("=" * 80)
    return {"meta_history": meta_history, "final_micro_batch": current_micro_batch}

def run_sequential_meta_pipeline(
    initial_micro_batch: int = DEFAULT_MICRO_BATCH_SIZE,
    initial_max_cycles: int = DEFAULT_TARGET_CYCLES,
    max_meta_passes: int = MAX_META_PASSES,
):
    """Executes Meta-Loop Governor sequentially across all task domains."""
    pipeline = ["distill", "bio", "sentiment", "agent"]
    print("=" * 80)
    print("LAUNCHING SEQUENTIAL META-PIPELINE GOVERNOR")
    print(f"Domain Order: {' -> '.join(pipeline)}")
    print("=" * 80)

    for dom in pipeline:
        print(f"\n>>> PIPELINE STAGE: Starting Meta-Governor for Domain '{dom.upper()}'...")
        run_meta_loop(
            domain=dom,
            initial_micro_batch=initial_micro_batch,
            initial_max_cycles=initial_max_cycles,
            max_meta_passes=max_meta_passes,
        )

if __name__ == "__main__":
    dom = sys.argv[1] if len(sys.argv) > 1 else "distill"
    mb_s = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else DEFAULT_MICRO_BATCH_SIZE
    max_c = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else DEFAULT_TARGET_CYCLES

    if dom in ("pipeline", "sequential", "--pipeline"):
        run_sequential_meta_pipeline(initial_micro_batch=mb_s, initial_max_cycles=max_c)
    else:
        run_meta_loop(domain=dom, initial_micro_batch=mb_s, initial_max_cycles=max_c)
