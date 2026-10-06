"""
Main entry point for Graph-Tracked Training Loop & Meta-Loop Governor.
Supports single domain meta-loops, sequential task pipeline meta-loops, custom cycle counts, and micro-batch sizing.

Usage:
    uv run python main.py --domain distill --max-cycles 25 --micro-batch 1500
    uv run python main.py --pipeline --max-cycles 25 --micro-batch 1500
    uv run python main.py [domain_or_cycles] [micro_batch_size]
"""

import sys
import argparse
from meta_loop import run_meta_loop, run_sequential_meta_pipeline, MAX_META_PASSES
from loop import DEFAULT_TARGET_CYCLES, DEFAULT_MICRO_BATCH_SIZE, DOMAIN_TAXONOMY

def main():
    parser = argparse.ArgumentParser(description="Graph-Tracked Training Loop & Meta-Governor for Laya Model.")
    parser.add_argument("--domain", type=str, default="distill", choices=list(DOMAIN_TAXONOMY.keys()), help="Target domain taxonomy to fine-tune.")
    parser.add_argument("--pipeline", action="store_true", help="Execute full sequential meta-pipeline across all domains (distill -> sentiment -> agent).")
    parser.add_argument("--max-cycles", type=int, default=DEFAULT_TARGET_CYCLES, help="Target max cycles per inner pass.")
    parser.add_argument("--micro-batch", type=int, default=DEFAULT_MICRO_BATCH_SIZE, help="Initial micro-batch size in samples per cycle.")
    parser.add_argument("--max-meta-passes", type=int, default=MAX_META_PASSES, help="Max meta-governor restarts per domain.")
    parser.add_argument("--llm-endpoint", type=str, default=None, help="Custom local LLM OpenAI-compatible endpoint URL (e.g. http://10.209.1.218:8080/v1/chat/completions).")
    parser.add_argument("--llm-timeout", type=int, default=300, help="Read timeout in seconds for local LLM requests (default: 300s).")
    parser.add_argument("--llm-model", type=str, default=None, help="Custom model name for local LLM payload.")
    parser.add_argument("positional_args", nargs="*", help="Optional positional args [domain/cycles] [micro_batch].")

    args = parser.parse_args()

    import os
    if args.llm_endpoint:
        os.environ["LLM_ENDPOINT"] = args.llm_endpoint
    if args.llm_timeout:
        os.environ["LLM_TIMEOUT"] = str(args.llm_timeout)
    if args.llm_model:
        os.environ["LLM_MODEL"] = args.llm_model

    domain = args.domain
    max_cycles = args.max_cycles
    micro_batch = args.micro_batch
    max_meta_passes = args.max_meta_passes

    # Handle positional backwards compatibility
    if args.positional_args:
        pos0 = args.positional_args[0]
        if pos0 in DOMAIN_TAXONOMY:
            domain = pos0
        elif pos0 in ("pipeline", "sequential"):
            args.pipeline = True
        elif pos0.isdigit():
            max_cycles = int(pos0)

        if len(args.positional_args) > 1 and args.positional_args[1].isdigit():
            micro_batch = int(args.positional_args[1])

    if args.pipeline:
        print(f"Initializing Sequential Task Meta-Pipeline across all domains...")
        print(f"  Target Max Cycles: {max_cycles} per stage")
        print(f"  Initial Micro-Batch Size: {micro_batch} samples/cycle")
        print(f"  Max Meta-Pass Restarts: {max_meta_passes}")
        run_sequential_meta_pipeline(
            initial_micro_batch=micro_batch,
            initial_max_cycles=max_cycles,
            max_meta_passes=max_meta_passes,
        )
    else:
        print(f"Initializing Meta-Loop Governor for Domain '{domain.upper()}'...")
        print(f"  Target Max Cycles: {max_cycles}")
        print(f"  Initial Micro-Batch Size: {micro_batch} samples/cycle")
        print(f"  Max Meta-Pass Restarts: {max_meta_passes}")
        run_meta_loop(
            domain=domain,
            initial_micro_batch=micro_batch,
            initial_max_cycles=max_cycles,
            max_meta_passes=max_meta_passes,
        )

if __name__ == "__main__":
    main()
