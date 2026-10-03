"""
Main entry point for Graph-Tracked Training Loop.
Supports single domain runs, sequential task pipelines, custom cycle counts, and micro-batch sizing.

Usage:
    uv run python main.py --domain distill --max-cycles 25 --micro-batch 1500
    uv run python main.py --pipeline --max-cycles 25 --micro-batch 1500
    uv run python main.py [domain_or_cycles] [micro_batch_size]
"""

import sys
import argparse
from loop import run_graph_loop, run_sequential_pipeline, DEFAULT_TARGET_CYCLES, DEFAULT_MICRO_BATCH_SIZE, DOMAIN_TAXONOMY

def main():
    parser = argparse.ArgumentParser(description="Graph-Tracked Training Loop Orchestrator for Laya Model.")
    parser.add_argument("--domain", type=str, default="distill", choices=list(DOMAIN_TAXONOMY.keys()), help="Target domain taxonomy to fine-tune.")
    parser.add_argument("--pipeline", action="store_true", help="Execute full sequential pipeline across all domains (distill -> bio -> sentiment -> agent).")
    parser.add_argument("--max-cycles", type=int, default=DEFAULT_TARGET_CYCLES, help="Target max cycles for this session.")
    parser.add_argument("--micro-batch", type=int, default=DEFAULT_MICRO_BATCH_SIZE, help="Micro-batch size in samples per cycle.")
    parser.add_argument("positional_args", nargs="*", help="Optional positional args [domain/cycles] [micro_batch].")

    args = parser.parse_args()

    domain = args.domain
    max_cycles = args.max_cycles
    micro_batch = args.micro_batch

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
        print(f"Initializing Sequential Task Pipeline across all domains...")
        print(f"  Target Max Cycles: {max_cycles} per stage")
        print(f"  Micro-Batch Size: {micro_batch} samples/cycle")
        run_sequential_pipeline(max_cycles=max_cycles, micro_batch_size=micro_batch)
    else:
        print(f"Initializing Graph-Tracked Training Loop for Domain '{domain.upper()}'...")
        print(f"  Target Max Cycles: {max_cycles}")
        print(f"  Micro-Batch Size: {micro_batch} samples/cycle")
        run_graph_loop(domain=domain, max_cycles=max_cycles, micro_batch_size=micro_batch)

if __name__ == "__main__":
    main()
