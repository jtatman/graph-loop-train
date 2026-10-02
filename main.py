"""
Main entry point for Graph-Tracked Training Loop.
Usage:
    uv run python main.py [max_cycles] [micro_batch_size]
Examples:
    uv run python main.py
    uv run python main.py 10 300
    uv run python main.py 5 200
"""

import sys
from loop import run_graph_loop, MAX_TOTAL_CYCLES, DEFAULT_MICRO_BATCH_SIZE

def main():
    max_cycles = MAX_TOTAL_CYCLES
    micro_batch_size = DEFAULT_MICRO_BATCH_SIZE

    if len(sys.argv) > 1:
        try:
            max_cycles = int(sys.argv[1])
        except ValueError:
            print(f"Invalid max_cycles argument '{sys.argv[1]}'. Using default ({MAX_TOTAL_CYCLES}).")

    if len(sys.argv) > 2:
        try:
            micro_batch_size = int(sys.argv[2])
        except ValueError:
            print(f"Invalid micro_batch_size argument '{sys.argv[2]}'. Using default ({DEFAULT_MICRO_BATCH_SIZE}).")

    print(f"Initializing Graph-Tracked Training Loop...")
    print(f"  Max Cycles: {max_cycles}")
    print(f"  Micro-Batch Size: {micro_batch_size} samples/cycle")
    
    run_graph_loop(max_cycles=max_cycles, micro_batch_size=micro_batch_size)

if __name__ == "__main__":
    main()
