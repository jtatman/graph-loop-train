"""
Main entry point for Graph-Tracked Training Loop.
Usage:
    uv run python main.py [max_cycles]
"""

import sys
from loop import run_graph_loop

def main():
    max_cycles = 20
    if len(sys.argv) > 1:
        try:
            max_cycles = int(sys.argv[1])
        except ValueError:
            print(f"Invalid max_cycles argument '{sys.argv[1]}'. Using default (20).")

    print(f"Initializing Graph-Tracked Training Loop (max_cycles={max_cycles})...")
    run_graph_loop(max_cycles=max_cycles)

if __name__ == "__main__":
    main()
