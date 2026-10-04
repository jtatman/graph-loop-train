# Graph-Tracked Training Loop for Laya

An autonomous, self-improving training loop designed to continuously optimize fine-tuning of the decision-oriented model [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya).

The loop dynamically explores datasets, hyperparameters, and HuggingFace search results, evaluated against a gold benchmark validation split with automatic checkpoint promotion and rollback.

---

## Architecture Overview

```
                  +-----------------------------------+
                  |         Loop Controller           |
                  |  (Local LLM: 10.209.1.214:8080)   |
                  +-----------------+-----------------+
                                    |
          +-------------------------+-------------------------+
          |                         |                         |
          v                         v                         v
+------------------+      +------------------+      +------------------+
| State Graph /    |      | HF Dataset Search|      | Execution Engine |
| History Tracking |      | (`hf` CLI / API) |      | (train_runner)   |
+--------+---------+      +------------------+      +--------+---------+
         |                                                   |
         +--------------------------+------------------------+
                                    |
                                    v
                        +-----------------------+
                        |  Eval & Rollback Gate |
                        | (Gold Benchmark F1)   |
                        +-----------+-----------+
                                    |
                      +-------------+-------------+
                      |                           |
               [Net Positive]              [Net Negative]
                      |                           |
                      v                           v
              Promote Checkpoint         Rollback Checkpoint &
              Update Best Score          Increment Stagnation
```

### Core Loop Rules & Mechanics

1. **Static Core**: Model architecture (`convaiinnovations/laya`), training software infrastructure (`train_runner.py`), and evaluation benchmark are static.
2. **Dynamic Variables**: Dataset choice, learning rate, batch size, training epochs, and HuggingFace dataset search queries.
3. **Primary Datasets**:
   - `tdavidson/hate_speech_offensive` (baseline benchmark)
   - `dnagpt/laya-bio`
   - `SargeDev/jev-distill-corpus-v3`
   - Dynamic HuggingFace datasets fetched via HF search.
4. **Evaluation Gate**:
   - Each training cycle evaluates against the gold benchmark validation split.
   - **Net Positive ($\Delta \text{Macro F1} > 0$)**: Save checkpoint to `checkpoints/best_head.safetensors`, update best score baseline, reset stagnation counter to `0`.
   - **Net Negative ($\Delta \text{Macro F1} \le 0$)**: Roll back head weights to `checkpoints/best_head.safetensors`, increment stagnation counter (`stale += 1`).
5. **Exit Conditions**:
   - Total training cycles > 20
   - Stagnation (non-improvement over 5 consecutive passes)
   - Unrecoverable catastrophic training failure.

---

## Environment & Setup

This repository uses **`uv`** for Python virtual environment and dependency management, configured with **PyTorch CPU** wheels to ensure memory safety on small GPUs/CPUs.

### Installation

```bash
# Sync virtual environment and dependencies using uv
uv sync
```

### Dependencies
- `torch>=2.14.0` (CPU index: `https://download.pytorch.org/whl/cpu`)
- `laya`
- `pandas`
- `scikit-learn`
- `safetensors`
- `huggingface-hub`
- `pyarrow`
- `requests`
- `datasets`

---

## Execution

### Single Launch Command

Launch the full autonomous graph loop until exit conditions are met:

```bash
uv run python main.py
```

To specify a custom maximum number of cycles (e.g., 10 cycles):

```bash
uv run python main.py 10
```

---

## File Structure

- `main.py` - Single-launch CLI entry point.
- `loop.py` - Graph loop orchestrator, state manager, evaluation & rollback gate.
- `llm_controller.py` - Local LLM client (`http://10.209.1.214:8080/v1`) for strategy decisions.
- `hf_search.py` - HuggingFace dataset search helper using `/usr/bin/hf` CLI and python API.
- `dataset_loader.py` - Dataset ingestion & column schema normalizer.
- `train_runner.py` - Parameterized Laya head training pass and gold benchmark evaluation.
- `train.py` - Baseline standalone training script.
- `pyproject.toml` - Project configuration & `uv` dependency declarations.
