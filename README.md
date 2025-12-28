# SYSTEMDS-BENCH-GPT

Backend-agnostic benchmarking suite for Large Language Model (LLM) inference systems.

SYSTEMDS-BENCH-GPT is a systems-oriented evaluation harness for comparing local LLM inference runtimes and hosted LLM APIs under controlled workloads, with a focus on latency, throughput, token efficiency, and runtime stability rather than leaderboard-style task accuracy.

---

## Scope

This repository implements a benchmarking framework to evaluate:

- local inference engines (e.g. MLX, vLLM, SystemDS)
- hosted LLM APIs (e.g. OpenAI-compatible endpoints)

The goal is to enable fair, reproducible, and extensible inference-system comparisons across backends and workloads.

---

## Supported Backends

Currently implemented:

- Local inference
  - MLX (`mlx-community/*` models)

- Hosted APIs
  - OpenAI-compatible APIs (e.g. `gpt-4.1-mini`)

Planned:
- vLLM backend adapter
- SystemDS inference runtime integration

---

## Repository Structure

- `backends/` — backend adapters (mlx, openai, etc.)
- `workloads/` — workload definitions
- `scripts/` — aggregation and report generation
- `results/` — per-run outputs (generated, ignored by git)
- `runner.py` — experiment runner
- `README.md` — project documentation


---

## Installation

Install Python dependencies:

**Create a virtual environment**
- python -m venv .venv

**Activate the virtual environment**
- source .venv/bin/activate

**Upgrade pip**
- python -m pip install --upgrade pip

**Install project dependencies**
- pip install -r requirements.txt

**For hosted API backends, ensure the required API keys are set:**

- export OPENAI_API_KEY="your_api_key_here"

---

## Running a Benchmark

### Example: local MLX summarization

python -u runner.py
--backend mlx
--workload workloads/summarization/config.yaml
--model mlx-community/Phi-3-mini-4k-instruct-4bit
--out results/run_mlx_$(date +%Y%m%d_%H%M%S)

### Example: OpenAI API summarization

python -u runner.py
--backend openai
--workload workloads/summarization/config.yaml
--model gpt-4.1-mini
--out results/run_openai_$(date +%Y%m%d_%H%M%S)

Each run produces:

- `samples.jsonl` — per-request outputs and metadata
- `metrics.json` — aggregated performance metrics
- `run_config.json` — exact configuration used
- `manifest.json` — timestamp and environment metadata

---

## Aggregating Results

Aggregate all completed runs into a single CSV summary:
python scripts/aggregate.py --out results_summary.csv

The aggregated output includes:
- mean, p50, and p95 latency
- throughput (requests/sec)
- input, output, and total token counts
- normalized token-based metrics when available

---

## Generating the HTML Report

Generate a static HTML report for inspection and sharing:
python scripts/report.py
open report.html

The report includes:
- latest runs (by timestamp)
- full run history
- derived normalization metrics such as tokens/sec and ms/token

The HTML report is a generated artifact and is not tracked in version control.

---

## Metrics Reported

For each run, the benchmark reports:

**Latency**
- mean latency
- p50 latency
- p95 latency

**Throughput**
- requests per second (req/s)

**Token accounting**
- input tokens
- output tokens
- total tokens
- average tokens per request

**Derived normalization**
- tokens/sec (total and output)
- ms/token (total and output)

These metrics allow fair comparison across backends and models.

---

## Current Limitations

This benchmark intentionally prioritizes performance and systems behavior over full task quality.

Known limitations:

- no accuracy metrics yet (e.g. ROUGE, F1)
- only summarization workload implemented
- no concurrency or batching sweeps
- TTFT (time-to-first-token) not measured
- no cost modeling for hosted APIs
- tokenization parity across backends is best-effort

Limitations are explicitly documented to avoid overclaiming.

---

## Intended Use

This benchmark is intended for:

- systems research and evaluation
- inference runtime comparison
- performance profiling under controlled workloads
- internal benchmarking and regression tracking

It is not intended as:

- a leaderboard
- a prompt engineering evaluation
- a replacement for task-quality benchmarks

---

## Resources

- **Purpose**: Systems-oriented benchmarking of LLM inference backends
- **Focus**: Latency, throughput, token efficiency, and runtime stability
- **Audience**: Systems researchers, inference engineers, and ML infrastructure developers
- **Status**: Active research prototype

---

## Roadmap

Planned extensions include:

- additional workloads (extraction / JSON, embeddings)
- lightweight accuracy evaluation per workload
- TTFT and decoding-time separation
- concurrency and batching sweeps
- cost-aware metrics for hosted APIs
- additional backend adapters (vLLM, SystemDS)

---

## License

This project is intended for research and educational use.  
License information will be added as the project matures.
