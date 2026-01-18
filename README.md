# SYSTEMDS-BENCH-GPT

Backend-agnostic benchmarking suite for Large Language Model (LLM) inference systems.

SYSTEMDS-BENCH-GPT is a systems-oriented evaluation harness for comparing local LLM inference runtimes and hosted LLM APIs under controlled workloads, with a focus on **latency, throughput, accuracy, cost, and resource usage**.

---

## Features

- **Multiple Backends**: OpenAI API, Ollama (local), vLLM (GPU server), MLX (Apple Silicon)
- **Real Datasets**: GSM8K (math), XSum (summarization), BoolQ (reasoning), CoNLL-2003 NER (JSON extraction)
- **Comprehensive Metrics**: Latency (mean, p50, p95), throughput, accuracy, cost, tokens, TTFT
- **HTML Reports**: Auto-generated reports with charts and visualizations
- **Extensible**: Easy to add new backends and workloads
- **Reproducible**: Shell scripts for easy benchmarking

---

## Supported Backends

| Backend | Description | Requirements |
|---------|-------------|--------------|
| `openai` | OpenAI API (GPT-4, etc.) | `OPENAI_API_KEY` environment variable |
| `ollama` | Local inference via Ollama | [Ollama](https://ollama.ai) installed and running |
| `vllm` | High-performance inference server | vLLM server running (requires GPU) |
| `mlx` | Apple Silicon optimized | macOS with Apple Silicon, `mlx-lm` package |

---

## Workloads

| Workload | Dataset | Description |
|----------|---------|-------------|
| `math` | GSM8K | Grade school math word problems |
| `summarization` | XSum, CNN/DM | Text summarization |
| `reasoning` | BoolQ, LogiQA | Logical reasoning / QA |
| `json_extraction` | CoNLL-2003 NER | Structured JSON extraction |

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/kubraaksux/systemds-bench-gpt.git
cd systemds-bench-gpt

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For OpenAI backend
export OPENAI_API_KEY="your-key-here"
```

### 2. Run Benchmarks

**Using shell scripts (recommended):**

```bash
# Run single benchmark
./benchmark.sh openai math
./benchmark.sh ollama reasoning
./benchmark.sh mlx summarization

# Run all workloads for a backend
./run_all_benchmarks.sh openai
./run_all_benchmarks.sh ollama
```

**Using Python directly:**

```bash
# OpenAI API
python runner.py \
  --backend openai \
  --workload workloads/math/config.yaml \
  --out results/openai_math

# Ollama (local)
ollama pull llama3.2
python runner.py \
  --backend ollama \
  --model llama3.2 \
  --workload workloads/math/config.yaml \
  --out results/ollama_math

# MLX (Apple Silicon)
python runner.py \
  --backend mlx \
  --model mlx-community/Phi-3-mini-4k-instruct-4bit \
  --workload workloads/summarization/config.yaml \
  --out results/mlx_summarization

# vLLM (requires GPU server)
python runner.py \
  --backend vllm \
  --model microsoft/phi-2 \
  --workload workloads/reasoning/config.yaml \
  --out results/vllm_reasoning
```

### 3. Generate Report

```bash
python scripts/report.py --out benchmark_report.html
open benchmark_report.html
```

---

## Repository Structure

```
systemds-bench-gpt/
├── backends/
│   ├── openai_backend.py   # OpenAI API adapter
│   ├── ollama_backend.py   # Ollama local inference
│   ├── vllm_backend.py     # vLLM server adapter
│   └── mlx_backend.py      # Apple Silicon MLX
├── workloads/
│   ├── math/               # GSM8K math problems
│   ├── summarization/      # XSum summarization
│   ├── reasoning/          # BoolQ reasoning
│   └── json_extraction/    # NER/structured extraction
├── scripts/
│   ├── aggregate.py        # CSV aggregation
│   └── report.py           # HTML report generation
├── notebooks/
│   └── vllm_colab.ipynb    # Google Colab for vLLM (GPU)
├── results/                # Benchmark outputs (gitignored)
├── runner.py               # Main benchmark runner
├── benchmark.sh            # Single benchmark script
├── run_all_benchmarks.sh   # Run all workloads
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Metrics Reported

### Latency
- **Mean latency**: Average response time
- **P50 latency**: Median response time
- **P95 latency**: 95th percentile response time
- **TTFT**: Time to first token (streaming)

### Throughput
- **Requests per second**: How many requests can be handled
- **Tokens per second**: Generation speed

### Accuracy
- **Per-workload accuracy**: Compared against ground truth
- **Accuracy count**: e.g., "8/10" correct

### Cost
- **Total cost (USD)**: For API-based backends
- **Cost per request**: Average cost per inference
- **Local backends**: Cost = $0 (only hardware costs)

### Token Accounting
- **Input tokens**: Prompt tokens
- **Output tokens**: Generated tokens
- **Total tokens**: Sum of input + output

---

## Output Files

Each run produces:

| File | Description |
|------|-------------|
| `samples.jsonl` | Per-request outputs with predictions, latencies, tokens |
| `metrics.json` | Aggregated performance metrics |
| `run_config.json` | Exact configuration used |
| `manifest.json` | Timestamp, environment, git hash |

---

## Backend Setup

### OpenAI
```bash
export OPENAI_API_KEY="sk-..."
python runner.py --backend openai --workload workloads/math/config.yaml --out results/test
```

### Ollama
```bash
# Install from https://ollama.ai
ollama pull llama3.2
python runner.py --backend ollama --model llama3.2 --workload workloads/math/config.yaml --out results/test
```

### vLLM (requires GPU)
```bash
# On a GPU machine, start vLLM server
pip install vllm
python -m vllm.entrypoints.openai.api_server --model microsoft/phi-2 --port 8000

# Run benchmark (from any machine that can reach the server)
python runner.py --backend vllm --model microsoft/phi-2 --workload workloads/math/config.yaml --out results/test

# Or use Google Colab (free GPU): see notebooks/vllm_colab.ipynb
```

### MLX (Apple Silicon only)
```bash
pip install mlx mlx-lm
python runner.py --backend mlx --model mlx-community/Phi-3-mini-4k-instruct-4bit --workload workloads/math/config.yaml --out results/test
```

---

## Sample Results

| Backend | Model | Workload | Accuracy | Latency (p50) | Cost |
|---------|-------|----------|----------|---------------|------|
| OpenAI | gpt-4.1-mini | math | 100% | 4.5s | $0.004 |
| OpenAI | gpt-4.1-mini | reasoning | 100% | 3.5s | $0.003 |
| MLX | Phi-3-mini-4bit | math | 20% | 10.0s | $0 |
| Ollama | llama3.2 | math | 10% | 5.9s | $0 |
| vLLM | microsoft/phi-2 | reasoning | 70% | 10.2s | $0 |

---

## Extending the Framework

### Adding a New Backend

Create `backends/mybackend_backend.py`:

```python
class MyBackend:
    def __init__(self, model: str):
        self.model = model
    
    def generate(self, prompts: list, config: dict) -> list:
        results = []
        for prompt in prompts:
            # Your inference logic here
            results.append({
                "text": "generated text",
                "latency_ms": 100.0,
                "ttft_ms": 10.0,
                "extra": {
                    "usage": {"input_tokens": 50, "output_tokens": 100, "total_tokens": 150},
                    "cost_usd": 0.0
                }
            })
        return results
```

### Adding a New Workload

Create `workloads/myworkload/`:
- `config.yaml` - Configuration
- `loader.py` - `load_samples()` and `accuracy_check()` functions
- `prompt.py` - `make_prompt()` function
- `__init__.py`

---

## Intended Use

This benchmark is intended for:
- Systems research and evaluation
- Inference runtime comparison
- Performance profiling under controlled workloads
- Cost-benefit analysis of local vs. hosted inference

---

## License

This project is developed as part of the SystemDS research group at TU Berlin.
License information will be added as the project matures.
