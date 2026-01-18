#!/bin/bash
# benchmark.sh - Run a single benchmark
# Usage: ./benchmark.sh <backend> <workload> [model]
# Examples:
#   ./benchmark.sh openai math
#   ./benchmark.sh ollama reasoning llama3.2
#   ./benchmark.sh mlx summarization mlx-community/Phi-3-mini-4k-instruct-4bit
#   ./benchmark.sh vllm math microsoft/phi-2

set -e

BACKEND="${1:-openai}"
WORKLOAD="${2:-math}"
MODEL="${3}"

# Default models for each backend
if [ -z "$MODEL" ]; then
    case "$BACKEND" in
        openai)
            MODEL="gpt-4.1-mini"
            ;;
        ollama)
            MODEL="llama3.2"
            ;;
        mlx)
            MODEL="mlx-community/Phi-3-mini-4k-instruct-4bit"
            ;;
        vllm)
            MODEL="microsoft/phi-2"
            ;;
        *)
            echo "Unknown backend: $BACKEND"
            exit 1
            ;;
    esac
fi

# Generate output directory name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/${BACKEND}_${WORKLOAD}_${TIMESTAMP}"

echo "========================================"
echo "SYSTEMDS-BENCH-GPT"
echo "========================================"
echo "Backend:  $BACKEND"
echo "Workload: $WORKLOAD"
echo "Model:    $MODEL"
echo "Output:   $OUTPUT_DIR"
echo "========================================"
echo ""

# Run the benchmark
python runner.py \
    --backend "$BACKEND" \
    --model "$MODEL" \
    --workload "workloads/${WORKLOAD}/config.yaml" \
    --out "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "Benchmark complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"

# Show metrics summary
if [ -f "$OUTPUT_DIR/metrics.json" ]; then
    echo ""
    echo "Metrics Summary:"
    python -c "
import json
with open('$OUTPUT_DIR/metrics.json') as f:
    m = json.load(f)
print(f\"  Accuracy: {m.get('accuracy_count', 'N/A')}\")
print(f\"  Latency (p50): {m.get('latency_ms_p50', 0):.0f}ms\")
print(f\"  Throughput: {m.get('throughput_req_per_s', 0):.3f} req/s\")
"
fi
