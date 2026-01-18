#!/bin/bash
# run_all_benchmarks.sh - Run all workloads for a single backend
# Usage: ./run_all_benchmarks.sh <backend> [model]
# Examples:
#   ./run_all_benchmarks.sh openai
#   ./run_all_benchmarks.sh ollama llama3.2
#   ./run_all_benchmarks.sh mlx mlx-community/Phi-3-mini-4k-instruct-4bit

set -e

BACKEND="${1:-openai}"
MODEL="${2}"

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

WORKLOADS=("math" "reasoning" "summarization" "json_extraction")

echo "========================================"
echo "SYSTEMDS-BENCH-GPT - Full Benchmark Suite"
echo "========================================"
echo "Backend: $BACKEND"
echo "Model:   $MODEL"
echo "Workloads: ${WORKLOADS[*]}"
echo "========================================"
echo ""

# Track results
declare -a RESULTS

for WORKLOAD in "${WORKLOADS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Running: $WORKLOAD"
    echo "----------------------------------------"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="results/${BACKEND}_${WORKLOAD}_${TIMESTAMP}"
    
    if python runner.py \
        --backend "$BACKEND" \
        --model "$MODEL" \
        --workload "workloads/${WORKLOAD}/config.yaml" \
        --out "$OUTPUT_DIR" 2>&1; then
        
        # Get metrics
        if [ -f "$OUTPUT_DIR/metrics.json" ]; then
            ACCURACY=$(python -c "import json; m=json.load(open('$OUTPUT_DIR/metrics.json')); print(m.get('accuracy_count', 'N/A'))")
            LATENCY=$(python -c "import json; m=json.load(open('$OUTPUT_DIR/metrics.json')); print(f\"{m.get('latency_ms_p50', 0):.0f}ms\")")
            RESULTS+=("$WORKLOAD: $ACCURACY accuracy, $LATENCY latency")
        fi
    else
        RESULTS+=("$WORKLOAD: FAILED")
    fi
done

echo ""
echo "========================================"
echo "BENCHMARK SUMMARY"
echo "========================================"
echo "Backend: $BACKEND ($MODEL)"
echo ""
for result in "${RESULTS[@]}"; do
    echo "  $result"
done
echo ""
echo "========================================"

# Generate report
echo ""
echo "Generating HTML report..."
python scripts/report.py --out benchmark_report.html
echo "Report saved to: benchmark_report.html"
