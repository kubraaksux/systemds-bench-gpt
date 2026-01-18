#!/usr/bin/env python3
"""
Run vLLM benchmarks on Modal cloud GPU from your local terminal.

Setup (one-time):
    pip install modal
    modal setup  # Creates account and authenticates

Usage:
    python run_vllm_cloud.py              # Run all workloads
    python run_vllm_cloud.py --workload math  # Run specific workload
"""

import modal
import json
import os
import argparse
from pathlib import Path

# Create Modal app
app = modal.App("systemds-bench-vllm")

# Define the container image with vLLM
vllm_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "vllm>=0.4.0",
        "torch",
        "transformers",
        "accelerate",
        "pyyaml",
        "numpy",
        "tqdm",
        "datasets",
        "requests",
        "psutil",
    )
)


@app.function(
    image=vllm_image,
    gpu="T4",  # Use T4 GPU (cheapest). Options: "T4", "A10G", "A100"
    timeout=1800,  # 30 minutes max
)
def run_vllm_benchmark(workload: str, model: str = "microsoft/phi-2", n_samples: int = 10):
    """Run a single benchmark on cloud GPU with vLLM."""
    import subprocess
    import time
    import requests
    import json
    import os
    
    print(f"Starting vLLM benchmark: {workload} with {model}")
    
    # Start vLLM server
    print("Starting vLLM server...")
    server = subprocess.Popen(
        [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--host", "127.0.0.1",
            "--port", "8000",
            "--dtype", "float16",
            "--max-model-len", "2048",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait for server to be ready
    max_wait = 120  # 2 minutes
    start = time.time()
    ready = False
    while time.time() - start < max_wait:
        try:
            resp = requests.get("http://127.0.0.1:8000/v1/models", timeout=5)
            if resp.status_code == 200:
                ready = True
                print("vLLM server is ready!")
                break
        except:
            pass
        time.sleep(5)
        print(f"Waiting for server... ({int(time.time() - start)}s)")
    
    if not ready:
        server.terminate()
        return {"error": "vLLM server failed to start"}
    
    # Run benchmark using vLLM backend
    from vllm import LLM, SamplingParams
    
    # Load workload config
    workload_configs = {
        "math": {
            "prompt_template": "Solve this math problem step by step. Give the final numerical answer.\n\nProblem: {question}\n\nSolution:",
            "max_tokens": 512,
        },
        "reasoning": {
            "prompt_template": "{puzzle}",
            "max_tokens": 256,
        },
        "summarization": {
            "prompt_template": "Summarize the following text in 1 sentence, keeping only the key point. Be concise and shorter than the original.\n\n{text}",
            "max_tokens": 128,
        },
        "json_extraction": {
            "prompt_template": "Extract the following information from the text and return as valid JSON.\n\nText: {text}\n\nExtract these fields: {schema}\n\nJSON:",
            "max_tokens": 256,
        },
    }
    
    config = workload_configs.get(workload, workload_configs["math"])
    
    # Load dataset
    from datasets import load_dataset
    
    results = []
    
    if workload == "math":
        dataset = load_dataset("gsm8k", "main", split="test")
        samples = list(dataset)[:n_samples]
        prompts = [config["prompt_template"].format(question=s["question"]) for s in samples]
        references = [s["answer"].split("####")[-1].strip() for s in samples]
    elif workload == "reasoning":
        dataset = load_dataset("google/boolq", split="validation")
        samples = list(dataset)[:n_samples]
        prompts = [f"Passage: {s['passage']}\n\nQuestion: {s['question']}\n\nAnswer with just 'Yes' or 'No'." for s in samples]
        references = ["Yes" if s["answer"] else "No" for s in samples]
    else:
        # Simple test prompts for other workloads
        prompts = [f"Test prompt {i}" for i in range(n_samples)]
        references = ["" for _ in range(n_samples)]
    
    # Generate with vLLM
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=config["max_tokens"],
    )
    
    llm = LLM(model=model, dtype="float16", max_model_len=2048)
    
    print(f"Running {len(prompts)} prompts...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.time() - start_time
    
    # Calculate metrics
    correct = 0
    latencies = []
    
    for i, output in enumerate(outputs):
        pred = output.outputs[0].text.strip()
        ref = references[i] if i < len(references) else ""
        
        # Simple accuracy check
        is_correct = ref.lower() in pred.lower() if ref else True
        if is_correct:
            correct += 1
        
        results.append({
            "prompt": prompts[i][:100] + "...",
            "prediction": pred[:200],
            "reference": ref,
            "correct": is_correct,
        })
    
    # Stop server
    server.terminate()
    
    metrics = {
        "workload": workload,
        "model": model,
        "n_samples": n_samples,
        "accuracy_mean": correct / n_samples,
        "accuracy_count": f"{correct}/{n_samples}",
        "total_time_s": total_time,
        "avg_latency_ms": (total_time / n_samples) * 1000,
        "throughput_req_per_s": n_samples / total_time,
    }
    
    print(f"\nResults for {workload}:")
    print(f"  Accuracy: {metrics['accuracy_count']}")
    print(f"  Avg Latency: {metrics['avg_latency_ms']:.0f}ms")
    print(f"  Throughput: {metrics['throughput_req_per_s']:.2f} req/s")
    
    return {"metrics": metrics, "samples": results[:5]}  # Return first 5 samples


@app.local_entrypoint()
def main(workload: str = "all", model: str = "microsoft/phi-2", n_samples: int = 10):
    """Run vLLM benchmarks from your local terminal."""
    
    workloads = ["math", "reasoning"] if workload == "all" else [workload]
    
    print("=" * 50)
    print("SYSTEMDS-BENCH-GPT: vLLM Cloud Benchmark")
    print("=" * 50)
    print(f"Model: {model}")
    print(f"Workloads: {workloads}")
    print(f"Samples: {n_samples}")
    print("=" * 50)
    
    all_results = {}
    
    for wl in workloads:
        print(f"\n>>> Running {wl}...")
        result = run_vllm_benchmark.remote(wl, model, n_samples)
        all_results[wl] = result
        
        if "error" not in result:
            # Save results locally
            out_dir = Path(f"results/vllm_{wl}_cloud")
            out_dir.mkdir(parents=True, exist_ok=True)
            
            with open(out_dir / "metrics.json", "w") as f:
                json.dump(result["metrics"], f, indent=2)
            
            print(f"Results saved to {out_dir}")
    
    print("\n" + "=" * 50)
    print("All benchmarks complete!")
    print("Run: python scripts/report.py && open report.html")
    print("=" * 50)
    
    return all_results


if __name__ == "__main__":
    print("Run with: modal run run_vllm_cloud.py")
    print("Or: modal run run_vllm_cloud.py --workload math")
