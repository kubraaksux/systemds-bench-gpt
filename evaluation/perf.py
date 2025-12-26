from typing import Dict, List
import numpy as np


def perf_metrics(latencies_ms: List[float], total_wall_s: float) -> Dict[str, float]:
    arr = np.array(latencies_ms, dtype=float)
    if len(arr) == 0:
        return {
            "n": 0.0,
            "latency_ms_mean": 0.0,
            "latency_ms_p50": 0.0,
            "latency_ms_p95": 0.0,
            "throughput_req_per_s": 0.0,
        }

    return {
        "n": float(len(arr)),
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "throughput_req_per_s": float(len(arr) / total_wall_s) if total_wall_s > 0 else 0.0,
    }
