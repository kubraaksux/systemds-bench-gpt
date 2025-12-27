import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone


import yaml

from evaluation.perf import perf_metrics
from workloads.summarization.loader import load_samples
from workloads.summarization.prompt import make_prompt

def json_safe(x):
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [json_safe(v) for v in x]
    # pydantic-like objects
    if hasattr(x, "model_dump"):
        return json_safe(x.model_dump())
    if hasattr(x, "dict"):
        return json_safe(x.dict())
    return str(x)

def write_manifest(out_dir: Path, workload_path: Path, backend: str, model: str) -> None:
    # git commit hash (best-effort)
    git_commit_hash = None
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        git_commit_hash = r.stdout.strip()
    except Exception:
        git_commit_hash = None

    # workload config hash
    workload_bytes = workload_path.read_bytes()
    workload_sha256 = hashlib.sha256(workload_bytes).hexdigest()

    manifest = {
        "git_commit_hash": git_commit_hash,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": {
            "os": platform.system(),
            "architecture": platform.machine(),
        },
        "backend": backend,
        "model": model,
        "workload_config_path": str(workload_path.resolve()),
        "workload_config_sha256": workload_sha256,
    }
    write_json(out_dir / "manifest.json", manifest)



def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="systemds-bench-gpt runner")
    parser.add_argument("--backend", required=True, choices=["mlx", "openai"])
    parser.add_argument("--workload", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg: Dict[str, Any] = yaml.safe_load(Path(args.workload).read_text(encoding="utf-8"))

    if args.backend == "mlx":
        if not args.model:
            raise RuntimeError("--model is required for mlx backend.")
        from backends.mlx_backend import MLXBackend
        backend = MLXBackend(args.model)
        backend_cfg = cfg.get("generation", {})
        backend_model = args.model
    else:
        from backends.openai_backend import OpenAIBackend
        backend = OpenAIBackend()
        backend_cfg = cfg.get("openai", {})
        if args.model:
            backend_cfg = {**backend_cfg, "model": args.model}
        backend_model = backend_cfg.get("model", "unknown")

    samples = load_samples(cfg)
    prompts = [make_prompt(s, cfg) for s in samples]

    t0 = time.perf_counter()
    outputs = backend.generate(prompts, backend_cfg)
    t1 = time.perf_counter()

    run_config = {
        "backend": args.backend,
        "backend_model": backend_model,
        "workload": cfg.get("name", "unknown"),
        "n_samples": len(samples),
        "backend_config": backend_cfg,
    }
    write_json(out_dir / "run_config.json", run_config)

    latencies = []
    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for s, o in zip(samples, outputs):
            lat = float(o.get("latency_ms", 0.0))
            latencies.append(lat)
            rec = {
                "id": s.sid,
                "prediction": o.get("text", ""),
                "reference": s.reference,
                "latency_ms": lat,
                "extra": json_safe(o.get("extra", {})),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    metrics = perf_metrics(latencies, total_wall_s=(t1 - t0))
    write_json(out_dir / "metrics.json", metrics)
    write_manifest(out_dir, Path(args.workload), args.backend, backend_model)


    print(f"OK: wrote {out_dir}")


if __name__ == "__main__":
    main()
