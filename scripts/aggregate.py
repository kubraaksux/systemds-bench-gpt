#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def iter_run_dirs(results_dir: Path) -> Iterable[Path]:
    if not results_dir.exists():
        return []
    for p in results_dir.iterdir():
        if p.is_dir() and (p / "metrics.json").exists() and (p / "run_config.json").exists():
            yield p


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def manifest_timestamp(run_dir: Path) -> str:
    """
    Returns UTC timestamp string from manifest.json if present, else "".
    Keeps it as a string (ISO 8601) so CSV stays simple.
    """
    mpath = run_dir / "manifest.json"
    if not mpath.exists():
        return ""
    try:
        m = read_json(mpath)
        ts = m.get("timestamp_utc")
        return "" if ts is None else str(ts)
    except Exception:
        return ""



def detect_has_tokens(run_dirs: Iterable[Path]) -> bool:
    for run_dir in run_dirs:
        samples_path = run_dir / "samples.jsonl"
        if not samples_path.exists():
            continue
        try:
            with samples_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    usage = (obj.get("extra") or {}).get("usage") or {}
                    if usage.get("total_tokens") is not None:
                        return True
        except Exception:
            # best-effort detection; ignore broken sample files
            continue
    return False


def token_stats(samples_path: Path) -> Tuple[Optional[int], Optional[float]]:
    """
    Returns (total_tokens, avg_tokens) if available, else (None, None).
    """
    if not samples_path.exists():
        return (None, None)

    total = 0
    count = 0
    try:
        with samples_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                usage = (obj.get("extra") or {}).get("usage") or {}
                tt = usage.get("total_tokens")
                if tt is None:
                    continue
                total += int(tt)
                count += 1
    except Exception:
        return (None, None)

    if count == 0:
        return (None, None)
    return (total, total / count)


def main() -> int:
    results_dir = Path("results")
    run_dirs = list(iter_run_dirs(results_dir))
    # sort by manifest timestamp if available; fallback to name
    run_dirs.sort(key=lambda p: (manifest_timestamp(p) == "", manifest_timestamp(p), p.name))


    if not run_dirs:
        print("Error: no valid run directories found under results/", file=sys.stderr)
        return 1

    # Detect whether any run contains token usage
    has_tokens = detect_has_tokens(run_dirs)

    header = [
        "run_dir",
        "ts",
        "backend",
        "backend_model",
        "n",
        "latency_ms_mean",
        "latency_ms_p50",
        "latency_ms_p95",
        "throughput_req_per_s",
    ]
    if has_tokens:
        header += ["total_tokens", "avg_tokens"]

    writer = csv.writer(sys.stdout)
    writer.writerow(header)

    for run_dir in run_dirs:
        try:
            metrics = read_json(run_dir / "metrics.json")
            cfg = read_json(run_dir / "run_config.json")

            row = [
                run_dir.name,
                manifest_timestamp(run_dir),
                cfg.get("backend", ""),
                cfg.get("backend_model", ""),
                metrics.get("n", ""),
                metrics.get("latency_ms_mean", ""),
                metrics.get("latency_ms_p50", ""),
                metrics.get("latency_ms_p95", ""),
                metrics.get("throughput_req_per_s", ""),
            ]

            if has_tokens:
                total, avg = token_stats(run_dir / "samples.jsonl")
                row += [
                    "" if total is None else total,
                    "" if avg is None else f"{avg:.4f}",
                ]

            writer.writerow(row)
        except Exception as e:
            print(f"Warning: skipping {run_dir.name}: {e}", file=sys.stderr)
            continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
