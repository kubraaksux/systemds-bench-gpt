#!/usr/bin/env python3
import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_run_dir(p: Path) -> bool:
    return p.is_dir() and (p / "metrics.json").exists() and (p / "run_config.json").exists()


def iter_run_dirs(results_dir: Path) -> List[Path]:
    if not results_dir.exists():
        return []

    seen = set()
    runs: List[Path] = []

    # direct children
    for p in results_dir.iterdir():
        if is_run_dir(p):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                runs.append(p)

    # one-level nesting
    for group in results_dir.iterdir():
        if not group.is_dir():
            continue
        for p in group.iterdir():
            if is_run_dir(p):
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    runs.append(p)

    return runs


def manifest_timestamp(run_dir: Path) -> str:
    mpath = run_dir / "manifest.json"
    if not mpath.exists():
        return ""
    try:
        m = read_json(mpath)
        ts = m.get("timestamp_utc")
        return "" if ts is None else str(ts)
    except Exception:
        return ""


def token_stats(samples_path: Path) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[int]]:
    if not samples_path.exists():
        return (None, None, None, None)

    total_tokens = 0
    total_in = 0
    total_out = 0
    count = 0
    saw_any = False

    try:
        with samples_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                usage = (obj.get("extra") or {}).get("usage") or {}
                tt = usage.get("total_tokens")
                it = usage.get("input_tokens")
                ot = usage.get("output_tokens")

                if tt is None and it is None and ot is None:
                    continue

                saw_any = True
                if tt is not None:
                    total_tokens += int(tt)
                if it is not None:
                    total_in += int(it)
                if ot is not None:
                    total_out += int(ot)

                count += 1
    except Exception:
        return (None, None, None, None)

    if not saw_any or count == 0:
        return (None, None, None, None)

    avg = (total_tokens / count) if total_tokens > 0 else None
    return (
        total_tokens if total_tokens > 0 else None,
        avg,
        total_in if total_in > 0 else None,
        total_out if total_out > 0 else None,
    )


def sort_key(run_dir: Path) -> Tuple[int, str, str]:
    ts = manifest_timestamp(run_dir)
    missing = 1 if ts == "" else 0
    return (missing, ts, run_dir.name)

def ts_sort_value(ts: str) -> str:
    # ISO timestamps sort lexicographically; make missing timestamps very old.
    return ts if ts else "0000-00-00T00:00:00"



def safe_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def fmt(x: Any) -> str:
    if x is None:
        return ""
    return html.escape(str(x))


def fmt_num(x: Any, digits: int = 4) -> str:
    v = safe_float(x)
    if v is None:
        return ""
    return f"{v:.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a static HTML report from results/ runs.")
    ap.add_argument("--results-dir", default="results", help="Directory containing run folders (default: results)")
    ap.add_argument("--out", default="report.html", help="Output HTML path (default: report.html)")
    ap.add_argument("--latest", type=int, default=10, help="How many latest runs to highlight (default: 10)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    run_dirs = iter_run_dirs(results_dir)
    run_dirs.sort(key=sort_key)


    if not run_dirs:
        print(f"Error: no valid run directories found under {results_dir}/", file=sys.stderr)
        return 1

    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        try:
            metrics = read_json(run_dir / "metrics.json")
            cfg = read_json(run_dir / "run_config.json")
            ts = manifest_timestamp(run_dir)
            total, avg, total_in, total_out = token_stats(run_dir / "samples.jsonl")

            rows.append(
                {
                    "run_dir": run_dir.name,
                    "ts": ts,
                    "backend": cfg.get("backend", ""),
                    "backend_model": cfg.get("backend_model", ""),
                    "workload": cfg.get("workload", ""),
                    "n": metrics.get("n", ""),
                    "lat_mean": metrics.get("latency_ms_mean", ""),
                    "lat_p50": metrics.get("latency_ms_p50", ""),
                    "lat_p95": metrics.get("latency_ms_p95", ""),
                    "thr": metrics.get("throughput_req_per_s", ""),
                    "total_tokens": total,
                    "avg_tokens": avg,
                    "total_input_tokens": total_in,
                    "total_output_tokens": total_out,
                }
            )
        except Exception as e:
            print(f"Warning: skipping {run_dir.name}: {e}", file=sys.stderr)


    rows_sorted = sorted(rows, key=lambda r: ts_sort_value(str(r.get("ts", "")))) 
    latest_rows = rows_sorted[-args.latest:] if args.latest > 0 else []


    gen_ts = datetime.now(timezone.utc).isoformat()


    def table_html(title: str, table_rows: List[Dict[str, Any]]) -> str:
        cols = [
            ("run_dir", "Run"),
            ("ts", "Timestamp (UTC)"),
            ("backend", "Backend"),
            ("backend_model", "Model"),
            ("workload", "Workload"),
            ("n", "n"),
            ("lat_mean", "lat mean (ms)"),
            ("lat_p50", "p50 (ms)"),
            ("lat_p95", "p95 (ms)"),
            ("thr", "throughput (req/s)"),
            ("total_tokens", "total tok"),
            ("avg_tokens", "avg tok"),
            ("total_input_tokens", "in tok"),
            ("total_output_tokens", "out tok"),
            ("tokps_total", "tok/s (total)"),
            ("mstok_total", "ms/tok (total)"),
            ("tokps_out", "tok/s (out)"),
            ("mstok_out", "ms/tok (out)"),
        ]

        out: List[str] = [f"<h2>{fmt(title)}</h2>", "<table>", "<thead><tr>"]
        for _, label in cols:
            out.append(f"<th>{fmt(label)}</th>")
        out.append("</tr></thead><tbody>")

        for r in table_rows:
            # Derived normalization metrics (best-effort)
            thr = safe_float(r.get("thr"))
            avg_tok = safe_float(r.get("avg_tokens"))
            n = safe_float(r.get("n"))
            out_total = safe_float(r.get("total_output_tokens"))

            avg_out = (out_total / n) if (out_total is not None and n is not None and n > 0) else None

            tokps_total = (thr * avg_tok) if (thr is not None and avg_tok is not None) else None
            mstok_total = (1000.0 / tokps_total) if (tokps_total is not None and tokps_total > 0) else None

            tokps_out = (thr * avg_out) if (thr is not None and avg_out is not None) else None
            mstok_out = (1000.0 / tokps_out) if (tokps_out is not None and tokps_out > 0) else None

            out.append("<tr>")
            out.append(f"<td>{fmt(r.get('run_dir'))}</td>")
            out.append(f"<td>{fmt(r.get('ts'))}</td>")
            out.append(f"<td>{fmt(r.get('backend'))}</td>")
            out.append(f"<td>{fmt(r.get('backend_model'))}</td>")
            out.append(f"<td>{fmt(r.get('workload'))}</td>")
            out.append(f"<td>{fmt(r.get('n'))}</td>")
            out.append(f"<td>{fmt_num(r.get('lat_mean'), 2)}</td>")
            out.append(f"<td>{fmt_num(r.get('lat_p50'), 2)}</td>")
            out.append(f"<td>{fmt_num(r.get('lat_p95'), 2)}</td>")
            out.append(f"<td>{fmt_num(r.get('thr'), 4)}</td>")
            out.append(f"<td>{fmt(r.get('total_tokens'))}</td>")
            out.append(f"<td>{fmt_num(r.get('avg_tokens'), 2)}</td>")
            out.append(f"<td>{fmt(r.get('total_input_tokens'))}</td>")
            out.append(f"<td>{fmt(r.get('total_output_tokens'))}</td>")
            out.append(f"<td>{fmt_num(tokps_total, 2)}</td>")
            out.append(f"<td>{fmt_num(mstok_total, 2)}</td>")
            out.append(f"<td>{fmt_num(tokps_out, 2)}</td>")
            out.append(f"<td>{fmt_num(mstok_out, 2)}</td>")
            out.append("</tr>")

        out.append("</tbody></table>")
        return "\n".join(out)

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>systemds-bench-gpt report</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; margin: 24px; }}
    h1 {{ margin: 0 0 6px 0; }}
    .meta {{ color: #555; margin-bottom: 18px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0 24px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 13px; }}
    th {{ background: #f6f6f6; }}
    tr:nth-child(even) {{ background: #fbfbfb; }}
    code {{ background: #f2f2f2; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>systemds-bench-gpt report</h1>
  <div class="meta">Generated (UTC): <code>{fmt(gen_ts)}</code> | Runs found: <code>{len(rows)}</code></div>

  {table_html("Latest runs", latest_rows)}
  {table_html("All runs", rows_sorted)}


</body>
</html>
"""

    Path(args.out).write_text(html_doc, encoding="utf-8")
    print(f"OK: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
