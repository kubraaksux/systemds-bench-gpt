#!/usr/bin/env python3
"""Generate HTML benchmark report with charts and visualizations."""
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
    for p in results_dir.iterdir():
        if is_run_dir(p):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                runs.append(p)
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


def cost_stats(samples_path: Path) -> Optional[float]:
    """Calculate total cost from samples."""
    if not samples_path.exists():
        return None
    total_cost = 0.0
    try:
        with samples_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    extra = obj.get("extra") or {}
                    cost = extra.get("cost_usd")
                    if cost is not None:
                        total_cost += float(cost)
                except Exception:
                    continue
    except Exception:
        return None
    return total_cost if total_cost > 0 else None


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


def fmt_num(x: Any, digits: int = 2) -> str:
    v = safe_float(x)
    if v is None:
        return ""
    return f"{v:.{digits}f}"


# Colors for backends
BACKEND_COLORS = {
    "openai": "#10a37f",
    "mlx": "#ff6b6b",
    "ollama": "#4ecdc4",
    "vllm": "#9b59b6",
}

# Colors for workloads
WORKLOAD_COLORS = {
    "math": "#3498db",
    "reasoning": "#e74c3c",
    "summarization": "#2ecc71",
    "json_extraction": "#f39c12",
}


def generate_bar_chart_svg(data: List[Tuple[str, float, str]], title: str, 
                            width: int = 500, height: int = 300,
                            value_suffix: str = "", show_values: bool = True) -> str:
    """Generate SVG bar chart. data = [(label, value, color), ...]"""
    if not data:
        return ""
    
    max_val = max(d[1] for d in data) if data else 1
    bar_height = 28
    gap = 8
    left_margin = 120
    right_margin = 80
    top_margin = 40
    chart_width = width - left_margin - right_margin
    chart_height = len(data) * (bar_height + gap)
    total_height = chart_height + top_margin + 20
    
    svg = [f'<svg width="{width}" height="{total_height}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<text x="{width//2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">{html.escape(title)}</text>')
    
    for i, (label, value, color) in enumerate(data):
        y = top_margin + i * (bar_height + gap)
        bar_width = (value / max_val) * chart_width if max_val > 0 else 0
        
        # Label
        svg.append(f'<text x="{left_margin - 8}" y="{y + bar_height//2 + 4}" text-anchor="end" font-size="11">{html.escape(label[:15])}</text>')
        
        # Bar
        svg.append(f'<rect x="{left_margin}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{color}" rx="3"/>')
        
        # Value
        if show_values:
            val_text = f"{value:.1f}{value_suffix}" if isinstance(value, float) else f"{value}{value_suffix}"
            svg.append(f'<text x="{left_margin + bar_width + 5}" y="{y + bar_height//2 + 4}" font-size="11">{val_text}</text>')
    
    svg.append('</svg>')
    return '\n'.join(svg)


def generate_grouped_bar_chart_svg(data: Dict[str, Dict[str, float]], title: str,
                                    group_colors: Dict[str, str],
                                    width: int = 600, height: int = 350,
                                    value_suffix: str = "") -> str:
    """Generate grouped bar chart. data = {category: {group: value}}"""
    if not data:
        return ""
    
    categories = list(data.keys())
    groups = set()
    for cat_data in data.values():
        groups.update(cat_data.keys())
    groups = sorted(groups)
    
    max_val = 0
    for cat_data in data.values():
        for v in cat_data.values():
            if v > max_val:
                max_val = v
    if max_val == 0:
        max_val = 1
    
    left_margin = 130
    right_margin = 20
    top_margin = 50
    bottom_margin = 60
    chart_width = width - left_margin - right_margin
    chart_height = height - top_margin - bottom_margin
    
    category_height = chart_height / len(categories) if categories else 1
    bar_height = min(20, (category_height - 10) / len(groups)) if groups else 20
    
    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<text x="{width//2}" y="25" text-anchor="middle" font-size="14" font-weight="bold">{html.escape(title)}</text>')
    
    for i, category in enumerate(categories):
        cat_y = top_margin + i * category_height
        
        # Category label
        svg.append(f'<text x="{left_margin - 8}" y="{cat_y + category_height//2}" text-anchor="end" font-size="11">{html.escape(category[:18])}</text>')
        
        for j, group in enumerate(groups):
            value = data[category].get(group, 0)
            bar_y = cat_y + j * (bar_height + 2) + 5
            bar_width = (value / max_val) * chart_width if max_val > 0 else 0
            color = group_colors.get(group, "#999")
            
            svg.append(f'<rect x="{left_margin}" y="{bar_y}" width="{bar_width}" height="{bar_height}" fill="{color}" rx="2"/>')
            
            if value > 0:
                val_text = f"{value:.1f}{value_suffix}" if isinstance(value, float) else f"{value}{value_suffix}"
                svg.append(f'<text x="{left_margin + bar_width + 3}" y="{bar_y + bar_height//2 + 4}" font-size="9">{val_text}</text>')
    
    svg.append('</svg>')
    
    # Legend as HTML
    legend = ['<div style="display: flex; flex-wrap: wrap; gap: 15px; margin-top: 10px; justify-content: center;">']
    for group in groups:
        color = group_colors.get(group, "#999")
        legend.append(f'<div style="display: flex; align-items: center; gap: 5px;">')
        legend.append(f'<div style="width: 14px; height: 14px; background: {color}; border-radius: 3px;"></div>')
        legend.append(f'<span style="font-size: 12px;">{html.escape(group)}</span>')
        legend.append('</div>')
    legend.append('</div>')
    
    return '\n'.join(svg) + '\n' + '\n'.join(legend)


def generate_accuracy_comparison_table(rows: List[Dict[str, Any]]) -> str:
    """Generate accuracy comparison table by workload and backend."""
    # Group by workload and backend, take latest
    data: Dict[str, Dict[str, Dict[str, Any]]] = {}  # workload -> backend -> metrics
    
    for r in rows:
        workload = r.get("workload", "")
        backend = r.get("backend", "")
        if not workload or not backend:
            continue
        
        if workload not in data:
            data[workload] = {}
        
        # Keep latest (rows should be sorted by timestamp)
        data[workload][backend] = r
    
    if not data:
        return ""
    
    workloads = sorted(data.keys())
    backends = sorted(set(b for w in data.values() for b in w.keys()))
    
    out = ['<h2>Accuracy Comparison by Workload</h2>']
    out.append('<table class="comparison-table">')
    out.append('<thead><tr><th>Workload</th>')
    for b in backends:
        out.append(f'<th>{html.escape(b)}</th>')
    out.append('</tr></thead><tbody>')
    
    for wl in workloads:
        out.append(f'<tr><td><strong>{html.escape(wl)}</strong></td>')
        for b in backends:
            if b in data[wl]:
                acc = data[wl][b].get("accuracy_mean")
                acc_count = data[wl][b].get("accuracy_count", "")
                if acc is not None:
                    pct = acc * 100
                    color = "#2ecc71" if pct >= 80 else "#f39c12" if pct >= 50 else "#e74c3c"
                    out.append(f'<td style="background: {color}22; color: {color}; font-weight: bold;">{pct:.0f}%<br><small>{acc_count}</small></td>')
                else:
                    out.append('<td>-</td>')
            else:
                out.append('<td>-</td>')
        out.append('</tr>')
    
    out.append('</tbody></table>')
    return '\n'.join(out)


def generate_latency_comparison_table(rows: List[Dict[str, Any]]) -> str:
    """Generate latency comparison table by workload and backend."""
    data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    for r in rows:
        workload = r.get("workload", "")
        backend = r.get("backend", "")
        if not workload or not backend:
            continue
        if workload not in data:
            data[workload] = {}
        data[workload][backend] = r
    
    if not data:
        return ""
    
    workloads = sorted(data.keys())
    backends = sorted(set(b for w in data.values() for b in w.keys()))
    
    out = ['<h2>Latency Comparison (p50 ms)</h2>']
    out.append('<table class="comparison-table">')
    out.append('<thead><tr><th>Workload</th>')
    for b in backends:
        out.append(f'<th>{html.escape(b)}</th>')
    out.append('</tr></thead><tbody>')
    
    for wl in workloads:
        out.append(f'<tr><td><strong>{html.escape(wl)}</strong></td>')
        for b in backends:
            if b in data[wl]:
                lat = safe_float(data[wl][b].get("lat_p50"))
                if lat is not None:
                    out.append(f'<td>{lat:.0f}ms</td>')
                else:
                    out.append('<td>-</td>')
            else:
                out.append('<td>-</td>')
        out.append('</tr>')
    
    out.append('</tbody></table>')
    return '\n'.join(out)


def generate_summary_cards(rows: List[Dict[str, Any]]) -> str:
    """Generate summary metric cards."""
    # Get unique backends and workloads
    backends = set(r.get("backend") for r in rows if r.get("backend"))
    workloads = set(r.get("workload") for r in rows if r.get("workload"))
    total_runs = len(rows)
    
    # Calculate averages
    accuracies = [r.get("accuracy_mean") for r in rows if r.get("accuracy_mean") is not None]
    avg_accuracy = sum(accuracies) / len(accuracies) * 100 if accuracies else 0
    
    latencies = [safe_float(r.get("lat_p50")) for r in rows if safe_float(r.get("lat_p50")) is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    out = ['<div class="summary-cards">']
    
    cards = [
        ("Total Runs", str(total_runs), "#3498db"),
        ("Backends", str(len(backends)), "#9b59b6"),
        ("Workloads", str(len(workloads)), "#e74c3c"),
        ("Avg Accuracy", f"{avg_accuracy:.0f}%", "#2ecc71"),
        ("Avg Latency", f"{avg_latency:.0f}ms", "#f39c12"),
    ]
    
    for label, value, color in cards:
        out.append(f'''
        <div class="card" style="border-left: 4px solid {color};">
            <div class="card-value">{value}</div>
            <div class="card-label">{label}</div>
        </div>
        ''')
    
    out.append('</div>')
    return '\n'.join(out)


def generate_charts_section(rows: List[Dict[str, Any]]) -> str:
    """Generate all charts."""
    out = ['<h2>Performance Charts</h2>', '<div class="charts-grid">']
    
    # Group data by workload and backend (latest only)
    latest: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        wl = r.get("workload", "")
        be = r.get("backend", "")
        if not wl or not be:
            continue
        if wl not in latest:
            latest[wl] = {}
        latest[wl][be] = r
    
    # Accuracy by Backend chart
    accuracy_data: Dict[str, Dict[str, float]] = {}
    for wl, backends in latest.items():
        accuracy_data[wl] = {}
        for be, r in backends.items():
            acc = r.get("accuracy_mean")
            if acc is not None:
                accuracy_data[wl][be] = acc * 100
    
    if accuracy_data:
        out.append('<div class="chart-container">')
        out.append(generate_grouped_bar_chart_svg(
            accuracy_data, "Accuracy by Workload (%)", 
            BACKEND_COLORS, value_suffix="%"
        ))
        out.append('</div>')
    
    # Latency by Backend chart
    latency_data: Dict[str, Dict[str, float]] = {}
    for wl, backends in latest.items():
        latency_data[wl] = {}
        for be, r in backends.items():
            lat = safe_float(r.get("lat_p50"))
            if lat is not None:
                latency_data[wl][be] = lat / 1000  # Convert to seconds
    
    if latency_data:
        out.append('<div class="chart-container">')
        out.append(generate_grouped_bar_chart_svg(
            latency_data, "Latency by Workload (p50, seconds)",
            BACKEND_COLORS, value_suffix="s"
        ))
        out.append('</div>')
    
    # Throughput chart
    throughput_data: Dict[str, Dict[str, float]] = {}
    for wl, backends in latest.items():
        throughput_data[wl] = {}
        for be, r in backends.items():
            thr = safe_float(r.get("thr"))
            if thr is not None:
                throughput_data[wl][be] = thr
    
    if throughput_data:
        out.append('<div class="chart-container">')
        out.append(generate_grouped_bar_chart_svg(
            throughput_data, "Throughput by Workload (req/s)",
            BACKEND_COLORS, value_suffix=" req/s"
        ))
        out.append('</div>')
    
    out.append('</div>')
    return '\n'.join(out)


def generate_detailed_table(title: str, table_rows: List[Dict[str, Any]]) -> str:
    """Generate detailed results table."""
    cols = [
        ("run_dir", "Run"),
        ("ts", "Timestamp"),
        ("backend", "Backend"),
        ("backend_model", "Model"),
        ("workload", "Workload"),
        ("n", "n"),
        ("accuracy_count", "Accuracy"),
        ("lat_mean", "Latency (mean)"),
        ("lat_p50", "p50"),
        ("lat_p95", "p95"),
        ("thr", "Throughput"),
        ("total_tokens", "Tokens"),
        ("cost", "Cost"),
    ]
    
    out = [f'<h2>{html.escape(title)}</h2>', '<div class="table-container"><table>']
    out.append('<thead><tr>')
    for _, label in cols:
        out.append(f'<th>{html.escape(label)}</th>')
    out.append('</tr></thead><tbody>')
    
    for r in table_rows:
        out.append('<tr>')
        out.append(f'<td><code>{html.escape(str(r.get("run_dir", ""))[:30])}</code></td>')
        ts = r.get("ts", "")
        if ts:
            ts = ts[:19].replace("T", " ")
        out.append(f'<td>{html.escape(ts)}</td>')
        
        backend = r.get("backend", "")
        color = BACKEND_COLORS.get(backend, "#666")
        out.append(f'<td><span class="badge" style="background: {color};">{html.escape(backend)}</span></td>')
        
        out.append(f'<td>{html.escape(str(r.get("backend_model", ""))[:25])}</td>')
        
        workload = r.get("workload", "")
        wl_color = WORKLOAD_COLORS.get(workload, "#666")
        out.append(f'<td><span class="badge" style="background: {wl_color};">{html.escape(workload)}</span></td>')
        
        out.append(f'<td>{fmt(r.get("n"))}</td>')
        
        acc = r.get("accuracy_mean")
        acc_count = r.get("accuracy_count", "")
        if acc is not None:
            pct = acc * 100
            acc_color = "#2ecc71" if pct >= 80 else "#f39c12" if pct >= 50 else "#e74c3c"
            out.append(f'<td style="color: {acc_color}; font-weight: bold;">{pct:.0f}% ({acc_count})</td>')
        else:
            out.append('<td>-</td>')
        
        out.append(f'<td>{fmt_num(r.get("lat_mean"))}ms</td>')
        out.append(f'<td>{fmt_num(r.get("lat_p50"))}ms</td>')
        out.append(f'<td>{fmt_num(r.get("lat_p95"))}ms</td>')
        out.append(f'<td>{fmt_num(r.get("thr"), 3)} req/s</td>')
        out.append(f'<td>{fmt(r.get("total_tokens"))}</td>')
        
        cost = r.get("cost")
        if cost is not None:
            out.append(f'<td>${cost:.4f}</td>')
        else:
            out.append('<td>-</td>')
        
        out.append('</tr>')
    
    out.append('</tbody></table></div>')
    return '\n'.join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate HTML benchmark report with charts.")
    ap.add_argument("--results-dir", default="results", help="Directory containing run folders")
    ap.add_argument("--out", default="report.html", help="Output HTML path")
    ap.add_argument("--latest", type=int, default=20, help="How many latest runs to show")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    run_dirs = iter_run_dirs(results_dir)
    
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
            cost = cost_stats(run_dir / "samples.jsonl")

            rows.append({
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
                "accuracy_mean": metrics.get("accuracy_mean"),
                "accuracy_count": metrics.get("accuracy_count", ""),
                "total_tokens": total,
                "avg_tokens": avg,
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
                "cost": cost,
            })
        except Exception as e:
            print(f"Warning: skipping {run_dir.name}: {e}", file=sys.stderr)

    # Sort by timestamp
    rows_sorted = sorted(rows, key=lambda r: r.get("ts", "") or "0000", reverse=True)
    latest_rows = rows_sorted[:args.latest]

    gen_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>systemds-bench-gpt Benchmark Report</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ 
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
        margin: 0; padding: 24px; 
        background: #f8f9fa;
        color: #333;
    }}
    .container {{ max-width: 1400px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px 0; color: #1a1a2e; }}
    h2 {{ margin: 30px 0 15px 0; color: #1a1a2e; border-bottom: 2px solid #eee; padding-bottom: 8px; }}
    .meta {{ color: #666; margin-bottom: 24px; font-size: 14px; }}
    
    .summary-cards {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 16px;
        margin-bottom: 30px;
    }}
    .card {{
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .card-value {{ font-size: 28px; font-weight: bold; color: #1a1a2e; }}
    .card-label {{ font-size: 13px; color: #666; margin-top: 4px; }}
    
    .charts-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
        gap: 24px;
        margin-bottom: 30px;
    }}
    .chart-container {{
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    .comparison-table {{
        width: 100%;
        border-collapse: collapse;
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 24px;
    }}
    .comparison-table th, .comparison-table td {{
        padding: 12px 16px;
        text-align: center;
        border-bottom: 1px solid #eee;
    }}
    .comparison-table th {{
        background: #f8f9fa;
        font-weight: 600;
        color: #1a1a2e;
    }}
    .comparison-table td:first-child {{
        text-align: left;
    }}
    
    .table-container {{
        overflow-x: auto;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    table {{ 
        border-collapse: collapse; 
        width: 100%; 
        font-size: 13px;
    }}
    th, td {{ 
        padding: 10px 12px; 
        text-align: left; 
        border-bottom: 1px solid #eee;
        white-space: nowrap;
    }}
    th {{ 
        background: #f8f9fa; 
        font-weight: 600;
        color: #1a1a2e;
        position: sticky;
        top: 0;
    }}
    tr:hover {{ background: #f8f9fa; }}
    
    .badge {{
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        color: white;
        font-size: 11px;
        font-weight: 500;
    }}
    
    code {{ 
        background: #f1f3f4; 
        padding: 2px 6px; 
        border-radius: 4px; 
        font-size: 12px;
    }}
    
    @media (max-width: 768px) {{
        .charts-grid {{ grid-template-columns: 1fr; }}
        .summary-cards {{ grid-template-columns: repeat(2, 1fr); }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>systemds-bench-gpt Benchmark Report</h1>
    <div class="meta">Generated: {gen_ts} | Total Runs: {len(rows)}</div>
    
    {generate_summary_cards(rows)}
    
    {generate_accuracy_comparison_table(rows_sorted)}
    
    {generate_latency_comparison_table(rows_sorted)}
    
    {generate_charts_section(rows_sorted)}
    
    {generate_detailed_table("Latest Benchmark Runs", latest_rows)}
    
  </div>
</body>
</html>
"""

    Path(args.out).write_text(html_doc, encoding="utf-8")
    print(f"OK: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
