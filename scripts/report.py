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
    found_any = False
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
                        found_any = True
                        total_cost += float(cost)
                except Exception:
                    continue
    except Exception:
        return None
    # Return 0.0 for local backends (they report cost_usd: 0.0)
    return total_cost if found_any else None


def timing_stats(samples_path: Path) -> Tuple[Optional[float], Optional[float]]:
    """Calculate TTFT and generation time means from samples."""
    if not samples_path.exists():
        return (None, None)
    ttft_vals = []
    gen_vals = []
    try:
        with samples_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ttft = obj.get("ttft_ms")
                    gen = obj.get("generation_ms")
                    if ttft is not None:
                        ttft_vals.append(float(ttft))
                    if gen is not None:
                        gen_vals.append(float(gen))
                except Exception:
                    continue
    except Exception:
        return (None, None)
    
    ttft_mean = sum(ttft_vals) / len(ttft_vals) if ttft_vals else None
    gen_mean = sum(gen_vals) / len(gen_vals) if gen_vals else None
    return (ttft_mean, gen_mean)


def safe_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def fmt(x: Any) -> str:
    if x is None:
        return "N/A"
    return html.escape(str(x))


def fmt_num(x: Any, digits: int = 2) -> str:
    v = safe_float(x)
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def fmt_pct(x: Any, digits: int = 1) -> str:
    v = safe_float(x)
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}%"


def fmt_cost(x: Any) -> str:
    v = safe_float(x)
    if v is None:
        return "N/A"
    if v == 0:
        return "$0.00"
    return f"${v:.4f}"


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
    
    costs = [safe_float(r.get("cost")) for r in rows if safe_float(r.get("cost")) is not None]
    total_cost = sum(costs) if costs else 0
    
    out = ['<div class="summary-cards">']
    
    cards = [
        ("Total Runs", str(total_runs), "#3498db"),
        ("Backends", str(len(backends)), "#9b59b6"),
        ("Workloads", str(len(workloads)), "#e74c3c"),
        ("Avg Accuracy", f"{avg_accuracy:.0f}%", "#2ecc71"),
        ("Avg Latency", f"{avg_latency:.0f}ms", "#f39c12"),
        ("Total Cost", f"${total_cost:.4f}", "#1abc9c"),
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


# Helper to show cost only for OpenAI (where it's actually tracked)
def fmt_cost_if_real(r: Dict[str, Any]) -> str:
    cost = r.get("cost")
    backend = r.get("backend", "")
    if backend == "openai" and cost is not None:
        return fmt_cost(cost)
    return "-"

def fmt_cost_per_1m_if_real(r: Dict[str, Any]) -> str:
    cost = r.get("cost_per_1m_tokens")
    backend = r.get("backend", "")
    if backend == "openai" and cost is not None:
        return fmt_cost(cost)
    return "-"

# Define ALL columns for the full table (includes TTFT for streaming backends)
FULL_TABLE_COLUMNS = [
    ("run_dir", "Run", lambda r: f'<code>{html.escape(str(r.get("run_dir", ""))[:25])}</code>'),
    ("ts", "Timestamp (UTC)", lambda r: html.escape((r.get("ts", "") or "")[:19].replace("T", " "))),
    ("backend", "Backend", lambda r: html.escape(r.get("backend", ""))),
    ("backend_model", "Model", lambda r: html.escape(str(r.get("backend_model", ""))[:20])),
    ("workload", "Workload", lambda r: html.escape(r.get("workload", ""))),
    ("n", "n", lambda r: fmt(r.get("n"))),
    ("accuracy", "Accuracy", lambda r: f'{r.get("accuracy_mean", 0)*100:.1f}% ({r.get("accuracy_count", "")})' if r.get("accuracy_mean") is not None else "N/A"),
    ("cost", "Cost ($)", fmt_cost_if_real),
    ("cost_per_1m", "$/1M tok", fmt_cost_per_1m_if_real),
    ("mem_peak", "Mem Peak (MB)", lambda r: fmt_num(r.get("mem_peak"), 1)),
    ("cpu_avg", "CPU Avg (%)", lambda r: fmt_num(r.get("cpu_avg"), 1)),
    ("lat_mean", "lat mean (ms)", lambda r: fmt_num(r.get("lat_mean"), 2)),
    ("lat_p50", "p50 (ms)", lambda r: fmt_num(r.get("lat_p50"), 2)),
    ("lat_p95", "p95 (ms)", lambda r: fmt_num(r.get("lat_p95"), 2)),
    ("lat_std", "Lat Std (ms)", lambda r: fmt_num(r.get("lat_std"), 2)),
    ("lat_cv", "Lat CV (%)", lambda r: fmt_pct(r.get("lat_cv"))),
    ("lat_min", "Lat Min (ms)", lambda r: fmt_num(r.get("lat_min"), 2)),
    ("lat_max", "Lat Max (ms)", lambda r: fmt_num(r.get("lat_max"), 2)),
    ("ttft_mean", "TTFT (ms)", lambda r: fmt_num(r.get("ttft_mean"), 2)),
    ("gen_mean", "Gen (ms)", lambda r: fmt_num(r.get("gen_mean"), 2)),
    ("thr", "throughput (req/s)", lambda r: fmt_num(r.get("thr"), 4)),
    ("total_tokens", "total tok", lambda r: fmt(r.get("total_tokens"))),
    ("avg_tokens", "avg tok", lambda r: fmt_num(r.get("avg_tokens"), 1)),
    ("total_input_tokens", "in tok", lambda r: fmt(r.get("total_input_tokens"))),
    ("total_output_tokens", "out tok", lambda r: fmt(r.get("total_output_tokens"))),
    ("toks_total", "tok/s (total)", lambda r: fmt_num(r.get("toks_total"), 2)),
    ("ms_per_tok_total", "ms/tok (total)", lambda r: fmt_num(r.get("ms_per_tok_total"), 2)),
    ("toks_out", "tok/s (out)", lambda r: fmt_num(r.get("toks_out"), 2)),
    ("ms_per_tok_out", "ms/tok (out)", lambda r: fmt_num(r.get("ms_per_tok_out"), 2)),
]


def generate_full_table(title: str, table_rows: List[Dict[str, Any]], table_id: str = "", is_h3: bool = False) -> str:
    """Generate full results table with all columns."""
    tag = "h3" if is_h3 else "h2"
    out = [f'<div class="table-header">']
    out.append(f'<{tag}>{html.escape(title)}</{tag}>')
    out.append(f'<div>')
    out.append(f'<button class="btn-small" onclick="printSection(\'{table_id}\')">Print</button>')
    out.append(f'<button class="btn-small" onclick="exportTableToCSV(\'{table_id}\', \'{table_id}.csv\')">CSV</button>')
    out.append(f'<button class="btn-small" onclick="copyTableToClipboard(\'{table_id}\')">Copy</button>')
    out.append(f'</div></div>')
    out.append(f'<div class="table-wrapper" id="{table_id}">')
    out.append('<table class="full-table">')
    out.append('<thead><tr>')
    for _, label, _ in FULL_TABLE_COLUMNS:
        out.append(f'<th>{html.escape(label)}</th>')
    out.append('</tr></thead><tbody>')
    
    for r in table_rows:
        out.append('<tr>')
        for _, _, render_fn in FULL_TABLE_COLUMNS:
            out.append(f'<td>{render_fn(r)}</td>')
        out.append('</tr>')
    
    out.append('</tbody></table></div>')
    return '\n'.join(out)


def generate_workload_tables(rows: List[Dict[str, Any]]) -> str:
    """Generate separate tables for each workload category."""
    # Group by workload
    by_workload: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        wl = r.get("workload", "unknown")
        if wl not in by_workload:
            by_workload[wl] = []
        by_workload[wl].append(r)
    
    out = ['<h2>Performance by Workload Category</h2>']
    
    for wl in sorted(by_workload.keys()):
        wl_rows = by_workload[wl]
        table_id = f"workload-{wl.replace('_', '-')}"
        out.append(generate_full_table(
            wl.replace("_", " ").title(), 
            wl_rows, 
            table_id,
            is_h3=True
        ))
    
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
            ttft_mean, gen_mean = timing_stats(run_dir / "samples.jsonl")
            
            # Calculate derived metrics
            lat_mean = safe_float(metrics.get("latency_ms_mean"))
            lat_std = safe_float(metrics.get("latency_ms_std"))
            lat_cv = (lat_std / lat_mean * 100) if lat_mean and lat_std else None
            
            # Token throughput metrics
            n = safe_float(metrics.get("n")) or 1
            total_time_s = (lat_mean * n / 1000) if lat_mean else None
            toks_total = (total / total_time_s) if total and total_time_s else None
            toks_out = (total_out / total_time_s) if total_out and total_time_s else None
            ms_per_tok_total = (1000 / toks_total) if toks_total else None
            ms_per_tok_out = (1000 / toks_out) if toks_out else None
            
            # Cost per 1M tokens
            cost_per_1m = (cost / total * 1_000_000) if cost and total else None

            rows.append({
                "run_dir": run_dir.name,
                "ts": ts,
                "backend": cfg.get("backend", ""),
                "backend_model": cfg.get("backend_model", ""),
                "workload": cfg.get("workload", ""),
                "n": metrics.get("n", ""),
                "lat_mean": metrics.get("latency_ms_mean"),
                "lat_p50": metrics.get("latency_ms_p50"),
                "lat_p95": metrics.get("latency_ms_p95"),
                "lat_std": lat_std,
                "lat_cv": lat_cv,
                "lat_min": metrics.get("latency_ms_min"),
                "lat_max": metrics.get("latency_ms_max"),
                "thr": metrics.get("throughput_req_per_s"),
                "accuracy_mean": metrics.get("accuracy_mean"),
                "accuracy_count": metrics.get("accuracy_count", ""),
                "total_tokens": total,
                "avg_tokens": avg,
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
                "cost": cost,
                "cost_per_1m_tokens": cost_per_1m,
                "mem_peak": metrics.get("memory_mb_peak"),
                "cpu_avg": metrics.get("cpu_percent_avg"),
                "ttft_mean": ttft_mean or metrics.get("ttft_ms_mean"),
                "gen_mean": gen_mean or metrics.get("generation_ms_mean"),
                "toks_total": toks_total,
                "toks_out": toks_out,
                "ms_per_tok_total": ms_per_tok_total,
                "ms_per_tok_out": ms_per_tok_out,
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
    .container {{ max-width: 100%; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px 0; color: #1a1a2e; }}
    h2 {{ margin: 30px 0 15px 0; color: #1a1a2e; border-bottom: 2px solid #eee; padding-bottom: 8px; }}
    h3 {{ margin: 20px 0 10px 0; color: #333; }}
    .meta {{ color: #666; margin-bottom: 24px; font-size: 14px; }}
    
    .summary-cards {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 16px;
        margin-bottom: 30px;
    }}
    .card {{
        background: white;
        padding: 16px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .card-value {{ font-size: 24px; font-weight: bold; color: #1a1a2e; }}
    .card-label {{ font-size: 12px; color: #666; margin-top: 4px; }}
    
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
    
    /* Full table with all columns - compact */
    .table-wrapper {{
        overflow-x: auto;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 24px;
    }}
    .table-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }}
    .table-header h2, .table-header h3 {{
        margin: 0;
    }}
    .btn-small {{
        padding: 6px 12px;
        background: #3498db;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 11px;
        margin-left: 8px;
    }}
    .btn-small:hover {{ background: #2980b9; }}
    .full-table {{ 
        border-collapse: collapse; 
        width: max-content;
        min-width: 100%;
        font-size: 9px;
    }}
    .full-table th, .full-table td {{ 
        padding: 4px 6px; 
        text-align: left; 
        border: 1px solid #ddd;
        white-space: nowrap;
    }}
    .full-table th {{ 
        background: #f0f0f0; 
        font-weight: 600;
        color: #1a1a2e;
        position: sticky;
        top: 0;
        font-size: 8px;
    }}
    .full-table tr:nth-child(even) {{ background: #fafafa; }}
    .full-table tr:hover {{ background: #f0f7ff; }}
    
    code {{ 
        background: #f1f3f4; 
        padding: 2px 4px; 
        border-radius: 3px; 
        font-size: 10px;
    }}
    
    /* Print/Export buttons */
    .toolbar {{
        display: flex;
        gap: 10px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }}
    .btn {{
        padding: 10px 20px;
        background: #3498db;
        color: white;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        font-size: 14px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .btn:hover {{ background: #2980b9; }}
    .btn-green {{ background: #2ecc71; }}
    .btn-green:hover {{ background: #27ae60; }}
    .btn-purple {{ background: #9b59b6; }}
    .btn-purple:hover {{ background: #8e44ad; }}
    
    /* Print styles for better screenshot/print */
    @media print {{
        .toolbar {{ display: none !important; }}
        body {{ 
            padding: 10px; 
            background: white;
            font-size: 9px;
        }}
        .summary-cards, .charts-grid, .chart-container {{ 
            break-inside: avoid; 
        }}
        .table-wrapper {{
            overflow: visible;
            box-shadow: none;
        }}
        .full-table {{
            font-size: 8px;
        }}
        .full-table th, .full-table td {{
            padding: 3px 4px;
        }}
        h2 {{ 
            break-before: page;
            margin-top: 10px;
        }}
    }}
    
    @page {{
        size: landscape;
        margin: 0.5cm;
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
    
    <div class="toolbar">
      <button class="btn" onclick="window.print()">
        <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M2.5 8a.5.5 0 1 0 0-1 .5.5 0 0 0 0 1z"/><path d="M5 1a2 2 0 0 0-2 2v2H2a2 2 0 0 0-2 2v3a2 2 0 0 0 2 2h1v1a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2v-1h1a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-1V3a2 2 0 0 0-2-2H5zM4 3a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2H4V3zm1 5a2 2 0 0 0-2 2v1H2a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v-1a2 2 0 0 0-2-2H5zm7 2v3a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1z"/></svg>
        Print Report
      </button>
      <button class="btn btn-green" onclick="exportTableToCSV('all-runs', 'benchmark_all_runs.csv')">
        <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/><path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"/></svg>
        Export CSV
      </button>
      <button class="btn btn-purple" onclick="copyTableToClipboard('all-runs')">
        <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M4 1.5H3a2 2 0 0 0-2 2V14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V3.5a2 2 0 0 0-2-2h-1v1h1a1 1 0 0 1 1 1V14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1h1v-1z"/><path d="M9.5 1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5h3zm-3-1A1.5 1.5 0 0 0 5 1.5v1A1.5 1.5 0 0 0 6.5 4h3A1.5 1.5 0 0 0 11 2.5v-1A1.5 1.5 0 0 0 9.5 0h-3z"/></svg>
        Copy Table
      </button>
    </div>
    
    {generate_summary_cards(rows)}
    
    {generate_accuracy_comparison_table(rows_sorted)}
    
    {generate_latency_comparison_table(rows_sorted)}
    
    {generate_charts_section(rows_sorted)}
    
    {generate_full_table("Latest Runs", latest_rows, "latest-runs")}
    
    {generate_full_table("All Runs", rows_sorted, "all-runs")}
    
    {generate_workload_tables(rows_sorted)}
    
  </div>
  
  <script>
    function exportTableToCSV(tableId, filename) {{
      const table = document.querySelector('#' + tableId + ' table');
      if (!table) {{ alert('Table not found'); return; }}
      
      let csv = [];
      const rows = table.querySelectorAll('tr');
      
      for (const row of rows) {{
        const cols = row.querySelectorAll('th, td');
        const rowData = [];
        for (const col of cols) {{
          let text = col.innerText.replace(/"/g, '""');
          rowData.push('"' + text + '"');
        }}
        csv.push(rowData.join(','));
      }}
      
      const csvContent = csv.join('\\n');
      const blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = filename;
      link.click();
    }}
    
    function copyTableToClipboard(tableId) {{
      const table = document.querySelector('#' + tableId + ' table');
      if (!table) {{ alert('Table not found'); return; }}
      
      let text = [];
      const rows = table.querySelectorAll('tr');
      
      for (const row of rows) {{
        const cols = row.querySelectorAll('th, td');
        const rowData = [];
        for (const col of cols) {{
          rowData.push(col.innerText);
        }}
        text.push(rowData.join('\\t'));
      }}
      
      navigator.clipboard.writeText(text.join('\\n')).then(() => {{
        alert('Table copied to clipboard! Paste in Excel or Google Sheets.');
      }});
    }}
    
    function printSection(tableId) {{
      const tableWrapper = document.getElementById(tableId);
      if (!tableWrapper) {{ alert('Table not found'); return; }}
      
      const printWindow = window.open('', '_blank');
      printWindow.document.write(`
        <html>
        <head>
          <title>Print - ${{tableId}}</title>
          <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 20px; }}
            table {{ border-collapse: collapse; width: 100%; font-size: 8px; }}
            th, td {{ border: 1px solid #ddd; padding: 4px 6px; text-align: left; white-space: nowrap; }}
            th {{ background: #f0f0f0; font-weight: bold; }}
            tr:nth-child(even) {{ background: #fafafa; }}
            @page {{ size: landscape; margin: 0.5cm; }}
          </style>
        </head>
        <body>
          <h2>${{tableId.replace(/-/g, ' ').replace(/workload /i, '')}}</h2>
          ${{tableWrapper.innerHTML}}
          <script>window.onload = function() {{ window.print(); window.close(); }}</` + `script>
        </body>
        </html>
      `);
      printWindow.document.close();
    }}
  </script>
</body>
</html>
"""

    Path(args.out).write_text(html_doc, encoding="utf-8")
    print(f"OK: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
