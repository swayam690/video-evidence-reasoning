"""Summarize and compare ZoomVid evaluation results across modules and budgets.

Scans an evaluation directory for JSON reports and prints/saves a consolidated
markdown comparison table.

Usage:
    python -m zoomvid.eval.report --eval-dir eval_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_all_reports(eval_dir: Path) -> List[Dict[str, Any]]:
    """Scan directory recursively for eval report JSON files."""
    reports = []
    for json_file in eval_dir.rglob("*.json"):
        if json_file.name in ("eval_summary.json", "manifest.json"):
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Verify it has standard EvalReport fields
            if "module_name" in data and "accuracy" in data and "token_budget" in data:
                reports.append(data)
        except Exception:
            continue
    return reports


def format_markdown_table(reports: List[Dict[str, Any]]) -> str:
    """Format reports into a sorted GitHub-flavored markdown comparison table."""
    if not reports:
        return "No evaluation reports found."

    # Sort by budget ascending, then accuracy descending
    sorted_reports = sorted(
        reports,
        key=lambda r: (r.get("token_budget", 0), -r.get("accuracy", 0.0), r.get("module_name", "")),
    )

    header = (
        "| Module | Budget | Samples | Accuracy | Avg Tokens | Avg Latency |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    )
    rows = []
    for r in sorted_reports:
        name = r.get("module_name", "Unknown")
        budget = r.get("token_budget", 0)
        samples = r.get("num_samples", 0)
        acc = r.get("accuracy", 0.0)
        tokens = r.get("avg_token_count", 0.0)
        lat = r.get("avg_latency_ms", 0.0)
        rows.append(
            f"| {name} | {budget} | {samples} | {acc:.2%} | {tokens:.0f} | {lat:.1f} ms |"
        )

    return header + "\n" + "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="ZoomVid: Summarize Evaluation Results")
    parser.add_argument("--eval-dir", type=str, default="eval_results", help="Evaluation results directory")
    parser.add_argument("--save-md", type=str, default=None, help="Optional output path for markdown table")
    args = parser.parse_args()

    eval_path = Path(args.eval_dir)
    if not eval_path.exists():
        print(f"Directory {eval_path} does not exist.")
        return

    reports = load_all_reports(eval_path)
    table = format_markdown_table(reports)

    print("\n" + "=" * 60)
    print(" ZoomVid Benchmark Comparison Table")
    print("=" * 60)
    print(table)
    print("=" * 60 + "\n")

    if args.save_md:
        out_file = Path(args.save_md)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"Saved summary table to {out_file}")


if __name__ == "__main__":
    main()
