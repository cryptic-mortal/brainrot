"""Generate TradingAgents allocation reports from SmartFolio outputs.

This helper reads an allocation CSV (Ticker, Weight) produced by SmartFolio,
invokes the TradingAgents library for each holding, and writes Markdown
summaries to ``explainability_results/trading_agent_reports`` by default.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class AllocationRow:
    ticker: str
    weight: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run TradingAgents reports for each SmartFolio allocation entry. "
            "Outputs Markdown files per ticker and an index summary."
        )
    )
    parser.add_argument(
        "--allocation-csv",
        default="allocation.csv",
        help="Path to the SmartFolio allocation CSV (defaults to allocation.csv in the repo root).",
    )
    parser.add_argument(
        "--trading-agent-root",
        default=None,
        help=(
            "Path to the trading_agent repository. If omitted, the script assumes "
            "../trading_agent relative to this file."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="explainability_results/trading_agent_reports",
        help="Directory where Markdown reports should be written.",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Override the as-of date (YYYY-MM-DD) passed to TradingAgents.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Number of calendar days the news agent should scan.",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=8,
        help="Maximum number of news headlines fetched per ticker.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM-generated summaries when the relevant API key is exported.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Optional override for the TradingAgents LLM model name.",
    )
    parser.add_argument(
        "--include-components",
        action="store_true",
        help="Include the full fundamental and news detail sections in each report.",
    )
    parser.add_argument(
        "--omit-metrics",
        action="store_true",
        help="Skip the fundamentals metrics table inside component sections.",
    )
    parser.add_argument(
        "--omit-articles",
        action="store_true",
        help="Skip the news headline table inside component sections.",
    )
    parser.add_argument(
        "--print-summaries",
        action="store_true",
        help="Echo each unified summary to stdout in addition to writing files.",
    )
    return parser.parse_args()


def resolve_trading_agent_root(explicit: Optional[str]) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parents[2] / "trading_agent"
    if not (root / "tradingagents").exists():
        raise RuntimeError(
            f"TradingAgents package not found under {root}. Use --trading-agent-root to point to the repo."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def load_allocations(csv_path: Path) -> List[AllocationRow]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Allocation file not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Allocation CSV is empty or missing headers.")
        expected = {"Ticker", "Weight"}
        missing = expected.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Allocation CSV must contain columns {sorted(expected)}; missing {sorted(missing)}."
            )
        rows: List[AllocationRow] = []
        for line in reader:
            ticker = (line.get("Ticker") or "").strip()
            weight_raw = line.get("Weight")
            if not ticker:
                continue
            try:
                weight = float(weight_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid weight for {ticker}: {weight_raw}") from exc
            rows.append(AllocationRow(ticker=ticker, weight=weight))
    if not rows:
        raise ValueError("Allocation CSV did not contain any ticker rows.")
    return rows


def warn_missing_llm_env() -> None:
    keys = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY")
    if any(os.environ.get(key) for key in keys):
        return
    print(
        "[WARN] --llm requested but no GOOGLE_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY is set.",
        file=sys.stderr,
    )


def generate_reports(
    allocations: Iterable[AllocationRow],
    *,
    as_of: Optional[str],
    lookback_days: int,
    max_articles: int,
    use_llm: bool,
    llm_model: Optional[str],
    include_components: bool,
    include_metrics: bool,
    include_articles: bool,
) -> List[tuple[AllocationRow, str]]:
    from tradingagents.combined_weight_agent import WeightSynthesisAgent

    agent = WeightSynthesisAgent()
    outputs: List[tuple[AllocationRow, str]] = []
    for row in allocations:
        try:
            report = agent.generate_report(
                ticker=row.ticker,
                weight=row.weight,
                as_of=as_of,
                lookback_days=lookback_days,
                max_articles=max_articles,
                use_llm=use_llm,
                llm_model=llm_model,
            )
        except Exception as err:  # noqa: BLE001
            print(f"[ERROR] Failed to generate report for {row.ticker}: {err}", file=sys.stderr)
            continue
        markdown = report.to_markdown(
            include_components=include_components,
            include_metrics=include_metrics,
            include_articles=include_articles,
        )
        outputs.append((row, markdown))
    return outputs


def write_reports(
    reports: Iterable[tuple[AllocationRow, str]],
    output_dir: Path,
) -> List[tuple[AllocationRow, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[tuple[AllocationRow, Path]] = []
    for row, markdown in reports:
        outfile = output_dir / f"{row.ticker}.md"
        outfile.write_text(markdown, encoding="utf-8")
        written.append((row, outfile))
    return written


def write_index(entries: Iterable[tuple[AllocationRow, Path]], output_dir: Path) -> None:
    entries_list = list(entries)
    if not entries_list:
        return
    index_path = output_dir / "index.md"
    lines = [
        "# TradingAgents Allocation Review\n",
        "\n",
        "| Ticker | Weight | Report |\n",
        "| --- | --- | --- |\n",
    ]
    for row, path in entries_list:
        rel_path = path.name
        lines.append(f"| {row.ticker} | {row.weight:.4f} | [{rel_path}]({rel_path}) |\n")
    index_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    allocation_path = Path(args.allocation_csv).expanduser()
    if not allocation_path.is_absolute():
        allocation_path = Path.cwd() / allocation_path

    trading_root = resolve_trading_agent_root(args.trading_agent_root)
    # Keep import side-effects local to ensure sys.path adjustments are applied.
    _ = trading_root  # placate linters about unused value

    allocations = load_allocations(allocation_path)

    if args.llm:
        warn_missing_llm_env()

    include_metrics = not args.omit_metrics
    include_articles = not args.omit_articles

    reports = generate_reports(
        allocations,
        as_of=args.as_of,
        lookback_days=args.lookback_days,
        max_articles=args.max_articles,
        use_llm=args.llm,
        llm_model=args.llm_model,
        include_components=args.include_components,
        include_metrics=include_metrics,
        include_articles=include_articles,
    )

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    written = write_reports(reports, output_dir)
    write_index(written, output_dir)

    if args.print_summaries:
        for row, markdown in reports:
            print(f"\n=== {row.ticker} ({row.weight:.2%}) ===")
            in_summary = False
            for line in markdown.splitlines():
                if line.startswith("## Unified Summary"):
                    in_summary = True
                    continue
                if in_summary and line.startswith("## "):
                    break
                if in_summary and line.startswith("- "):
                    print(line)

    print(f"Generated {len(written)} TradingAgents report(s) under {output_dir}")


if __name__ == "__main__":
    main()
