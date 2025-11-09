"""LLM-powered explanation generator for SmartFolio interpretability outputs.

This script mirrors the trading_agent LLM utilities: it loads an
``explainability_snapshot_*.json`` file, assembles a structured context, and
requests a narrative from the configured LLM (Gemini/OpenAI).

Usage example:

```
/home/pushpendras0026/inter/venv310/bin/python tools/explainability_llm_agent.py \
  --snapshot explainability_results/explainability_snapshot_custom.json \
  --llm --print
```

Set ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY`` (for Gemini) or ``OPENAI_API_KEY``
before invoking ``--llm``. The output is printed to stdout and optionally saved
to ``explainability_results/explainability_narrative.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


SYSTEM_PROMPT = (
    "You are the SmartFolio interpretability analyst responsible for explaining allocations "
    "produced by a mimic decision-tree model labelled 'model_name'. Your audience is a "
    "portfolio manager who needs a practical, data-grounded explanation of why the policy "
    "allocates capital the way it does. Use crisp language, connect tree splits and attention "
    "patterns to portfolio outcomes, and flag any data gaps instead of guessing."
)


@dataclass
class SnapshotContext:
    metadata: Dict[str, str]
    tree_summary: Dict[str, object]
    attention_summary: Dict[str, object]
    synopsis_tree: Dict[str, object]
    synopsis_attention: Dict[str, object]
    raw_json: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a natural-language explanation for a SmartFolio explainability snapshot."
    )
    parser.add_argument(
        "--snapshot",
        default="explainability_results/explainability_snapshot_custom.json",
        help="Path to the explainability snapshot JSON file.",
    )
    parser.add_argument(
        "--trading-agent-root",
        default=None,
        help="Path to the trading_agent repo (defaults to ../trading_agent).",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use the configured LLM (Gemini/OpenAI) for the narrative.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Optional override for the LLM model name.",
    )
    parser.add_argument(
        "--output",
        default="explainability_results/explainability_narrative.md",
        help="File to write the narrative to (default inside explainability_results).",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Echo the narrative to stdout.",
    )
    return parser.parse_args()


def ensure_trading_agent_import(root_override: Optional[str]) -> None:
    if "tradingagents" in sys.modules:
        return
    if root_override:
        root = Path(root_override).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parents[2] / "trading_agent"
    if not (root / "tradingagents").exists():
        raise RuntimeError(
            f"Unable to locate trading_agent package under {root}. Provide --trading-agent-root."
        )
    sys.path.insert(0, str(root))


def load_snapshot(path: Path) -> SnapshotContext:
    if not path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {path}")
    raw_text = path.read_text(encoding="utf-8")
    data = json.loads(raw_text)

    metadata = {
        "generated_at": str(data.get("generated_at")),
        "market": str(data.get("market")),
        "model_path": str(data.get("model_path")),
        "dataset_dir": str(data.get("dataset_dir")),
    }

    tree_summary = data.get("tree_summary", {})
    attention_summary = data.get("attention_summary", {})
    synopsis = data.get("synopsis", {})
    synopsis_tree = synopsis.get("tree", {}) if isinstance(synopsis, dict) else {}
    synopsis_attention = synopsis.get("attention", {}) if isinstance(synopsis, dict) else {}

    return SnapshotContext(
        metadata=metadata,
        tree_summary=tree_summary if isinstance(tree_summary, dict) else {},
        attention_summary=attention_summary if isinstance(attention_summary, dict) else {},
        synopsis_tree=synopsis_tree if isinstance(synopsis_tree, dict) else {},
        synopsis_attention=synopsis_attention if isinstance(synopsis_attention, dict) else {},
        raw_json=raw_text,
    )


def _format_percent(value: Optional[float], default: str = "n/a") -> str:
    if not isinstance(value, (int, float)):
        return default
    return f"{float(value):.2%}"


def describe_top_assets(context: SnapshotContext, limit: int = 5) -> str:
    top_tickers = context.synopsis_tree.get("top_assets")
    if isinstance(top_tickers, list) and top_tickers:
        rows = []
        for item in top_tickers[:limit]:
            ticker = item.get("ticker")
            weight = _format_percent(item.get("avg_weight"))
            rank = item.get("rank")
            rows.append(f"#{rank}: {ticker} (avg weight {weight})")
        return "Top weighted positions from the surrogate tree:\n" + "\n".join(rows)

    tickers = context.tree_summary.get("top_tickers")
    weights = context.tree_summary.get("avg_weights")
    if isinstance(tickers, list) and isinstance(weights, list) and tickers:
        pairs = zip(tickers[:limit], weights[:limit])
        rows = [f"{t} (avg weight {_format_percent(w)})" for t, w in pairs]
        return "Top weighted positions from the surrogate tree:\n" + "\n".join(rows)
    return "Top weighted positions unavailable in snapshot."


def describe_attention_edges(context: SnapshotContext, limit: int = 3) -> str:
    top_edges = context.synopsis_attention.get("top_edges")
    if not isinstance(top_edges, dict):
        return "Attention edge details missing."

    lines: List[str] = []
    for label, edges in top_edges.items():
        if not isinstance(edges, list) or not edges:
            continue
        subset = edges[:limit]
        entries = []
        for edge in subset:
            source = edge.get("source_ticker") or edge.get("source_index")
            target = edge.get("target_ticker") or edge.get("target_index")
            value = edge.get("mean_attention")
            if isinstance(value, (int, float)):
                entries.append(f"{source} → {target} ({float(value):.3f})")
            else:
                entries.append(f"{source} → {target} (attention n/a)")
        if entries:
            lines.append(f"{label.title()} attention focus: " + ", ".join(entries))
    return "\n".join(lines) if lines else "Attention edge details missing."


def describe_semantic_mix(context: SnapshotContext) -> str:
    semantic = context.synopsis_attention.get("semantic_mean")
    if isinstance(semantic, dict) and semantic:
        parts = [f"{label}: {share:.1%}" for label, share in semantic.items()]
        return "Semantic attention distribution: " + ", ".join(parts)

    labels = context.attention_summary.get("semantic_labels")
    shares = context.attention_summary.get("semantic_mean")
    if isinstance(labels, list) and isinstance(shares, list) and labels:
        pairs = [f"{lab}: {float(val):.1%}" for lab, val in zip(labels, shares)]
        return "Semantic attention distribution: " + ", ".join(pairs)
    return "Semantic attention distribution unavailable."


def assemble_prompt(context: SnapshotContext) -> str:
    metadata_lines = [
        f"Generated at: {context.metadata.get('generated_at')}",
        f"Market: {context.metadata.get('market')}",
        f"Model path: {context.metadata.get('model_path')}",
        f"Dataset dir: {context.metadata.get('dataset_dir')}",
    ]

    global_r2 = context.tree_summary.get("global_r2")
    num_stocks = context.tree_summary.get("num_stocks")
    input_dim = context.tree_summary.get("input_dim")

    tree_highlights = [
        f"Global R^2: {float(global_r2):.3f}" if isinstance(global_r2, (int, float)) else "Global R^2 unavailable",
        f"Universe size: {num_stocks}" if isinstance(num_stocks, int) else "Universe size unavailable",
        f"Per-stock feature count: {input_dim}" if isinstance(input_dim, int) else "Feature count unavailable",
        describe_top_assets(context),
    ]

    attention_highlights = [
        describe_semantic_mix(context),
        describe_attention_edges(context),
    ]

    prompt = "\n".join(
        [
            SYSTEM_PROMPT,
            "\nModel provenance: This explainability snapshot comes from a mimic tree-based model labelled \"model_name\" that approximates the SmartFolio HGAT policy. Explain why the allocations emerge the way they do, grounding each point in the snapshot data.",
            "\nSnapshot metadata:\n" + "\n".join(metadata_lines),
            "\nDecision tree surrogate notes:\n" + "\n".join(tree_highlights),
            "\nAttention head summary:\n" + "\n".join(attention_highlights),
            "\nInstructions: Produce 4-6 bullets. Each bullet must begin with a bold heading followed by one or two sentences."
            "\nCover in order: (1) overall signal quality + limitations, (2) how the mimic tree's splits and top stocks explain the observed allocations, "
            "(3) what the attention channels reveal about cross-stock relationships, (4) concrete implications for portfolio positioning and risk controls, "
            "and (5) any missing or inconsistent data that might weaken the story. If information is absent, say so explicitly instead of inventing details.",
            "Avoid raw JSON in the final answer; translate everything into plain language.\n"
            "Full explainability snapshot JSON follows verbatim:\n" + context.raw_json,
        ]
    )

    return prompt


def llm_narrative(prompt: str, *, model: Optional[str], max_points: int = 6) -> Optional[List[str]]:
    try:
        from tradingagents import llm_client  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover - import checked earlier
        raise RuntimeError(
            "tradingagents.llm_client could not be imported. Set --trading-agent-root to the repo path."
        ) from exc

    return llm_client.generate_bullets(
        prompt,
        max_points=max_points,
        model=model,
    )


def fallback_narrative(context: SnapshotContext) -> List[str]:
    lines = [
        "**Executive Summary:** LLM output unavailable; using deterministic narrative about the mimic tree model ('model_name'). Treat these points as descriptive signals for why the allocation looks the way it does.",
    ]
    global_r2 = context.tree_summary.get("global_r2")
    if isinstance(global_r2, (int, float)):
        summary_tickers = context.tree_summary.get("top_tickers")
        if isinstance(summary_tickers, list) and summary_tickers:
            front_three = ", ".join(summary_tickers[:3])
        else:
            front_three = "the highlighted holdings"
        lines.append(
            f"**Tree Perspective:** The decision tree explains roughly {float(global_r2):.1%} of the variance. Its top-weight stocks suggest the policy leans on stable names such as {front_three}."
        )
    semantic = describe_semantic_mix(context)
    lines.append(
        f"**Attention Perspective:** {semantic}. Dominant attention channels hint at which relationship graphs (positive, negative, industry, self) influenced the HGAT policy."
    )
    edges = describe_attention_edges(context)
    lines.append(
        f"**Action Items:** Monitor the highlighted attention edges ({edges}) to ensure they align with risk expectations, and combine with qualitative review before adjusting allocations."
    )
    return lines


def format_output(sections: Iterable[str]) -> str:
    return "\n".join(f"- {section}" for section in sections)


def main() -> None:
    args = parse_args()

    snapshot_path = Path(args.snapshot).expanduser()
    if not snapshot_path.is_absolute():
        snapshot_path = Path.cwd() / snapshot_path

    ensure_trading_agent_import(args.trading_agent_root)
    context = load_snapshot(snapshot_path)

    prompt = assemble_prompt(context)

    debug_prompt_path = Path(args.output).expanduser()
    if not debug_prompt_path.is_absolute():
        debug_prompt_path = Path.cwd() / debug_prompt_path
    debug_prompt_path = debug_prompt_path.with_name("input_tree_llm.txt")
    debug_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    debug_prompt_path.write_text(prompt + "\n", encoding="utf-8")

    sections: List[str]
    if args.llm:
        sections = llm_narrative(prompt, model=args.llm_model)
        if sections is None:
            from tradingagents import llm_client  # type: ignore[import-untyped]

            detail = llm_client.LAST_LLM_ERROR or "LLM call returned no content"
            raise RuntimeError(f"LLM explanation failed: {detail}")
        if not sections:
            from tradingagents import llm_client  # type: ignore[import-untyped]

            detail = llm_client.LAST_LLM_ERROR or "LLM call produced an empty response"
            raise RuntimeError(f"LLM explanation failed: {detail}")
        if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            print(
                "[WARN] --llm set but no API key detected in environment; export GOOGLE_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY.",
                file=sys.stderr,
            )
    else:
        sections = fallback_narrative(context)

    output_text = format_output(sections)

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text + "\n", encoding="utf-8")

    if args.print:
        print(output_text)

    print(f"Narrative written to {output_path}")


if __name__ == "__main__":
    main()
