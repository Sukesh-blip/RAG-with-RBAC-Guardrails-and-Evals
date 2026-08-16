"""
Cost monitoring: a LangChain callback that captures token usage from every
LLM call in a request, plus a persistent log and a threshold alert.

Pricing verified against Groq's published rates (per 1M tokens) as of
Aug 2026 - update MODEL_PRICING if Groq changes rates.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

USAGE_LOG_PATH = Path(__file__).parent / "usage_log.jsonl"

# USD per 1M tokens (input, output)
MODEL_PRICING = {
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
}

# Alert if a single request's estimated cost exceeds this (USD)
PER_REQUEST_ALERT_THRESHOLD = 0.01
# Alert if the running daily total exceeds this (USD)
DAILY_ALERT_THRESHOLD = 1.00


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return 0.0
    return (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]


class TokenUsageCallback(BaseCallbackHandler):
    """
    Attach to LLM calls via config={"callbacks": [tracker]} on .invoke().
    One instance should be created per logical request (e.g. one per
    run_agent() call) so it accumulates usage across every node - scope
    check, router, critic, generate - that request touches.
    """

    def __init__(self):
        self.calls: list[dict] = []

    def on_llm_end(self, response, **kwargs) -> None:
        try:
            generation = response.generations[0][0]
            model = getattr(generation.message, "response_metadata", {}).get("model_name", "unknown")
            usage = getattr(generation.message, "usage_metadata", None)

            if usage:
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
            else:
                # Fallback for providers/paths that don't set usage_metadata
                token_usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
                input_tokens = token_usage.get("prompt_tokens", 0)
                output_tokens = token_usage.get("completion_tokens", 0)

            self.calls.append({
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            })
        except (IndexError, AttributeError, KeyError):
            pass

    def total_tokens(self) -> int:
        return sum(c["input_tokens"] + c["output_tokens"] for c in self.calls)

    def total_cost(self) -> float:
        return sum(
            estimate_cost(c["model"], c["input_tokens"], c["output_tokens"])
            for c in self.calls
        )

    def summary(self) -> dict:
        return {
            "num_llm_calls": len(self.calls),
            "total_tokens": self.total_tokens(),
            "estimated_cost_usd": round(self.total_cost(), 6),
        }


def log_usage(role: str, query: str, summary: dict) -> list[str]:
    """
    Appends a usage record to the persistent log. Returns a list of any
    alert messages triggered (per-request or daily threshold exceeded).
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "query": query[:100],
        **summary,
    }

    with open(USAGE_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    alerts = []
    if record["estimated_cost_usd"] > PER_REQUEST_ALERT_THRESHOLD:
        alerts.append(
            f"Single request cost ${record['estimated_cost_usd']:.4f} exceeds "
            f"per-request threshold ${PER_REQUEST_ALERT_THRESHOLD}"
        )

    today_total = _today_total_cost()
    if today_total > DAILY_ALERT_THRESHOLD:
        alerts.append(
            f"Today's cumulative cost ${today_total:.4f} exceeds daily "
            f"threshold ${DAILY_ALERT_THRESHOLD}"
        )

    return alerts


def _today_total_cost() -> float:
    if not USAGE_LOG_PATH.exists():
        return 0.0

    today = datetime.now(timezone.utc).date().isoformat()
    total = 0.0
    with open(USAGE_LOG_PATH) as f:
        for line in f:
            try:
                record = json.loads(line)
                if record["timestamp"].startswith(today):
                    total += record.get("estimated_cost_usd", 0)
            except (json.JSONDecodeError, KeyError):
                continue
    return total


def get_usage_summary() -> dict:
    """Returns aggregate usage stats for the /cost/usage endpoint."""
    if not USAGE_LOG_PATH.exists():
        return {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "today_cost_usd": 0.0,
        }

    total_tokens = 0
    total_cost = 0.0
    total_requests = 0

    with open(USAGE_LOG_PATH) as f:
        for line in f:
            try:
                record = json.loads(line)
                total_requests += 1
                total_tokens += record.get("total_tokens", 0)
                total_cost += record.get("estimated_cost_usd", 0)
            except json.JSONDecodeError:
                continue

    return {
        "total_requests": total_requests,
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "today_cost_usd": round(_today_total_cost(), 6),
    }