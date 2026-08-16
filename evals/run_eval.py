"""
Eval runner: sends every item in test_set.json through the actual agent,
scores answer quality with Ragas, and separately verifies RBAC/scope/PII
behaviors that a generic faithfulness score wouldn't catch on its own.

Exits with a non-zero code if anything regresses - this is what CI checks.
"""

import json
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from agents.graph import run_agent_with_context
from rbac.access_control import get_allowed_doc_roles

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
RAGAS_MODEL = os.getenv("RAGAS_MODEL", "openai/gpt-oss-20b")
TEST_SET_PATH = Path(__file__).parent / "test_set.json"
RESULTS_PATH = Path(__file__).parent / "results.json"

# Minimum acceptable scores - tune these as the project matures
FAITHFULNESS_THRESHOLD = 0.7
RELEVANCY_THRESHOLD = 0.6


def load_test_set() -> list[dict]:
    return json.loads(TEST_SET_PATH.read_text())


def run_all(test_items: list[dict]) -> list[dict]:
    results = []
    for item in test_items:
        print(f"Running: [{item['category']}] {item['question'][:60]}...")
        outcome = run_agent_with_context(item["role"], item["question"])
        results.append({**item, **outcome})
    return results


def check_behavioral_regressions(results: list[dict]) -> list[str]:
    """
    Category-specific pass/fail checks Ragas metrics don't cover:
    did out-of-scope get refused, did RBAC actually block unauthorized
    content from being retrieved, did legitimate questions still work.
    """
    failures = []

    for r in results:
        cat = r["category"]

        if cat == "out_of_scope" and not r["out_of_scope"]:
            failures.append(f"REGRESSION: out-of-scope question was answered: '{r['question']}'")

        if cat == "rbac_adversarial":
            allowed = set(get_allowed_doc_roles(r["role"]))
            leaked_roles = [
                role for role in r.get("context_roles", []) if role not in allowed
            ]
            if leaked_roles:
                failures.append(
                    f"REGRESSION: RBAC leak - role '{r['role']}' retrieved unauthorized "
                    f"content tagged {leaked_roles}: '{r['question']}'"
                )

        if cat in ("finance_factual", "hr_factual", "general_factual", "cross_role"):
            if r["out_of_scope"] or not r["contexts"]:
                failures.append(f"REGRESSION: legitimate in-scope question got refused: '{r['question']}'")

    return failures


def run_ragas_eval(results: list[dict]) -> dict:
    """Scores answer quality on the non-refusal items using Ragas + Groq."""
    scorable = [
        r for r in results
        if r["category"] not in ("out_of_scope", "rbac_adversarial") and r["contexts"]
    ]

    if not scorable:
        print("No scorable items for Ragas (all refused/empty) - skipping quality scoring.")
        return {}

    eval_data = [
        {
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": r["contexts"],
            "reference": r["reference"],
        }
        for r in scorable
    ]

    dataset = EvaluationDataset.from_list(eval_data)

    evaluator_llm = LangchainLLMWrapper(ChatGroq(model=RAGAS_MODEL, temperature=0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(max_workers=1, timeout=180, max_retries=3),
    )

    return scores.to_pandas().to_dict(orient="records")


def main():
    test_items = load_test_set()
    results = run_all(test_items)

    behavioral_failures = check_behavioral_regressions(results)
    ragas_scores = run_ragas_eval(results)

    RESULTS_PATH.write_text(json.dumps({
        "results": [{k: v for k, v in r.items() if k != "contexts"} for r in results],
        "ragas_scores": ragas_scores,
        "behavioral_failures": behavioral_failures,
    }, indent=2))

    print("\n" + "=" * 60)
    print("BEHAVIORAL CHECKS (RBAC / scope / refusal correctness)")
    print("=" * 60)
    if behavioral_failures:
        for f in behavioral_failures:
            print(f"  ❌ {f}")
    else:
        print("  ✅ All behavioral checks passed.")

    print("\n" + "=" * 60)
    print("RAGAS QUALITY SCORES (faithfulness / answer relevancy)")
    print("=" * 60)

    quality_failures = []
    skipped_faithfulness = 0
    skipped_relevancy = 0

    if ragas_scores:
        raw_faithfulness = [r.get("faithfulness") for r in ragas_scores]
        raw_relevancy = [r.get("answer_relevancy") for r in ragas_scores]

        faithfulness_scores = [
            v for v in raw_faithfulness if v is not None and not math.isnan(v)
        ]
        relevancy_scores = [
            v for v in raw_relevancy if v is not None and not math.isnan(v)
        ]

        skipped_faithfulness = len(raw_faithfulness) - len(faithfulness_scores)
        skipped_relevancy = len(raw_relevancy) - len(relevancy_scores)

        avg_faithfulness = (
            sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
        )
        avg_relevancy = (
            sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0
        )

        print(f"  Average faithfulness: {avg_faithfulness:.2f} (threshold: {FAITHFULNESS_THRESHOLD})"
              + (f"  [{skipped_faithfulness} item(s) timed out, excluded]" if skipped_faithfulness else ""))
        print(f"  Average answer relevancy: {avg_relevancy:.2f} (threshold: {RELEVANCY_THRESHOLD})"
              + (f"  [{skipped_relevancy} item(s) timed out, excluded]" if skipped_relevancy else ""))

        if not faithfulness_scores:
            quality_failures.append("All faithfulness scores failed/timed out - could not evaluate")
        elif avg_faithfulness < FAITHFULNESS_THRESHOLD:
            quality_failures.append(f"Average faithfulness {avg_faithfulness:.2f} below threshold {FAITHFULNESS_THRESHOLD}")

        if not relevancy_scores:
            quality_failures.append("All relevancy scores failed/timed out - could not evaluate")
        elif avg_relevancy < RELEVANCY_THRESHOLD:
            quality_failures.append(f"Average relevancy {avg_relevancy:.2f} below threshold {RELEVANCY_THRESHOLD}")

    all_failures = behavioral_failures + quality_failures
    print(f"\nResults saved to {RESULTS_PATH}")

    if all_failures:
        print(f"\n❌ EVAL FAILED — {len(all_failures)} issue(s) found.")
        sys.exit(1)
    else:
        print("\n✅ EVAL PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()