"""
Evaluation framework for comparing vector-only vs graph-enhanced retrieval.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field

from backend.llm_provider import BaseLLMProvider
from backend.retrieval.graph_rag import vector_only_query, graph_enhanced_query
from backend.graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


JUDGE_PROMPT = """You are an evaluation judge. Compare the given answer to the expected answer and rate it.

## Question
{question}

## Expected Answer
{expected}

## Actual Answer
{actual}

## Rating Criteria
Rate the actual answer on a scale of 1-5:
1 = Completely wrong or irrelevant
2 = Partially relevant but mostly incorrect
3 = Somewhat correct but missing key information
4 = Mostly correct with minor omissions
5 = Fully correct and comprehensive

Return a JSON object:
{{"score": <1-5>, "reasoning": "Brief explanation"}}"""


@dataclass
class EvalResult:
    question: str
    question_type: str
    expected_answer: str
    vector_answer: str = ""
    graph_answer: str = ""
    vector_score: int = 0
    graph_score: int = 0
    vector_latency_ms: float = 0
    graph_latency_ms: float = 0
    vector_reasoning: str = ""
    graph_reasoning: str = ""


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)
    vector_avg_score: float = 0
    graph_avg_score: float = 0
    vector_avg_latency_ms: float = 0
    graph_avg_latency_ms: float = 0

    def compute_summary(self) -> None:
        if not self.results:
            return
        n = len(self.results)
        self.vector_avg_score = sum(r.vector_score for r in self.results) / n
        self.graph_avg_score = sum(r.graph_score for r in self.results) / n
        self.vector_avg_latency_ms = sum(r.vector_latency_ms for r in self.results) / n
        self.graph_avg_latency_ms = sum(r.graph_latency_ms for r in self.results) / n

    def to_markdown(self) -> str:
        self.compute_summary()
        lines = [
            "# GraphRAG Evaluation Report\n",
            "## Summary\n",
            "| Metric | Vector-Only | Graph-Enhanced |",
            "|--------|-------------|----------------|",
            f"| Avg Score (1-5) | {self.vector_avg_score:.2f} | {self.graph_avg_score:.2f} |",
            f"| Avg Latency (ms) | {self.vector_avg_latency_ms:.0f} | {self.graph_avg_latency_ms:.0f} |",
            "",
            "## Per-Question Results\n",
            "| # | Type | Question | Vector Score | Graph Score | Winner |",
            "|---|------|----------|-------------|-------------|--------|",
        ]

        for i, r in enumerate(self.results, 1):
            winner = "Graph" if r.graph_score > r.vector_score else (
                "Vector" if r.vector_score > r.graph_score else "Tie"
            )
            q_short = r.question[:60] + "..." if len(r.question) > 60 else r.question
            lines.append(
                f"| {i} | {r.question_type} | {q_short} | {r.vector_score} | {r.graph_score} | {winner} |"
            )

        # Breakdown by question type
        single_hop = [r for r in self.results if r.question_type == "single_hop"]
        multi_hop = [r for r in self.results if r.question_type == "multi_hop"]

        lines.append("\n## Breakdown by Question Type\n")
        if single_hop:
            v_avg = sum(r.vector_score for r in single_hop) / len(single_hop)
            g_avg = sum(r.graph_score for r in single_hop) / len(single_hop)
            lines.append(f"**Single-hop** ({len(single_hop)} questions): Vector={v_avg:.2f}, Graph={g_avg:.2f}")
        if multi_hop:
            v_avg = sum(r.vector_score for r in multi_hop) / len(multi_hop)
            g_avg = sum(r.graph_score for r in multi_hop) / len(multi_hop)
            lines.append(f"**Multi-hop** ({len(multi_hop)} questions): Vector={v_avg:.2f}, Graph={g_avg:.2f}")

        return "\n".join(lines)


async def evaluate(
    eval_file: Path,
    llm: BaseLLMProvider,
    qdrant_client,
    neo4j: Neo4jClient,
) -> EvalReport:
    """Run evaluation comparing vector-only vs graph-enhanced retrieval."""
    with open(eval_file, "r") as f:
        eval_data = json.load(f)

    questions = eval_data.get("questions", [])
    report = EvalReport()

    for i, q in enumerate(questions):
        logger.info("Evaluating question %d/%d: %s", i + 1, len(questions), q["question"][:80])

        result = EvalResult(
            question=q["question"],
            question_type=q.get("type", "single_hop"),
            expected_answer=q.get("expected_answer", ""),
        )

        # Vector-only query
        t0 = time.time()
        try:
            vector_response = await vector_only_query(q["question"], llm, qdrant_client)
            result.vector_answer = vector_response.answer
        except Exception as e:
            result.vector_answer = f"Error: {str(e)}"
        result.vector_latency_ms = (time.time() - t0) * 1000

        # Graph-enhanced query
        t0 = time.time()
        try:
            graph_response = await graph_enhanced_query(q["question"], llm, qdrant_client, neo4j)
            result.graph_answer = graph_response.answer
        except Exception as e:
            result.graph_answer = f"Error: {str(e)}"
        result.graph_latency_ms = (time.time() - t0) * 1000

        # Judge both answers
        if result.expected_answer:
            for mode in ("vector", "graph"):
                actual = result.vector_answer if mode == "vector" else result.graph_answer
                judge_prompt = JUDGE_PROMPT.format(
                    question=result.question,
                    expected=result.expected_answer,
                    actual=actual,
                )
                try:
                    judge_result = await llm.generate_structured(judge_prompt)
                    score = int(judge_result.get("score", 0))
                    reasoning = judge_result.get("reasoning", "")
                    if mode == "vector":
                        result.vector_score = score
                        result.vector_reasoning = reasoning
                    else:
                        result.graph_score = score
                        result.graph_reasoning = reasoning
                except Exception as e:
                    logger.warning("Judge failed for %s mode: %s", mode, str(e))

        report.results.append(result)

    report.compute_summary()
    logger.info(
        "Evaluation complete: Vector avg=%.2f, Graph avg=%.2f",
        report.vector_avg_score, report.graph_avg_score,
    )
    return report
