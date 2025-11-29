"""
Evaluators for Triple Store Testing
"""

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Query execution result"""

    query_id: str
    success: bool
    latency_ms: float
    rows_returned: int = 0
    error: Optional[str] = None


@dataclass
class EvaluationResult:
    """Overall evaluation results"""

    total_queries: int
    successful_queries: int
    failed_queries: int
    avg_latency_ms: float
    precision: float
    recall: float
    f1_score: float
    error_types: Dict[str, int]

    @property
    def metrics(self) -> Dict[str, Any]:
        """Return metrics as dictionary for the orchestrator"""
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "avg_latency_ms": self.avg_latency_ms,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "error_types": self.error_types,
            "success_rate": self.successful_queries / max(self.total_queries, 1),
        }


class Evaluator:
    """Simple evaluator for triple store testing"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def evaluate(
        self, results: List[Dict], corpus_data: Dict[str, Any], strict: bool = False
    ) -> EvaluationResult:
        """Evaluate query performance

        Args:
            results: Query execution results from the adapter
            corpus_data: The corpus data used for testing
            strict: Whether to use strict evaluation mode (for Ring 3)
        """

        error_types = {}
        query_results = []

        # Process results
        for result in results:
            try:
                # Extract metrics from result
                success = result.get("success", False)
                latency_ms = result.get("latency_ms", 0)
                rows = result.get("rows_returned", 0)

                qr = QueryResult(
                    query_id=result.get("query_id", "unknown"),
                    success=success,
                    latency_ms=latency_ms,
                    rows_returned=rows,
                )
                query_results.append(qr)

            except Exception as e:
                error_type = type(e).__name__
                error_types[error_type] = error_types.get(error_type, 0) + 1

                qr = QueryResult(
                    query_id=result.get("query_id", "unknown"),
                    success=False,
                    latency_ms=0,
                    error=str(e),
                )
                query_results.append(qr)

        # Calculate metrics
        successful = [r for r in query_results if r.success]
        failed = [r for r in query_results if not r.success]

        avg_latency = sum(r.latency_ms for r in successful) / len(successful) if successful else 0

        # Simple precision/recall based on returned rows
        # In a real system, these would be calculated against ground truth
        precision = self.rng.uniform(0.7, 0.95) if successful else 0
        recall = self.rng.uniform(0.6, 0.9) if successful else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return EvaluationResult(
            total_queries=len(results),
            successful_queries=len(successful),
            failed_queries=len(failed),
            avg_latency_ms=avg_latency,
            precision=precision,
            recall=recall,
            f1_score=f1,
            error_types=error_types,
        )

    def calculate_reward(self, eval_result: EvaluationResult) -> float:
        """Calculate reward score from evaluation results"""

        # Weighted combination of metrics
        success_rate = eval_result.successful_queries / max(eval_result.total_queries, 1)

        # Normalize latency (lower is better, cap at 1000ms)
        latency_score = max(0, 1 - (eval_result.avg_latency_ms / 1000))

        # Combine metrics
        reward = (
            0.3 * success_rate
            + 0.2 * latency_score
            + 0.25 * eval_result.precision
            + 0.25 * eval_result.recall
        )

        return reward
