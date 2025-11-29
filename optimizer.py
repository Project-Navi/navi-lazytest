"""
Optimizer for Triple Store Configuration Tuning
Uses Upper Confidence Bound (UCB) algorithm for multi-armed bandit optimization
"""

import json
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set

import numpy as np
from pydantic import BaseModel, confloat, conint

# ============================================================================
# Configuration Space Definition
# ============================================================================


class OptimizerConfig(BaseModel):
    """
    Configuration space for the triple store optimizer.
    Uses Pydantic for validation and type safety.
    """

    # Query execution parameters
    batch_size: conint(ge=1, le=1000)
    query_timeout_ms: conint(ge=100, le=30000)
    max_concurrent_queries: conint(ge=1, le=20)

    # Caching parameters
    cache_enabled: bool
    cache_size_mb: conint(ge=0, le=1024)
    cache_ttl_seconds: conint(ge=0, le=3600)

    # Graph traversal parameters
    max_hops: conint(ge=1, le=5)
    traversal_strategy: Literal["bfs", "dfs", "bidirectional"]
    early_termination: bool

    # Vector search parameters
    vector_k: conint(ge=5, le=100)
    vector_threshold: confloat(ge=0.0, le=1.0)
    vector_ef: conint(ge=50, le=500)  # HNSW exploration factor

    # Fusion parameters
    fusion_strategy: Literal["rrf", "linear", "learned"]
    vector_weight: confloat(ge=0.0, le=1.0)
    graph_weight: confloat(ge=0.0, le=1.0)
    text_weight: confloat(ge=0.0, le=1.0)

    # Validators temporarily disabled for compatibility
    # TODO: Re-enable with proper Pydantic v2 syntax

    class Config:
        frozen = True  # Make instances immutable and hashable


# ============================================================================
# Run Status and Results
# ============================================================================


class RunStatus(Enum):
    """Status of a configuration run"""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    OOM = "out_of_memory"
    CONNECTION_ERROR = "connection_error"
    CRITICAL_FAILURE = "critical_failure"
    VALIDATION_ERROR = "validation_error"


@dataclass
class RunResult:
    """Result from running a configuration"""

    config: OptimizerConfig
    status: RunStatus
    composite_reward: float
    metrics: Dict[str, float]
    timestamp: datetime
    duration_ms: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.dict(),
            "status": self.status.value,
            "composite_reward": self.composite_reward,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
        }


@dataclass
class HallOfFameEntry:
    """Entry in the hall of fame"""

    config: OptimizerConfig
    avg_reward: float
    best_reward: float
    worst_reward: float
    run_count: int
    first_seen: datetime
    last_seen: datetime
    metrics_summary: Dict[str, float]


# ============================================================================
# UCB Optimizer
# ============================================================================


class UCBOptimizer:
    """
    Upper Confidence Bound optimizer for configuration tuning.
    Uses sliding window for adaptability to non-stationary environments.
    """

    def __init__(
        self,
        config_space: List[OptimizerConfig],
        exploration_factor: float = 1.0,
        window_size: int = 30,
        quarantine_duration: int = 100,
        hall_of_fame_size: int = 10,
        seed: int = 42,
    ):
        """
        Initialize the UCB optimizer.

        Args:
            config_space: List of configurations to explore
            exploration_factor: UCB exploration parameter (c)
            window_size: Size of sliding window for reward tracking
            quarantine_duration: Number of iterations to quarantine failed configs
            hall_of_fame_size: Number of top configs to track
            seed: Random seed for reproducibility
        """
        self.config_space = config_space
        self.exploration_factor = exploration_factor
        self.window_size = window_size
        self.quarantine_duration = quarantine_duration
        self.hall_of_fame_size = hall_of_fame_size

        # Initialize random state
        self.rng = random.Random(seed)

        # Tracking dictionaries
        self.run_history: Dict[OptimizerConfig, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.total_runs = 0
        self.config_runs: Dict[OptimizerConfig, int] = defaultdict(int)

        # Failure tracking
        self.blacklist: Set[OptimizerConfig] = set()
        self.quarantine: Dict[OptimizerConfig, int] = {}  # Config -> iterations remaining

        # Hall of Fame
        self.hall_of_fame: List[HallOfFameEntry] = []

        # Cold start tracking
        self.unvisited_configs = set(config_space)

    def calculate_ucb_score(self, config: OptimizerConfig) -> float:
        """
        Calculate UCB score for a configuration.

        UCB = mean_reward + c * sqrt(2 * ln(total_runs) / config_runs)
        """
        if config not in self.run_history or len(self.run_history[config]) == 0:
            # Unvisited config gets maximum score
            return float("inf")

        # Calculate mean reward from sliding window
        rewards = [r.composite_reward for r in self.run_history[config]]
        mean_reward = np.mean(rewards)

        # Calculate confidence bonus
        config_runs = len(rewards)
        if self.total_runs > 0 and config_runs > 0:
            confidence_bonus = self.exploration_factor * math.sqrt(
                2 * math.log(self.total_runs) / config_runs
            )
        else:
            confidence_bonus = float("inf")

        return mean_reward + confidence_bonus

    def select_next_config(self) -> Optional[OptimizerConfig]:
        """
        Select the next configuration to test using UCB algorithm.

        Returns:
            Next config to test, or None if all configs are blacklisted
        """
        # Cold start: Round-robin through unvisited configs first
        if self.unvisited_configs:
            config = self.rng.choice(list(self.unvisited_configs))
            self.unvisited_configs.remove(config)
            return config

        # Filter out blacklisted and quarantined configs
        available_configs = [
            c for c in self.config_space if c not in self.blacklist and c not in self.quarantine
        ]

        if not available_configs:
            return None

        # Calculate UCB scores for all available configs
        scores = [(c, self.calculate_ucb_score(c)) for c in available_configs]

        # Select config with highest UCB score
        best_config = max(scores, key=lambda x: x[1])[0]

        return best_config

    def update(self, result: RunResult):
        """
        Update optimizer state with run result.

        Args:
            result: Result from configuration run
        """
        self.total_runs += 1
        config = result.config

        # Handle failures
        if result.status != RunStatus.SUCCESS:
            self._handle_failure(config, result.status)
            return

        # Update run history (sliding window)
        self.run_history[config].append(result)
        self.config_runs[config] += 1

        # Update hall of fame
        self._update_hall_of_fame(result)

        # Update quarantine (decrement counters)
        self._update_quarantine()

    def _handle_failure(self, config: OptimizerConfig, status: RunStatus):
        """Handle configuration failures based on status"""

        # Permanent blacklist for critical failures
        if status in [RunStatus.CRITICAL_FAILURE, RunStatus.VALIDATION_ERROR]:
            self.blacklist.add(config)
            # Remove from other tracking
            if config in self.quarantine:
                del self.quarantine[config]
            if config in self.run_history:
                del self.run_history[config]

        # Quarantine for transient failures
        elif status in [RunStatus.TIMEOUT, RunStatus.OOM, RunStatus.CONNECTION_ERROR]:
            if config not in self.quarantine:
                self.quarantine[config] = self.quarantine_duration

    def _update_quarantine(self):
        """Decrement quarantine counters and release configs"""
        configs_to_release = []

        for config, remaining in self.quarantine.items():
            if remaining <= 1:
                configs_to_release.append(config)
            else:
                self.quarantine[config] = remaining - 1

        # Release configs from quarantine
        for config in configs_to_release:
            del self.quarantine[config]

    def _update_hall_of_fame(self, result: RunResult):
        """Update hall of fame with new result"""
        config = result.config

        # Check if config already in hall of fame
        existing_entry = None
        for entry in self.hall_of_fame:
            if entry.config == config:
                existing_entry = entry
                break

        if existing_entry:
            # Update existing entry
            existing_entry.run_count += 1
            existing_entry.last_seen = result.timestamp
            existing_entry.best_reward = max(existing_entry.best_reward, result.composite_reward)
            existing_entry.worst_reward = min(existing_entry.worst_reward, result.composite_reward)

            # Update average
            rewards = [r.composite_reward for r in self.run_history[config]]
            existing_entry.avg_reward = np.mean(rewards)

            # Update metrics summary
            for key, value in result.metrics.items():
                if key not in existing_entry.metrics_summary:
                    existing_entry.metrics_summary[key] = []
                existing_entry.metrics_summary[key] = value
        else:
            # Create new entry
            new_entry = HallOfFameEntry(
                config=config,
                avg_reward=result.composite_reward,
                best_reward=result.composite_reward,
                worst_reward=result.composite_reward,
                run_count=1,
                first_seen=result.timestamp,
                last_seen=result.timestamp,
                metrics_summary=result.metrics,
            )
            self.hall_of_fame.append(new_entry)

        # Sort hall of fame by average reward
        self.hall_of_fame.sort(key=lambda x: x.avg_reward, reverse=True)

        # Keep only top N entries
        self.hall_of_fame = self.hall_of_fame[: self.hall_of_fame_size]

    def get_best_config(self) -> Optional[OptimizerConfig]:
        """Get the current best performing configuration"""
        if self.hall_of_fame:
            return self.hall_of_fame[0].config
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics"""
        return {
            "total_runs": self.total_runs,
            "configs_evaluated": len(self.run_history),
            "blacklisted_configs": len(self.blacklist),
            "quarantined_configs": len(self.quarantine),
            "hall_of_fame_size": len(self.hall_of_fame),
            "exploration_factor": self.exploration_factor,
            "window_size": self.window_size,
        }

    def save_state(self, filepath: str):
        """Save optimizer state to file"""
        state = {
            "total_runs": self.total_runs,
            "config_runs": {c.dict(): count for c, count in self.config_runs.items()},
            "run_history": {
                c.dict(): [r.to_dict() for r in history] for c, history in self.run_history.items()
            },
            "blacklist": [c.dict() for c in self.blacklist],
            "quarantine": {c.dict(): remaining for c, remaining in self.quarantine.items()},
            "hall_of_fame": [
                {
                    "config": entry.config.dict(),
                    "avg_reward": entry.avg_reward,
                    "best_reward": entry.best_reward,
                    "worst_reward": entry.worst_reward,
                    "run_count": entry.run_count,
                    "first_seen": entry.first_seen.isoformat(),
                    "last_seen": entry.last_seen.isoformat(),
                    "metrics_summary": entry.metrics_summary,
                }
                for entry in self.hall_of_fame
            ],
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, filepath: str, config_space: List[OptimizerConfig]) -> "UCBOptimizer":
        """Load optimizer state from file"""
        with open(filepath, "r") as f:
            state = json.load(f)

        # Create new optimizer
        optimizer = cls(config_space=config_space)

        # Restore state
        optimizer.total_runs = state["total_runs"]

        # Note: This is simplified - in production you'd need to reconstruct
        # the OptimizerConfig objects from the dictionaries

        return optimizer


# ============================================================================
# Reward Calculator
# ============================================================================


class RewardCalculator:
    """Calculate composite reward from metrics"""

    def __init__(
        self,
        recall_weight: float = 0.3,
        ndcg_weight: float = 0.2,
        edge_parity_weight: float = 0.2,
        latency_weight: float = 0.1,
        governance_weight: float = 0.2,
        latency_slo_ms: float = 500,
    ):
        """
        Initialize reward calculator.

        Args:
            recall_weight: Weight for recall@10
            ndcg_weight: Weight for nDCG@10
            edge_parity_weight: Weight for graph edge parity
            latency_weight: Weight for latency penalty
            governance_weight: Weight for governance violations
            latency_slo_ms: Service level objective for P95 latency
        """
        self.recall_weight = recall_weight
        self.ndcg_weight = ndcg_weight
        self.edge_parity_weight = edge_parity_weight
        self.latency_weight = latency_weight
        self.governance_weight = governance_weight
        self.latency_slo_ms = latency_slo_ms

        # Validate weights sum to 1.0
        total_weight = sum(
            [recall_weight, ndcg_weight, edge_parity_weight, latency_weight, governance_weight]
        )
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    def calculate(self, metrics: Dict[str, float]) -> float:
        """
        Calculate composite reward from metrics.

        Args:
            metrics: Dictionary of metric values

        Returns:
            Composite reward value (higher is better)
        """
        # Extract metrics (with defaults)
        recall = metrics.get("recall_at_10", 0.0)
        ndcg = metrics.get("ndcg_at_10", 0.0)
        edge_parity = metrics.get("edge_parity", 0.0)
        p95_latency = metrics.get("p95_latency_ms", self.latency_slo_ms * 2)
        pii_leaks = metrics.get("pii_leak_count", 0)
        access_violations = metrics.get("access_violation_count", 0)
        total_queries = metrics.get("total_queries", 1)

        # Normalize latency penalty
        latency_penalty = max(0, (p95_latency - self.latency_slo_ms) / self.latency_slo_ms)
        latency_penalty = min(1.0, latency_penalty)  # Cap at 1.0

        # Calculate governance violation rate
        governance_violations = (pii_leaks + access_violations) / max(1, total_queries)
        governance_violations = min(1.0, governance_violations)  # Cap at 1.0

        # Calculate composite reward
        reward = (
            self.recall_weight * recall
            + self.ndcg_weight * ndcg
            + self.edge_parity_weight * edge_parity
            - self.latency_weight * latency_penalty
            - self.governance_weight * governance_violations
        )

        # Ensure reward is in [0, 1] range
        return max(0.0, min(1.0, reward))


# ============================================================================
# Config Space Generator
# ============================================================================


def generate_config_space(seed: int = 42) -> List[OptimizerConfig]:
    """
    Generate a discretized configuration space.

    Returns:
        List of configurations to explore
    """
    rng = random.Random(seed)
    configs = []

    # Define discrete options for each parameter
    batch_sizes = [10, 50, 100, 200, 500]
    timeouts = [500, 1000, 2000, 5000, 10000]
    cache_sizes = [0, 64, 128, 256, 512]
    max_hops_options = [1, 2, 3, 4]
    vector_ks = [10, 20, 50, 100]
    vector_thresholds = [0.3, 0.5, 0.7, 0.9]

    # Generate a subset of the full cartesian product
    # (Full product would be too large)

    # Strategy 1: Key configurations (manually selected)
    key_configs = [
        # Fast config (low latency)
        OptimizerConfig(
            batch_size=10,
            query_timeout_ms=500,
            max_concurrent_queries=5,
            cache_enabled=True,
            cache_size_mb=128,
            cache_ttl_seconds=300,
            max_hops=1,
            traversal_strategy="bfs",
            early_termination=True,
            vector_k=10,
            vector_threshold=0.7,
            vector_ef=50,
            fusion_strategy="rrf",
            vector_weight=0.5,
            graph_weight=0.3,
            text_weight=0.2,
        ),
        # Balanced config
        OptimizerConfig(
            batch_size=100,
            query_timeout_ms=2000,
            max_concurrent_queries=10,
            cache_enabled=True,
            cache_size_mb=256,
            cache_ttl_seconds=600,
            max_hops=2,
            traversal_strategy="bidirectional",
            early_termination=False,
            vector_k=20,
            vector_threshold=0.5,
            vector_ef=100,
            fusion_strategy="linear",
            vector_weight=0.4,
            graph_weight=0.4,
            text_weight=0.2,
        ),
        # Deep search config (high recall)
        OptimizerConfig(
            batch_size=200,
            query_timeout_ms=5000,
            max_concurrent_queries=5,
            cache_enabled=True,
            cache_size_mb=512,
            cache_ttl_seconds=1800,
            max_hops=3,
            traversal_strategy="dfs",
            early_termination=False,
            vector_k=50,
            vector_threshold=0.3,
            vector_ef=200,
            fusion_strategy="learned",
            vector_weight=0.3,
            graph_weight=0.5,
            text_weight=0.2,
        ),
    ]

    configs.extend(key_configs)

    # Strategy 2: Random sampling from parameter space
    for _ in range(20):  # Generate 20 random configs
        try:
            # Random fusion weights that sum to 1
            weights = rng.random(), rng.random(), rng.random()
            weights = [w / sum(weights) for w in weights]

            config = OptimizerConfig(
                batch_size=rng.choice(batch_sizes),
                query_timeout_ms=rng.choice(timeouts),
                max_concurrent_queries=rng.randint(1, 20),
                cache_enabled=rng.choice([True, False]),
                cache_size_mb=rng.choice(cache_sizes),
                cache_ttl_seconds=rng.randint(60, 3600),
                max_hops=rng.choice(max_hops_options),
                traversal_strategy=rng.choice(["bfs", "dfs", "bidirectional"]),
                early_termination=rng.choice([True, False]),
                vector_k=rng.choice(vector_ks),
                vector_threshold=rng.choice(vector_thresholds),
                vector_ef=rng.randint(50, 300),
                fusion_strategy=rng.choice(["rrf", "linear", "learned"]),
                vector_weight=round(weights[0], 2),
                graph_weight=round(weights[1], 2),
                text_weight=round(weights[2], 2),
            )
            configs.append(config)
        except ValueError:
            # Skip invalid configs
            continue

    return configs
