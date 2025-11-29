"""
Orchestrator for Self-Improving Spiral Test Suite
Coordinates all components to run the spiral loop
"""

import json
import logging
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evaluator import Evaluator

# Import all spiral components
from .generators import SyntheticCorpusGenerator
from .mutators import CompositeMutator
from .observers import MetricsCollector, PerformanceTracker, ReceiptStore, TestReceipt
from .optimizer import (
    OptimizerConfig,
    RewardCalculator,
    RunResult,
    RunStatus,
    UCBOptimizer,
    generate_config_space,
)
from .query_suite import QuerySuite

# ============================================================================
# State Management
# ============================================================================


class OrchestratorState(Enum):
    """Orchestrator state machine states"""

    INITIALIZING = "initializing"
    GENERATING = "generating"
    MUTATING = "mutating"
    LOADING = "loading"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    OBSERVING = "observing"
    OPTIMIZING = "optimizing"
    PROGRESSING = "progressing"  # Ring progression
    CLEANUP = "cleanup"
    FAILED = "failed"
    COMPLETED = "completed"


@dataclass
class OrchestratorContext:
    """
    Context for orchestrator execution.
    Tracks all state needed for checkpoint/resume.
    """

    # Identity
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.now)

    # Ring management
    current_ring: int = 1
    iteration_in_ring: int = 0
    global_iteration: int = 0

    # Performance tracking
    best_config: Optional[Dict[str, Any]] = None
    best_reward: float = 0.0
    recent_rewards: List[float] = field(default_factory=list)

    # State machine
    current_state: OrchestratorState = OrchestratorState.INITIALIZING
    last_successful_state: Optional[OrchestratorState] = None

    # Current iteration data
    current_config: Optional[Dict[str, Any]] = None
    current_corpus: Optional[Dict[str, Any]] = None
    current_metrics: Optional[Dict[str, float]] = None

    # Failure tracking
    consecutive_failures: int = 0
    total_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "current_ring": self.current_ring,
            "iteration_in_ring": self.iteration_in_ring,
            "global_iteration": self.global_iteration,
            "best_config": self.best_config,
            "best_reward": self.best_reward,
            "recent_rewards": self.recent_rewards[-20:],  # Keep last 20
            "current_state": self.current_state.value,
            "last_successful_state": self.last_successful_state.value
            if self.last_successful_state
            else None,
            "current_config": self.current_config,
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrchestratorContext":
        """Create from dictionary"""
        context = cls()
        context.run_id = data["run_id"]
        context.start_time = datetime.fromisoformat(data["start_time"])
        context.current_ring = data["current_ring"]
        context.iteration_in_ring = data["iteration_in_ring"]
        context.global_iteration = data["global_iteration"]
        context.best_config = data.get("best_config")
        context.best_reward = data.get("best_reward", 0.0)
        context.recent_rewards = data.get("recent_rewards", [])
        context.current_state = OrchestratorState(data["current_state"])

        if data.get("last_successful_state"):
            context.last_successful_state = OrchestratorState(data["last_successful_state"])

        context.current_config = data.get("current_config")
        context.consecutive_failures = data.get("consecutive_failures", 0)
        context.total_failures = data.get("total_failures", 0)

        return context

    def save(self, filepath: Path):
        """Save state to file (atomic write)"""
        temp_file = filepath.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        temp_file.replace(filepath)

    @classmethod
    def load(cls, filepath: Path) -> Optional["OrchestratorContext"]:
        """Load state from file"""
        if not filepath.exists():
            return None

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logging.error(f"Failed to load orchestrator state: {e}")
            return None


# ============================================================================
# Test Target Adapter Interface
# ============================================================================


class TestTargetAdapter(ABC):
    """
    Abstract adapter for test target implementations.

    Each concrete adapter manages the full lifecycle of one test target instance
    (database, service, API, etc.). Implement this interface to integrate SPIRAL
    with any system you want to test and optimize.
    """

    @abstractmethod
    def setup(self) -> None:
        """
        Initialize the test target instance.
        This might start a Docker container, create database schemas, etc.
        """
        pass

    @abstractmethod
    def teardown(self) -> None:
        """
        Clean up the test target instance.
        Stop containers, remove data volumes, close connections.
        """
        pass

    @abstractmethod
    def configure(self, config: OptimizerConfig) -> None:
        """
        Apply runtime configuration to the test target.

        Args:
            config: Optimizer configuration with parameters like batch_size, cache_size, etc.
        """
        pass

    @abstractmethod
    def load_corpus(self, corpus: Dict[str, Any]) -> None:
        """
        Load synthetic corpus into the test target.

        Args:
            corpus: Dictionary with entities, chunks, triples, embeddings
        """
        pass

    @abstractmethod
    def execute_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single query against the test target.

        Args:
            query: Query specification

        Returns:
            Query results including timing and result set
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the test target is healthy and ready.

        Returns:
            True if healthy, False otherwise
        """
        pass

    @abstractmethod
    def get_resource_usage(self) -> Dict[str, float]:
        """
        Get current resource usage metrics.

        Returns:
            Dictionary with memory_mb, cpu_percent, connections_active, etc.
        """
        pass


# Backwards compatibility alias
TripleStoreAdapter = TestTargetAdapter


# ============================================================================
# Mock Adapter for Dry Run
# ============================================================================


class MockAdapter(TestTargetAdapter):
    """Mock adapter for dry-run mode and testing"""

    def __init__(self, failure_rate: float = 0.0, base_latency_ms: float = 50.0):
        """
        Initialize mock adapter.

        Args:
            failure_rate: Probability of query failure (0.0 to 1.0)
            base_latency_ms: Base query latency in milliseconds
        """
        self.failure_rate = failure_rate
        self.base_latency_ms = base_latency_ms
        self.is_setup = False
        self.config = None
        self.corpus_loaded = False
        self.query_count = 0

    def setup(self) -> None:
        """Simulate setup"""
        time.sleep(0.1)  # Simulate startup time
        self.is_setup = True
        logging.info("MockAdapter: Setup complete")

    def teardown(self) -> None:
        """Simulate teardown"""
        time.sleep(0.05)  # Simulate cleanup time
        self.is_setup = False
        self.corpus_loaded = False
        self.query_count = 0
        logging.info("MockAdapter: Teardown complete")

    def configure(self, config: OptimizerConfig) -> None:
        """Apply configuration"""
        self.config = config
        logging.info(f"MockAdapter: Applied config with batch_size={config.batch_size}")

    def load_corpus(self, corpus: Dict[str, Any]) -> None:
        """Simulate corpus loading"""
        time.sleep(0.2)  # Simulate load time
        self.corpus_loaded = True
        entity_count = len(corpus.get("entities", []))
        triple_count = len(corpus.get("triples", []))
        logging.info(f"MockAdapter: Loaded {entity_count} entities, {triple_count} triples")

    def execute_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate query execution"""
        import random

        self.query_count += 1

        # Simulate failure
        if random.random() < self.failure_rate:
            raise Exception("Simulated query failure")

        # Simulate variable latency
        latency_ms = self.base_latency_ms * (1 + random.gauss(0, 0.2))
        time.sleep(latency_ms / 1000)

        # Return mock results
        return {
            "query_id": query.get("id", f"q{self.query_count}"),
            "success": True,
            "latency_ms": latency_ms,
            "result_count": random.randint(0, 100),
            "results": [],
        }

    def health_check(self) -> bool:
        """Check health"""
        return self.is_setup

    def get_resource_usage(self) -> Dict[str, float]:
        """Get mock resource usage"""
        import random

        return {
            "memory_mb": 100 + random.random() * 50,
            "cpu_percent": 10 + random.random() * 20,
            "connections_active": self.query_count % 10,
            "cache_hits": self.query_count * 0.7,
            "cache_misses": self.query_count * 0.3,
        }


# ============================================================================
# Spiral Orchestrator
# ============================================================================


class SpiralOrchestrator:
    """
    Main orchestrator for the self-improving spiral test suite.
    Coordinates all components and manages the spiral loop.
    """

    def __init__(
        self,
        adapter: TripleStoreAdapter,
        config_path: Optional[Path] = None,
        state_path: Optional[Path] = None,
        receipts_dir: Optional[Path] = None,
        dry_run: bool = False,
        seed: int = 42,
    ):
        """
        Initialize orchestrator.

        Args:
            adapter: Triple store adapter to use
            config_path: Path to configuration file
            state_path: Path for state persistence
            receipts_dir: Directory for test receipts
            dry_run: Use mock adapter if True
            seed: Random seed for reproducibility
        """
        self.adapter = adapter if not dry_run else MockAdapter()
        self.config_path = config_path or Path("orchestrator_config.yaml")
        self.state_path = state_path or Path("orchestrator_state.json")
        self.receipts_dir = receipts_dir or Path("receipts")
        self.seed = seed

        # Load or create context
        self.context = OrchestratorContext.load(self.state_path) or OrchestratorContext()

        # Initialize components
        self._init_components()

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _init_components(self):
        """Initialize all spiral components"""
        # Generator
        self.generator = SyntheticCorpusGenerator(seed=self.seed)

        # Mutator (intensity based on ring)
        self.mutator = None  # Created per ring

        # Query suite
        self.query_suite = QuerySuite(seed=self.seed)

        # Evaluator
        self.evaluator = Evaluator()

        # Optimizer
        config_space = generate_config_space(seed=self.seed)
        self.optimizer = UCBOptimizer(
            config_space=config_space, exploration_factor=1.0, window_size=30
        )

        # Reward calculator
        self.reward_calculator = RewardCalculator()

        # Receipt store
        self.receipt_store = ReceiptStore(self.receipts_dir)

        # Performance tracker
        self.performance_tracker = PerformanceTracker(self.receipt_store)

        # Metrics collector (created per iteration)
        self.metrics_collector = None

    def run(self, max_iterations: Optional[int] = None):
        """
        Run the spiral loop.

        Args:
            max_iterations: Maximum total iterations (across all rings)
        """
        self.logger.info(f"Starting spiral orchestrator, ring {self.context.current_ring}")

        try:
            while True:
                # Check stopping criteria
                if max_iterations and self.context.global_iteration >= max_iterations:
                    self.logger.info(f"Reached max iterations: {max_iterations}")
                    break

                if self.context.current_ring > 3:
                    self.logger.info("Completed all rings")
                    break

                # Run one iteration
                self._run_iteration()

                # Check ring progression
                if self._should_progress_ring():
                    self._progress_ring()

                # Save state after each iteration
                self.context.save(self.state_path)

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}\n{traceback.format_exc()}")
            self.context.current_state = OrchestratorState.FAILED
        finally:
            self._cleanup()

    def _run_iteration(self):
        """Run a single iteration of the spiral loop"""
        iteration_start = time.time()
        run_id = (
            f"{self.context.run_id}_r{self.context.current_ring}_i{self.context.iteration_in_ring}"
        )

        self.logger.info(
            f"Starting iteration {self.context.global_iteration} "
            f"(Ring {self.context.current_ring}, iteration {self.context.iteration_in_ring})"
        )

        # Create metrics collector for this iteration
        self.metrics_collector = MetricsCollector(run_id)

        try:
            # State machine execution
            self._execute_state_machine()

            # Create and store receipt
            receipt = self._create_receipt(iteration_start)
            self.receipt_store.store_receipt(receipt)

            # Check for performance degradation
            warning = self.performance_tracker.check_degradation(receipt)
            if warning:
                self.logger.warning(warning)

            # Detect anomalies
            anomalies = self.performance_tracker.detect_anomalies(receipt)
            for anomaly in anomalies:
                self.logger.warning(f"Anomaly detected: {anomaly}")

            # Update context
            self.context.iteration_in_ring += 1
            self.context.global_iteration += 1
            self.context.recent_rewards.append(receipt.composite_reward)

            if receipt.composite_reward > self.context.best_reward:
                self.context.best_reward = receipt.composite_reward
                self.context.best_config = self.context.current_config
                self.logger.info(f"New best reward: {receipt.composite_reward:.3f}")

            # Reset failure counter on success
            self.context.consecutive_failures = 0

        except Exception as e:
            self.logger.error(f"Iteration failed: {e}")
            self.context.consecutive_failures += 1
            self.context.total_failures += 1

            # Check if we should abort
            if self.context.consecutive_failures > 5:
                raise Exception("Too many consecutive failures, aborting")

    def _execute_state_machine(self):
        """Execute the state machine for one iteration"""
        states = [
            (OrchestratorState.INITIALIZING, self._state_initialize),
            (OrchestratorState.GENERATING, self._state_generate),
            (OrchestratorState.MUTATING, self._state_mutate),
            (OrchestratorState.LOADING, self._state_load),
            (OrchestratorState.EXECUTING, self._state_execute),
            (OrchestratorState.EVALUATING, self._state_evaluate),
            (OrchestratorState.OBSERVING, self._state_observe),
            (OrchestratorState.OPTIMIZING, self._state_optimize),
        ]

        for state, handler in states:
            self.context.current_state = state
            self.logger.debug(f"Entering state: {state.value}")

            try:
                handler()
                self.context.last_successful_state = state
            except Exception as e:
                self.logger.error(f"Failed in state {state.value}: {e}")
                self.context.current_state = OrchestratorState.CLEANUP
                self._state_cleanup()
                raise

    def _state_initialize(self):
        """Initialize state: Setup triple store"""
        self.adapter.setup()

        # Wait for health check
        max_attempts = 30
        for i in range(max_attempts):
            if self.adapter.health_check():
                break
            time.sleep(1)
        else:
            raise Exception("Triple store failed to become healthy")

    def _state_generate(self):
        """Generate state: Create synthetic corpus"""
        # Generate corpus based on ring
        if self.context.current_ring == 1:
            entity_count = 100
            chunk_count = 50
            triple_count = 200
        elif self.context.current_ring == 2:
            entity_count = 500
            chunk_count = 250
            triple_count = 1000
        else:  # Ring 3
            entity_count = 2000
            chunk_count = 1000
            triple_count = 5000

        self.context.current_corpus = {
            "entities": self.generator.generate_entities(entity_count),
            "chunks": self.generator.generate_chunks(chunk_count),
            "triples": self.generator.generate_triples(triple_count),
            "manifest": self.generator.get_manifest(),
        }

    def _state_mutate(self):
        """Mutate state: Apply mutations based on ring"""
        # Create mutator for current ring
        intensity = 0.1 * self.context.current_ring  # 0.1, 0.2, 0.3
        self.mutator = CompositeMutator(
            seed=self.seed + self.context.global_iteration,
            intensity=intensity,
            ring=self.context.current_ring,
        )

        # Apply mutations
        self.context.current_corpus = self.mutator.mutate(self.context.current_corpus)

        # Store mutation manifest
        self.context.current_corpus["mutations"] = self.mutator.get_mutation_summary()

    def _state_load(self):
        """Load state: Load corpus into triple store"""
        self.adapter.load_corpus(self.context.current_corpus)

    def _state_execute(self):
        """Execute state: Run query suite"""
        # Get next configuration from optimizer
        config = self.optimizer.select_next_config()
        if config is None:
            raise Exception("No valid configurations available")

        self.context.current_config = config.dict()

        # Apply configuration
        self.adapter.configure(config)

        # Get queries for current ring
        queries = self.query_suite.get_queries(canonical_only=(self.context.current_ring == 1))

        # Execute queries
        results = []
        for query in queries:
            # Instantiate query with actual data
            query = self.query_suite.instantiate_query(query, self.context.current_corpus)

            with self.metrics_collector.observe_query(query.id, query.category.value) as obs:
                try:
                    result = self.adapter.execute_query(query.query_params)
                    obs.result_count = result.get("result_count", 0)
                    obs.bytes_processed = result.get("bytes_processed", 0)
                    results.append(result)
                except TimeoutError:
                    raise  # Re-raise for context manager to handle
                except Exception as e:
                    self.logger.warning(f"Query {query.id} failed: {e}")
                    raise

        self.context.current_results = results

    def _state_evaluate(self):
        """Evaluate state: Calculate metrics"""
        # Evaluate results
        eval_results = self.evaluator.evaluate(
            self.context.current_results,
            self.context.current_corpus,
            strict=(self.context.current_ring == 3),  # Strict mode for Ring 3
        )

        # Get metrics from collector
        collector_metrics = self.metrics_collector.get_summary()

        # Combine metrics
        self.context.current_metrics = {**eval_results.metrics, **collector_metrics}

    def _state_observe(self):
        """Observe state: Collect resource usage"""
        # Get resource snapshot
        resources = self.adapter.get_resource_usage()
        self.metrics_collector.record_resource_snapshot(
            memory_mb=resources.get("memory_mb", 0),
            cpu_percent=resources.get("cpu_percent", 0),
            connections_active=resources.get("connections_active", 0),
            cache_hits=resources.get("cache_hits", 0),
            cache_misses=resources.get("cache_misses", 0),
        )

    def _state_optimize(self):
        """Optimize state: Update optimizer with results"""
        # Calculate composite reward
        reward = self.reward_calculator.calculate(self.context.current_metrics)

        # Create run result
        result = RunResult(
            config=OptimizerConfig(**self.context.current_config),
            status=RunStatus.SUCCESS,
            composite_reward=reward,
            metrics=self.context.current_metrics,
            timestamp=datetime.now(),
            duration_ms=self.metrics_collector.get_summary()["duration_seconds"] * 1000,
        )

        # Update optimizer
        self.optimizer.update(result)

    def _state_cleanup(self):
        """Cleanup state: Teardown triple store"""
        try:
            self.adapter.teardown()
        except Exception as e:
            self.logger.error(f"Teardown failed: {e}")

    def _should_progress_ring(self) -> bool:
        """Check if we should progress to the next ring"""
        # Max iterations per ring
        max_per_ring = {1: 50, 2: 100, 3: 150}
        if self.context.iteration_in_ring >= max_per_ring.get(self.context.current_ring, 100):
            return True

        # Plateau detection
        if len(self.context.recent_rewards) >= 20:
            recent = self.context.recent_rewards[-20:]
            if max(recent) - min(recent) < 0.01:  # Less than 1% variation
                self.logger.info("Performance plateau detected")
                return True

        return False

    def _progress_ring(self):
        """Progress to the next ring"""
        self.logger.info(
            f"Progressing from Ring {self.context.current_ring} to Ring {self.context.current_ring + 1}"
        )
        self.context.current_ring += 1
        self.context.iteration_in_ring = 0
        self.context.recent_rewards = []

    def _create_receipt(self, iteration_start: float) -> TestReceipt:
        """Create test receipt for current iteration"""
        metrics = self.context.current_metrics or {}
        corpus_size = {
            "entities": len(self.context.current_corpus.get("entities", [])),
            "chunks": len(self.context.current_corpus.get("chunks", [])),
            "triples": len(self.context.current_corpus.get("triples", [])),
        }

        return TestReceipt(
            run_id=self.metrics_collector.run_id,
            ring=self.context.current_ring,
            timestamp=datetime.now(),
            duration_ms=(time.time() - iteration_start) * 1000,
            config=self.context.current_config,
            seed=self.seed,
            corpus_size=corpus_size,
            mutations_applied=self.context.current_corpus.get("mutations", {}).get("mutations", []),
            query_count=metrics.get("query_count", 0),
            success_count=metrics.get("success_count", 0),
            failure_count=metrics.get("failure_count", 0),
            timeout_count=metrics.get("timeout_count", 0),
            latency_percentiles=metrics.get("latency_percentiles", {}),
            throughput_qps=metrics.get("throughput_qps", 0),
            recall_at_k={"10": metrics.get("recall_at_10", 0)},
            ndcg_at_k={"10": metrics.get("ndcg_at_10", 0)},
            edge_parity=metrics.get("edge_parity", 0),
            pii_leaks=metrics.get("pii_leaks", 0),
            access_violations=metrics.get("access_violations", 0),
            peak_memory_mb=metrics.get("peak_memory_mb", 0),
            avg_cpu_percent=metrics.get("avg_cpu_percent", 0),
            total_bytes_processed=metrics.get("total_bytes_processed", 0),
            composite_reward=self.reward_calculator.calculate(metrics),
        )

    def _cleanup(self):
        """Final cleanup"""
        try:
            self.adapter.teardown()
        except:
            pass

        # Save final state
        self.context.current_state = OrchestratorState.COMPLETED
        self.context.save(self.state_path)

        # Generate final report
        report = self.performance_tracker.generate_report()
        self.logger.info(f"Final report: {json.dumps(report, indent=2)}")


# ============================================================================
# Entry Point
# ============================================================================


def main():
    """Main entry point for the orchestrator"""
    import argparse

    parser = argparse.ArgumentParser(description="Self-Improving Spiral Test Suite Orchestrator")
    parser.add_argument("--dry-run", action="store_true", help="Use mock adapter for testing")
    parser.add_argument("--max-iterations", type=int, help="Maximum iterations to run")
    parser.add_argument("--ring", type=int, choices=[1, 2, 3], help="Start at specific ring")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--state-file", type=Path, help="State file path")
    parser.add_argument("--receipts-dir", type=Path, help="Receipts directory")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create adapter
    if args.dry_run:
        adapter = MockAdapter()
    else:
        # TODO: Create real adapter based on config
        raise NotImplementedError("Real adapters not implemented yet")

    # Create and run orchestrator
    orchestrator = SpiralOrchestrator(
        adapter=adapter,
        state_path=args.state_file,
        receipts_dir=args.receipts_dir,
        dry_run=args.dry_run,
        seed=args.seed,
    )

    # Override ring if specified
    if args.ring:
        orchestrator.context.current_ring = args.ring

    # Run the spiral
    orchestrator.run(max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()
