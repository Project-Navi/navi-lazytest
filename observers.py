"""
Observers for Triple Store Testing
Metrics collection, receipt storage, and performance tracking
"""

import hashlib
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# For accurate percentile calculations
try:
    from hdrhistogram import HdrHistogram
except ImportError:
    HdrHistogram = None
    print("Warning: hdrhistogram not installed. Using numpy percentiles as fallback.")

# ============================================================================
# Data Models
# ============================================================================


@dataclass
class QueryObservation:
    """Single query execution observation"""

    query_id: str
    query_type: str
    start_time: float
    end_time: float
    success: bool
    error_message: Optional[str] = None
    result_count: int = 0
    bytes_processed: int = 0

    @property
    def duration_ms(self) -> float:
        """Query duration in milliseconds"""
        return (self.end_time - self.start_time) * 1000


@dataclass
class ResourceSnapshot:
    """System resource snapshot"""

    timestamp: float
    memory_mb: float
    cpu_percent: float
    connections_active: int
    cache_hits: int
    cache_misses: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
            "connections_active": self.connections_active,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }


@dataclass
class TestReceipt:
    """
    Immutable record of a test run.
    Includes configuration, metrics, mutations, and results.
    """

    run_id: str
    ring: int
    timestamp: datetime
    duration_ms: float

    # Configuration
    config: Dict[str, Any]

    # Test setup
    seed: int
    corpus_size: Dict[str, int]  # entities, chunks, triples counts
    mutations_applied: List[Dict[str, Any]]

    # Metrics
    query_count: int
    success_count: int
    failure_count: int
    timeout_count: int

    # Performance metrics
    latency_percentiles: Dict[str, float]  # P50, P95, P99, etc.
    throughput_qps: float

    # Quality metrics
    recall_at_k: Dict[str, float]
    ndcg_at_k: Dict[str, float]
    edge_parity: float

    # Governance metrics
    pii_leaks: int
    access_violations: int

    # Resource usage
    peak_memory_mb: float
    avg_cpu_percent: float
    total_bytes_processed: int

    # Composite scores
    composite_reward: float
    ucb_score: Optional[float] = None

    # Metadata
    hostname: str = field(
        default_factory=lambda: os.uname().nodename
        if hasattr(os, "uname")
        else os.environ.get("COMPUTERNAME", "unknown")
    )
    python_version: str = field(default_factory=lambda: __import__("sys").version)

    def calculate_hash(self) -> str:
        """Calculate SHA256 hash of receipt for integrity"""
        # Convert to JSON-serializable dict
        receipt_dict = {
            "run_id": self.run_id,
            "ring": self.ring,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "config": self.config,
            "seed": self.seed,
            "corpus_size": self.corpus_size,
            "mutations_applied": self.mutations_applied,
            "query_count": self.query_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "timeout_count": self.timeout_count,
            "latency_percentiles": self.latency_percentiles,
            "throughput_qps": self.throughput_qps,
            "recall_at_k": self.recall_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "edge_parity": self.edge_parity,
            "pii_leaks": self.pii_leaks,
            "access_violations": self.access_violations,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_cpu_percent": self.avg_cpu_percent,
            "total_bytes_processed": self.total_bytes_processed,
            "composite_reward": self.composite_reward,
            "ucb_score": self.ucb_score,
            "hostname": self.hostname,
            "python_version": self.python_version,
        }

        # Sort keys for consistent hashing
        json_str = json.dumps(receipt_dict, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def to_json(self) -> str:
        """Convert receipt to JSON string"""
        receipt_dict = self.__dict__.copy()
        receipt_dict["timestamp"] = receipt_dict["timestamp"].isoformat()
        receipt_dict["hash"] = self.calculate_hash()
        return json.dumps(receipt_dict, indent=2, sort_keys=True)

    def save_to_file(self, directory: Path):
        """Save receipt to JSON file"""
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"receipt_{self.run_id}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = directory / filename

        with open(filepath, "w") as f:
            f.write(self.to_json())

        return filepath


# ============================================================================
# Metrics Collector
# ============================================================================


class MetricsCollector:
    """
    Collects metrics during test execution.
    Uses HdrHistogram for accurate percentile calculations.
    """

    def __init__(self, run_id: str):
        """
        Initialize metrics collector.

        Args:
            run_id: Unique identifier for this test run
        """
        self.run_id = run_id
        self.start_time = time.time()

        # Query observations
        self.observations: List[QueryObservation] = []
        self.lock = Lock()

        # Latency histogram
        if HdrHistogram:
            # Track latencies from 1ms to 60s with 3 significant figures
            self.latency_histogram = HdrHistogram(1, 60000, 3)
        else:
            self.latencies = []

        # Resource snapshots
        self.resource_snapshots: List[ResourceSnapshot] = []

        # Counters
        self.query_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.timeout_count = 0

        # Governance counters
        self.pii_leak_count = 0
        self.access_violation_count = 0

        # Bytes processed
        self.total_bytes_processed = 0

    @contextmanager
    def observe_query(self, query_id: str, query_type: str):
        """
        Context manager to observe a single query execution.

        Usage:
            with collector.observe_query("q1", "vector_search") as obs:
                # Execute query
                result = execute_query()
                obs.result_count = len(result)
        """
        observation = QueryObservation(
            query_id=query_id,
            query_type=query_type,
            start_time=time.time(),
            end_time=0,
            success=False,
        )

        try:
            yield observation
            observation.success = True
        except TimeoutError as e:
            observation.error_message = str(e)
            self.timeout_count += 1
        except Exception as e:
            observation.error_message = str(e)
            self.failure_count += 1
        finally:
            observation.end_time = time.time()

            with self.lock:
                self.observations.append(observation)
                self.query_count += 1

                if observation.success:
                    self.success_count += 1

                    # Record latency
                    latency_ms = observation.duration_ms
                    if HdrHistogram and self.latency_histogram:
                        self.latency_histogram.record_value(int(latency_ms))
                    else:
                        self.latencies.append(latency_ms)

                # Update bytes processed
                self.total_bytes_processed += observation.bytes_processed

    def record_resource_snapshot(
        self,
        memory_mb: float,
        cpu_percent: float,
        connections_active: int,
        cache_hits: int = 0,
        cache_misses: int = 0,
    ):
        """Record a system resource snapshot"""
        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            connections_active=connections_active,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

        with self.lock:
            self.resource_snapshots.append(snapshot)

    def record_governance_violation(self, violation_type: str):
        """Record a governance violation"""
        with self.lock:
            if violation_type == "pii_leak":
                self.pii_leak_count += 1
            elif violation_type == "access_violation":
                self.access_violation_count += 1

    def get_latency_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles"""
        percentiles = {}

        if HdrHistogram and self.latency_histogram:
            percentiles["p50"] = self.latency_histogram.get_value_at_percentile(50)
            percentiles["p75"] = self.latency_histogram.get_value_at_percentile(75)
            percentiles["p90"] = self.latency_histogram.get_value_at_percentile(90)
            percentiles["p95"] = self.latency_histogram.get_value_at_percentile(95)
            percentiles["p99"] = self.latency_histogram.get_value_at_percentile(99)
            percentiles["p999"] = self.latency_histogram.get_value_at_percentile(99.9)
            percentiles["max"] = self.latency_histogram.get_max_value()
            percentiles["min"] = self.latency_histogram.get_min_value()
            percentiles["mean"] = self.latency_histogram.get_mean()
        elif self.latencies:
            # Fallback to numpy
            latencies = np.array(self.latencies)
            percentiles["p50"] = np.percentile(latencies, 50)
            percentiles["p75"] = np.percentile(latencies, 75)
            percentiles["p90"] = np.percentile(latencies, 90)
            percentiles["p95"] = np.percentile(latencies, 95)
            percentiles["p99"] = np.percentile(latencies, 99)
            percentiles["p999"] = np.percentile(latencies, 99.9)
            percentiles["max"] = np.max(latencies)
            percentiles["min"] = np.min(latencies)
            percentiles["mean"] = np.mean(latencies)

        return percentiles

    def get_throughput(self) -> float:
        """Calculate queries per second"""
        if not self.observations:
            return 0.0

        duration = time.time() - self.start_time
        if duration > 0:
            return self.query_count / duration
        return 0.0

    def get_resource_summary(self) -> Dict[str, float]:
        """Get resource usage summary"""
        if not self.resource_snapshots:
            return {
                "peak_memory_mb": 0.0,
                "avg_cpu_percent": 0.0,
                "max_connections": 0,
                "cache_hit_rate": 0.0,
            }

        memories = [s.memory_mb for s in self.resource_snapshots]
        cpus = [s.cpu_percent for s in self.resource_snapshots]
        connections = [s.connections_active for s in self.resource_snapshots]

        total_cache_hits = sum(s.cache_hits for s in self.resource_snapshots)
        total_cache_misses = sum(s.cache_misses for s in self.resource_snapshots)
        total_cache_ops = total_cache_hits + total_cache_misses

        return {
            "peak_memory_mb": max(memories),
            "avg_cpu_percent": np.mean(cpus),
            "max_connections": max(connections),
            "cache_hit_rate": total_cache_hits / total_cache_ops if total_cache_ops > 0 else 0.0,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get complete metrics summary"""
        return {
            "run_id": self.run_id,
            "duration_seconds": time.time() - self.start_time,
            "query_count": self.query_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "timeout_count": self.timeout_count,
            "success_rate": self.success_count / self.query_count if self.query_count > 0 else 0,
            "latency_percentiles": self.get_latency_percentiles(),
            "throughput_qps": self.get_throughput(),
            "pii_leaks": self.pii_leak_count,
            "access_violations": self.access_violation_count,
            "total_bytes_processed": self.total_bytes_processed,
            **self.get_resource_summary(),
        }


# ============================================================================
# Receipt Store
# ============================================================================


class ReceiptStore:
    """
    Hybrid storage for test receipts.
    JSON files as source of truth, SQLite for queries.
    """

    def __init__(self, base_directory: Path, db_path: Optional[Path] = None):
        """
        Initialize receipt store.

        Args:
            base_directory: Directory for JSON receipts
            db_path: Path to SQLite database (optional)
        """
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path or self.base_directory / "receipts.db"
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS receipts (
                    run_id TEXT PRIMARY KEY,
                    ring INTEGER,
                    timestamp DATETIME,
                    duration_ms REAL,
                    config_hash TEXT,
                    seed INTEGER,
                    corpus_entities INTEGER,
                    corpus_chunks INTEGER,
                    corpus_triples INTEGER,
                    mutation_count INTEGER,
                    query_count INTEGER,
                    success_count INTEGER,
                    failure_count INTEGER,
                    timeout_count INTEGER,
                    p50_latency REAL,
                    p95_latency REAL,
                    p99_latency REAL,
                    throughput_qps REAL,
                    recall_at_10 REAL,
                    ndcg_at_10 REAL,
                    edge_parity REAL,
                    pii_leaks INTEGER,
                    access_violations INTEGER,
                    peak_memory_mb REAL,
                    avg_cpu_percent REAL,
                    composite_reward REAL,
                    ucb_score REAL,
                    hostname TEXT,
                    receipt_hash TEXT,
                    json_path TEXT
                )
            """
            )

            # Create indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON receipts(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ring ON receipts(ring)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_composite_reward ON receipts(composite_reward)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_config_hash ON receipts(config_hash)")

    def store_receipt(self, receipt: TestReceipt) -> Path:
        """
        Store receipt in both JSON and SQLite.

        Args:
            receipt: Test receipt to store

        Returns:
            Path to saved JSON file
        """
        # Save JSON file (source of truth)
        json_path = receipt.save_to_file(self.base_directory)

        # Index in SQLite for queries
        self._index_receipt(receipt, json_path)

        return json_path

    def _index_receipt(self, receipt: TestReceipt, json_path: Path):
        """Index receipt in SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            # Calculate config hash for grouping similar configs
            config_hash = hashlib.md5(
                json.dumps(receipt.config, sort_keys=True).encode()
            ).hexdigest()

            conn.execute(
                """
                INSERT OR REPLACE INTO receipts VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """,
                (
                    receipt.run_id,
                    receipt.ring,
                    receipt.timestamp,
                    receipt.duration_ms,
                    config_hash,
                    receipt.seed,
                    receipt.corpus_size.get("entities", 0),
                    receipt.corpus_size.get("chunks", 0),
                    receipt.corpus_size.get("triples", 0),
                    len(receipt.mutations_applied),
                    receipt.query_count,
                    receipt.success_count,
                    receipt.failure_count,
                    receipt.timeout_count,
                    receipt.latency_percentiles.get("p50", 0),
                    receipt.latency_percentiles.get("p95", 0),
                    receipt.latency_percentiles.get("p99", 0),
                    receipt.throughput_qps,
                    receipt.recall_at_k.get("10", 0),
                    receipt.ndcg_at_k.get("10", 0),
                    receipt.edge_parity,
                    receipt.pii_leaks,
                    receipt.access_violations,
                    receipt.peak_memory_mb,
                    receipt.avg_cpu_percent,
                    receipt.composite_reward,
                    receipt.ucb_score,
                    receipt.hostname,
                    receipt.calculate_hash(),
                    str(json_path),
                ),
            )

    def query_receipts(
        self, ring: Optional[int] = None, min_reward: Optional[float] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query receipts from SQLite.

        Args:
            ring: Filter by ring number
            min_reward: Minimum composite reward
            limit: Maximum number of results

        Returns:
            List of receipt summaries
        """
        query = "SELECT * FROM receipts WHERE 1=1"
        params = []

        if ring is not None:
            query += " AND ring = ?"
            params.append(ring)

        if min_reward is not None:
            query += " AND composite_reward >= ?"
            params.append(min_reward)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_performance_trend(
        self, metric: str = "composite_reward", window: int = 100
    ) -> List[Tuple[datetime, float]]:
        """
        Get performance trend over time.

        Args:
            metric: Metric to track
            window: Number of recent runs to include

        Returns:
            List of (timestamp, value) tuples
        """
        valid_metrics = [
            "composite_reward",
            "p95_latency",
            "throughput_qps",
            "recall_at_10",
            "ndcg_at_10",
            "edge_parity",
            "peak_memory_mb",
        ]

        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric. Must be one of {valid_metrics}")

        query = f"""
            SELECT timestamp, {metric}
            FROM receipts
            ORDER BY timestamp DESC
            LIMIT ?
        """

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, (window,))
            return [(datetime.fromisoformat(row[0]), row[1]) for row in cursor.fetchall()]


# ============================================================================
# Performance Tracker
# ============================================================================


class PerformanceTracker:
    """
    Track performance metrics over time for trending and alerting.
    """

    def __init__(self, receipt_store: ReceiptStore):
        """
        Initialize performance tracker.

        Args:
            receipt_store: Receipt store for querying historical data
        """
        self.receipt_store = receipt_store
        self.logger = logging.getLogger(__name__)

    def check_degradation(
        self, current_receipt: TestReceipt, window: int = 20, threshold: float = 0.1
    ) -> Optional[str]:
        """
        Check if performance has degraded compared to recent history.

        Args:
            current_receipt: Current test receipt
            window: Number of recent runs to compare against
            threshold: Degradation threshold (0.1 = 10% worse)

        Returns:
            Warning message if degradation detected, None otherwise
        """
        # Get recent performance trend
        trend = self.receipt_store.get_performance_trend(metric="composite_reward", window=window)

        if len(trend) < 5:  # Not enough history
            return None

        # Calculate baseline (average of recent runs)
        recent_rewards = [value for _, value in trend[:window]]
        baseline = np.mean(recent_rewards)

        # Check for degradation
        current_reward = current_receipt.composite_reward
        degradation = (baseline - current_reward) / baseline

        if degradation > threshold:
            return (
                f"Performance degradation detected: "
                f"Current reward {current_reward:.3f} is "
                f"{degradation:.1%} worse than baseline {baseline:.3f}"
            )

        return None

    def detect_anomalies(self, current_receipt: TestReceipt, z_threshold: float = 3.0) -> List[str]:
        """
        Detect anomalous metrics using z-score.

        Args:
            current_receipt: Current test receipt
            z_threshold: Z-score threshold for anomaly detection

        Returns:
            List of anomaly warnings
        """
        anomalies = []

        # Metrics to check for anomalies
        metrics_to_check = [
            ("p95_latency", "latency_percentiles", "p95"),
            ("throughput_qps", "throughput_qps", None),
            ("peak_memory_mb", "peak_memory_mb", None),
        ]

        for metric_name, receipt_field, subfield in metrics_to_check:
            # Get historical values
            trend = self.receipt_store.get_performance_trend(metric=metric_name, window=50)

            if len(trend) < 10:  # Not enough history
                continue

            # Calculate statistics
            historical_values = [value for _, value in trend]
            mean = np.mean(historical_values)
            std = np.std(historical_values)

            if std == 0:  # No variance
                continue

            # Get current value
            if subfield:
                current_value = getattr(current_receipt, receipt_field).get(subfield, 0)
            else:
                current_value = getattr(current_receipt, receipt_field)

            # Calculate z-score
            z_score = abs((current_value - mean) / std)

            if z_score > z_threshold:
                anomalies.append(
                    f"Anomaly in {metric_name}: value {current_value:.2f} "
                    f"is {z_score:.1f} std devs from mean {mean:.2f}"
                )

        return anomalies

    def generate_report(self, ring: Optional[int] = None, last_n_days: int = 7) -> Dict[str, Any]:
        """
        Generate performance report.

        Args:
            ring: Filter by ring number
            last_n_days: Include runs from last N days

        Returns:
            Performance report dictionary
        """
        # Query recent receipts
        receipts = self.receipt_store.query_receipts(ring=ring, limit=1000)

        if not receipts:
            return {"error": "No receipts found"}

        # Filter by date
        cutoff = datetime.now() - timedelta(days=last_n_days)
        receipts = [r for r in receipts if datetime.fromisoformat(r["timestamp"]) > cutoff]

        if not receipts:
            return {"error": f"No receipts in last {last_n_days} days"}

        # Calculate statistics
        rewards = [r["composite_reward"] for r in receipts]
        latencies = [r["p95_latency"] for r in receipts]
        throughputs = [r["throughput_qps"] for r in receipts]

        return {
            "period": f"Last {last_n_days} days",
            "ring": ring,
            "total_runs": len(receipts),
            "composite_reward": {
                "mean": np.mean(rewards),
                "std": np.std(rewards),
                "min": np.min(rewards),
                "max": np.max(rewards),
                "trend": "improving" if rewards[-1] > rewards[0] else "declining",
            },
            "p95_latency_ms": {
                "mean": np.mean(latencies),
                "std": np.std(latencies),
                "min": np.min(latencies),
                "max": np.max(latencies),
            },
            "throughput_qps": {
                "mean": np.mean(throughputs),
                "std": np.std(throughputs),
                "min": np.min(throughputs),
                "max": np.max(throughputs),
            },
            "success_rate": np.mean(
                [r["success_count"] / r["query_count"] for r in receipts if r["query_count"] > 0]
            ),
            "governance_violations": sum(r["pii_leaks"] + r["access_violations"] for r in receipts),
        }
