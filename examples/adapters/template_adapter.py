"""
Navi LazyTest - Adapter Template
================================

This is a fully-documented template for creating your own LazyTest adapter.
Copy this file, fill in the connection and query logic, and you're ready to go.

An adapter bridges LazyTest to your test target (database, API, service, etc.).
The orchestrator calls your adapter methods in a specific lifecycle:

    1. setup()          - Initialize connections, create schemas
    2. configure()      - Apply runtime parameters from OptimizerConfig
    3. load_corpus()    - Load test data into your target
    4. execute_query()  - Run queries (called many times per iteration)
    5. health_check()   - Verify target is responsive
    6. get_resource_usage() - Report memory, CPU, connections
    7. teardown()       - Clean up resources

Quick Start:
    1. Copy this file to your project
    2. Replace YourSystemAdapter with your adapter name
    3. Implement the TODO sections with your system-specific code
    4. Run: runner.run(max_iterations=5)

Example Usage:
    from your_adapter import YourSystemAdapter
    from navi_lazytest import LazyTestRunner

    adapter = YourSystemAdapter(connection_string="...")
    runner = LazyTestRunner(adapter=adapter)
    runner.run(max_iterations=100)

For more examples, see:
    - examples/adapters/sqlite_adapter.py  (relational DB)
    - examples/adapters/qdrant_adapter.py  (vector store)
    - examples/adapters/neo4j_adapter.py   (graph database)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

# Import the base class and configuration from navi_lazytest
from navi_lazytest import TestTargetAdapter, OptimizerConfig

logger = logging.getLogger(__name__)


class YourSystemAdapter(TestTargetAdapter):
    """
    Template adapter for integrating your system with LazyTest.

    This adapter demonstrates best practices for:
    - Connection lifecycle management
    - Configuration application
    - Corpus data loading
    - Query execution with timing
    - Health monitoring
    - Resource usage reporting

    Attributes:
        connection_string: Connection URL or path to your system.
        options: Additional configuration options.
        _client: Internal client/connection object (set during setup).
        _config: Current OptimizerConfig (set during configure).
        _is_ready: Whether the adapter is set up and ready.

    Example:
        >>> adapter = YourSystemAdapter("localhost:5432")
        >>> with adapter:
        ...     adapter.configure(config)
        ...     adapter.load_corpus(corpus)
        ...     result = adapter.execute_query(query)
    """

    def __init__(
        self,
        connection_string: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the adapter with connection parameters.

        This method should NOT establish connections - that happens in setup().
        Only store configuration needed to connect later.

        Args:
            connection_string: URL, path, or connection string for your system.
                Examples:
                - "postgresql://user:pass@localhost:5432/db"
                - "redis://localhost:6379"
                - "/path/to/sqlite.db"
                - "http://api.example.com"

            options: Optional dictionary of additional settings.
                Examples:
                - {"pool_size": 10, "timeout_seconds": 30}
                - {"ssl_enabled": True, "verify_certs": False}
        """
        self.connection_string = connection_string
        self.options = options or {}

        # Internal state - initialized in setup()
        self._client: Any = None  # TODO: Replace 'Any' with your client type
        self._config: Optional[OptimizerConfig] = None
        self._is_ready: bool = False

        # Metrics tracking (optional but recommended)
        self._query_count: int = 0
        self._total_bytes_loaded: int = 0

    # =========================================================================
    # Context Manager Support (Optional but Recommended)
    # =========================================================================

    def __enter__(self) -> "YourSystemAdapter":
        """Enable 'with' statement usage for automatic cleanup."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ensure teardown is called even if an exception occurs."""
        self.teardown()

    # =========================================================================
    # Required Methods - You MUST implement these
    # =========================================================================

    def setup(self) -> None:
        """
        Initialize the test target instance.

        Called once at the start of a test run. Use this to:
        - Establish database connections
        - Create connection pools
        - Initialize client libraries
        - Create schemas or indexes
        - Start containers if needed

        Raises:
            ConnectionError: If unable to connect to the target system.
            RuntimeError: If setup fails for any other reason.

        Example Implementation:
            def setup(self) -> None:
                self._client = YourClient.connect(
                    self.connection_string,
                    **self.options
                )
                self._client.create_schema_if_not_exists()
                self._is_ready = True
                logger.info("Connected to %s", self.connection_string)
        """
        # TODO: Replace with your connection logic
        #
        # Example for a database:
        # self._client = create_connection(self.connection_string)
        # self._client.execute("CREATE TABLE IF NOT EXISTS ...")
        #
        # Example for an API:
        # self._client = httpx.Client(base_url=self.connection_string)
        # self._client.get("/health")  # Verify connectivity
        #
        # Example for a vector store:
        # self._client = qdrant_client.QdrantClient(url=self.connection_string)
        # self._client.get_collections()  # Verify connectivity

        logger.info("YourSystemAdapter: Setting up connection to %s", self.connection_string)
        time.sleep(0.1)  # TODO: Remove - placeholder for actual setup
        self._is_ready = True
        logger.info("YourSystemAdapter: Setup complete")

    def teardown(self) -> None:
        """
        Clean up the test target instance.

        Called at the end of a test run. Use this to:
        - Close database connections
        - Drain connection pools
        - Clean up temporary data
        - Stop containers if started in setup()

        This method should be idempotent - safe to call multiple times.
        It should NOT raise exceptions (log warnings instead).

        Example Implementation:
            def teardown(self) -> None:
                if self._client is not None:
                    try:
                        self._client.close()
                    except Exception as e:
                        logger.warning("Error during teardown: %s", e)
                    finally:
                        self._client = None
                self._is_ready = False
        """
        # TODO: Replace with your cleanup logic
        #
        # Always check if client exists before closing
        # Always set state to not-ready after cleanup
        # Never raise exceptions - log warnings instead

        if self._client is not None:
            logger.info("YourSystemAdapter: Closing connection")
            # self._client.close()  # TODO: Uncomment for real client
            self._client = None

        self._is_ready = False
        self._query_count = 0
        self._total_bytes_loaded = 0
        logger.info("YourSystemAdapter: Teardown complete")

    def configure(self, config: OptimizerConfig) -> None:
        """
        Apply runtime configuration to the test target.

        Called before each test iteration with a new configuration to test.
        The OptimizerConfig contains parameters that LazyTest is optimizing.

        You should apply any relevant parameters to your system. Not all
        parameters will be applicable to every adapter - ignore what doesn't
        apply to your system.

        Args:
            config: Configuration parameters to apply. Key fields include:
                - batch_size (1-1000): Batch size for operations
                - query_timeout_ms (100-30000): Query timeout in milliseconds
                - max_concurrent_queries (1-20): Max parallel queries
                - cache_enabled (bool): Whether caching is on
                - cache_size_mb (0-1024): Cache size if enabled
                - cache_ttl_seconds (0-3600): Cache TTL if enabled
                - max_hops (1-5): Graph traversal depth (for graph DBs)
                - traversal_strategy ("bfs"/"dfs"/"bidirectional"): Graph traversal
                - vector_k (5-100): Number of nearest neighbors (for vector DBs)
                - vector_threshold (0.0-1.0): Similarity threshold
                - vector_ef (50-500): HNSW exploration factor
                - fusion_strategy ("rrf"/"linear"/"learned"): Result fusion

        Example Implementation:
            def configure(self, config: OptimizerConfig) -> None:
                self._config = config

                # Apply timeout
                self._client.set_timeout(config.query_timeout_ms / 1000)

                # Apply caching if supported
                if hasattr(self._client, 'cache'):
                    self._client.cache.enabled = config.cache_enabled
                    self._client.cache.max_size_mb = config.cache_size_mb

                logger.info("Applied config: batch=%d, timeout=%dms",
                           config.batch_size, config.query_timeout_ms)
        """
        self._config = config

        # TODO: Apply relevant configuration to your system
        #
        # Example for a database:
        # self._client.set_timeout(config.query_timeout_ms / 1000)
        # self._client.set_batch_size(config.batch_size)
        #
        # Example for a vector store:
        # self._search_params = {"k": config.vector_k, "ef": config.vector_ef}
        #
        # Example for a graph database:
        # self._max_hops = config.max_hops
        # self._traversal = config.traversal_strategy

        logger.info(
            "YourSystemAdapter: Applied config batch_size=%d, timeout=%dms",
            config.batch_size,
            config.query_timeout_ms,
        )

    def load_corpus(self, corpus: Dict[str, Any]) -> None:
        """
        Load synthetic corpus into the test target.

        Called once per iteration after configure(). The corpus contains
        synthetic data generated by LazyTest for testing your system.

        Args:
            corpus: Dictionary containing test data with keys:
                - "entities": List of entity dicts with id, name, type, properties
                - "chunks": List of text chunks with id, text, embedding, metadata
                - "triples": List of relationship triples (subject, predicate, object)
                - "manifest": Generation metadata (seed, counts, etc.)

        The exact structure depends on what your system needs:
        - Relational DBs: Use entities and their properties
        - Vector stores: Use chunks with embeddings
        - Graph DBs: Use entities as nodes and triples as edges
        - Multi-store: Use all data types

        Example Implementation:
            def load_corpus(self, corpus: Dict[str, Any]) -> None:
                # Clear previous data
                self._client.truncate("test_table")

                # Load entities
                entities = corpus.get("entities", [])
                for batch in batched(entities, self._config.batch_size):
                    self._client.insert_many("test_table", batch)

                # Load embeddings for vector search
                chunks = corpus.get("chunks", [])
                for chunk in chunks:
                    if chunk.get("embedding"):
                        self._client.upsert_vector(
                            id=chunk["id"],
                            vector=chunk["embedding"],
                            payload={"text": chunk["text"]}
                        )

                logger.info("Loaded %d entities, %d chunks",
                           len(entities), len(chunks))
        """
        # TODO: Replace with your data loading logic
        #
        # Tips:
        # - Clear previous test data before loading new data
        # - Use batch_size from self._config for efficient loading
        # - Track bytes/records loaded for resource reporting
        # - Handle missing keys gracefully (use .get() with defaults)

        entities = corpus.get("entities", [])
        chunks = corpus.get("chunks", [])
        triples = corpus.get("triples", [])

        logger.info(
            "YourSystemAdapter: Loading corpus with %d entities, %d chunks, %d triples",
            len(entities),
            len(chunks),
            len(triples),
        )

        # Example: batch insert entities
        # batch_size = self._config.batch_size if self._config else 100
        # for i in range(0, len(entities), batch_size):
        #     batch = entities[i:i + batch_size]
        #     self._client.insert_batch(batch)

        self._total_bytes_loaded = sum(
            len(str(e)) for e in entities
        ) + sum(
            len(str(c)) for c in chunks
        )

        time.sleep(0.1)  # TODO: Remove - placeholder for actual loading
        logger.info("YourSystemAdapter: Corpus loaded successfully")

    def execute_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single query against the test target.

        This is the core method that runs your actual test logic. It's called
        many times per iteration with different query types.

        Args:
            query: Query specification with keys:
                - "id": Unique query identifier (str)
                - "category": Query type/category (str)
                - "query_text": Human-readable query description (str)
                - "query_params": Parameters for execution (dict)
                - "expected_characteristics": Expected result properties (dict)

        Returns:
            Dict containing at minimum:
                - "success" (bool): Whether query succeeded
                - "latency_ms" (float): Execution time in milliseconds
                - "result_count" (int): Number of results returned
                - "results" (list): Actual result data (can be empty list)
                - "error" (str, optional): Error message if success=False

        Raises:
            Exception: Any exception will be caught by the orchestrator
                      and recorded as a failed query.

        Example Implementation:
            def execute_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
                query_id = query.get("id", "unknown")
                params = query.get("query_params", {})

                start = time.perf_counter()
                try:
                    results = self._client.execute(
                        params.get("sql", "SELECT 1"),
                        timeout=self._config.query_timeout_ms / 1000
                    )
                    latency_ms = (time.perf_counter() - start) * 1000

                    return {
                        "success": True,
                        "latency_ms": latency_ms,
                        "result_count": len(results),
                        "results": results,
                    }
                except TimeoutError:
                    return {
                        "success": False,
                        "latency_ms": self._config.query_timeout_ms,
                        "result_count": 0,
                        "results": [],
                        "error": "Query timeout",
                    }
                except Exception as e:
                    latency_ms = (time.perf_counter() - start) * 1000
                    return {
                        "success": False,
                        "latency_ms": latency_ms,
                        "result_count": 0,
                        "results": [],
                        "error": str(e),
                    }
        """
        self._query_count += 1
        query_id = query.get("id", f"q{self._query_count}")
        query_params = query.get("query_params", {})

        start_time = time.perf_counter()

        try:
            # TODO: Replace with your query execution logic
            #
            # Example for SQL:
            # results = self._client.execute(query_params.get("sql"))
            #
            # Example for vector search:
            # results = self._client.search(
            #     vector=query_params.get("vector"),
            #     limit=self._config.vector_k if self._config else 10,
            # )
            #
            # Example for graph traversal:
            # results = self._client.traverse(
            #     start_node=query_params.get("start_node"),
            #     max_hops=self._config.max_hops if self._config else 2,
            # )

            # Placeholder - simulate query execution
            time.sleep(0.01)  # TODO: Remove - replace with real query
            results: List[Any] = []

            latency_ms = (time.perf_counter() - start_time) * 1000

            return {
                "query_id": query_id,
                "success": True,
                "latency_ms": latency_ms,
                "result_count": len(results),
                "results": results,
            }

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.warning("Query %s failed: %s", query_id, e)

            return {
                "query_id": query_id,
                "success": False,
                "latency_ms": latency_ms,
                "result_count": 0,
                "results": [],
                "error": str(e),
            }

    def health_check(self) -> bool:
        """
        Check if the test target is healthy and ready.

        Called periodically to verify the target system is responsive.
        Should be a lightweight check that returns quickly.

        Returns:
            True if the system is healthy and ready to accept queries.
            False if the system is unhealthy or unresponsive.

        Example Implementation:
            def health_check(self) -> bool:
                if not self._is_ready or self._client is None:
                    return False
                try:
                    # Lightweight health check - ping or simple query
                    self._client.ping()  # Or: self._client.execute("SELECT 1")
                    return True
                except Exception as e:
                    logger.warning("Health check failed: %s", e)
                    return False
        """
        # TODO: Replace with your health check logic
        #
        # Keep it lightweight - this is called frequently
        # Should timeout quickly if the system is unresponsive
        #
        # Examples:
        # - Database: Execute "SELECT 1" or use ping()
        # - API: GET /health endpoint
        # - Vector store: List collections
        # - Graph DB: Count nodes (with limit)

        if not self._is_ready:
            return False

        if self._client is None:
            return False

        try:
            # self._client.ping()  # TODO: Uncomment for real client
            return True
        except Exception as e:
            logger.warning("Health check failed: %s", e)
            return False

    def get_resource_usage(self) -> Dict[str, float]:
        """
        Get current resource usage metrics.

        Called after query batches to monitor system resource consumption.
        Used for optimizing configurations and detecting resource issues.

        Returns:
            Dictionary with resource metrics. Common keys include:
                - "memory_mb" (float): Memory usage in megabytes
                - "cpu_percent" (float): CPU utilization percentage
                - "connections_active" (float): Number of active connections
                - "cache_hits" (float): Number of cache hits
                - "cache_misses" (float): Number of cache misses
                - "disk_usage_mb" (float): Disk usage in megabytes
                - "queue_depth" (float): Number of pending operations

            All values should be floats. Return 0.0 for unavailable metrics.

        Example Implementation:
            def get_resource_usage(self) -> Dict[str, float]:
                stats = self._client.get_stats()
                return {
                    "memory_mb": stats.memory_bytes / (1024 * 1024),
                    "cpu_percent": stats.cpu_percent,
                    "connections_active": float(stats.active_connections),
                    "cache_hits": float(stats.cache_hits),
                    "cache_misses": float(stats.cache_misses),
                }
        """
        # TODO: Replace with your resource monitoring logic
        #
        # Options for getting metrics:
        # - Query system stats from your client (preferred)
        # - Use psutil for process-level metrics
        # - Query external monitoring endpoints
        #
        # Example with psutil:
        # import psutil
        # process = psutil.Process()
        # return {
        #     "memory_mb": process.memory_info().rss / (1024 * 1024),
        #     "cpu_percent": process.cpu_percent(),
        #     "connections_active": len(process.connections()),
        # }

        return {
            "memory_mb": 0.0,  # TODO: Get actual memory usage
            "cpu_percent": 0.0,  # TODO: Get actual CPU usage
            "connections_active": 0.0,  # TODO: Get actual connection count
            "cache_hits": 0.0,  # TODO: Get actual cache hits
            "cache_misses": 0.0,  # TODO: Get actual cache misses
            "bytes_loaded": float(self._total_bytes_loaded),
            "query_count": float(self._query_count),
        }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    """
    Quick test of the template adapter with MockAdapter-like behavior.

    Run this file directly to verify the adapter interface works:
        python template_adapter.py

    Expected output:
        YourSystemAdapter: Setup complete
        YourSystemAdapter: Applied config...
        YourSystemAdapter: Corpus loaded...
        Query result: {'success': True, ...}
        Health: True
        Resources: {'memory_mb': 0.0, ...}
        YourSystemAdapter: Teardown complete
    """
    import sys
    from pathlib import Path

    # Add parent to path for local testing
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from navi_lazytest import OptimizerConfig

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    # Create test config
    config = OptimizerConfig(
        batch_size=100,
        query_timeout_ms=1000,
        max_concurrent_queries=5,
        cache_enabled=True,
        cache_size_mb=256,
        cache_ttl_seconds=300,
        max_hops=2,
        traversal_strategy="bfs",
        early_termination=True,
        vector_k=10,
        vector_threshold=0.7,
        vector_ef=100,
        fusion_strategy="rrf",
        vector_weight=0.4,
        graph_weight=0.3,
        text_weight=0.3,
    )

    # Test the adapter
    print("\n=== Testing Template Adapter ===\n")

    adapter = YourSystemAdapter(
        connection_string="localhost:5432",
        options={"pool_size": 5},
    )

    # Test with context manager
    with adapter:
        # Configure
        adapter.configure(config)

        # Load test corpus
        test_corpus = {
            "entities": [{"id": "e1", "name": "Test", "type": "example"}],
            "chunks": [{"id": "c1", "text": "Test chunk", "embedding": [0.1] * 384}],
            "triples": [{"subject": "e1", "predicate": "IS_A", "object": "example"}],
        }
        adapter.load_corpus(test_corpus)

        # Execute a query
        result = adapter.execute_query({
            "id": "test_query",
            "category": "ENTITY_LOOKUP",
            "query_text": "Find test entity",
            "query_params": {"entity_id": "e1"},
        })
        print(f"Query result: {result}")

        # Check health
        print(f"Health: {adapter.health_check()}")

        # Get resources
        print(f"Resources: {adapter.get_resource_usage()}")

    print("\n=== Template Adapter Test Complete ===\n")
