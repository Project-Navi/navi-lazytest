"""
Triple Store Adapter for SPIRAL Testing Framework

Example adapter demonstrating integration with triple-store-db.
Provides a synchronous interface to the asynchronous triple store implementation,
managing the async event loop in a background thread for seamless integration.

This is an OPTIONAL adapter - SPIRAL works with any TestTargetAdapter implementation.
"""

import asyncio
import logging
import sys
import time
from concurrent.futures import Future
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional

from ..orchestrator import OptimizerConfig, TestTargetAdapter

# Optional triple-store import - only needed if using this adapter
try:
    from triple_store.core.store import TripleStore
    from triple_store.models.config import StoreConfig
    TRIPLE_STORE_AVAILABLE = True
except ImportError:
    TripleStore = None
    StoreConfig = None
    TRIPLE_STORE_AVAILABLE = False

logger = logging.getLogger(__name__)


class RealTripleStoreAdapter(TestTargetAdapter):
    """
    Synchronous adapter that wraps the asynchronous Triple Store,
    managing the asyncio event loop in a background thread.

    This allows the synchronous test suite to interact with the
    async triple store without refactoring the entire test framework.
    """

    def __init__(self, config: Optional[StoreConfig] = None):
        """
        Initialize the adapter with optional store configuration.

        Args:
            config: Triple store configuration. If None, uses defaults.
        """
        self._provided_config = config  # Store provided config, delay creation
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[Thread] = None
        self.store: Optional[TripleStore] = None
        self._initialized = False

        # Performance tracking
        self.query_count = 0
        self.total_latency_ms = 0.0

    def setup(self) -> None:
        """
        Initialize the triple store instance.
        Starts the async event loop and initializes all database connections.
        """
        if self._initialized:
            logger.warning("Triple store already initialized")
            return

        logger.info("Setting up Real Triple Store Adapter...")

        # Create new event loop in background thread
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Load configuration from environment variables
        self.config = self._provided_config or StoreConfig.from_env()
        logger.info(f"Loaded config with Neo4j user: {self.config.neo4j.user}")
        logger.info(f"Neo4j URI: {self.config.neo4j.uri}")

        # Initialize the triple store
        self.store = TripleStore(self.config)
        self._run_coro_threadsafe(self.store.initialize())

        self._initialized = True
        logger.info("Real Triple Store Adapter setup complete")

    def teardown(self) -> None:
        """
        Clean up the triple store instance.
        Shuts down all connections and stops the event loop.
        """
        if not self._initialized:
            return

        logger.info("Tearing down Real Triple Store Adapter...")

        if self.store and self._loop and self._loop.is_running():
            # Shutdown the store
            self._run_coro_threadsafe(self.store.shutdown())

            # Stop the event loop
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5.0)

        self._loop = None
        self._thread = None
        self.store = None
        self._initialized = False

        logger.info("Real Triple Store Adapter teardown complete")

    def configure(self, config: OptimizerConfig) -> None:
        """
        Apply runtime configuration to the triple store.

        Args:
            config: Optimizer configuration with batch_size, cache_size, etc.
        """
        if not self.store:
            raise RuntimeError("Store not initialized. Call setup() first.")

        # Map optimizer config to store settings
        # The store might not directly support all optimizer params,
        # so we do best-effort mapping
        logger.info(
            f"Applying config: batch_size={config.batch_size}, "
            f"cache_size={config.cache_size}, "
            f"query_timeout={config.query_timeout_ms}ms"
        )

        # Store config updates if the store supports runtime reconfiguration
        # For now, we just log the configuration
        pass

    def load_corpus(self, corpus: Dict[str, Any]) -> None:
        """
        Load synthetic corpus into the triple store.

        Args:
            corpus: Dictionary with entities, chunks, triples, and manifest
        """
        if not self.store:
            raise RuntimeError("Store not initialized. Call setup() first.")

        start_time = time.time()

        # Load entities
        entities = corpus.get("entities", [])
        for entity_data in entities:
            # Convert to dict if needed, ensuring required fields
            if not isinstance(entity_data, dict):
                entity_data = {
                    "id": entity_data.id,
                    "name": entity_data.name,
                    "type": entity_data.type,
                    "properties": {},
                }

            # Use create_entity which expects: name, entity_type, properties, embedding
            self._run_coro_threadsafe(
                self.store.create_entity(
                    name=entity_data.get("name", ""),
                    entity_type=entity_data.get("type", "unknown"),
                    properties=entity_data.get("properties", {}),
                    embedding=entity_data.get("embedding", None),
                )
            )

        # Load chunks (with embeddings if available)
        chunks = corpus.get("chunks", [])
        for chunk_data in chunks:
            # For now, we store chunks as entities with type="chunk"
            if not isinstance(chunk_data, dict):
                chunk_data = {"content": str(chunk_data)}

            self._run_coro_threadsafe(
                self.store.create_entity(
                    name=chunk_data.get("content", "")[:100],  # Use first 100 chars as name
                    entity_type="chunk",
                    properties=chunk_data,
                    embedding=chunk_data.get("embedding", None),
                )
            )

        # Load triples (relationships)
        # The store uses PostgreSQL for entities and Neo4j for relationships
        # We need to create relationships directly through Neo4j adapter
        triples = corpus.get("triples", [])
        for triple_data in triples:
            if isinstance(triple_data, dict):
                # Create relationship through Neo4j
                self._run_coro_threadsafe(
                    self.store.neo4j.create_relationship(
                        source_id=triple_data.get("subjectId", ""),
                        target_id=triple_data.get("objectId", ""),
                        relationship_type=triple_data.get("predicate", "RELATED_TO"),
                        properties=triple_data.get("properties", {}),
                    )
                )

        load_time = (time.time() - start_time) * 1000
        logger.info(
            f"Loaded corpus: {len(entities)} entities, "
            f"{len(chunks)} chunks, {len(triples)} triples "
            f"in {load_time:.2f}ms"
        )

    def execute_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a query against the triple store.

        Args:
            query: Query specification with type, parameters, etc.

        Returns:
            Query results including data, latency, and success status
        """
        if not self.store:
            raise RuntimeError("Store not initialized. Call setup() first.")

        start_time = time.time()
        self.query_count += 1

        try:
            query_type = query.get("type", "unknown")
            params = query.get("params", {})

            # Route query based on type using available TripleStore methods
            if query_type == "entity_lookup":
                result = self._run_coro_threadsafe(self.store.get_entity(params.get("entity_id")))
            elif query_type == "relationship_query":
                # Use find_by_relationship for relationship queries
                result = self._run_coro_threadsafe(
                    self.store.find_by_relationship(
                        entity_id=params.get("subject_id"),
                        relationship_type=params.get("predicate", "RELATED_TO"),
                        direction=params.get("direction", "outgoing"),
                    )
                )
            elif query_type == "semantic_search":
                # Use search_similar for semantic search
                result = self._run_coro_threadsafe(
                    self.store.search_similar(
                        query_text=params.get("query_text"),
                        limit=params.get("limit", 10),
                        threshold=params.get("threshold", 0.7),
                    )
                )
            elif query_type == "path_finding":
                # Neo4j adapter might have path finding
                result = self._run_coro_threadsafe(
                    self.store.neo4j.find_shortest_path(
                        start_id=params.get("start_id"),
                        end_id=params.get("end_id"),
                        max_depth=params.get("max_depth", 3),
                    )
                )
            else:
                # Default to entity lookup
                result = self._run_coro_threadsafe(
                    self.store.get_entity(params.get("entity_id", "unknown"))
                )

            latency_ms = (time.time() - start_time) * 1000
            self.total_latency_ms += latency_ms

            return {
                "success": True,
                "data": result,
                "latency_ms": latency_ms,
                "query_type": query_type,
            }

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Query execution failed: {e}")

            return {
                "success": False,
                "error": str(e),
                "latency_ms": latency_ms,
                "query_type": query.get("type", "unknown"),
            }

    def execute_queries(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute multiple queries in sequence.

        Args:
            queries: List of query specifications

        Returns:
            List of query results
        """
        return [self.execute_query(q) for q in queries]

    def health_check(self) -> bool:
        """
        Check if the triple store is healthy.

        Returns:
            True if all databases are connected and responding
        """
        if not self.store:
            return False

        try:
            # Get health status from the store
            health = self._run_coro_threadsafe(self.store.get_health_status())

            # Check if all adapters are healthy
            return all(
                [
                    health.get("postgres", {}).get("connected", False),
                    health.get("neo4j", {}).get("connected", False),
                    health.get("qdrant", {}).get("connected", False),
                ]
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_resource_usage(self) -> Dict[str, float]:
        """
        Get current resource usage metrics from all databases.

        Returns:
            Dictionary with memory_mb, cpu_percent, connections, etc.
        """
        if not self.store:
            return {
                "memory_mb": 0.0,
                "cpu_percent": 0.0,
                "connections_active": 0,
                "cache_hits": 0,
                "cache_misses": 0,
            }

        # Get metrics from the store using available method
        metrics = self._run_coro_threadsafe(self.store.get_health_status())

        # Add adapter-level metrics
        metrics.update(
            {
                "total_queries": self.query_count,
                "avg_latency_ms": self.total_latency_ms / max(self.query_count, 1),
            }
        )

        return metrics

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------

    def _run_loop(self):
        """Run the event loop in the background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_coro_threadsafe(self, coro) -> Any:
        """
        Submit a coroutine to the event loop and wait for the result.

        Args:
            coro: Coroutine to execute

        Returns:
            The result of the coroutine

        Raises:
            RuntimeError: If the event loop is not running
        """
        if not self._loop or not self._loop.is_running():
            raise RuntimeError("Event loop is not running")

        future: Future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=60.0)  # 60 second timeout for slow connections

    def __enter__(self):
        """Context manager entry - setup the adapter."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - teardown the adapter."""
        self.teardown()
