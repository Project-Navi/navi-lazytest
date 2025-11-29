"""
Query Suite for Triple Store Testing
Ring 1: Basic breadth and depth query patterns
"""

import json
import random
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class QueryCategory(Enum):
    """Categories for organizing test queries"""

    # Breadth queries - wide but shallow
    ENTITY_LOOKUP = "entity_lookup"
    ENTITY_SEARCH = "entity_search"
    VECTOR_SIMILARITY = "vector_similarity"
    GRAPH_NEIGHBORS = "graph_neighbors"

    # Depth queries - narrow but deep
    MULTI_HOP_TRAVERSAL = "multi_hop_traversal"
    PATH_FINDING = "path_finding"
    SUBGRAPH_EXTRACTION = "subgraph_extraction"
    RECURSIVE_EXPANSION = "recursive_expansion"

    # Hybrid queries - both breadth and depth
    FUSION_SEARCH = "fusion_search"
    CONTEXTUAL_RETRIEVAL = "contextual_retrieval"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"

    # Edge cases
    EMPTY_RESULTS = "empty_results"
    LARGE_RESULTS = "large_results"
    MALFORMED = "malformed"


@dataclass
class Query:
    """A single test query with metadata"""

    id: str
    category: QueryCategory
    query_text: str
    query_params: Dict[str, Any]
    expected_characteristics: Dict[str, Any]
    complexity_score: float  # 0-1, for prioritization
    is_canonical: bool  # True for core queries that must always pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.value,
            "query_text": self.query_text,
            "query_params": self.query_params,
            "expected": self.expected_characteristics,
            "complexity": self.complexity_score,
            "canonical": self.is_canonical,
        }


class QuerySuite:
    """
    Manages the test query collection for the triple store.
    Ring 1: 20-30 queries covering basic breadth and depth
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.queries: List[Query] = []
        self._build_suite()

    def _build_suite(self):
        """Build the complete query suite"""
        self._add_breadth_queries()
        self._add_depth_queries()
        self._add_hybrid_queries()
        self._add_edge_case_queries()

    def _add_breadth_queries(self):
        """Add queries that test wide but shallow retrieval"""

        # Entity lookups
        self.queries.append(
            Query(
                id="breadth_entity_exact",
                category=QueryCategory.ENTITY_LOOKUP,
                query_text="Find entity by exact name",
                query_params={"entity_name": "PLACEHOLDER_ENTITY", "exact_match": True},
                expected_characteristics={
                    "result_count": 1,
                    "response_time_ms": 50,
                    "uses_index": True,
                },
                complexity_score=0.1,
                is_canonical=True,
            )
        )

        self.queries.append(
            Query(
                id="breadth_entity_fuzzy",
                category=QueryCategory.ENTITY_SEARCH,
                query_text="Search entities with fuzzy matching",
                query_params={
                    "search_term": "PLACEHOLDER_TERM",
                    "fuzzy_threshold": 0.7,
                    "limit": 20,
                },
                expected_characteristics={
                    "min_results": 1,
                    "max_results": 20,
                    "response_time_ms": 200,
                    "relevance_ordered": True,
                },
                complexity_score=0.3,
                is_canonical=True,
            )
        )

        # Vector similarity
        self.queries.append(
            Query(
                id="breadth_vector_knn",
                category=QueryCategory.VECTOR_SIMILARITY,
                query_text="Find k-nearest neighbors by embedding",
                query_params={"query_text": "PLACEHOLDER_TEXT", "k": 10, "threshold": 0.5},
                expected_characteristics={
                    "result_count": 10,
                    "response_time_ms": 300,
                    "similarity_scores": True,
                    "score_range": [0.5, 1.0],
                },
                complexity_score=0.4,
                is_canonical=True,
            )
        )

        # Graph neighbors
        self.queries.append(
            Query(
                id="breadth_1hop_neighbors",
                category=QueryCategory.GRAPH_NEIGHBORS,
                query_text="Get all 1-hop neighbors of entity",
                query_params={
                    "entity_id": "PLACEHOLDER_ID",
                    "hop_count": 1,
                    "relation_filter": None,
                },
                expected_characteristics={
                    "min_results": 0,
                    "max_results": 100,
                    "response_time_ms": 150,
                    "includes_relations": True,
                },
                complexity_score=0.2,
                is_canonical=True,
            )
        )

        self.queries.append(
            Query(
                id="breadth_typed_relations",
                category=QueryCategory.GRAPH_NEIGHBORS,
                query_text="Get neighbors by relation type",
                query_params={
                    "entity_id": "PLACEHOLDER_ID",
                    "relation_types": ["isa", "partof", "relatedto"],
                    "direction": "outgoing",
                },
                expected_characteristics={"filtered_by_type": True, "response_time_ms": 200},
                complexity_score=0.3,
                is_canonical=False,
            )
        )

    def _add_depth_queries(self):
        """Add queries that test deep traversal"""

        # Multi-hop traversal
        self.queries.append(
            Query(
                id="depth_3hop_traversal",
                category=QueryCategory.MULTI_HOP_TRAVERSAL,
                query_text="Traverse 3 hops from entity",
                query_params={"start_entity": "PLACEHOLDER_ID", "max_hops": 3, "max_paths": 10},
                expected_characteristics={
                    "traversal_depth": 3,
                    "response_time_ms": 500,
                    "path_structure": True,
                },
                complexity_score=0.6,
                is_canonical=True,
            )
        )

        # Path finding
        self.queries.append(
            Query(
                id="depth_shortest_path",
                category=QueryCategory.PATH_FINDING,
                query_text="Find shortest path between entities",
                query_params={
                    "source": "PLACEHOLDER_SOURCE",
                    "target": "PLACEHOLDER_TARGET",
                    "max_depth": 5,
                },
                expected_characteristics={
                    "finds_optimal": True,
                    "response_time_ms": 800,
                    "path_length": [1, 5],
                },
                complexity_score=0.7,
                is_canonical=True,
            )
        )

        # Subgraph extraction
        self.queries.append(
            Query(
                id="depth_subgraph_community",
                category=QueryCategory.SUBGRAPH_EXTRACTION,
                query_text="Extract subgraph around entity",
                query_params={"center_entity": "PLACEHOLDER_ID", "radius": 2, "max_nodes": 50},
                expected_characteristics={
                    "returns_graph": True,
                    "node_count": [5, 50],
                    "edge_count": [4, 200],
                    "response_time_ms": 1000,
                },
                complexity_score=0.8,
                is_canonical=False,
            )
        )

        # Recursive expansion
        self.queries.append(
            Query(
                id="depth_recursive_ancestors",
                category=QueryCategory.RECURSIVE_EXPANSION,
                query_text="Find all ancestors recursively",
                query_params={
                    "entity_id": "PLACEHOLDER_ID",
                    "relation": "partof",
                    "max_depth": None,  # Unlimited
                },
                expected_characteristics={
                    "handles_cycles": True,
                    "response_time_ms": 600,
                    "complete_closure": True,
                },
                complexity_score=0.7,
                is_canonical=False,
            )
        )

    def _add_hybrid_queries(self):
        """Add queries combining vector, graph, and text search"""

        self.queries.append(
            Query(
                id="hybrid_fusion_search",
                category=QueryCategory.FUSION_SEARCH,
                query_text="Combined vector + graph search",
                query_params={
                    "query": "PLACEHOLDER_QUERY",
                    "vector_weight": 0.5,
                    "graph_weight": 0.3,
                    "text_weight": 0.2,
                    "limit": 20,
                },
                expected_characteristics={
                    "uses_multiple_indexes": True,
                    "response_time_ms": 400,
                    "score_fusion": True,
                },
                complexity_score=0.6,
                is_canonical=True,
            )
        )

        self.queries.append(
            Query(
                id="hybrid_contextual_retrieval",
                category=QueryCategory.CONTEXTUAL_RETRIEVAL,
                query_text="Retrieve with graph context",
                query_params={
                    "query": "PLACEHOLDER_QUERY",
                    "expand_context": True,
                    "context_hops": 1,
                    "aggregate_chunks": True,
                },
                expected_characteristics={
                    "enriched_results": True,
                    "response_time_ms": 600,
                    "includes_provenance": True,
                },
                complexity_score=0.7,
                is_canonical=True,
            )
        )

        self.queries.append(
            Query(
                id="hybrid_knowledge_synthesis",
                category=QueryCategory.KNOWLEDGE_SYNTHESIS,
                query_text="Synthesize knowledge from multiple sources",
                query_params={
                    "topic": "PLACEHOLDER_TOPIC",
                    "synthesis_depth": 2,
                    "source_diversity": 0.7,
                    "max_sources": 10,
                },
                expected_characteristics={
                    "multi_source": True,
                    "response_time_ms": 1200,
                    "coherent_narrative": True,
                },
                complexity_score=0.9,
                is_canonical=False,
            )
        )

    def _add_edge_case_queries(self):
        """Add queries that test edge cases and error handling"""

        self.queries.append(
            Query(
                id="edge_empty_results",
                category=QueryCategory.EMPTY_RESULTS,
                query_text="Query with no matches",
                query_params={"entity_name": "NONEXISTENT_ENTITY_XYZ123", "strict": True},
                expected_characteristics={
                    "result_count": 0,
                    "response_time_ms": 50,
                    "graceful_empty": True,
                },
                complexity_score=0.1,
                is_canonical=True,
            )
        )

        self.queries.append(
            Query(
                id="edge_large_fanout",
                category=QueryCategory.LARGE_RESULTS,
                query_text="Query with large result set",
                query_params={"entity_type": "common_type", "limit": 1000, "offset": 0},
                expected_characteristics={
                    "handles_pagination": True,
                    "response_time_ms": 2000,
                    "memory_efficient": True,
                },
                complexity_score=0.5,
                is_canonical=False,
            )
        )

        self.queries.append(
            Query(
                id="edge_malformed_query",
                category=QueryCategory.MALFORMED,
                query_text="Malformed query parameters",
                query_params={
                    "invalid_field": "test",
                    "hop_count": -1,  # Invalid
                    "limit": "not_a_number",  # Wrong type
                },
                expected_characteristics={
                    "error_handling": True,
                    "validation_message": True,
                    "no_crash": True,
                },
                complexity_score=0.2,
                is_canonical=True,
            )
        )

    def get_queries(
        self,
        categories: Optional[List[QueryCategory]] = None,
        canonical_only: bool = False,
        max_complexity: float = 1.0,
    ) -> List[Query]:
        """
        Get filtered subset of queries

        Args:
            categories: Filter by category
            canonical_only: Only return canonical queries
            max_complexity: Maximum complexity score

        Returns:
            Filtered list of queries
        """
        queries = self.queries

        if categories:
            queries = [q for q in queries if q.category in categories]

        if canonical_only:
            queries = [q for q in queries if q.is_canonical]

        queries = [q for q in queries if q.complexity_score <= max_complexity]

        return queries

    def get_canonical_queries(self) -> List[Query]:
        """Get only the canonical queries that must always pass"""
        return self.get_queries(canonical_only=True)

    def instantiate_query(self, query: Query, corpus_data: Dict[str, Any]) -> Query:
        """
        Replace placeholders in query with actual data from corpus

        Args:
            query: Query template with placeholders
            corpus_data: Actual data to use (entities, texts, etc.)

        Returns:
            Query with placeholders replaced
        """
        import copy

        instantiated = copy.deepcopy(query)

        # Replace placeholders based on query needs
        if "PLACEHOLDER_ENTITY" in str(instantiated.query_params):
            entity = self.rng.choice(corpus_data.get("entities", ["TestEntity"]))
            instantiated.query_params = self._replace_in_dict(
                instantiated.query_params,
                "PLACEHOLDER_ENTITY",
                entity.name if hasattr(entity, "name") else str(entity),
            )

        if "PLACEHOLDER_ID" in str(instantiated.query_params):
            entity = self.rng.choice(corpus_data.get("entities", [{"id": "test-1"}]))
            instantiated.query_params = self._replace_in_dict(
                instantiated.query_params,
                "PLACEHOLDER_ID",
                entity.id if hasattr(entity, "id") else entity.get("id", "test-1"),
            )

        if "PLACEHOLDER_TEXT" in str(instantiated.query_params):
            chunks = corpus_data.get("chunks", [])
            if not chunks:
                # Provide a default chunk if none available
                chunks = [{"content": "test text", "text": "test text"}]
            chunk = self.rng.choice(chunks)
            # Try 'text' first, then 'content', then fallback
            text_value = chunk.get("text", chunk.get("content", "test"))
            instantiated.query_params = self._replace_in_dict(
                instantiated.query_params, "PLACEHOLDER_TEXT", text_value
            )

        # Handle source/target for path queries
        if "PLACEHOLDER_SOURCE" in str(instantiated.query_params):
            entities = corpus_data.get("entities", [])
            if len(entities) >= 2:
                source, target = self.rng.sample(entities, 2)
                instantiated.query_params = self._replace_in_dict(
                    instantiated.query_params,
                    "PLACEHOLDER_SOURCE",
                    source.id if hasattr(source, "id") else source.get("id"),
                )
                instantiated.query_params = self._replace_in_dict(
                    instantiated.query_params,
                    "PLACEHOLDER_TARGET",
                    target.id if hasattr(target, "id") else target.get("id"),
                )

        return instantiated

    def _replace_in_dict(self, d: Dict, placeholder: str, value: Any) -> Dict:
        """Recursively replace placeholder in dictionary"""
        result = {}
        for k, v in d.items():
            if isinstance(v, str) and placeholder in v:
                result[k] = v.replace(placeholder, str(value))
            elif isinstance(v, dict):
                result[k] = self._replace_in_dict(v, placeholder, value)
            else:
                result[k] = v
        return result

    def save_to_file(self, filepath: str):
        """Save query suite to JSON file"""
        suite_data = {
            "seed": self.seed,
            "timestamp": datetime.now().isoformat(),
            "query_count": len(self.queries),
            "queries": [q.to_dict() for q in self.queries],
        }

        with open(filepath, "w") as f:
            json.dump(suite_data, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "QuerySuite":
        """Load query suite from JSON file"""
        with open(filepath, "r") as f:
            data = json.load(f)

        suite = cls(seed=data["seed"])
        suite.queries = []

        for q_data in data["queries"]:
            suite.queries.append(
                Query(
                    id=q_data["id"],
                    category=QueryCategory(q_data["category"]),
                    query_text=q_data["query_text"],
                    query_params=q_data["query_params"],
                    expected_characteristics=q_data["expected"],
                    complexity_score=q_data["complexity"],
                    is_canonical=q_data["canonical"],
                )
            )

        return suite
