"""
Data Generators for Spiral Test Suite
Deterministic, reproducible synthetic data generation
"""

import hashlib
import json
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Define simple models for testing
# These match the database schema but are simplified for test data generation


@dataclass
class Entity:
    """Simple entity model for testing"""

    id: str
    name: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Simple chunk model for testing"""

    id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Triple:
    """Simple triple model for testing"""

    subject: str
    predicate: str
    object: str
    properties: Dict[str, Any] = field(default_factory=dict)


# Define enums as simple strings for testing
class EntityType:
    SERVICE = "service"
    DATABASE = "database"
    LIBRARY = "library"
    FRAMEWORK = "framework"
    CONCEPT = "concept"
    OTHER = "other"


class PredicateKind:
    USES = "uses"
    DEPENDS_ON = "depends_on"
    IMPLEMENTS = "implements"
    OWNS = "owns"
    AUTHORED_BY = "authored_by"
    IS_A = "is_a"
    DEFINES = "defines"
    CALLS = "calls"
    RELATED_TO = "related_to"


@dataclass
class GeneratorManifest:
    """Manifest for reproducible data generation"""

    seed: int
    timestamp: str
    counts: Dict[str, int]
    distributions: Dict[str, Any]
    perturbations: List[str]
    embedder_model: str = "text-embedding-3-small"
    embedder_version: str = "v1"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "GeneratorManifest":
        return cls(**json.loads(json_str))

    def get_hash(self) -> str:
        """Deterministic hash for this configuration"""
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:12]


class DataGenerator:
    """Base class for deterministic data generation"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.manifest = GeneratorManifest(
            seed=seed,
            timestamp=datetime.utcnow().isoformat(),
            counts={},
            distributions={},
            perturbations=[],
        )

    def reset(self):
        """Reset generator to initial state"""
        self.rng = random.Random(self.seed)

    def generate_uuid(self) -> str:
        """Generate deterministic UUID"""
        return str(uuid.UUID(int=self.rng.getrandbits(128), version=4))

    def generate_hash(self, content: str) -> str:
        """Generate content hash"""
        return hashlib.sha256(f"{self.seed}:{content}".encode()).hexdigest()

    def get_manifest(self) -> Dict[str, Any]:
        """Get generation manifest as dictionary"""
        return asdict(self.manifest)


class SyntheticCorpusGenerator(DataGenerator):
    """Generate synthetic corpus with controlled properties"""

    # Templates for generating realistic content
    TECH_CONCEPTS = [
        "microservices",
        "kubernetes",
        "docker",
        "database",
        "api",
        "authentication",
        "caching",
        "queue",
        "serverless",
        "monitoring",
        "testing",
        "deployment",
        "security",
        "encryption",
        "backup",
    ]

    PREDICATES_MAP = {
        "technical": [PredicateKind.USES, PredicateKind.DEPENDS_ON, PredicateKind.IMPLEMENTS],
        "organizational": [PredicateKind.OWNS, PredicateKind.AUTHORED_BY],
        "structural": [PredicateKind.IS_A, PredicateKind.DEFINES, PredicateKind.CALLS],
    }

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.entities_cache = {}
        self.chunks_cache = []
        self.triples_cache = []

    def generate_entities(
        self,
        count: int = 100,
        alias_rate: float = 0.3,
        type_distribution: Optional[Dict[EntityType, float]] = None,
    ) -> List[Entity]:
        """Generate entities with controlled properties"""

        if type_distribution is None:
            type_distribution = {
                EntityType.SERVICE: 0.3,
                EntityType.DATABASE: 0.1,
                EntityType.LIBRARY: 0.2,
                EntityType.FRAMEWORK: 0.1,
                EntityType.CONCEPT: 0.2,
                EntityType.OTHER: 0.1,
            }

        entities = []

        for i in range(count):
            # Choose type based on distribution
            entity_type = self.rng.choices(
                list(type_distribution.keys()), weights=list(type_distribution.values())
            )[0]

            # Generate name
            base_name = self.rng.choice(self.TECH_CONCEPTS)
            suffix = f"_{i}" if self.rng.random() < 0.3 else ""
            name = f"{base_name}{suffix}"

            # Create entity
            entity = {
                "id": self.generate_uuid(),
                "name": name,
                "type": entity_type,
                "tags": self._generate_tags(entity_type),
                "description": self._generate_description(name, entity_type),
                "createdAt": datetime.utcnow().isoformat(),
            }

            entities.append(entity)
            self.entities_cache[entity["id"]] = entity

            # Generate aliases
            if self.rng.random() < alias_rate:
                aliases = self._generate_aliases(name, entity["id"])
                entity["aliases"] = aliases

        self.manifest.counts["entities"] = len(entities)
        self.manifest.distributions["entity_types"] = type_distribution
        self.manifest.distributions["alias_rate"] = alias_rate

        return entities

    def generate_chunks(
        self, count: int = 1000, files: int = 100, chunk_size_range: Tuple[int, int] = (100, 500)
    ) -> List[Dict]:
        """Generate chunks with realistic content"""

        chunks = []
        run_id = self.generate_uuid()

        # Generate source files
        source_files = []
        for f in range(files):
            file_path = f"src/module_{f}/file_{self.rng.randint(1, 100)}.py"
            source_file = {
                "id": self.generate_uuid(),
                "runId": run_id,
                "path": file_path,
                "sha256": self.generate_hash(file_path),
                "size": self.rng.randint(1000, 10000),
                "language": "python",
                "createdAt": datetime.utcnow().isoformat(),
            }
            source_files.append(source_file)

        # Generate chunks
        chunks_per_file = count // files
        for file in source_files:
            for c in range(chunks_per_file):
                content = self._generate_code_content(self.rng.randint(*chunk_size_range))

                chunk = {
                    "id": self.generate_uuid(),
                    "chunkId": f"chunk_{file['id'][:8]}_{c}",
                    "runId": run_id,
                    "fileId": file["id"],
                    "repoPath": file["path"],
                    "startLine": c * 20,
                    "endLine": (c + 1) * 20,
                    "content": content,
                    "contentHash": self.generate_hash(content),
                    "blessingScore": self.rng.uniform(0.3, 0.9),
                    "blessingTier": self.rng.choice(["FINAL", "TENTATIVE", "REFUSAL"]),
                    "phase": self.rng.choice(["COMPOST", "REFLECTION", "BECOMING"]),
                    "qdrantProjected": False,
                    "neo4jProjected": False,
                    "createdAt": datetime.utcnow().isoformat(),
                }

                chunks.append(chunk)
                self.chunks_cache.append(chunk)

        self.manifest.counts["chunks"] = len(chunks)
        self.manifest.counts["files"] = len(source_files)
        self.manifest.distributions["chunk_size_range"] = chunk_size_range

        return chunks

    def generate_triples(
        self, density: float = 2.0, graph_shape: str = "small_world"
    ) -> List[Dict]:
        """Generate triples with controlled graph topology"""

        if not self.entities_cache:
            raise ValueError("Generate entities first")

        triples = []
        entity_ids = list(self.entities_cache.keys())
        run_id = self.generate_uuid()

        # Calculate number of triples based on density
        num_triples = int(len(entity_ids) * density)

        # Generate based on graph shape
        if graph_shape == "small_world":
            triples = self._generate_small_world_triples(entity_ids, num_triples, run_id)
        elif graph_shape == "tree":
            triples = self._generate_tree_triples(entity_ids, num_triples, run_id)
        elif graph_shape == "dag":
            triples = self._generate_dag_triples(entity_ids, num_triples, run_id)
        else:
            # Random graph
            for _ in range(num_triples):
                triple = self._generate_random_triple(entity_ids, run_id)
                triples.append(triple)

        self.triples_cache = triples
        self.manifest.counts["triples"] = len(triples)
        self.manifest.distributions["triple_density"] = density
        self.manifest.distributions["graph_shape"] = graph_shape

        return triples

    def generate_pii_fixtures(self, count: int = 50) -> List[Dict]:
        """Generate controlled PII data for testing redaction"""

        pii_patterns = {
            "email": lambda: f"user{self.rng.randint(1, 1000)}@example.com",
            "phone": lambda: f"+1-555-{self.rng.randint(100, 999)}-{self.rng.randint(1000, 9999)}",
            "ssn": lambda: f"{self.rng.randint(100, 999)}-{self.rng.randint(10, 99)}-{self.rng.randint(1000, 9999)}",
            "api_key": lambda: f"sk_test_{self.generate_hash('key')[:32]}",
            "credit_card": lambda: f"{self.rng.randint(4000, 4999)}-****-****-{self.rng.randint(1000, 9999)}",
        }

        fixtures = []
        for i in range(count):
            pii_type = self.rng.choice(list(pii_patterns.keys()))
            value = pii_patterns[pii_type]()

            fixture = {
                "id": self.generate_uuid(),
                "type": pii_type,
                "value": value,
                "context": self._generate_pii_context(pii_type, value),
                "should_redact": True,
                "createdAt": datetime.utcnow().isoformat(),
            }
            fixtures.append(fixture)

        self.manifest.counts["pii_fixtures"] = len(fixtures)
        self.manifest.perturbations.append("pii_injection")

        return fixtures

    # Helper methods
    def _generate_tags(self, entity_type: EntityType) -> List[str]:
        """Generate relevant tags for entity type"""
        tag_pool = {
            EntityType.SERVICE: ["api", "backend", "frontend", "microservice"],
            EntityType.DATABASE: ["sql", "nosql", "cache", "persistent"],
            EntityType.LIBRARY: ["utility", "framework", "package", "dependency"],
            EntityType.FRAMEWORK: ["web", "testing", "orm", "middleware"],
            EntityType.CONCEPT: ["pattern", "architecture", "design", "methodology"],
            EntityType.OTHER: ["misc", "tool", "config", "script"],
        }

        tags = tag_pool.get(entity_type, ["other"])
        return self.rng.sample(tags, k=min(2, len(tags)))

    def _generate_description(self, name: str, entity_type: EntityType) -> str:
        """Generate description for entity"""
        templates = [
            f"A {entity_type.lower()} component for {name} functionality",
            f"Manages {name} operations and related processes",
            f"Implementation of {name} with optimized performance",
            f"Core {entity_type.lower()} handling {name} requirements",
        ]
        return self.rng.choice(templates)

    def _generate_aliases(self, name: str, entity_id: str) -> List[Dict]:
        """Generate aliases for entity"""
        aliases = []
        num_aliases = self.rng.randint(1, 3)

        for _ in range(num_aliases):
            alias_variations = [
                name.lower(),
                name.upper(),
                name.replace("_", "-"),
                f"{name}_v2",
                name[:3] + name[-3:] if len(name) > 6 else name,
            ]

            alias = {
                "name": self.rng.choice(alias_variations),
                "entityId": entity_id,
                "confidence": self.rng.uniform(0.7, 1.0),
                "source": self.rng.choice(["import", "reference", "documentation"]),
            }
            aliases.append(alias)

        return aliases

    def _generate_code_content(self, size: int) -> str:
        """Generate realistic code content"""
        lines = []
        indent = 0

        # Generate imports
        for _ in range(self.rng.randint(1, 5)):
            module = self.rng.choice(self.TECH_CONCEPTS)
            lines.append(f"import {module}")

        lines.append("")

        # Generate class or function
        if self.rng.random() < 0.5:
            class_name = self.rng.choice(self.TECH_CONCEPTS).capitalize()
            lines.append(f"class {class_name}:")
            indent = 4
        else:
            func_name = self.rng.choice(self.TECH_CONCEPTS).lower()
            lines.append(f"def {func_name}():")
            indent = 4

        # Add content until size is reached
        current_size = sum(len(line) for line in lines)
        while current_size < size:
            line_type = self.rng.choice(["comment", "code", "docstring"])

            if line_type == "comment":
                comment = f"# {self.rng.choice(['TODO', 'FIXME', 'NOTE'])}: Handle {self.rng.choice(self.TECH_CONCEPTS)}"
                lines.append(" " * indent + comment)
            elif line_type == "code":
                var = self.rng.choice(["result", "data", "response", "config"])
                value = self.rng.choice(["None", "[]", "{}", "''", "0"])
                lines.append(" " * indent + f"{var} = {value}")
            else:
                lines.append(" " * indent + '"""')
                lines.append(" " * indent + f"Process {self.rng.choice(self.TECH_CONCEPTS)}")
                lines.append(" " * indent + '"""')

            current_size = sum(len(line) for line in lines)

        return "\n".join(lines)

    def _generate_small_world_triples(
        self, entity_ids: List[str], num_triples: int, run_id: str
    ) -> List[Dict]:
        """Generate small-world graph topology"""
        triples = []

        # Create local clusters
        cluster_size = max(3, len(entity_ids) // 10)
        for i in range(0, len(entity_ids), cluster_size):
            cluster = entity_ids[i : i + cluster_size]

            # Connect within cluster
            for j in range(len(cluster) - 1):
                if len(triples) < num_triples:
                    triple = self._create_triple(
                        cluster[j], cluster[j + 1], PredicateKind.RELATED_TO, run_id
                    )
                    triples.append(triple)

        # Add long-range connections
        while len(triples) < num_triples:
            src = self.rng.choice(entity_ids)
            dst = self.rng.choice(entity_ids)
            if src != dst:
                predicate = self.rng.choice(
                    [
                        PredicateKind.USES,
                        PredicateKind.DEPENDS_ON,
                        PredicateKind.IMPLEMENTS,
                        PredicateKind.OWNS,
                        PredicateKind.AUTHORED_BY,
                        PredicateKind.IS_A,
                        PredicateKind.DEFINES,
                        PredicateKind.CALLS,
                        PredicateKind.RELATED_TO,
                    ]
                )
                triple = self._create_triple(src, dst, predicate, run_id)
                triples.append(triple)

        return triples

    def _generate_tree_triples(
        self, entity_ids: List[str], num_triples: int, run_id: str
    ) -> List[Dict]:
        """Generate tree topology"""
        triples = []

        # Create tree structure
        for i in range(1, min(len(entity_ids), num_triples + 1)):
            parent_idx = (i - 1) // 2
            triple = self._create_triple(
                entity_ids[parent_idx], entity_ids[i], PredicateKind.IS_A, run_id
            )
            triples.append(triple)

        return triples

    def _generate_dag_triples(
        self, entity_ids: List[str], num_triples: int, run_id: str
    ) -> List[Dict]:
        """Generate DAG topology"""
        triples = []

        # Sort entities for DAG property
        sorted_ids = sorted(entity_ids)

        for i in range(len(sorted_ids)):
            # Connect to future nodes only (DAG property)
            num_connections = min(3, len(sorted_ids) - i - 1)
            if num_connections > 0:
                targets = self.rng.sample(
                    sorted_ids[i + 1 :], k=min(num_connections, len(sorted_ids[i + 1 :]))
                )
                for target in targets:
                    if len(triples) < num_triples:
                        predicate = self.rng.choice(
                            [PredicateKind.DEPENDS_ON, PredicateKind.USES, PredicateKind.CALLS]
                        )
                        triple = self._create_triple(sorted_ids[i], target, predicate, run_id)
                        triples.append(triple)

        return triples

    def _generate_random_triple(self, entity_ids: List[str], run_id: str) -> Dict:
        """Generate a random triple"""
        subject = self.rng.choice(entity_ids)
        object = self.rng.choice([e for e in entity_ids if e != subject])
        predicate = self.rng.choice(
            [
                PredicateKind.USES,
                PredicateKind.DEPENDS_ON,
                PredicateKind.IMPLEMENTS,
                PredicateKind.OWNS,
                PredicateKind.AUTHORED_BY,
                PredicateKind.IS_A,
                PredicateKind.DEFINES,
                PredicateKind.CALLS,
                PredicateKind.RELATED_TO,
            ]
        )

        return self._create_triple(subject, object, predicate, run_id)

    def _create_triple(
        self, subject_id: str, object_id: str, predicate: PredicateKind, run_id: str
    ) -> Dict:
        """Create a triple dictionary"""
        chunk = self.rng.choice(self.chunks_cache) if self.chunks_cache else None

        triple_str = f"{subject_id}:{predicate}:{object_id}"

        return {
            "id": self.generate_uuid(),
            "chunkRowId": chunk["id"] if chunk else self.generate_uuid(),
            "ingestRunId": run_id,
            "subjectId": subject_id,
            "predicate": predicate,
            "objectId": object_id,
            "confidence": self.rng.uniform(0.6, 1.0),
            "source": "synthetic_generator",
            "tripleHash": self.generate_hash(triple_str),
            "createdAt": datetime.utcnow().isoformat(),
        }

    def _generate_pii_context(self, pii_type: str, value: str) -> str:
        """Generate context containing PII"""
        contexts = {
            "email": f"Contact the user at {value} for more information",
            "phone": f"Call {value} to schedule an appointment",
            "ssn": f"SSN: {value} - Confidential",
            "api_key": f"API_KEY={value} # DO NOT COMMIT",
            "credit_card": f"Payment method: {value}",
        }
        return contexts.get(pii_type, f"Contains {pii_type}: {value}")


# Golden dataset for exact correctness testing
class GoldenDatasetGenerator(SyntheticCorpusGenerator):
    """Generate dataset with known correct answers"""

    def generate_golden_queries(self) -> List[Dict]:
        """Generate queries with expected results"""

        if not self.entities_cache or not self.triples_cache:
            raise ValueError("Generate entities and triples first")

        queries = []

        # Type 1: Direct entity lookup
        for _ in range(5):
            entity = self.rng.choice(list(self.entities_cache.values()))
            query = {
                "id": self.generate_uuid(),
                "type": "entity_lookup",
                "query": f"Find information about {entity['name']}",
                "expected_entities": [entity["id"]],
                "expected_triples": [
                    t["id"]
                    for t in self.triples_cache
                    if t["subjectId"] == entity["id"] or t["objectId"] == entity["id"]
                ],
            }
            queries.append(query)

        # Type 2: Relationship queries
        for _ in range(5):
            triple = self.rng.choice(self.triples_cache)
            subject = self.entities_cache[triple["subjectId"]]
            object = self.entities_cache[triple["objectId"]]

            query = {
                "id": self.generate_uuid(),
                "type": "relationship",
                "query": f"How is {subject['name']} related to {object['name']}?",
                "expected_entities": [triple["subjectId"], triple["objectId"]],
                "expected_triples": [triple["id"]],
                "expected_path": [triple["subjectId"], triple["objectId"]],
            }
            queries.append(query)

        # Type 3: Semantic search
        for _ in range(10):
            chunk = self.rng.choice(self.chunks_cache)
            keywords = chunk["content"].split()[:5]

            query = {
                "id": self.generate_uuid(),
                "type": "semantic_search",
                "query": " ".join(self.rng.sample(keywords, k=min(3, len(keywords)))),
                "expected_chunks": [chunk["id"]],
                "expected_entities": [],
            }
            queries.append(query)

        return queries
