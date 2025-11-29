"""
Simplified Mutators for Triple Store Testing
"""

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MutationRecord:
    """Record of applied mutations"""

    type: str
    count: int
    intensity: float


class CompositeMutator:
    """Simplified composite mutator for testing"""

    def __init__(self, seed: int = 42, intensity: float = 0.1, ring: int = 1):
        self.seed = seed
        self.intensity = intensity
        self.ring = ring
        self.rng = random.Random(seed)
        self.mutations_applied = []

    def mutate(self, data: Any) -> Any:
        """Apply mutations based on ring and intensity"""
        mutated = copy.deepcopy(data)

        # Simple mutations for testing
        if self.ring >= 1 and "entities" in mutated:
            # Inject some typos
            for entity in mutated.get("entities", []):
                if self.rng.random() < self.intensity:
                    if "name" in entity:
                        entity["name"] = entity["name"] + "_mutated"
                    self.mutations_applied.append(MutationRecord("typo", 1, self.intensity))

        if self.ring >= 2 and "triples" in mutated:
            # Remove some relations
            triples = mutated.get("triples", [])
            num_to_remove = int(len(triples) * self.intensity * 0.1)
            for _ in range(num_to_remove):
                if triples:
                    triples.pop()
                    self.mutations_applied.append(
                        MutationRecord("missing_relation", 1, self.intensity)
                    )

        return mutated

    def get_mutation_summary(self) -> Dict[str, Any]:
        """Get summary of mutations applied"""
        return {
            "mutations": [
                {"type": m.type, "count": m.count, "intensity": m.intensity}
                for m in self.mutations_applied
            ],
            "total_mutations": len(self.mutations_applied),
            "ring": self.ring,
            "seed": self.seed,
        }
