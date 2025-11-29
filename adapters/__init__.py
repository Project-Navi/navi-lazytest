"""
SPIRAL Testing Framework - Adapters

Concrete implementations of TestTargetAdapter for various systems.
"""

# Example adapter for triple-store (optional import)
try:
    from .triple_store import RealTripleStoreAdapter
except ImportError:
    RealTripleStoreAdapter = None

__all__ = ["RealTripleStoreAdapter"]
