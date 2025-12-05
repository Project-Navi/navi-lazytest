"""
Navi LazyTest - Example Adapters
================================

Example adapters demonstrating how to integrate various systems with LazyTest.

Quick Start:
    1. Copy template_adapter.py to your project
    2. Rename the class and fill in your connection/query logic
    3. Run with: runner.run(max_iterations=5)

Available Adapters:
    template_adapter    - Documented template (start here!)
    sqlite_adapter      - SQLite/relational database (coming soon)
    qdrant_adapter      - Qdrant vector store (coming soon)
    neo4j_adapter       - Neo4j graph database (coming soon)
"""

from .template_adapter import YourSystemAdapter

__all__ = ["YourSystemAdapter"]
