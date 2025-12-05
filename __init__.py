"""
Navi LazyTest - Set it and forget it testing

Copyright (c) 2024-2025 Project Navi. All rights reserved.

SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-PNEUL-D-2.2

This software is dual-licensed:
- AGPL-3.0-or-later for open source use
- PNEUL-D v2.2 commercial license (contact legal@projectnavi.ai)

A self-improving test framework that learns from each iteration to optimize
system configuration using UCB (Upper Confidence Bound) bandit algorithms.

Built for lazy developers who hate babysitting tests.

Core Concepts:
- Ring-based progressive testing (Basic → Chaos → Governance)
- UCB optimization for configuration space exploration
- Deterministic synthetic data generation
- Comprehensive observability with receipt system
- Chaos engineering through controlled mutations

Usage:
    from navi_lazytest import LazyTestRunner, TestTargetAdapter, MockAdapter

    # Create your adapter (or use MockAdapter for dry runs)
    adapter = MockAdapter()

    # Run it and walk away
    runner = LazyTestRunner(adapter=adapter)
    runner.run(max_iterations=100)
"""

__version__ = "1.0.0"

# Core orchestration
from .orchestrator import (
    SpiralOrchestrator as LazyTestRunner,  # New name
    SpiralOrchestrator,  # Backwards compat
    TestTargetAdapter,
    TripleStoreAdapter,  # Backwards compatibility alias
    MockAdapter,
    OrchestratorState,
    OrchestratorContext,
)

# Optimization
from .optimizer import (
    UCBOptimizer,
    OptimizerConfig,
    RewardCalculator,
    RunResult,
    RunStatus,
    generate_config_space,
)

# Data generation
from .generators import SyntheticCorpusGenerator

# Chaos engineering
from .mutators import CompositeMutator

# Query templates
from .query_suite import QuerySuite, Query, QueryCategory

# Metrics and observability
from .observers import (
    MetricsCollector,
    PerformanceTracker,
    ReceiptStore,
    TestReceipt,
)

# Evaluation
from .evaluator import Evaluator

__all__ = [
    # Version
    "__version__",
    # Core
    "LazyTestRunner",
    "SpiralOrchestrator",  # Backwards compat
    "TestTargetAdapter",
    "TripleStoreAdapter",  # Backwards compat
    "MockAdapter",
    "OrchestratorState",
    "OrchestratorContext",
    # Optimization
    "UCBOptimizer",
    "OptimizerConfig",
    "RewardCalculator",
    "RunResult",
    "RunStatus",
    "generate_config_space",
    # Generation
    "SyntheticCorpusGenerator",
    # Mutation
    "CompositeMutator",
    # Queries
    "QuerySuite",
    "Query",
    "QueryCategory",
    # Observability
    "MetricsCollector",
    "PerformanceTracker",
    "ReceiptStore",
    "TestReceipt",
    # Evaluation
    "Evaluator",
]
