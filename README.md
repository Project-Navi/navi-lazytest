# Navi LazyTest

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![License: Commercial](https://img.shields.io/badge/License-Commercial-green.svg)](docs/legal/PNEUL-D_v2.2.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Set it and forget it testing.**

A self-improving test framework that learns from each iteration to optimize system configuration using UCB (Upper Confidence Bound) bandit algorithms.

Built for lazy developers who hate babysitting tests.

## Philosophy

> "I built this because I'm lazy and I hate how current Python testing works."

LazyTest embodies one principle: **you shouldn't have to think about it.**

- Run it, walk away, come back to optimized configs
- Failures inform learning, not just red/green
- Progressive difficulty so you don't configure complexity
- Never throws away what it learned

## Who Should Use LazyTest

**Infrastructure teams** who need to optimize CI under limited budget.

**Data-platform and RAG infra owners** (vector DBs, graph DBs, relational stores) who want to automatically discover brittle configurations and prioritize high-value tests.

**Product teams** that need an auditable, repeatable mechanism to discover configuration tradeoffs and quarantine flaky or dangerous setups.

### Why It Fits Multi-Store Stacks

LazyTest treats any test target as a pluggable adapter — whether your target is a relational DB, a vector store (Qdrant/pgvector), or a graph DB (Neo4j). Combined with RRF-style fusion and configurable rewards, LazyTest can help you optimize retrieval and indexing configs across the entire retrieval stack.

## Features

- **Self-Improving**: UCB optimization learns optimal configurations
- **Progressive Rings**: Three levels of increasing complexity and chaos
- **Deterministic**: Seeded generation for reproducible tests
- **Observable**: Full receipt system with metrics database
- **Resilient**: Quarantine/blacklist for flaky configs
- **Extensible**: Bring your own adapter for any test target

## Installation

```bash
pip install navi-lazytest

# With HDR histogram support (recommended)
pip install navi-lazytest[hdr]

# Full installation
pip install navi-lazytest[full]
```

## Quick Start

```python
from navi_lazytest import LazyTestRunner, MockAdapter

# Dry run with mock adapter
adapter = MockAdapter()
runner = LazyTestRunner(adapter=adapter)
runner.run(max_iterations=10)

# That's it. Walk away.
```

## How It Works

### Testing Rings

```
Ring 1 (Basic)     → Ring 2 (Chaos)      → Ring 3 (Governance)
   ↓                    ↓                      ↓
Standard tests      Mutation tests        Compliance tests
100 entities        500 entities          2000 entities
Low mutations       Medium mutations      High mutations
```

Progression happens automatically when performance plateaus. You don't decide when - the framework does.

### UCB Optimization

The framework uses Upper Confidence Bound to balance exploration vs exploitation:

```
UCB = mean_reward + c * sqrt(2 * ln(total_runs) / config_runs)
      └─exploit──┘   └────────────explore────────────────┘
```

Translation: it tries new configs enough to learn, then exploits what works.

### The Learning Loop

```
1. Select config (UCB picks intelligently)
2. Run tests with that config
3. Measure reward (success rate, latency, quality)
4. Update UCB statistics
5. Repeat
6. You drink coffee
```

## Implementing Your Adapter

### 10-Minute Adapter Checklist

Getting started is fast. Follow this checklist:

1. **Copy the template**
   ```bash
   cp examples/adapters/template_adapter.py my_adapter.py
   ```

2. **Fill in connection logic** in `setup()` and `teardown()`

3. **Implement query execution** in `execute_query()` — return `{success, latency_ms, result_count, results}`

4. **Run a quick test**
   ```python
   from my_adapter import MyAdapter
   from navi_lazytest import LazyTestRunner

   runner = LazyTestRunner(adapter=MyAdapter("localhost:5432"))
   runner.run(max_iterations=5)
   ```

5. **Check your receipts** in `receipts/` — you're done!

See [`examples/adapters/template_adapter.py`](examples/adapters/template_adapter.py) for a fully-documented starting point.

### Required Methods

| Method | Purpose |
|--------|---------|
| `setup()` | Initialize connections, create schemas |
| `teardown()` | Clean up resources, close connections |
| `configure(config)` | Apply OptimizerConfig parameters |
| `load_corpus(corpus)` | Load test data (entities, chunks, triples) |
| `execute_query(query)` | Run a query, return results with timing |
| `health_check()` | Return True if system is responsive |
| `get_resource_usage()` | Return dict with memory_mb, cpu_percent, etc. |

### Minimal Example

```python
from navi_lazytest import TestTargetAdapter, OptimizerConfig

class MyDatabaseAdapter(TestTargetAdapter):
    def setup(self) -> None:
        self.db = connect_to_database()

    def teardown(self) -> None:
        self.db.close()

    def configure(self, config: OptimizerConfig) -> None:
        self.db.set_batch_size(config.batch_size)

    def load_corpus(self, corpus: dict) -> None:
        for entity in corpus["entities"]:
            self.db.insert(entity)

    def execute_query(self, query: dict) -> dict:
        start = time.time()
        result = self.db.query(query["params"])
        return {
            "success": True,
            "data": result,
            "latency_ms": (time.time() - start) * 1000
        }

    def health_check(self) -> bool:
        return self.db.ping()

    def get_resource_usage(self) -> dict:
        return {"memory_mb": get_memory(), "connections": self.db.pool_size}
```

### Supported Adapters

| Adapter | Target | Status |
|---------|--------|--------|
| [`template_adapter`](examples/adapters/template_adapter.py) | Template/Starting Point | ✅ Available |
| `sqlite_adapter` | SQLite / Relational DBs | 🔜 Coming Soon |
| `qdrant_adapter` | Qdrant / Vector Stores | 🔜 Coming Soon |
| `neo4j_adapter` | Neo4j / Graph DBs | 🔜 Coming Soon |

## CLI Usage

```bash
# Dry run
lazytest --dry-run --max-iterations 5

# Let it rip
lazytest --ring 1 --max-iterations 100

# Resume from where you left off
lazytest --resume

# Verbose (if you must)
lazytest --dry-run --log-level DEBUG
```

## What You Get

- **JSON receipts** for each iteration (audit trail)
- **SQLite database** with queryable metrics
- **Performance trends** with anomaly detection
- **Hall of Fame** tracking top configurations
- **Your weekend back**

## Configuration Space

The optimizer explores 23+ parameters:

```python
OptimizerConfig:
    batch_size: 1-1000
    query_timeout_ms: 100-30000
    cache_enabled: bool
    cache_size_mb: 0-1024
    max_hops: 1-5
    traversal_strategy: bfs/dfs/bidirectional
    vector_k: 5-100
    fusion_strategy: rrf/linear/learned
    # ... you get the idea
```

You don't tune these. LazyTest does.

## Why Not Just Pytest?

| You Want | Pytest | LazyTest |
|----------|--------|----------|
| Run tests | ✅ | ✅ |
| Remember what worked | ❌ | ✅ |
| Tune configs for you | ❌ | ✅ |
| Get smarter over time | ❌ | ✅ |
| Let you be lazy | ❌ | ✅ |

## License

**Dual Licensed:**
- **AGPL-3.0** — Free for open source use ([full text](https://www.gnu.org/licenses/agpl-3.0.html))
- **Commercial (PNEUL-D v2.2)** — For proprietary use, contact [legal@projectnavi.ai](mailto:legal@projectnavi.ai)

See [LICENSE](LICENSE) and [docs/legal/](docs/legal/) for full terms.

### Ethical Support Program

Open source projects may qualify for enhanced support by voluntarily aligning with our ethical principles. This creates no additional license obligations — it's a separate service relationship. See [ETHICAL_SUPPORT_FRAMEWORK.md](docs/legal/ETHICAL_SUPPORT_FRAMEWORK.md).

---

*Part of the [Project Navi](https://github.com/project-navi) ecosystem.*
