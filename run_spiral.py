#!/usr/bin/env python
"""
Run the Self-Improving Spiral Test Suite
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging

# Import orchestrator
from tests.spiral.orchestrator import MockAdapter, SpiralOrchestrator


def main():
    """Main entry point for running the spiral test suite"""

    parser = argparse.ArgumentParser(
        description="Self-Improving Spiral Test Suite for Triple Store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run with mock adapter (for testing)
  python run_spiral.py --dry-run --max-iterations 5

  # Start from Ring 1 with 10 iterations
  python run_spiral.py --ring 1 --max-iterations 10

  # Resume from saved state
  python run_spiral.py --resume

  # Verbose logging
  python run_spiral.py --dry-run --log-level DEBUG
        """,
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Use mock adapter for testing the orchestrator"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=10, help="Maximum iterations to run (default: 10)"
    )
    parser.add_argument(
        "--ring", type=int, choices=[1, 2, 3], help="Start at specific ring (default: 1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from saved state")
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("orchestrator_state.json"),
        help="State file path (default: orchestrator_state.json)",
    )
    parser.add_argument(
        "--receipts-dir",
        type=Path,
        default=Path("receipts"),
        help="Directory for test receipts (default: ./receipts)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument("--log-file", type=Path, help="Optional log file path")

    args = parser.parse_args()

    # Setup logging
    log_handlers = [logging.StreamHandler()]
    if args.log_file:
        log_handlers.append(logging.FileHandler(args.log_file))

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=log_handlers,
    )

    logger = logging.getLogger(__name__)

    # Print banner
    print("=" * 70)
    print("🌀 Self-Improving Spiral Test Suite for Triple Store 🌀")
    print("=" * 70)
    print(f"Mode: {'DRY RUN (Mock)' if args.dry_run else 'LIVE'}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"Seed: {args.seed}")
    print(f"State File: {args.state_file}")
    print(f"Receipts Directory: {args.receipts_dir}")
    print(f"Log Level: {args.log_level}")
    print("=" * 70)
    print()

    try:
        # Create adapter
        if args.dry_run:
            logger.info("Creating mock adapter for dry run")
            adapter = MockAdapter(failure_rate=0.05, base_latency_ms=50.0)  # 5% query failure rate
        else:
            # Import real adapter
            from triple_store.models.config import StoreConfig

            from tests.spiral.triple_store_adapter import RealTripleStoreAdapter

            logger.info("Creating real triple store adapter")
            # Create config from environment or defaults
            store_config = StoreConfig.from_env()  # Load from environment variables
            adapter = RealTripleStoreAdapter(config=store_config)

        # Create orchestrator
        logger.info("Initializing orchestrator...")
        orchestrator = SpiralOrchestrator(
            adapter=adapter,
            state_path=args.state_file,
            receipts_dir=args.receipts_dir,
            dry_run=args.dry_run,
            seed=args.seed,
        )

        # Handle resume vs fresh start
        if args.resume and args.state_file.exists():
            logger.info(f"Resuming from saved state: {args.state_file}")
        else:
            if args.ring:
                orchestrator.context.current_ring = args.ring
                logger.info(f"Starting at Ring {args.ring}")
            else:
                logger.info("Starting fresh from Ring 1")

        # Run the spiral
        logger.info("Starting spiral loop...")
        print("\n" + "=" * 70)
        print("SPIRAL LOOP STARTING")
        print("=" * 70 + "\n")

        orchestrator.run(max_iterations=args.max_iterations)

        print("\n" + "=" * 70)
        print("SPIRAL LOOP COMPLETED")
        print("=" * 70)

        # Print summary
        print("\nSummary:")
        print(f"  Total Iterations: {orchestrator.context.global_iteration}")
        print(f"  Final Ring: {orchestrator.context.current_ring}")
        print(f"  Best Reward: {orchestrator.context.best_reward:.3f}")
        if orchestrator.context.best_config:
            print("  Best Config:")
            for key, value in orchestrator.context.best_config.items():
                print(f"    {key}: {value}")

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\n\nInterrupted by user. State has been saved.")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
