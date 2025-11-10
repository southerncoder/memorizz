"""CLI commands for Memorizz."""

import sys


def setup_oracle():
    """Run Oracle database setup."""
    try:
        from memorizz.memory_provider.oracle import setup_oracle_user

        return setup_oracle_user()
    except ImportError as e:
        print(f"✗ Failed to import setup module: {e}")
        print("\nPlease ensure memorizz[oracle] is installed:")
        print("  pip install memorizz[oracle]")
        return False


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Memorizz CLI")
        print("\nAvailable commands:")
        print("  setup-oracle    Set up Oracle database schema")
        print("\nUsage:")
        print("  memorizz setup-oracle")
        print("  python -m memorizz.cli setup-oracle")
        sys.exit(1)

    command = sys.argv[1]

    if command == "setup-oracle":
        success = setup_oracle()
        sys.exit(0 if success else 1)
    else:
        print(f"✗ Unknown command: {command}")
        print("Run 'memorizz' or 'python -m memorizz.cli' for help")
        sys.exit(1)


if __name__ == "__main__":
    main()
