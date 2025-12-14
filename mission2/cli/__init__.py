"""CLI module for Connect4."""

def main() -> None:
    """Entry point wrapper to avoid importing CLI module at package import time."""
    from .main import main as _main

    _main()


__all__ = ["main"]
