"""
CLI entry point for ribbon_solver_group_theory.

Usage:
    python -m ribbon_solver_group_theory --symmetries "sin²(x) + cos²(x)"
    python -m ribbon_solver_group_theory --orbit "exp(x)"
    python -m ribbon_solver_group_theory --formula pi
"""

from .discover import main

if __name__ == "__main__":
    main()
