#!/usr/bin/env python3
"""
Command-line interface for Sokoban solver.
"""

import sys
import argparse
from puzzle import Puzzle
from solver import solve_and_display, solve


# Collection of sample puzzles
SAMPLE_PUZZLES = {
    "trivial": """
#####
#@$.#
#####
""",
    "simple": """
#######
# . . #
# $ $ #
#  @  #
#######
""",
    "medium": """
########
#   .  #
# @$$  #
#   . ##
########
""",
    "harder": """
  #####
  #   #
  #$  #
###@$##
#  $  #
# ...##
########
""",
    "challenge": """
#########
#   #   #
# $   $ #
### # ###
# $ @ $ #
# .   . #
## . . ##
#########
""",
}


def solve_from_file(filepath: str, verbose: bool = True, max_states: int = 100000):
    """
    Load and solve a puzzle from a file.

    Args:
        filepath: Path to puzzle file
        verbose: Whether to display solution steps
        max_states: Maximum states to explore
    """
    try:
        with open(filepath, 'r') as f:
            puzzle_string = f.read()

        puzzle = Puzzle(puzzle_string)

        if verbose:
            result = solve_and_display(puzzle)
        else:
            result = solve(puzzle, max_states=max_states)
            print(f"Solved: {result.solved}")
            if result.solved:
                print(f"Solution length: {result.solution_length}")
                print(f"States explored: {result.states_explored}")
            else:
                print(f"Reason: {result.message}")

    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def solve_sample(name: str, verbose: bool = True):
    """
    Solve a built-in sample puzzle.

    Args:
        name: Name of the sample puzzle
        verbose: Whether to display solution steps
    """
    if name not in SAMPLE_PUZZLES:
        print(f"Error: Unknown sample puzzle '{name}'")
        print(f"Available puzzles: {', '.join(SAMPLE_PUZZLES.keys())}")
        sys.exit(1)

    puzzle_string = SAMPLE_PUZZLES[name]
    puzzle = Puzzle(puzzle_string)

    print(f"Solving sample puzzle: {name}")
    print("=" * 60)
    solve_and_display(puzzle, verbose=verbose)


def list_samples():
    """List all available sample puzzles."""
    print("Available sample puzzles:")
    print()
    for name, puzzle_string in SAMPLE_PUZZLES.items():
        puzzle = Puzzle(puzzle_string)
        print(f"{name}:")
        print(f"  Size: {puzzle.rows}x{puzzle.cols}")
        print(f"  Boxes: {len(puzzle.initial_state.box_positions)}")
        print(f"  Goals: {len(puzzle.goals)}")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sokoban Solver - Solve Sokoban puzzles using A* search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --sample trivial          Solve a trivial 1-box puzzle
  %(prog)s --sample medium           Solve a medium puzzle
  %(prog)s --list                    List all sample puzzles
  %(prog)s --file puzzle.txt         Solve puzzle from file
  %(prog)s --file puzzle.txt --quiet Solve without showing steps
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--sample',
        type=str,
        help='Solve a sample puzzle (trivial, simple, medium, harder, challenge)'
    )
    group.add_argument(
        '--file',
        type=str,
        help='Solve puzzle from file'
    )
    group.add_argument(
        '--list',
        action='store_true',
        help='List all available sample puzzles'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only show summary, not step-by-step solution'
    )

    parser.add_argument(
        '--max-states',
        type=int,
        default=100000,
        help='Maximum states to explore (default: 100000)'
    )

    args = parser.parse_args()

    if args.list:
        list_samples()
    elif args.sample:
        solve_sample(args.sample, verbose=not args.quiet)
    elif args.file:
        solve_from_file(args.file, verbose=not args.quiet, max_states=args.max_states)


if __name__ == "__main__":
    main()
