#!/usr/bin/env python3
"""
Simple verification script to demonstrate the solver works correctly.
This is a minimal test without UI.
"""

from puzzle import Puzzle
from solver import solve

def test_simple_puzzle():
    """Test a simple puzzle to verify solver works."""
    
    puzzle_string = """
#####
#@$.#
#####
"""
    
    print("="*60)
    print("SOKOBAN SOLVER VERIFICATION")
    print("="*60)
    print("\nTesting with simple puzzle:")
    print(puzzle_string)
    
    puzzle = Puzzle(puzzle_string)
    
    print("Initial state:")
    print(puzzle.display())
    print()
    
    print("Running A* solver...")
    result = solve(puzzle)
    
    if result.solved:
        print(f"\n✓ SUCCESS!")
        print(f"  Solution found: {result.solution_length} moves")
        print(f"  States explored: {result.states_explored}")
        print(f"\nSolution:")
        for i, move in enumerate(result.moves, 1):
            print(f"  {i}. {move}")
        return True
    else:
        print(f"\n✗ FAILED!")
        print(f"  {result.message}")
        return False


def test_medium_puzzle():
    """Test a medium difficulty puzzle."""
    
    puzzle_string = """
########
#   .  #
# @$$  #
#   . ##
########
"""
    
    print("\n" + "="*60)
    print("Testing medium difficulty puzzle:")
    print("="*60)
    print(puzzle_string)
    
    puzzle = Puzzle(puzzle_string)
    
    print("Initial state:")
    print(puzzle.display())
    print()
    
    print("Running A* solver...")
    result = solve(puzzle, max_states=10000)
    
    if result.solved:
        print(f"\n✓ SUCCESS!")
        print(f"  Solution found: {result.solution_length} moves")
        print(f"  States explored: {result.states_explored}")
        print(f"\nSolution:")
        for i, move in enumerate(result.moves, 1):
            print(f"  {i}. {move}")
        return True
    else:
        print(f"\n✗ FAILED!")
        print(f"  {result.message}")
        return False


def test_hard_puzzle():
    """Test a harder puzzle."""
    
    puzzle_string = """
  #####
  #   #
  #$  #
###@$ #
#  $  #
# ...##
########
"""
    
    print("\n" + "="*60)
    print("Testing hard puzzle:")
    print("="*60)
    print(puzzle_string)
    
    puzzle = Puzzle(puzzle_string)
    
    print("Initial state:")
    print(puzzle.display())
    print()
    
    print("Running A* solver...")
    result = solve(puzzle, max_states=50000)
    
    if result.solved:
        print(f"\n✓ SUCCESS!")
        print(f"  Solution found: {result.solution_length} moves")
        print(f"  States explored: {result.states_explored}")
        print(f"\nSolution:")
        for i, move in enumerate(result.moves, 1):
            print(f"  {i}. {move}")
        return True
    else:
        print(f"\n✗ FAILED!")
        print(f"  {result.message}")
        return False


def main():
    results = []
    
    results.append(("Simple", test_simple_puzzle()))
    results.append(("Medium", test_medium_puzzle()))
    results.append(("Hard", test_hard_puzzle()))
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name} puzzle: {'PASSED' if success else 'FAILED'}")
    
    all_passed = all(success for _, success in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Solver is working correctly!")
    else:
        print("✗ SOME TESTS FAILED - Solver needs fixing!")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
