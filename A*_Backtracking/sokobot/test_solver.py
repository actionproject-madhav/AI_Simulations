#!/usr/bin/env python3
"""
Test the Sokoban solver on various puzzles to ensure it works correctly.
"""

from puzzle import Puzzle
from solver import solve_and_display

# Test puzzles
PUZZLES = {
    "trivial": """
#####
#@$.#
#####
""",
    "corner": """
#######
#  .  #
# $@$ #
#  .  #
#######
""",
    "line_push": """
#########
#  ...  #
# @ $$$ #
#       #
#########
""",
    "tutorial": """
#####
#@$.#
#####
""",
    "easy": """
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
}


def test_puzzle(name, puzzle_string, verbose=True):
    """Test a single puzzle."""
    print("=" * 60)
    print(f"Testing puzzle: {name}")
    print("=" * 60)
    
    try:
        puzzle = Puzzle(puzzle_string)
        result = solve_and_display(puzzle, verbose=verbose)
        
        if result.solved:
            print(f"✓ SUCCESS: Found solution with {result.solution_length} moves")
            print(f"  States explored: {result.states_explored}")
            return True
        else:
            print(f"✗ FAILED: {result.message}")
            print(f"  States explored: {result.states_explored}")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SOKOBAN SOLVER TEST SUITE")
    print("=" * 60 + "\n")
    
    results = {}
    
    for name, puzzle_string in PUZZLES.items():
        success = test_puzzle(name, puzzle_string, verbose=True)
        results[name] = success
        print("\n")
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for s in results.values() if s)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
