"""
Integration tests - end-to-end solver tests.
"""

from puzzle import Puzzle
from solver import solve
import time


def test_all_samples():
    """Test all sample puzzles from main.py."""
    print("Testing all sample puzzles...")
    print("=" * 60)

    samples = {
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
    }

    results = []

    for name, puzzle_string in samples.items():
        print(f"\nTesting {name}...")
        puzzle = Puzzle(puzzle_string)

        start_time = time.time()
        result = solve(puzzle, max_states=50000)
        elapsed = time.time() - start_time

        status = "✓" if result.solved else "✗"
        print(f"  {status} Solved: {result.solved}")
        if result.solved:
            print(f"  Solution length: {result.solution_length}")
            print(f"  States explored: {result.states_explored}")
            print(f"  Time: {elapsed:.3f}s")
        else:
            print(f"  Reason: {result.message}")

        results.append({
            'name': name,
            'solved': result.solved,
            'length': result.solution_length if result.solved else None,
            'states': result.states_explored,
            'time': elapsed
        })

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for r in results:
        status = "✓" if r['solved'] else "✗"
        length_str = f"{r['length']} moves" if r['solved'] else "N/A"
        print(f"{status} {r['name']:10s} | {length_str:10s} | "
              f"{r['states']:5d} states | {r['time']:.3f}s")

    print("=" * 60)

    # Check that at least the simple ones solved
    assert results[0]['solved'], "Trivial puzzle should solve"
    assert results[1]['solved'], "Simple puzzle should solve"
    assert results[2]['solved'], "Medium puzzle should solve"

    print("\n✓ ALL INTEGRATION TESTS PASSED!\n")


def test_performance_limits():
    """Test that solver respects max_states limit."""
    print("Testing max_states limit...")

    # Create a puzzle that might be complex
    puzzle_string = """
#########
#   #   #
# $   $ #
### # ###
# $ @ $ #
# .   . #
## . . ##
#########
"""

    puzzle = Puzzle(puzzle_string)

    # Limit to very few states
    result = solve(puzzle, max_states=10)

    print(f"  States explored: {result.states_explored}")
    print(f"  Solved: {result.solved}")

    # Should either solve quickly or hit the limit
    assert result.states_explored <= 11  # May explore 1 more before checking

    print("✓ Max states limit works\n")


def test_unsolvable_detection():
    """Test that unsolvable puzzles are detected."""
    print("Testing unsolvable puzzle detection...")

    # Puzzle where box is in a corner deadlock from the start
    unsolvable = """
#####
#@$ #
### #
# . #
#####
"""

    puzzle = Puzzle(unsolvable)
    result = solve(puzzle, max_states=1000)

    print(f"  Solved: {result.solved}")
    print(f"  States explored: {result.states_explored}")

    # Should not solve (box is in corner deadlock)
    # Corner deadlock detection should prevent any moves
    assert not result.solved

    print("✓ Unsolvable puzzle correctly detected\n")


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("INTEGRATION TESTS")
    print("=" * 60 + "\n")

    test_all_samples()
    test_performance_limits()
    test_unsolvable_detection()

    print("=" * 60)
    print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
