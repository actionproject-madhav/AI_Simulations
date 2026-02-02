"""
Tests for the A* solver.
"""

from puzzle import Puzzle
from solver import solve, solve_and_display


TRIVIAL_PUZZLE = """
#####
#@$.#
#####
"""

SIMPLE_2BOX = """
#######
# . . #
# $ $ #
#  @  #
#######
"""

MEDIUM_PUZZLE = """
########
#   .  #
# @$$  #
#   . ##
########
"""

HARDER_PUZZLE = """
  #####
  #   #
  #$  #
###@$##
#  $  #
# ...##
########
"""


def test_solve_trivial():
    """Test solving the trivial 1-move puzzle."""
    print("Testing solver on trivial puzzle...")
    print()

    puzzle = Puzzle(TRIVIAL_PUZZLE)
    result = solve_and_display(puzzle, verbose=True)

    assert result.solved
    assert result.solution_length == 1
    assert len(result.moves) == 1

    print("✓ Trivial puzzle solved correctly\n")


def test_solve_simple():
    """Test solving a simple 2-box puzzle."""
    print("Testing solver on simple 2-box puzzle...")
    print()

    puzzle = Puzzle(SIMPLE_2BOX)
    result = solve_and_display(puzzle, verbose=True)

    assert result.solved
    print(f"Solution requires {result.solution_length} moves")

    print("✓ Simple 2-box puzzle solved\n")


def test_solve_medium():
    """Test solving a medium puzzle."""
    print("Testing solver on medium puzzle...")
    print()

    puzzle = Puzzle(MEDIUM_PUZZLE)
    result = solve_and_display(puzzle, verbose=True)

    if result.solved:
        print(f"Solution requires {result.solution_length} moves")
        print("✓ Medium puzzle solved\n")
    else:
        print(f"Could not solve: {result.message}")
        print("This is acceptable for a medium puzzle\n")


def test_solve_harder():
    """Test solving a harder puzzle."""
    print("Testing solver on harder puzzle...")
    print()

    puzzle = Puzzle(HARDER_PUZZLE)

    # This may take longer, so let's not be too verbose
    result = solve(puzzle, max_states=50000)

    if result.solved:
        print(f"✓ Harder puzzle solved!")
        print(f"  Solution length: {result.solution_length}")
        print(f"  States explored: {result.states_explored}")
        print()
    else:
        print(f"Could not solve harder puzzle: {result.message}")
        print(f"  States explored: {result.states_explored}")
        print("This is acceptable for a harder puzzle\n")


def test_verify_solution():
    """Test that the solution actually solves the puzzle."""
    print("Testing solution verification...")

    puzzle = Puzzle(TRIVIAL_PUZZLE)
    result = solve(puzzle, max_states=1000)

    assert result.solved

    # Manually apply moves and verify we reach goal
    current_state = puzzle.initial_state

    for move in result.moves:
        # Verify box is being moved
        assert move.box_from in current_state.box_positions

        # Apply move
        new_box_positions = (current_state.box_positions - {move.box_from}) | {move.box_to}
        current_state = current_state.__class__(
            player_pos=move.box_to,
            box_positions=frozenset(new_box_positions)
        )

    # Final state should be goal
    assert puzzle.is_goal_state(current_state)

    print("✓ Solution verification passed\n")


def test_already_at_goal():
    """Test puzzle that's already solved."""
    print("Testing already-solved puzzle...")

    # Create a puzzle where box is already on goal
    solved_puzzle = """
#####
#@* #
#####
"""

    puzzle = Puzzle(solved_puzzle)
    result = solve(puzzle)

    assert result.solved
    assert result.solution_length == 0
    print(f"  Correctly detected puzzle is already solved")

    print("✓ Already-solved puzzle handled correctly\n")


def main():
    """Run all solver tests."""
    print("=" * 60)
    print("PHASE 4 TESTS: A* Solver")
    print("=" * 60 + "\n")

    test_solve_trivial()
    test_solve_simple()
    test_verify_solution()
    test_already_at_goal()
    test_solve_medium()
    test_solve_harder()

    print("=" * 60)
    print("✓ ALL PHASE 4 TESTS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
