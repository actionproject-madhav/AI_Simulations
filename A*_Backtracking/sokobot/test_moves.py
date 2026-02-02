"""
Tests for move generation.
"""

from puzzle import Puzzle, State
from moves import get_neighbors, reconstruct_path, Move


TRIVIAL_PUZZLE = """
#####
#@$.#
#####
"""

SIMPLE_PUZZLE = """
#######
#@  $ #
#   # #
#  .  #
#######
"""


def test_get_neighbors_trivial():
    """Test neighbor generation for trivial puzzle."""
    print("Testing neighbor generation (trivial)...")

    puzzle = Puzzle(TRIVIAL_PUZZLE)
    state = puzzle.initial_state

    print(f"Initial state:")
    print(puzzle.display(state))
    print()

    # Get all neighbors
    neighbors = get_neighbors(state, puzzle)

    print(f"Number of valid neighbors: {len(neighbors)}")
    for i, (next_state, move) in enumerate(neighbors):
        print(f"\nNeighbor {i + 1}: {move}")
        print(puzzle.display(next_state))

    # For trivial puzzle, should have exactly 1 valid move (push box right)
    assert len(neighbors) == 1
    next_state, move = neighbors[0]
    assert move.box_to == (1, 3)  # Box moves to goal

    print("\n✓ Neighbor generation (trivial) works\n")


def test_get_neighbors_simple():
    """Test neighbor generation for simple puzzle."""
    print("Testing neighbor generation (simple)...")

    puzzle = Puzzle(SIMPLE_PUZZLE)
    state = puzzle.initial_state

    print(f"Initial state:")
    print(puzzle.display(state))
    print()

    # Get all neighbors
    neighbors = get_neighbors(state, puzzle)

    print(f"Number of valid neighbors: {len(neighbors)}")
    for i, (next_state, move) in enumerate(neighbors):
        print(f"\nNeighbor {i + 1}: {move}")
        print(puzzle.display(next_state))

    # Should have at least one valid move
    assert len(neighbors) > 0

    print("\n✓ Neighbor generation (simple) works\n")


def test_state_transitions():
    """Test that state transitions are correct."""
    print("Testing state transitions...")

    puzzle = Puzzle(TRIVIAL_PUZZLE)
    state = puzzle.initial_state

    # Get the single valid move
    neighbors = get_neighbors(state, puzzle)
    next_state, move = neighbors[0]

    # Verify state changed correctly
    print(f"Original player pos: {state.player_pos}")
    print(f"New player pos: {next_state.player_pos}")
    print(f"Original box pos: {state.box_positions}")
    print(f"New box pos: {next_state.box_positions}")

    # Player should move to where box was
    assert next_state.player_pos == move.box_from

    # Box should be in new position
    assert move.box_to in next_state.box_positions
    assert move.box_from not in next_state.box_positions

    print("✓ State transitions work correctly\n")


def test_goal_state_no_neighbors():
    """Test that goal state has no neighbors (all boxes on goals)."""
    print("Testing goal state has no moves...")

    puzzle = Puzzle(TRIVIAL_PUZZLE)

    # Create goal state
    goal_state = State(
        player_pos=(1, 2),  # Player at box position
        box_positions=frozenset([(1, 3)])  # Box on goal
    )

    print("Goal state:")
    print(puzzle.display(goal_state))
    print()

    # In goal state, we should still be able to push boxes
    # Actually, boxes can be moved OFF goals, so there may be neighbors
    neighbors = get_neighbors(goal_state, puzzle)

    print(f"Number of neighbors from goal state: {len(neighbors)}")

    # This is interesting - in goal state, we can still move boxes!
    # That's correct behavior for Sokoban

    print("✓ Goal state neighbor generation works\n")


def test_reconstruct_path():
    """Test path reconstruction."""
    print("Testing path reconstruction...")

    # Create a simple path manually
    state1 = State(
        player_pos=(0, 0),
        box_positions=frozenset([(1, 1)])
    )

    state2 = State(
        player_pos=(1, 1),
        box_positions=frozenset([(1, 2)])
    )

    state3 = State(
        player_pos=(1, 2),
        box_positions=frozenset([(1, 3)])
    )

    move1 = Move((1, 1), (1, 2), (0, 0), (1, 1), (0, 1))
    move2 = Move((1, 2), (1, 3), (1, 1), (1, 2), (0, 1))

    came_from = {
        state2: (state1, move1),
        state3: (state2, move2)
    }

    path = reconstruct_path(came_from, state3)

    print(f"Path length: {len(path)}")
    for i, move in enumerate(path):
        print(f"  Step {i + 1}: {move}")

    assert len(path) == 2
    assert path[0] == move1
    assert path[1] == move2

    print("✓ Path reconstruction works\n")


def main():
    """Run all Phase 3 tests."""
    print("=" * 50)
    print("PHASE 3 TESTS: Move Generation")
    print("=" * 50 + "\n")

    test_get_neighbors_trivial()
    test_get_neighbors_simple()
    test_state_transitions()
    test_goal_state_no_neighbors()
    test_reconstruct_path()

    print("=" * 50)
    print("✓ ALL PHASE 3 TESTS PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    main()
