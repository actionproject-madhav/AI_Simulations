"""
Tests for puzzle parsing and display.
"""

from puzzle import Puzzle, State


# Sample puzzles for testing
TRIVIAL_PUZZLE = """
#####
#@$.#
#####
"""

SIMPLE_PUZZLE = """
  #####
  #   #
  #$  #
### #$##
#   $ .#
# .@.  #
########
"""

MEDIUM_PUZZLE = """
########
#  #   #
#  $   #
# $@$ .#
#  $ #.#
#.     #
########
"""


def test_parse_trivial():
    """Test parsing the trivial 1-box puzzle."""
    print("Testing trivial puzzle parsing...")
    puzzle = Puzzle(TRIVIAL_PUZZLE)

    print(f"Grid dimensions: {puzzle.rows}x{puzzle.cols}")
    print(f"Walls: {len(puzzle.walls)}")
    print(f"Goals: {len(puzzle.goals)}")
    print(f"Initial player position: {puzzle.initial_state.player_pos}")
    print(f"Initial box positions: {puzzle.initial_state.box_positions}")

    assert puzzle.rows == 3
    assert puzzle.cols == 5
    assert puzzle.initial_state.player_pos == (1, 1)
    assert len(puzzle.initial_state.box_positions) == 1
    assert len(puzzle.goals) == 1

    print("✓ Trivial puzzle parsed correctly\n")


def test_display_trivial():
    """Test displaying the trivial puzzle."""
    print("Testing trivial puzzle display...")
    puzzle = Puzzle(TRIVIAL_PUZZLE)

    display = puzzle.display()
    print(display)

    # Check that player, box, and goal are visible
    assert '@' in display
    assert '$' in display
    assert '.' in display

    print("✓ Trivial puzzle displays correctly\n")


def test_parse_simple():
    """Test parsing a simple multi-box puzzle."""
    print("Testing simple puzzle parsing...")
    puzzle = Puzzle(SIMPLE_PUZZLE)

    print(f"Grid dimensions: {puzzle.rows}x{puzzle.cols}")
    print(f"Walls: {len(puzzle.walls)}")
    print(f"Goals: {len(puzzle.goals)}")
    print(f"Boxes: {len(puzzle.initial_state.box_positions)}")

    assert len(puzzle.initial_state.box_positions) == len(puzzle.goals)

    print("✓ Simple puzzle parsed correctly\n")


def test_display_simple():
    """Test displaying the simple puzzle."""
    print("Testing simple puzzle display...")
    puzzle = Puzzle(SIMPLE_PUZZLE)

    display = puzzle.display()
    print(display)
    print()

    print("✓ Simple puzzle displays correctly\n")


def test_goal_state():
    """Test goal state detection."""
    print("Testing goal state detection...")
    puzzle = Puzzle(TRIVIAL_PUZZLE)

    # Initial state should not be goal
    assert not puzzle.is_goal_state(puzzle.initial_state)

    # Create a goal state (box on goal)
    goal_state = State(
        player_pos=(1, 1),
        box_positions=frozenset(puzzle.goals)
    )
    assert puzzle.is_goal_state(goal_state)

    print("✓ Goal state detection works\n")


def test_state_equality():
    """Test that states are comparable and hashable."""
    print("Testing state equality and hashing...")

    state1 = State(
        player_pos=(1, 1),
        box_positions=frozenset([(2, 2), (3, 3)])
    )

    state2 = State(
        player_pos=(1, 1),
        box_positions=frozenset([(2, 2), (3, 3)])
    )

    state3 = State(
        player_pos=(1, 2),
        box_positions=frozenset([(2, 2), (3, 3)])
    )

    assert state1 == state2
    assert state1 != state3
    assert hash(state1) == hash(state2)

    # Test that states can be added to sets
    state_set = {state1, state2, state3}
    assert len(state_set) == 2  # state1 and state2 are the same

    print("✓ State equality and hashing works\n")


def test_wall_and_bounds():
    """Test wall and bounds checking."""
    print("Testing wall and bounds checking...")
    puzzle = Puzzle(TRIVIAL_PUZZLE)

    # Test wall detection
    assert puzzle.is_wall((0, 0))  # Top-left corner
    assert not puzzle.is_wall((1, 1))  # Player starting position

    # Test bounds checking
    assert puzzle.is_in_bounds((0, 0))
    assert puzzle.is_in_bounds((2, 4))
    assert not puzzle.is_in_bounds((-1, 0))
    assert not puzzle.is_in_bounds((3, 0))

    # Test valid floor
    assert puzzle.is_valid_floor((1, 1))
    assert not puzzle.is_valid_floor((0, 0))  # Wall

    print("✓ Wall and bounds checking works\n")


def main():
    """Run all tests."""
    print("=" * 50)
    print("PHASE 1 TESTS: Core Data Structures")
    print("=" * 50 + "\n")

    test_parse_trivial()
    test_display_trivial()
    test_parse_simple()
    test_display_simple()
    test_goal_state()
    test_state_equality()
    test_wall_and_bounds()

    print("=" * 50)
    print("✓ ALL PHASE 1 TESTS PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    main()
