"""
Helper functions for Sokoban solver.
"""

from typing import Tuple, Set
from collections import deque
from puzzle import Puzzle, State


# Direction vectors: (delta_row, delta_col)
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)

DIRECTIONS = [UP, DOWN, LEFT, RIGHT]


def add_positions(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> Tuple[int, int]:
    """Add two position tuples."""
    return (pos1[0] + pos2[0], pos1[1] + pos2[1])


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """
    Calculate Manhattan distance between two positions.

    Args:
        pos1: First position (row, col)
        pos2: Second position (row, col)

    Returns:
        Manhattan distance as integer
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def can_player_reach(
    start_pos: Tuple[int, int],
    target_pos: Tuple[int, int],
    puzzle: Puzzle,
    box_positions: Set[Tuple[int, int]]
) -> bool:
    """
    Check if player can reach target position from start position using BFS.
    Boxes are treated as obstacles.

    Args:
        start_pos: Player's current position
        target_pos: Position player wants to reach
        puzzle: The puzzle object
        box_positions: Current box positions (obstacles)

    Returns:
        True if target is reachable, False otherwise
    """
    if start_pos == target_pos:
        return True

    visited = {start_pos}
    queue = deque([start_pos])

    while queue:
        current = queue.popleft()

        for direction in DIRECTIONS:
            next_pos = add_positions(current, direction)

            # Check if next position is valid and unvisited
            if (puzzle.is_valid_floor(next_pos) and
                next_pos not in box_positions and
                next_pos not in visited):

                if next_pos == target_pos:
                    return True

                visited.add(next_pos)
                queue.append(next_pos)

    return False


def is_corner_deadlock(
    box_pos: Tuple[int, int],
    puzzle: Puzzle
) -> bool:
    """
    Check if a box is in a corner deadlock.
    A box in a corner (walls on two perpendicular sides) that is not a goal
    is stuck forever.

    Args:
        box_pos: Position of the box to check
        puzzle: The puzzle object

    Returns:
        True if box is in corner deadlock, False otherwise
    """
    # If box is on a goal, it's not a deadlock
    if puzzle.is_goal(box_pos):
        return False

    r, c = box_pos

    # Check all four corner configurations
    # Top-left: wall above AND wall to the left
    if puzzle.is_wall((r - 1, c)) and puzzle.is_wall((r, c - 1)):
        return True

    # Top-right: wall above AND wall to the right
    if puzzle.is_wall((r - 1, c)) and puzzle.is_wall((r, c + 1)):
        return True

    # Bottom-left: wall below AND wall to the left
    if puzzle.is_wall((r + 1, c)) and puzzle.is_wall((r, c - 1)):
        return True

    # Bottom-right: wall below AND wall to the right
    if puzzle.is_wall((r + 1, c)) and puzzle.is_wall((r, c + 1)):
        return True

    return False


def is_valid_push(
    box_pos: Tuple[int, int],
    direction: Tuple[int, int],
    puzzle: Puzzle,
    state: State
) -> bool:
    """
    Check if pushing a box in a given direction is valid.

    A push is valid if:
    1. The new box position is valid floor (not wall, in bounds)
    2. The new box position doesn't have another box
    3. The player can reach the position behind the box to push from
    4. The box doesn't end up in a corner deadlock

    Args:
        box_pos: Position of box to push
        direction: Direction to push (UP, DOWN, LEFT, RIGHT)
        puzzle: The puzzle object
        state: Current game state

    Returns:
        True if push is valid, False otherwise
    """
    # Calculate new box position after push
    new_box_pos = add_positions(box_pos, direction)

    # Check 1: New box position must be valid floor
    if not puzzle.is_valid_floor(new_box_pos):
        return False

    # Check 2: New box position must not have another box
    if new_box_pos in state.box_positions:
        return False

    # Check 3: Player must be able to reach the push-from position
    # The position the player needs to be to push is opposite the direction
    opposite_direction = (-direction[0], -direction[1])
    push_from_pos = add_positions(box_pos, opposite_direction)

    # Push-from position must be valid floor
    if not puzzle.is_valid_floor(push_from_pos):
        return False

    # Player must be able to reach push-from position
    # For reachability check, treat all boxes except the one being pushed as obstacles
    other_boxes = state.box_positions - {box_pos}
    if not can_player_reach(state.player_pos, push_from_pos, puzzle, other_boxes):
        return False

    # Check 4: Box must not end up in corner deadlock
    if is_corner_deadlock(new_box_pos, puzzle):
        return False

    return True


def calculate_heuristic(state: State, puzzle: Puzzle) -> int:
    """
    Calculate heuristic cost for A* search.
    Uses sum of Manhattan distances from each box to nearest goal.

    Args:
        state: Current game state
        puzzle: The puzzle object

    Returns:
        Heuristic cost estimate
    """
    total = 0
    for box_pos in state.box_positions:
        # Find minimum distance to any goal
        min_dist = min(
            manhattan_distance(box_pos, goal_pos)
            for goal_pos in puzzle.goals
        )
        total += min_dist
    return total
