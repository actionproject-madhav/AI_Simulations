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

    # Player must be able to reach push-from position without walking through any box
    if not can_player_reach(state.player_pos, push_from_pos, puzzle, state.box_positions):
        return False

    # Check 4: Box must not end up in corner deadlock
    if is_corner_deadlock(new_box_pos, puzzle):
        return False

    return True


def hungarian_algorithm(cost_matrix):
    """
    Minimum cost assignment using Hungarian algorithm.
    Uses a simplified but correct implementation with augmenting paths.

    Args:
        cost_matrix: 2D list where cost_matrix[i][j] is cost of assigning worker i to job j

    Returns:
        Minimum total cost
    """
    if not cost_matrix or not cost_matrix[0]:
        return 0

    n = len(cost_matrix)
    m = len(cost_matrix[0])

    # For small matrices, use brute force
    if n <= 3 and m <= 3:
        from itertools import permutations
        min_cost = float('inf')

        # If n != m, we need to handle it differently
        if n <= m:
            for perm in permutations(range(m), n):
                cost = sum(cost_matrix[i][perm[i]] for i in range(n))
                min_cost = min(min_cost, cost)
        else:
            for perm in permutations(range(n), m):
                cost = sum(cost_matrix[perm[j]][j] for j in range(m))
                min_cost = min(min_cost, cost)

        return min_cost

    # For larger matrices, use proper Hungarian algorithm with augmenting paths
    size = max(n, m)

    # Pad matrix to square
    INF = float('inf')
    matrix = [[INF] * size for _ in range(size)]
    for i in range(n):
        for j in range(m):
            matrix[i][j] = cost_matrix[i][j]

    # Subtract row minimums
    for i in range(size):
        row_min = min(matrix[i])
        if row_min != INF:
            for j in range(size):
                if matrix[i][j] != INF:
                    matrix[i][j] -= row_min

    # Subtract column minimums
    for j in range(size):
        col_min = min(matrix[i][j] for i in range(size))
        if col_min != INF:
            for i in range(size):
                if matrix[i][j] != INF:
                    matrix[i][j] -= col_min

    # Find assignment using augmenting paths
    row_match = [-1] * size
    col_match = [-1] * size

    def find_augmenting_path(row, visited_rows, visited_cols):
        visited_rows.add(row)
        for col in range(size):
            if col in visited_cols or matrix[row][col] != 0:
                continue
            visited_cols.add(col)

            if col_match[col] == -1 or find_augmenting_path(col_match[col], visited_rows, visited_cols):
                row_match[row] = col
                col_match[col] = row
                return True
        return False

    # Find maximum matching
    for i in range(size):
        find_augmenting_path(i, set(), set())

    # Calculate total cost
    total_cost = 0
    for i in range(n):
        if row_match[i] != -1 and row_match[i] < m:
            total_cost += cost_matrix[i][row_match[i]]

    return total_cost


def calculate_heuristic(state: State, puzzle: Puzzle) -> int:
    """
    Calculate heuristic cost for A* search using Hungarian algorithm.
    Finds optimal assignment of boxes to goals to minimize total distance.

    Args:
        state: Current game state
        puzzle: The puzzle object

    Returns:
        Heuristic cost estimate
    """
    boxes = list(state.box_positions)
    goals = list(puzzle.goals)

    if not boxes:
        return 0

    # Create cost matrix: cost[i][j] = distance from box i to goal j
    cost_matrix = []
    for box_pos in boxes:
        row = []
        for goal_pos in goals:
            row.append(manhattan_distance(box_pos, goal_pos))
        cost_matrix.append(row)

    # Use Hungarian algorithm to find minimum cost assignment
    return hungarian_algorithm(cost_matrix)
