"""
Move generation for Sokoban solver.
"""

from typing import List, Tuple
from puzzle import Puzzle, State
from helpers import is_valid_push, add_positions, DIRECTIONS


class Move:
    """Represents a single push move in the solution."""

    def __init__(
        self,
        box_from: Tuple[int, int],
        box_to: Tuple[int, int],
        player_from: Tuple[int, int],
        player_to: Tuple[int, int],
        direction: Tuple[int, int]
    ):
        self.box_from = box_from
        self.box_to = box_to
        self.player_from = player_from
        self.player_to = player_to
        self.direction = direction

    def __repr__(self):
        direction_names = {
            (-1, 0): "UP",
            (1, 0): "DOWN",
            (0, -1): "LEFT",
            (0, 1): "RIGHT"
        }
        dir_name = direction_names.get(self.direction, "UNKNOWN")
        return f"Push box from {self.box_from} to {self.box_to} ({dir_name})"


def get_neighbors(state: State, puzzle: Puzzle) -> List[Tuple[State, Move]]:
    """
    Generate all valid neighbor states from current state.

    For each box, try pushing it in all four directions.
    A push is valid if:
    - New box position is valid floor
    - New box position doesn't have another box
    - Player can reach the push-from position
    - Box doesn't end up in deadlock

    Args:
        state: Current game state
        puzzle: The puzzle object

    Returns:
        List of (next_state, move) tuples
    """
    neighbors = []

    # Try pushing each box
    for box_pos in state.box_positions:
        # Try each direction
        for direction in DIRECTIONS:
            # Check if this push is valid
            if is_valid_push(box_pos, direction, puzzle, state):
                # Calculate new positions
                new_box_pos = add_positions(box_pos, direction)

                # Player moves from current position to where the box was
                new_player_pos = box_pos

                # Create new box positions (remove old box, add new box)
                new_box_positions = (state.box_positions - {box_pos}) | {new_box_pos}

                # Create new state
                new_state = State(
                    player_pos=new_player_pos,
                    box_positions=frozenset(new_box_positions)
                )

                # Create move object for tracking
                move = Move(
                    box_from=box_pos,
                    box_to=new_box_pos,
                    player_from=state.player_pos,
                    player_to=new_player_pos,
                    direction=direction
                )

                neighbors.append((new_state, move))

    return neighbors


def reconstruct_path(
    came_from: dict,
    current_state: State
) -> List[Move]:
    """
    Reconstruct the solution path from the came_from dictionary.

    Args:
        came_from: Dictionary mapping state -> (previous_state, move)
        current_state: The goal state

    Returns:
        List of moves from initial state to goal state
    """
    path = []
    current = current_state

    while current in came_from:
        prev_state, move = came_from[current]
        path.append(move)
        current = prev_state

    # Reverse to get path from start to goal
    path.reverse()
    return path
