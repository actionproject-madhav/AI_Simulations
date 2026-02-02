"""
Sokoban Puzzle representation and parsing.
"""

from typing import Set, Tuple, List, FrozenSet
from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    """
    Represents a game state for A* search.
    Immutable and hashable for use in sets/dicts.
    """
    player_pos: Tuple[int, int]
    box_positions: FrozenSet[Tuple[int, int]]

    def __hash__(self):
        return hash((self.player_pos, self.box_positions))

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return (self.player_pos == other.player_pos and
                self.box_positions == other.box_positions)


class Puzzle:
    """
    Represents a Sokoban puzzle with the grid and static elements.
    """

    # Character mappings
    WALL = '#'
    FLOOR = ' '
    BOX = '$'
    GOAL = '.'
    PLAYER = '@'
    BOX_ON_GOAL = '*'
    PLAYER_ON_GOAL = '+'

    def __init__(self, puzzle_string: str):
        """
        Parse a puzzle from string format.

        Args:
            puzzle_string: Multi-line string with Sokoban puzzle
        """
        lines = puzzle_string.strip().split('\n')
        self.grid = [list(line) for line in lines]
        self.rows = len(self.grid)
        self.cols = max(len(row) for row in self.grid) if self.grid else 0

        # Normalize grid - pad shorter rows with spaces
        for row in self.grid:
            while len(row) < self.cols:
                row.append(self.FLOOR)

        # Extract static elements
        self.walls: Set[Tuple[int, int]] = set()
        self.goals: Set[Tuple[int, int]] = set()
        initial_player_pos: Tuple[int, int] = None
        initial_box_positions: Set[Tuple[int, int]] = set()

        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.grid[r][c]

                if cell == self.WALL:
                    self.walls.add((r, c))
                elif cell == self.GOAL:
                    self.goals.add((r, c))
                elif cell == self.PLAYER:
                    initial_player_pos = (r, c)
                elif cell == self.BOX:
                    initial_box_positions.add((r, c))
                elif cell == self.BOX_ON_GOAL:
                    self.goals.add((r, c))
                    initial_box_positions.add((r, c))
                elif cell == self.PLAYER_ON_GOAL:
                    self.goals.add((r, c))
                    initial_player_pos = (r, c)

        if initial_player_pos is None:
            raise ValueError("Puzzle must contain a player (@)")
        if not initial_box_positions:
            raise ValueError("Puzzle must contain at least one box ($)")
        if not self.goals:
            raise ValueError("Puzzle must contain at least one goal (.)")
        if len(initial_box_positions) != len(self.goals):
            raise ValueError(f"Number of boxes ({len(initial_box_positions)}) must match number of goals ({len(self.goals)})")

        self.initial_state = State(
            player_pos=initial_player_pos,
            box_positions=frozenset(initial_box_positions)
        )

    def is_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if position is a wall."""
        return pos in self.walls

    def is_goal(self, pos: Tuple[int, int]) -> bool:
        """Check if position is a goal."""
        return pos in self.goals

    def is_in_bounds(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_valid_floor(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid floor (not wall, in bounds)."""
        return self.is_in_bounds(pos) and not self.is_wall(pos)

    def display(self, state: State = None) -> str:
        """
        Display the puzzle in text format.

        Args:
            state: Optional state to display. If None, shows initial state.

        Returns:
            String representation of the puzzle
        """
        if state is None:
            state = self.initial_state

        # Create a copy of the grid for display
        display_grid = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                pos = (r, c)
                if pos in self.walls:
                    row.append(self.WALL)
                elif pos in self.goals:
                    row.append(self.GOAL)
                else:
                    row.append(self.FLOOR)
            display_grid.append(row)

        # Place boxes
        for box_pos in state.box_positions:
            r, c = box_pos
            if (r, c) in self.goals:
                display_grid[r][c] = self.BOX_ON_GOAL
            else:
                display_grid[r][c] = self.BOX

        # Place player
        pr, pc = state.player_pos
        if (pr, pc) in self.goals:
            display_grid[pr][pc] = self.PLAYER_ON_GOAL
        else:
            display_grid[pr][pc] = self.PLAYER

        # Convert to string
        return '\n'.join(''.join(row) for row in display_grid)

    def is_goal_state(self, state: State) -> bool:
        """Check if all boxes are on goals."""
        return state.box_positions == frozenset(self.goals)

    def __str__(self):
        """String representation showing initial state."""
        return self.display()
