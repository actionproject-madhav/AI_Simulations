"""
A* search solver for Sokoban puzzles.
"""

import heapq
from typing import Optional, List, Dict
from puzzle import Puzzle, State
from moves import get_neighbors, reconstruct_path, Move
from helpers import calculate_heuristic


class SolverResult:
    """Result from solving a puzzle."""

    def __init__(
        self,
        solved: bool,
        moves: Optional[List[Move]] = None,
        states_explored: int = 0,
        solution_length: int = 0,
        message: str = ""
    ):
        self.solved = solved
        self.moves = moves or []
        self.states_explored = states_explored
        self.solution_length = solution_length
        self.message = message

    def __repr__(self):
        if self.solved:
            return (f"SolverResult(solved=True, solution_length={self.solution_length}, "
                    f"states_explored={self.states_explored})")
        else:
            return f"SolverResult(solved=False, message='{self.message}')"


def solve(puzzle: Puzzle, max_states: int = 100000) -> SolverResult:
    """
    Solve a Sokoban puzzle using A* search.

    Args:
        puzzle: The puzzle to solve
        max_states: Maximum number of states to explore before giving up

    Returns:
        SolverResult with solution or failure information
    """
    initial_state = puzzle.initial_state

    # Check if already at goal
    if puzzle.is_goal_state(initial_state):
        return SolverResult(
            solved=True,
            moves=[],
            states_explored=0,
            solution_length=0,
            message="Already at goal state"
        )

    # Priority queue: (f_score, counter, state)
    # counter ensures FIFO ordering for equal f_scores
    counter = 0
    initial_f = calculate_heuristic(initial_state, puzzle)
    frontier = [(initial_f, counter, initial_state)]
    counter += 1

    # Track visited states
    visited = {initial_state}

    # Track how we reached each state: state -> (previous_state, move)
    came_from: Dict[State, tuple] = {}

    # Track g_scores (cost from start)
    g_score = {initial_state: 0}

    # Stats
    states_explored = 0

    while frontier:
        # Get state with lowest f_score
        current_f, _, current_state = heapq.heappop(frontier)
        current_g = g_score[current_state]

        states_explored += 1

        # Check if we've exceeded max states
        if states_explored > max_states:
            return SolverResult(
                solved=False,
                states_explored=states_explored,
                message=f"Exceeded maximum states ({max_states})"
            )

        # Check if goal reached
        if puzzle.is_goal_state(current_state):
            moves = reconstruct_path(came_from, current_state)
            return SolverResult(
                solved=True,
                moves=moves,
                states_explored=states_explored,
                solution_length=len(moves),
                message="Solution found"
            )

        # Generate neighbors
        for next_state, move in get_neighbors(current_state, puzzle):
            # Calculate tentative g_score (each move costs 1)
            tentative_g = current_g + 1

            # If we found a better path to this state, update it
            if next_state not in g_score or tentative_g < g_score[next_state]:
                # Record this path
                came_from[next_state] = (current_state, move)
                g_score[next_state] = tentative_g

                # Calculate f_score = g + h
                h_score = calculate_heuristic(next_state, puzzle)
                f_score = tentative_g + h_score

                # Add to frontier if not visited
                if next_state not in visited:
                    visited.add(next_state)
                    heapq.heappush(frontier, (f_score, counter, next_state))
                    counter += 1

    # No solution found
    return SolverResult(
        solved=False,
        states_explored=states_explored,
        message="No solution exists"
    )


def solve_and_display(puzzle: Puzzle, verbose: bool = True) -> SolverResult:
    """
    Solve a puzzle and optionally display the solution step by step.

    Args:
        puzzle: The puzzle to solve
        verbose: If True, print solution steps

    Returns:
        SolverResult
    """
    if verbose:
        print("Initial puzzle:")
        print(puzzle.display())
        print()

    result = solve(puzzle)

    if verbose:
        if result.solved:
            print(f"✓ Solution found!")
            print(f"  Steps: {result.solution_length}")
            print(f"  States explored: {result.states_explored}")
            print()

            # Display solution steps
            current_state = puzzle.initial_state
            print(f"Step 0 (Initial):")
            print(puzzle.display(current_state))
            print()

            for i, move in enumerate(result.moves):
                # Apply move to get next state
                new_box_positions = (current_state.box_positions - {move.box_from}) | {move.box_to}
                current_state = State(
                    player_pos=move.box_from,  # Player moves to where box was
                    box_positions=frozenset(new_box_positions)
                )

                print(f"Step {i + 1}: {move}")
                print(puzzle.display(current_state))
                print()

        else:
            print(f"✗ No solution found")
            print(f"  Reason: {result.message}")
            print(f"  States explored: {result.states_explored}")

    return result
