"""
Motion planning on a rectangular grid using A* search
"""

from random import random
from random import seed
from queue import PriorityQueue
from copy import deepcopy


class State(object):

    def __init__(self, start_position, goal_position, start_grid):
        self.position = start_position
        self.goal = goal_position
        self.grid = start_grid
        self.total_moves = 0
        self.parent = None  # For path reconstruction

    def manhattan_distance(self):
        """
        Calculate Manhattan distance from current position to goal
        This is our heuristic function h(n)
        """
        return abs(self.position[0] - self.goal[0]) + abs(self.position[1] - self.goal[1])

    def __lt__(self, other):
        """
        Less-than comparison for priority queue
        Required when priorities are equal
        """
        return self.total_moves < other.total_moves

    def __eq__(self, other):
        """
        Equality comparison based on position
        """
        return self.position == other.position

    def __hash__(self):
        """
        Hash based on position for use in sets/dicts
        """
        return hash(self.position)


def create_grid():

    """
    Create and return a randomized grid

    0's in the grid indcate free squares
    1's indicate obstacles

    DON'T MODIFY THIS ROUTINE.
    DON'T MODIFY THIS ROUTINE.
    DON'T MODIFY THIS ROUTINE.
    DON'T MODIFY THIS ROUTINE.
    ARE YOU MODIFYING THIS ROUTINE?
    IF SO, STOP IT.
    """

    # Start with a num_rows by num_cols grid of all zeros
    grid = [[0 for c in range(num_cols)] for r in range(num_rows)]

    # Put ones around the boundary
    grid[0] = [1 for c in range(num_cols)]
    grid[num_rows - 1] = [1 for c in range(num_cols)]

    for r in range(num_rows):
        grid[r][0] = 1
        grid[r][num_cols - 1] = 1

    # Sprinkle in obstacles randomly
    for r in range(1, num_rows - 1):
        for c in range(2, num_cols - 2):
            if random() < obstacle_prob:
                grid[r][c] = 1;

    # Make sure the goal and start spaces are clear
    grid[1][1] = 0
    grid[num_rows - 2][num_cols - 2] = 0

    return grid


def print_grid(grid):

    """
    Print a grid, putting spaces in place of zeros for readability

    DON'T MODIFY THIS ROUTINE.
    DON'T MODIFY THIS ROUTINE.
    DON'T MODIFY THIS ROUTINE.
    DON'T MODIFY THIS ROUTINE.
    ARE YOU MODIFYING THIS ROUTINE?
    IF SO, STOP IT.
    """

    for r in range(num_rows):
        for c in range(num_cols):
            if grid[r][c] == 0:
                print(' ', end='')
            else:
                print(grid[r][c], end='')
        print('')

    print('')

    return 


def main():
    """
    Use A* search to find a path from the upper left to the lower right
    of the puzzle grid

    Complete this method to implement the search
    At the end, print the solution state
    
    Each State object has a copy of the grid
    
    When you make a move by generating a new State, put a * on its grid
    to show the solution path
    """

  
    # Setup the randomized grid
    grid = create_grid()
    print_grid(grid)

    # Initialize the starting state and priority queue
    start_position = (1, 1)
    goal_position = (num_rows - 2, num_cols - 2)
    start_state = State(start_position, goal_position, grid)

    # A* priority: f(n) = g(n) + h(n)
    priority = start_state.total_moves + start_state.manhattan_distance()

    queue = PriorityQueue()

    # Insert as a tuple
    # The queue orders elements by the first tuple value
    # A call to queue.get() returns the tuple with the minimum first value
    queue.put((priority, start_state))

    # Track visited positions to avoid revisiting
    visited = set()
    visited.add(start_position)

    # Track best cost to reach each position (for A* optimality)
    g_score = {start_position: 0}

    # Possible moves: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # A* search loop
    while not queue.empty():
        # Get state with lowest f-score
        current_priority, current_state = queue.get()

        # Check if we reached the goal
        if current_state.position == goal_position:
            # Reconstruct path by collecting all positions
            path_positions = []
            path_state = current_state
            while path_state is not None:
                path_positions.append(path_state.position)
                path_state = path_state.parent

            # Mark the path on a fresh grid copy
            solution_grid = deepcopy(grid)
            for r, c in path_positions:
                solution_grid[r][c] = '*'

            # Print solution
            print('Solution found!')
            print(f'Path length: {current_state.total_moves} moves')
            print_grid(solution_grid)
            return

        # Explore neighbors
        r, c = current_state.position

        for dr, dc in directions:
            new_r, new_c = r + dr, c + dc
            new_position = (new_r, new_c)

            # Check if the new position is valid (not a wall/obstacle)
            if grid[new_r][new_c] == 1:
                continue

            # Calculate tentative g-score (cost from start)
            tentative_g = current_state.total_moves + 1

            # Skip if we've found a better path to this position already
            if new_position in g_score and tentative_g >= g_score[new_position]:
                continue

            # Create new state
            new_grid = deepcopy(current_state.grid)
            new_state = State(new_position, goal_position, new_grid)
            new_state.total_moves = tentative_g
            new_state.parent = current_state

            # Update g_score and visited
            g_score[new_position] = tentative_g
            visited.add(new_position)

            # Calculate f-score and add to queue
            f_score = tentative_g + new_state.manhattan_distance()
            queue.put((f_score, new_state))

    # No solution found
    print('No path exists from start to goal.')
    print('')


if __name__ == '__main__':

    seed(0)

    #--- Easy mode

    # Global variables
    # Saves us the trouble of continually passing them as parameters 
    num_rows = 8
    num_cols = 16
    obstacle_prob = .20

    for trial in range(5):
        print('\n\n-----Easy trial ' + str(trial + 1) + '-----')
        main()

    #--- Uncomment the following sets of trials when you're ready

    #--- Hard mode
    num_rows = 15
    num_cols = 30
    obstacle_prob = .30

    for trial in range(5):
        print('\n\n-----Harder trial ' + str(trial + 1) + '-----')
        main()

    #--- INSANE mode
    num_rows = 20
    num_cols = 60
    obstacle_prob = .35

    for trial in range(5):
        print('\n\n-----INSANE trial ' + str(trial + 1) + '-----')
        main()