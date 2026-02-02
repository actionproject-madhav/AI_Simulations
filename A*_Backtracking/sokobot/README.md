# Sokoban Solver

An interactive Sokoban puzzle solver using A* search algorithm.

## Features

- **A* Search Algorithm**: Efficient pathfinding with Manhattan distance heuristic
- **Deadlock Detection**: Automatically detects corner deadlocks to prune impossible states
- **Player Reachability**: Uses BFS to verify player can reach push positions
- **Step-by-Step Solutions**: Displays the solution moves visually
- **Sample Puzzles**: Includes 5 built-in puzzles of varying difficulty
- **File Support**: Load and solve custom puzzles from files

## Installation

No special dependencies required - uses only Python standard library.

Requires Python 3.7+

## Usage

### Command Line Interface

List available sample puzzles:
```bash
python3 main.py --list
```

Solve a sample puzzle:
```bash
python3 main.py --sample trivial
python3 main.py --sample medium
python3 main.py --sample challenge
```

Solve without showing step-by-step (summary only):
```bash
python3 main.py --sample medium --quiet
```

Solve a puzzle from a file:
```bash
python3 main.py --file my_puzzle.txt
```

Set maximum states to explore:
```bash
python3 main.py --file puzzle.txt --max-states 50000
```

### Puzzle File Format

Puzzles use standard Sokoban text format:

```
#####
#@$.#
#####
```

- `#` - Wall
- ` ` - Floor (space)
- `$` - Box
- `.` - Goal/Storage location
- `@` - Player
- `*` - Box on goal
- `+` - Player on goal

**Important**: The number of boxes must equal the number of goals.

### Python API

```python
from puzzle import Puzzle
from solver import solve_and_display

# Create puzzle from string
puzzle_string = """
#####
#@$.#
#####
"""

puzzle = Puzzle(puzzle_string)

# Solve and display
result = solve_and_display(puzzle)

print(f"Solved: {result.solved}")
print(f"Moves: {result.solution_length}")
print(f"States explored: {result.states_explored}")
```

## Project Structure

```
sokobot/
├── puzzle.py          # Puzzle representation and parsing
├── helpers.py         # Helper functions (heuristics, validation)
├── moves.py           # Move generation
├── solver.py          # A* search algorithm
├── main.py            # Command-line interface
├── solver.md          # Design document
├── README.md          # This file
└── test_*.py          # Test files
```

## Algorithm Details

### State Representation

- **State**: (player_position, box_positions)
- **Goal**: All boxes on goal positions
- **Action**: Push a box in one direction (up, down, left, right)

### A* Search

- **g(n)**: Number of pushes from initial state
- **h(n)**: Sum of Manhattan distances from each box to nearest goal
- **f(n) = g(n) + h(n)**: Total estimated cost

### Optimizations

1. **Corner Deadlock Detection**: Rejects boxes pushed into corners (unless the corner is a goal)
2. **Player Reachability**: Validates that the player can reach the push position before generating a move
3. **State Hashing**: Uses frozenset for efficient state comparison and duplicate detection

## Performance

The solver handles:
- Puzzles up to 8x8 grid
- Up to 5 boxes
- Solutions up to 50 moves
- Typically solves in < 30 seconds for moderate complexity

For harder puzzles, increase `--max-states` limit.

## Testing

Run all tests:

```bash
python3 test_puzzle.py    # Phase 1: Core structures
python3 test_helpers.py   # Phase 2: Helper functions
python3 test_moves.py     # Phase 3: Move generation
python3 test_solver.py    # Phase 4: A* solver
```

All tests should pass.

## Sample Puzzles

### Trivial (1 box, 1 move)
```
#####
#@$.#
#####
```

### Simple (2 boxes, 2 moves)
```
#######
# . . #
# $ $ #
#  @  #
#######
```

### Medium (2 boxes, 3 moves)
```
########
#   .  #
# @$$  #
#   . ##
########
```

### Harder (3 boxes, 7 moves)
```
  #####
  #   #
  #$  #
###@$##
#  $  #
# ...##
########
```

### Challenge (4 boxes)
```
#########
#   #   #
# $   $ #
### # ###
# $ @ $ #
# .   . #
## . . ##
#########
```

## Future Enhancements

Potential improvements:
- Better heuristics (Hungarian algorithm for box-goal assignment)
- More deadlock patterns (edge deadlocks, frozen boxes)
- Web interface with puzzle designer
- Animation of solution
- Puzzle difficulty rating
- Solution optimality verification

## License

Educational project - feel free to use and modify.

## References

- [Sokoban Wiki](https://www.sokobano.de/)
- [Sokoban Online](https://www.sokobanonline.com/)
- Original game by Hiroyuki Imabayashi (1981)
