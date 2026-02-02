#!/usr/bin/env python3
"""
Flask web server for interactive Sokoban game.
"""

from flask import Flask, render_template, jsonify, request
from puzzle import Puzzle, State
from solver import solve

app = Flask(__name__)

# Store current game state in memory (for demo purposes)
# In production, you'd use sessions or a database
current_puzzle = None
current_state = None
puzzle_history = []  # Track moves for undo


# Sample puzzles
PUZZLES = {
    "tutorial": """
#####
#@$.#
#####
""",
    "easy": """
#######
# . . #
# $ $ #
#  @  #
#######
""",
    "medium": """
########
#   .  #
# @$$  #
#   . ##
########
""",
    "hard": """
  #####
  #   #
  #$  #
###@$##
#  $  #
# ...##
########
""",
    "challenge": """
#########
#   #   #
# $   $ #
### # ###
# $ @ $ #
# .   . #
## . . ##
#########
""",
}


@app.route('/')
def index():
    """Serve the main game page."""
    return render_template('index.html')


@app.route('/api/puzzles')
def get_puzzles():
    """Get list of available puzzles."""
    puzzle_list = []
    for name, puzzle_string in PUZZLES.items():
        p = Puzzle(puzzle_string)
        puzzle_list.append({
            'name': name,
            'size': f"{p.rows}x{p.cols}",
            'boxes': len(p.initial_state.box_positions)
        })
    return jsonify(puzzle_list)


@app.route('/api/load/<puzzle_name>')
def load_puzzle(puzzle_name):
    """Load a specific puzzle."""
    global current_puzzle, current_state, puzzle_history

    if puzzle_name not in PUZZLES:
        return jsonify({'error': 'Puzzle not found'}), 404

    puzzle_string = PUZZLES[puzzle_name]
    current_puzzle = Puzzle(puzzle_string)
    current_state = current_puzzle.initial_state
    puzzle_history = [current_state]

    return jsonify({
        'puzzle': serialize_puzzle(current_puzzle, current_state),
        'message': f'Loaded {puzzle_name} puzzle'
    })


@app.route('/api/move', methods=['POST'])
def make_move():
    """Handle player move."""
    global current_state, puzzle_history

    if current_puzzle is None:
        return jsonify({'error': 'No puzzle loaded'}), 400

    data = request.json
    direction = data.get('direction')

    # Direction mapping
    directions = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }

    if direction not in directions:
        return jsonify({'error': 'Invalid direction'}), 400

    delta = directions[direction]
    new_state = try_move(current_state, delta, current_puzzle)

    if new_state is None:
        return jsonify({
            'success': False,
            'message': 'Invalid move'
        })

    current_state = new_state
    puzzle_history.append(current_state)

    # Check if won
    won = current_puzzle.is_goal_state(current_state)

    return jsonify({
        'success': True,
        'puzzle': serialize_puzzle(current_puzzle, current_state),
        'won': won,
        'moves': len(puzzle_history) - 1
    })


@app.route('/api/undo', methods=['POST'])
def undo_move():
    """Undo last move."""
    global current_state, puzzle_history

    if len(puzzle_history) <= 1:
        return jsonify({'error': 'Nothing to undo'}), 400

    puzzle_history.pop()
    current_state = puzzle_history[-1]

    return jsonify({
        'success': True,
        'puzzle': serialize_puzzle(current_puzzle, current_state),
        'moves': len(puzzle_history) - 1
    })


@app.route('/api/reset', methods=['POST'])
def reset_puzzle():
    """Reset puzzle to initial state."""
    global current_state, puzzle_history

    if current_puzzle is None:
        return jsonify({'error': 'No puzzle loaded'}), 400

    current_state = current_puzzle.initial_state
    puzzle_history = [current_state]

    return jsonify({
        'success': True,
        'puzzle': serialize_puzzle(current_puzzle, current_state),
        'moves': 0
    })


@app.route('/api/solve', methods=['POST'])
def solve_puzzle():
    """Get AI solution for current puzzle."""
    if current_puzzle is None:
        return jsonify({'error': 'No puzzle loaded'}), 400

    # Solve from initial state
    result = solve(current_puzzle, max_states=50000)

    if not result.solved:
        return jsonify({
            'solved': False,
            'message': result.message
        })

    # Convert moves to direction strings
    solution_moves = []
    for move in result.moves:
        # Determine direction from box movement
        dr = move.box_to[0] - move.box_from[0]
        dc = move.box_to[1] - move.box_from[1]

        if dr == -1:
            direction = 'up'
        elif dr == 1:
            direction = 'down'
        elif dc == -1:
            direction = 'left'
        else:
            direction = 'right'

        solution_moves.append({
            'direction': direction,
            'box_from': move.box_from,
            'box_to': move.box_to
        })

    return jsonify({
        'solved': True,
        'moves': solution_moves,
        'length': result.solution_length,
        'states_explored': result.states_explored
    })


def try_move(state, delta, puzzle):
    """
    Try to move player in given direction.
    Returns new state if valid, None otherwise.
    """
    pr, pc = state.player_pos
    new_pr, new_pc = pr + delta[0], pc + delta[1]
    new_pos = (new_pr, new_pc)

    # Check if new position is valid
    if not puzzle.is_valid_floor(new_pos):
        return None

    # Check if there's a box at new position
    if new_pos in state.box_positions:
        # Try to push the box
        box_new_pr, box_new_pc = new_pr + delta[0], new_pc + delta[1]
        box_new_pos = (box_new_pr, box_new_pc)

        # Check if box can be pushed
        if not puzzle.is_valid_floor(box_new_pos):
            return None
        if box_new_pos in state.box_positions:
            return None  # Can't push into another box

        # Push the box
        new_box_positions = (state.box_positions - {new_pos}) | {box_new_pos}
        return State(
            player_pos=new_pos,
            box_positions=frozenset(new_box_positions)
        )
    else:
        # Just move player
        return State(
            player_pos=new_pos,
            box_positions=state.box_positions
        )


def serialize_puzzle(puzzle, state):
    """Convert puzzle and state to JSON-serializable format."""
    # Create grid representation
    grid = []
    for r in range(puzzle.rows):
        row = []
        for c in range(puzzle.cols):
            pos = (r, c)
            cell = {
                'type': 'floor',
                'hasGoal': pos in puzzle.goals,
                'hasBox': pos in state.box_positions,
                'hasPlayer': pos == state.player_pos
            }

            if pos in puzzle.walls:
                cell['type'] = 'wall'

            row.append(cell)
        grid.append(row)

    return {
        'grid': grid,
        'rows': puzzle.rows,
        'cols': puzzle.cols,
        'won': puzzle.is_goal_state(state)
    }


if __name__ == '__main__':
    print("Starting Sokoban game server...")
    print("Open http://localhost:5001 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5001)
