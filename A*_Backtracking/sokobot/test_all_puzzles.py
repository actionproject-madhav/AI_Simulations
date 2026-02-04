#!/usr/bin/env python3
"""
Test the Sokoban solver on all puzzles from the app.
"""

from puzzle import Puzzle
from solver import solve

# Puzzles from app.py
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
###@$ #
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


def test_puzzle(name, puzzle_string, max_states=100000):
    """Test a single puzzle."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)
    
    try:
        puzzle = Puzzle(puzzle_string)
        print(f"Puzzle size: {puzzle.rows}x{puzzle.cols}")
        print(f"Boxes: {len(puzzle.initial_state.box_positions)}")
        print(f"Goals: {len(puzzle.goals)}")
        print("\nInitial state:")
        print(puzzle.display())
        print("\nSolving...")
        
        result = solve(puzzle, max_states=max_states)
        
        if result.solved:
            print(f"\n✓ SUCCESS!")
            print(f"  Solution length: {result.solution_length} moves")
            print(f"  States explored: {result.states_explored}")
            
            # Verify solution
            from puzzle import State
            current_state = puzzle.initial_state
            for i, move in enumerate(result.moves):
                # Apply move
                new_box_positions = (current_state.box_positions - {move.box_from}) | {move.box_to}
                current_state = State(
                    player_pos=move.box_from,
                    box_positions=frozenset(new_box_positions)
                )
            
            # Check if final state is goal
            if puzzle.is_goal_state(current_state):
                print(f"  ✓ Solution verified - reaches goal state!")
            else:
                print(f"  ✗ WARNING: Solution does NOT reach goal state!")
                return False
                
            return True
        else:
            print(f"\n✗ NO SOLUTION FOUND")
            print(f"  Reason: {result.message}")
            print(f"  States explored: {result.states_explored}")
            return False
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SOKOBAN SOLVER - COMPREHENSIVE TEST")
    print("="*60)
    
    results = {}
    
    # Test each puzzle with different max_states limits
    for name, puzzle_string in PUZZLES.items():
        # Adjust max_states based on difficulty
        max_states = {
            "tutorial": 1000,
            "easy": 5000,
            "medium": 10000,
            "hard": 50000,
            "challenge": 100000
        }.get(name, 10000)
        
        success = test_puzzle(name, puzzle_string, max_states=max_states)
        results[name] = success
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for s in results.values() if s)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All puzzles solved successfully!")
    else:
        print(f"\n⚠️  {total - passed} puzzle(s) failed")
    
    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
