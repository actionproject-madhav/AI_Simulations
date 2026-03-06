// Monte Carlo Tree Search (MCTS) AI Engine for Ultimate Tic-Tac-Toe
// Uses UCB1 policy for tree descent and random rollouts for simulation

import {
  makeMove,
  getAllLegalMoves,
  getGameStatus,
  cloneState
} from './gameEngine';

// MCTS Node class
class MCTSNode {
  constructor(state, parent = null, move = null) {
    this.state = state; // Game state {board, metaBoard, activeBoards, currentPlayer}
    this.parent = parent;
    this.move = move; // {localBoardIdx, cellIdx}
    this.children = [];
    this.visits = 0;
    this.wins = 0; // Wins from the perspective of the player who just moved to reach this node
    this.untriedMoves = getAllLegalMoves(
      state.board,
      state.metaBoard,
      state.activeBoards
    );
  }

  /**
   * Check if node is fully expanded (all moves tried)
   */
  isFullyExpanded() {
    return this.untriedMoves.length === 0;
  }

  /**
   * Check if node is terminal (game over)
   */
  isTerminal() {
    const status = getGameStatus(this.state.board, this.state.metaBoard);
    return status !== 'in_progress';
  }

  /**
   * Get the best child using UCB1 formula
   */
  bestChild(explorationConstant = Math.sqrt(2)) {
    let bestScore = -Infinity;
    let bestChild = null;

    for (const child of this.children) {
      // UCB1 formula: wins/visits + C * sqrt(ln(parent.visits) / visits)
      const exploitation = child.wins / child.visits;
      const exploration = explorationConstant * Math.sqrt(Math.log(this.visits) / child.visits);
      const ucb1Score = exploitation + exploration;

      if (ucb1Score > bestScore) {
        bestScore = ucb1Score;
        bestChild = child;
      }
    }

    return bestChild;
  }

  /**
   * Get child with most visits (most robust choice)
   */
  mostVisitedChild() {
    let mostVisits = -1;
    let bestChild = null;

    for (const child of this.children) {
      if (child.visits > mostVisits) {
        mostVisits = child.visits;
        bestChild = child;
      }
    }

    return bestChild;
  }
}

/**
 * Selection phase: Traverse tree using UCB1 until we reach a leaf
 */
function select(node) {
  while (!node.isTerminal() && node.isFullyExpanded()) {
    node = node.bestChild();
  }
  return node;
}

/**
 * Expansion phase: Add one new child node
 */
function expand(node) {
  if (node.untriedMoves.length === 0) {
    return node;
  }

  // Pick a random untried move
  const moveIndex = Math.floor(Math.random() * node.untriedMoves.length);
  const move = node.untriedMoves[moveIndex];

  // Remove from untried moves
  node.untriedMoves.splice(moveIndex, 1);

  // Create new state by making the move
  const newState = makeMove(
    node.state.board,
    node.state.metaBoard,
    node.state.activeBoards,
    move.localBoardIdx,
    move.cellIdx,
    node.state.currentPlayer
  );

  // Create and add child node
  const childNode = new MCTSNode(newState, node, move);
  node.children.push(childNode);

  return childNode;
}

/**
 * Simulation phase: Random rollout to terminal state
 */
function simulate(state) {
  let currentState = cloneState(state);

  // Play random moves until game ends
  let iterations = 0;
  const maxIterations = 200; // Prevent infinite loops

  while (iterations < maxIterations) {
    const status = getGameStatus(currentState.board, currentState.metaBoard);

    if (status !== 'in_progress') {
      // Game over, return result from AI's perspective (AI is 'O')
      if (status === 'o_wins') return 1;
      if (status === 'x_wins') return -1;
      return 0; // draw
    }

    // Get legal moves
    const legalMoves = getAllLegalMoves(
      currentState.board,
      currentState.metaBoard,
      currentState.activeBoards
    );

    if (legalMoves.length === 0) {
      return 0; // No moves available, draw
    }

    // Pick random move
    const randomMove = legalMoves[Math.floor(Math.random() * legalMoves.length)];

    // Make move
    currentState = makeMove(
      currentState.board,
      currentState.metaBoard,
      currentState.activeBoards,
      randomMove.localBoardIdx,
      randomMove.cellIdx,
      currentState.currentPlayer
    );

    iterations++;
  }

  // Max iterations reached, evaluate as draw
  return 0;
}

/**
 * Backpropagation phase: Update statistics up the tree
 */
function backpropagate(node, result) {
  while (node !== null) {
    node.visits += 1;

    // Update wins based on whose turn it was when the node was created
    // If the node's player is 'O' (AI), add the result as-is
    // If the node's player is 'X' (human), flip the result
    // Note: node.state.currentPlayer is the player ABOUT TO MOVE from this state
    // So we need to look at the PREVIOUS player (who just moved to reach this state)

    const playerWhoMovedHere = node.state.currentPlayer === 'X' ? 'O' : 'X';

    if (playerWhoMovedHere === 'O') {
      // AI just moved to reach this node
      node.wins += result;
    } else {
      // Human just moved to reach this node
      node.wins -= result;
    }

    node = node.parent;
  }
}

/**
 * Main MCTS function: Returns best move for AI
 */
function mcts(rootState, iterations) {
  const root = new MCTSNode(rootState);

  for (let i = 0; i < iterations; i++) {
    // Selection
    let node = select(root);

    // Expansion
    if (!node.isTerminal()) {
      node = expand(node);
    }

    // Simulation
    const result = simulate(node.state);

    // Backpropagation
    backpropagate(node, result);
  }

  // Return the most visited child (most robust)
  const bestChild = root.mostVisitedChild();
  return bestChild ? bestChild.move : null;
}

/**
 * Get AI move with specified difficulty level
 * @param {Array} board - Current board state
 * @param {Array} metaBoard - Current meta-board state
 * @param {Array} activeBoards - Current active boards
 * @param {string} difficulty - 'Easy', 'Medium', or 'Hard'
 * @returns {Promise<Object>} - Move object {localBoardIdx, cellIdx}
 */
export async function getAIMove(board, metaBoard, activeBoards, difficulty) {
  // Map difficulty to iteration count
  const iterations = {
    'Easy': 1000,
    'Medium': 5000,
    'Hard': 10000
  }[difficulty] || 5000;

  const startTime = Date.now();

  // Run MCTS
  const state = {
    board,
    metaBoard,
    activeBoards,
    currentPlayer: 'O'
  };

  const move = mcts(state, iterations);

  // Ensure minimum delay of 500ms for better UX
  const elapsed = Date.now() - startTime;
  const remainingDelay = Math.max(0, 500 - elapsed);

  if (remainingDelay > 0) {
    await new Promise(resolve => setTimeout(resolve, remainingDelay));
  }

  return move;
}
