// AI Engine - Minimax algorithm with alpha-beta pruning

import {
  getAllLegalMoves,
  makeMove,
  isCheckmate,
  isStalemate,
  getPieceColor,
  getPieceType,
} from './chessEngine';

// Piece values for evaluation
const PIECE_VALUES = {
  pawn: 100,
  knight: 320,
  bishop: 330,
  rook: 500,
  queen: 900,
  king: 20000,
};

// Position bonus tables (encourages better piece placement)
const PAWN_TABLE = [
  [0, 0, 0, 0, 0, 0, 0, 0],
  [50, 50, 50, 50, 50, 50, 50, 50],
  [10, 10, 20, 30, 30, 20, 10, 10],
  [5, 5, 10, 25, 25, 10, 5, 5],
  [0, 0, 0, 20, 20, 0, 0, 0],
  [5, -5, -10, 0, 0, -10, -5, 5],
  [5, 10, 10, -20, -20, 10, 10, 5],
  [0, 0, 0, 0, 0, 0, 0, 0]
];

const KNIGHT_TABLE = [
  [-50, -40, -30, -30, -30, -30, -40, -50],
  [-40, -20, 0, 0, 0, 0, -20, -40],
  [-30, 0, 10, 15, 15, 10, 0, -30],
  [-30, 5, 15, 20, 20, 15, 5, -30],
  [-30, 0, 15, 20, 20, 15, 0, -30],
  [-30, 5, 10, 15, 15, 10, 5, -30],
  [-40, -20, 0, 5, 5, 0, -20, -40],
  [-50, -40, -30, -30, -30, -30, -40, -50]
];

const BISHOP_TABLE = [
  [-20, -10, -10, -10, -10, -10, -10, -20],
  [-10, 0, 0, 0, 0, 0, 0, -10],
  [-10, 0, 5, 10, 10, 5, 0, -10],
  [-10, 5, 5, 10, 10, 5, 5, -10],
  [-10, 0, 10, 10, 10, 10, 0, -10],
  [-10, 10, 10, 10, 10, 10, 10, -10],
  [-10, 5, 0, 0, 0, 0, 5, -10],
  [-20, -10, -10, -10, -10, -10, -10, -20]
];

const ROOK_TABLE = [
  [0, 0, 0, 0, 0, 0, 0, 0],
  [5, 10, 10, 10, 10, 10, 10, 5],
  [-5, 0, 0, 0, 0, 0, 0, -5],
  [-5, 0, 0, 0, 0, 0, 0, -5],
  [-5, 0, 0, 0, 0, 0, 0, -5],
  [-5, 0, 0, 0, 0, 0, 0, -5],
  [-5, 0, 0, 0, 0, 0, 0, -5],
  [0, 0, 0, 5, 5, 0, 0, 0]
];

const QUEEN_TABLE = [
  [-20, -10, -10, -5, -5, -10, -10, -20],
  [-10, 0, 0, 0, 0, 0, 0, -10],
  [-10, 0, 5, 5, 5, 5, 0, -10],
  [-5, 0, 5, 5, 5, 5, 0, -5],
  [0, 0, 5, 5, 5, 5, 0, -5],
  [-10, 5, 5, 5, 5, 5, 0, -10],
  [-10, 0, 5, 0, 0, 0, 0, -10],
  [-20, -10, -10, -5, -5, -10, -10, -20]
];

const KING_MIDDLE_GAME_TABLE = [
  [-30, -40, -40, -50, -50, -40, -40, -30],
  [-30, -40, -40, -50, -50, -40, -40, -30],
  [-30, -40, -40, -50, -50, -40, -40, -30],
  [-30, -40, -40, -50, -50, -40, -40, -30],
  [-20, -30, -30, -40, -40, -30, -30, -20],
  [-10, -20, -20, -20, -20, -20, -20, -10],
  [20, 20, 0, 0, 0, 0, 20, 20],
  [20, 30, 10, 0, 0, 10, 30, 20]
];

// Get position bonus for a piece
const getPositionBonus = (piece, row, col) => {
  const type = getPieceType(piece);
  const color = getPieceColor(piece);

  // Flip the table for black pieces (they're on opposite side)
  const adjustedRow = color === 'black' ? row : 7 - row;

  switch (type) {
    case 'pawn':
      return PAWN_TABLE[adjustedRow][col];
    case 'knight':
      return KNIGHT_TABLE[adjustedRow][col];
    case 'bishop':
      return BISHOP_TABLE[adjustedRow][col];
    case 'rook':
      return ROOK_TABLE[adjustedRow][col];
    case 'queen':
      return QUEEN_TABLE[adjustedRow][col];
    case 'king':
      return KING_MIDDLE_GAME_TABLE[adjustedRow][col];
    default:
      return 0;
  }
};

// Evaluate board position (positive = good for white, negative = good for black)
export const evaluateBoard = (board) => {
  let score = 0;

  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const piece = board[row][col];
      if (!piece) continue;

      const type = getPieceType(piece);
      const color = getPieceColor(piece);
      const pieceValue = PIECE_VALUES[type] || 0;
      const positionBonus = getPositionBonus(piece, row, col);

      const totalValue = pieceValue + positionBonus;

      if (color === 'white') {
        score += totalValue;
      } else {
        score -= totalValue;
      }
    }
  }

  return score;
};

// Minimax algorithm with alpha-beta pruning
const minimax = (board, depth, alpha, beta, isMaximizing, gameState = null) => {
  const color = isMaximizing ? 'white' : 'black';

  // Check terminal conditions
  if (isCheckmate(board, color, gameState)) {
    return isMaximizing ? -Infinity : Infinity;
  }
  if (isStalemate(board, color, gameState)) {
    return 0;
  }

  // Base case: depth limit reached
  if (depth === 0) {
    return evaluateBoard(board);
  }

  const moves = getAllLegalMoves(board, color, gameState);

  if (isMaximizing) {
    let maxEval = -Infinity;

    for (const move of moves) {
      const { board: newBoard, gameState: newGameState } = makeMove(
        board,
        move.from[0],
        move.from[1],
        move.to[0],
        move.to[1],
        gameState
      );
      const evaluation = minimax(newBoard, depth - 1, alpha, beta, false, newGameState);

      maxEval = Math.max(maxEval, evaluation);
      alpha = Math.max(alpha, evaluation);

      if (beta <= alpha) {
        break; // Beta cutoff
      }
    }

    return maxEval;
  } else {
    let minEval = Infinity;

    for (const move of moves) {
      const { board: newBoard, gameState: newGameState } = makeMove(
        board,
        move.from[0],
        move.from[1],
        move.to[0],
        move.to[1],
        gameState
      );
      const evaluation = minimax(newBoard, depth - 1, alpha, beta, true, newGameState);

      minEval = Math.min(minEval, evaluation);
      beta = Math.min(beta, evaluation);

      if (beta <= alpha) {
        break; // Alpha cutoff
      }
    }

    return minEval;
  }
};

// Get the best move for the AI (black)
export const getBestMove = (board, depth = 3, gameState = null) => {
  const moves = getAllLegalMoves(board, 'black', gameState);

  if (moves.length === 0) return null;

  let bestMove = null;
  let bestValue = Infinity; // Black wants to minimize

  for (const move of moves) {
    const { board: newBoard, gameState: newGameState } = makeMove(
      board,
      move.from[0],
      move.from[1],
      move.to[0],
      move.to[1],
      gameState
    );
    const boardValue = minimax(newBoard, depth - 1, -Infinity, Infinity, true, newGameState);

    if (boardValue < bestValue) {
      bestValue = boardValue;
      bestMove = move;
    }
  }

  return bestMove;
};

// Get best move with a delay (for better UX)
export const getAIMove = async (board, depth = 3, minDelay = 500, gameState = null) => {
  const startTime = Date.now();

  return new Promise((resolve) => {
    // Run minimax in background
    setTimeout(() => {
      const bestMove = getBestMove(board, depth, gameState);
      const elapsed = Date.now() - startTime;
      const remainingDelay = Math.max(0, minDelay - elapsed);

      // Add remaining delay for better UX
      setTimeout(() => {
        resolve(bestMove);
      }, remainingDelay);
    }, 0);
  });
};
