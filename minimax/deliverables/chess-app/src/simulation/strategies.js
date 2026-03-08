/**
 * Strategy implementations for 3-Board Chess
 */

import { getLegalMoves, getPieceType, getPieceColor, getGameStatus, isInCheck } from '../utils/chessEngine.js';
import { getAIMove, evaluateBoard } from '../utils/aiEngine.js';

/**
 * Base Strategy Class
 */
export class BaseStrategy {
  constructor(name, options = {}) {
    this.name = name;
    this.depth = options.depth || 3;
  }

  /**
   * Select a move for the given board position
   * @param {Array} board - Current board state
   * @param {String} color - Color to move ('white' or 'black')
   * @param {Object} gameState - Game state (castling, en passant, etc.)
   * @param {Array} allBoards - All 3 boards (for context)
   * @param {Number} activeBoardIndex - Which board is active
   * @returns {Object} - {from: [row, col], to: [row, col]}
   */
  async selectMove(board, color, gameState, allBoards, activeBoardIndex) {
    throw new Error('selectMove must be implemented by strategy');
  }

  /**
   * Get all legal moves for the current player
   */
  getAllLegalMoves(board, color, gameState) {
    const moves = [];
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const piece = board[row][col];
        if (piece && getPieceColor(piece) === color) {
          const pieceMoves = getLegalMoves(board, row, col, gameState);
          pieceMoves.forEach(([toRow, toCol]) => {
            moves.push({
              from: [row, col],
              to: [toRow, toCol],
              piece: piece
            });
          });
        }
      }
    }
    return moves;
  }
}

/**
 * Random Strategy - Picks random legal moves (baseline)
 */
export class RandomStrategy extends BaseStrategy {
  constructor() {
    super('Random');
  }

  async selectMove(board, color, gameState) {
    const legalMoves = this.getAllLegalMoves(board, color, gameState);
    if (legalMoves.length === 0) return null;

    const randomIndex = Math.floor(Math.random() * legalMoves.length);
    return legalMoves[randomIndex];
  }
}

/**
 * Minimax Strategy - Uses existing AI engine with configurable depth
 */
export class MinimaxStrategy extends BaseStrategy {
  constructor(depth = 3) {
    super(`Minimax-D${depth}`, { depth });
  }

  async selectMove(board, color, gameState) {
    const move = await getAIMove(board, this.depth, 0, gameState);
    return move;
  }
}

/**
 * Aggressive Strategy - Prioritizes checkmate threats and attacks
 */
export class AggressiveStrategy extends BaseStrategy {
  constructor(depth = 3) {
    super(`Aggressive-D${depth}`, { depth });
  }

  async selectMove(board, color, gameState) {
    const legalMoves = this.getAllLegalMoves(board, color, gameState);
    if (legalMoves.length === 0) return null;

    // Prioritize moves that give check or create checkmate threats
    const scoredMoves = legalMoves.map(move => ({
      move,
      score: this.evaluateAggressiveMove(board, move, color, gameState)
    }));

    // Sort by score (highest first)
    scoredMoves.sort((a, b) => b.score - a.score);

    // Pick from top 3 moves (add some randomness)
    const topMoves = scoredMoves.slice(0, 3);
    const selectedMove = topMoves[Math.floor(Math.random() * topMoves.length)];

    return selectedMove.move;
  }

  evaluateAggressiveMove(board, move, color, gameState) {
    let score = 0;

    // Simulate the move
    const testBoard = JSON.parse(JSON.stringify(board));
    const testGameState = JSON.parse(JSON.stringify(gameState));

    const [fromRow, fromCol] = move.from;
    const [toRow, toCol] = move.to;

    const captured = testBoard[toRow][toCol];
    testBoard[toRow][toCol] = testBoard[fromRow][fromCol];
    testBoard[fromRow][fromCol] = '';

    // Bonus for captures
    if (captured) {
      const pieceValues = { '♔': 0, '♚': 0, '♕': 900, '♛': 900, '♖': 500, '♜': 500, '♗': 330, '♝': 330, '♘': 320, '♞': 320, '♙': 100, '♟': 100 };
      score += pieceValues[captured] || 0;
    }

    // Huge bonus for checks
    const opponentColor = color === 'white' ? 'black' : 'white';
    if (isInCheck(testBoard, opponentColor)) {
      score += 500;
    }

    // Check if this leads to checkmate
    const status = getGameStatus(testBoard, opponentColor, testGameState);
    if (status === 'white_wins' || status === 'black_wins') {
      score += 10000; // Checkmate!
    }

    // Bonus for moving pieces toward opponent king
    const opponentKing = color === 'white' ? '♚' : '♔';
    let kingPos = null;
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        if (testBoard[r][c] === opponentKing) {
          kingPos = [r, c];
          break;
        }
      }
      if (kingPos) break;
    }

    if (kingPos) {
      const distBefore = Math.abs(fromRow - kingPos[0]) + Math.abs(fromCol - kingPos[1]);
      const distAfter = Math.abs(toRow - kingPos[0]) + Math.abs(toCol - kingPos[1]);
      score += (distBefore - distAfter) * 10; // Reward moving closer to king
    }

    return score;
  }
}

/**
 * Defensive Strategy - Prioritizes safety and solid positions
 */
export class DefensiveStrategy extends BaseStrategy {
  constructor(depth = 3) {
    super(`Defensive-D${depth}`, { depth });
  }

  async selectMove(board, color, gameState) {
    const legalMoves = this.getAllLegalMoves(board, color, gameState);
    if (legalMoves.length === 0) return null;

    const scoredMoves = legalMoves.map(move => ({
      move,
      score: this.evaluateDefensiveMove(board, move, color, gameState)
    }));

    scoredMoves.sort((a, b) => b.score - a.score);

    const topMoves = scoredMoves.slice(0, 3);
    const selectedMove = topMoves[Math.floor(Math.random() * topMoves.length)];

    return selectedMove.move;
  }

  evaluateDefensiveMove(board, move, color, gameState) {
    let score = 0;

    const testBoard = JSON.parse(JSON.stringify(board));
    const [fromRow, fromCol] = move.from;
    const [toRow, toCol] = move.to;

    const movingPiece = testBoard[fromRow][fromCol];
    const captured = testBoard[toRow][toCol];

    testBoard[toRow][toCol] = movingPiece;
    testBoard[fromRow][fromCol] = '';

    // Bonus for keeping king safe
    if (isInCheck(testBoard, color)) {
      score -= 1000; // Avoid putting ourselves in check
    }

    // Bonus for maintaining material
    const pieceValues = { '♔': 0, '♚': 0, '♕': 900, '♛': 900, '♖': 500, '♜': 500, '♗': 330, '♝': 330, '♘': 320, '♞': 320, '♙': 100, '♟': 100 };
    if (captured) {
      score += pieceValues[captured] || 0;
    }

    // Penalty for exposing valuable pieces
    const movingValue = pieceValues[movingPiece] || 0;
    if (movingValue > 300) {
      // Check if destination square is attacked
      score -= 50; // Slight penalty for moving valuable pieces
    }

    // Use board evaluation for overall position quality
    const evaluation = evaluateBoard(testBoard, color);
    score += evaluation;

    return score;
  }
}

/**
 * Trap Builder Strategy - Focus on creating checkmate-in-1 threats across multiple boards
 * This is the hypothesized optimal strategy for this variant
 */
export class TrapBuilderStrategy extends BaseStrategy {
  constructor(depth = 3) {
    super(`TrapBuilder-D${depth}`, { depth });
  }

  async selectMove(board, color, gameState, allBoards, activeBoardIndex) {
    const legalMoves = this.getAllLegalMoves(board, color, gameState);
    if (legalMoves.length === 0) return null;

    // Count how many boards already have traps
    const trapCount = this.countTrapBoards(allBoards, color);

    const scoredMoves = legalMoves.map(move => ({
      move,
      score: this.evaluateTrapMove(board, move, color, gameState, trapCount)
    }));

    scoredMoves.sort((a, b) => b.score - a.score);

    const topMoves = scoredMoves.slice(0, 3);
    const selectedMove = topMoves[Math.floor(Math.random() * topMoves.length)];

    return selectedMove.move;
  }

  countTrapBoards(allBoards, color) {
    let count = 0;
    for (const boardData of allBoards) {
      if (this.hasCheckmateIn1(boardData.board, color, boardData.gameState)) {
        count++;
      }
    }
    return count;
  }

  hasCheckmateIn1(board, color, gameState) {
    const allMoves = this.getAllLegalMoves(board, color, gameState);

    for (const move of allMoves) {
      const testBoard = JSON.parse(JSON.stringify(board));
      const testGameState = JSON.parse(JSON.stringify(gameState));

      const [fromRow, fromCol] = move.from;
      const [toRow, toCol] = move.to;

      testBoard[toRow][toCol] = testBoard[fromRow][fromCol];
      testBoard[fromRow][fromCol] = '';

      const opponentColor = color === 'white' ? 'black' : 'white';
      const status = getGameStatus(testBoard, opponentColor, testGameState);

      if (status === 'white_wins' || status === 'black_wins') {
        return true; // Found checkmate in 1
      }
    }

    return false;
  }

  evaluateTrapMove(board, move, color, gameState, currentTrapCount) {
    let score = 0;

    const testBoard = JSON.parse(JSON.stringify(board));
    const testGameState = JSON.parse(JSON.stringify(gameState));

    const [fromRow, fromCol] = move.from;
    const [toRow, toCol] = move.to;

    testBoard[toRow][toCol] = testBoard[fromRow][fromCol];
    testBoard[fromRow][fromCol] = '';

    // Immediate checkmate = highest priority
    const opponentColor = color === 'white' ? 'black' : 'white';
    const status = getGameStatus(testBoard, opponentColor, testGameState);
    if (status === 'white_wins' || status === 'black_wins') {
      return 100000; // CHECKMATE!
    }

    // Check if this move creates a checkmate-in-1 threat
    if (this.hasCheckmateIn1(testBoard, color, testGameState)) {
      score += 5000; // Very high value for creating traps
    }

    // Bonus for checks (potential threats)
    if (isInCheck(testBoard, opponentColor)) {
      score += 800;
    }

    // Material considerations (secondary)
    const captured = board[toRow][toCol];
    const pieceValues = { '♔': 0, '♚': 0, '♕': 900, '♛': 900, '♖': 500, '♜': 500, '♗': 330, '♝': 330, '♘': 320, '♞': 320, '♙': 100, '♟': 100 };
    if (captured) {
      score += (pieceValues[captured] || 0) * 0.5; // Less weight on material
    }

    // Bonus for creating multiple trap boards (the key hypothesis!)
    if (currentTrapCount < 2) {
      score += 1000; // Encourage building up to 2 trap boards
    }

    return score;
  }
}

export default {
  RandomStrategy,
  MinimaxStrategy,
  AggressiveStrategy,
  DefensiveStrategy,
  TrapBuilderStrategy
};
