/**
 * Headless 3-Board Chess Game Simulator
 * Runs games without UI for strategy discovery
 */

import { createInitialGameState, makeMove, getGameStatus, isInCheck } from '../utils/chessEngine.js';
import { getAIMove } from '../utils/aiEngine.js';

const initialBoardState = [
  ['♜', '♞', '♝', '♛', '♚', '♝', '♞', '♜'],
  ['♟', '♟', '♟', '♟', '♟', '♟', '♟', '♟'],
  ['', '', '', '', '', '', '', ''],
  ['', '', '', '', '', '', '', ''],
  ['', '', '', '', '', '', '', ''],
  ['', '', '', '', '', '', '', ''],
  ['♙', '♙', '♙', '♙', '♙', '♙', '♙', '♙'],
  ['♖', '♘', '♗', '♕', '♔', '♗', '♘', '♖'],
];

/**
 * Simulates a complete 3-board chess game
 */
export class GameSimulator {
  constructor(player1Strategy, player2Strategy, options = {}) {
    this.player1Strategy = player1Strategy; // Strategy for player 1
    this.player2Strategy = player2Strategy; // Strategy for player 2
    this.maxTurnPairs = options.maxTurnPairs || 100; // Prevent infinite games
    this.verbose = options.verbose || false; // Log game progress
    this.aiDepth = options.aiDepth || 3; // Minimax depth
    this.aiDelay = 0; // No delay for simulations

    this.reset();
  }

  reset() {
    // Initialize 3 boards
    this.boards = [
      {
        board: JSON.parse(JSON.stringify(initialBoardState)),
        gameState: createInitialGameState(),
        currentTurn: 'white',
        status: 'in_progress'
      },
      {
        board: JSON.parse(JSON.stringify(initialBoardState)),
        gameState: createInitialGameState(),
        currentTurn: 'white',
        status: 'in_progress'
      },
      {
        board: JSON.parse(JSON.stringify(initialBoardState)),
        gameState: createInitialGameState(),
        currentTurn: 'white',
        status: 'in_progress'
      }
    ];

    this.turnPairCount = 0;
    this.moveHistory = [];
    this.gameOver = false;
    this.winner = null;
    this.gameLog = [];
  }

  /**
   * Pick random board (0-2)
   */
  pickRandomBoard() {
    return Math.floor(Math.random() * 3);
  }

  /**
   * Pick random player assignment (player1 = white/black)
   */
  pickRandomSide() {
    return Math.random() < 0.5 ? 'white' : 'black';
  }

  /**
   * Get AI move using strategy
   */
  async getMove(boardIndex, color, strategy) {
    const boardData = this.boards[boardIndex];

    // Use the strategy to pick a move
    const move = await strategy.selectMove(
      boardData.board,
      color,
      boardData.gameState,
      this.boards, // Pass all boards for strategy context
      boardIndex
    );

    return move;
  }

  /**
   * Execute a move on a board
   */
  executeMove(boardIndex, move) {
    const boardData = this.boards[boardIndex];
    const { from, to } = move;

    const result = makeMove(
      boardData.board,
      from[0],
      from[1],
      to[0],
      to[1],
      boardData.gameState
    );

    boardData.board = result.board;
    boardData.gameState = result.gameState;

    return result;
  }

  /**
   * Update board status (check, checkmate, etc.)
   */
  updateBoardStatus(boardIndex) {
    const boardData = this.boards[boardIndex];
    const status = getGameStatus(
      boardData.board,
      boardData.currentTurn,
      boardData.gameState
    );
    boardData.status = status;
    return status;
  }

  /**
   * Switch turn on a board
   */
  switchTurn(boardIndex) {
    const boardData = this.boards[boardIndex];
    boardData.currentTurn = boardData.currentTurn === 'white' ? 'black' : 'white';
  }

  /**
   * Log game event
   */
  log(message) {
    this.gameLog.push(message);
    if (this.verbose) {
      console.log(message);
    }
  }

  /**
   * Play a complete turn pair (white move + black move on one board)
   */
  async playTurnPair() {
    this.turnPairCount++;

    // Step 1: Random board and color assignment
    const activeBoardIndex = this.pickRandomBoard();
    const player1Color = this.pickRandomSide();
    const player2Color = player1Color === 'white' ? 'black' : 'white';

    this.log(`\n=== Turn Pair ${this.turnPairCount} ===`);
    this.log(`Board: ${activeBoardIndex + 1}, Player1=${player1Color}, Player2=${player2Color}`);

    const boardData = this.boards[activeBoardIndex];

    // Ensure board starts at white's turn (should always be the case)
    if (boardData.currentTurn !== 'white') {
      this.log(`ERROR: Board ${activeBoardIndex + 1} not at white's turn!`);
      return false;
    }

    // Step 2: White's move
    const whitePlayer = player1Color === 'white' ? 'player1' : 'player2';
    const whiteStrategy = whitePlayer === 'player1' ? this.player1Strategy : this.player2Strategy;

    this.log(`White's turn (${whitePlayer})`);
    const whiteMove = await this.getMove(activeBoardIndex, 'white', whiteStrategy);

    if (!whiteMove) {
      this.log(`White has no legal moves!`);
      const status = this.updateBoardStatus(activeBoardIndex);
      if (status === 'stalemate') {
        this.gameOver = true;
        this.winner = 'draw';
        return false;
      }
      return false;
    }

    this.executeMove(activeBoardIndex, whiteMove);
    this.moveHistory.push({
      turnPair: this.turnPairCount,
      board: activeBoardIndex,
      color: 'white',
      player: whitePlayer,
      move: whiteMove
    });
    this.log(`White: ${JSON.stringify(whiteMove.from)} -> ${JSON.stringify(whiteMove.to)}`);

    // Check if white just won
    this.switchTurn(activeBoardIndex);
    const statusAfterWhite = this.updateBoardStatus(activeBoardIndex);

    if (statusAfterWhite === 'white_wins') {
      this.gameOver = true;
      this.winner = whitePlayer;
      this.log(`GAME OVER: White wins! Winner: ${whitePlayer}`);
      return false;
    }

    // Step 3: Black's move
    const blackPlayer = player1Color === 'black' ? 'player1' : 'player2';
    const blackStrategy = blackPlayer === 'player1' ? this.player1Strategy : this.player2Strategy;

    this.log(`Black's turn (${blackPlayer})`);
    const blackMove = await this.getMove(activeBoardIndex, 'black', blackStrategy);

    if (!blackMove) {
      this.log(`Black has no legal moves!`);
      const status = this.updateBoardStatus(activeBoardIndex);
      if (status === 'stalemate') {
        this.gameOver = true;
        this.winner = 'draw';
        return false;
      }
      return false;
    }

    this.executeMove(activeBoardIndex, blackMove);
    this.moveHistory.push({
      turnPair: this.turnPairCount,
      board: activeBoardIndex,
      color: 'black',
      player: blackPlayer,
      move: blackMove
    });
    this.log(`Black: ${JSON.stringify(blackMove.from)} -> ${JSON.stringify(blackMove.to)}`);

    // Check if black just won
    this.switchTurn(activeBoardIndex);
    const statusAfterBlack = this.updateBoardStatus(activeBoardIndex);

    if (statusAfterBlack === 'black_wins') {
      this.gameOver = true;
      this.winner = blackPlayer;
      this.log(`GAME OVER: Black wins! Winner: ${blackPlayer}`);
      return false;
    }

    if (statusAfterBlack === 'stalemate') {
      this.gameOver = true;
      this.winner = 'draw';
      this.log(`GAME OVER: Stalemate`);
      return false;
    }

    return true; // Turn pair completed successfully
  }

  /**
   * Play a complete game
   */
  async playGame() {
    this.reset();
    this.log(`Starting new game: ${this.player1Strategy.name} vs ${this.player2Strategy.name}`);

    while (!this.gameOver && this.turnPairCount < this.maxTurnPairs) {
      const continueGame = await this.playTurnPair();
      if (!continueGame) {
        break;
      }
    }

    if (this.turnPairCount >= this.maxTurnPairs) {
      this.log(`Game reached max turn pairs (${this.maxTurnPairs}). Declaring draw.`);
      this.winner = 'draw';
    }

    this.log(`\nGame finished after ${this.turnPairCount} turn pairs. Winner: ${this.winner}`);

    return this.getGameResult();
  }

  /**
   * Get game result summary
   */
  getGameResult() {
    return {
      winner: this.winner,
      turnPairs: this.turnPairCount,
      moveHistory: this.moveHistory,
      finalBoards: this.boards.map(b => ({
        board: b.board,
        status: b.status,
        currentTurn: b.currentTurn
      })),
      gameLog: this.gameLog
    };
  }

  /**
   * Get current game state snapshot (for analysis)
   */
  getSnapshot() {
    return {
      turnPair: this.turnPairCount,
      boards: this.boards.map(b => ({
        board: JSON.parse(JSON.stringify(b.board)),
        gameState: JSON.parse(JSON.stringify(b.gameState)),
        currentTurn: b.currentTurn,
        status: b.status
      }))
    };
  }
}

export default GameSimulator;
