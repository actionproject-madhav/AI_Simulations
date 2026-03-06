// Ultimate Tic-Tac-Toe Game Engine
// Rules: X goes first (any of 81 cells). The cell you pick determines which
// small board the opponent must play on next (cell index = board index).
// Winning a small board marks it (X/O); no more moves there. If sent to a
// board that's already won or full, the player may choose any active board.
// Win = 3 in a row on the meta board; draw when no winner possible.

/**
 * Creates the initial game state.
 * X goes first and may move to any of the 81 positions (all 9 boards active).
 * @returns {Object} Initial game state with empty boards
 */
export function createInitialGameState() {
  return {
    // 9 local boards, each with 9 cells (81 cells total). Board/cell index 0 = top-left, 8 = center.
    board: Array(9).fill(null).map(() => Array(9).fill('')),
    // Meta-board: '' = open, 'X'/'O' = won, 'D' = draw (full, no winner)
    metaBoard: Array(9).fill(''),
    // Indices of boards that may be played on. Initially all 9 (X can play anywhere).
    activeBoards: [0, 1, 2, 3, 4, 5, 6, 7, 8],
    currentPlayer: 'X'
  };
}

/**
 * Gets the 9 cells for a specific local board
 * @param {Array} board - The full 9x9 board
 * @param {number} localBoardIdx - Index of the local board (0-8)
 * @returns {Array} Array of 9 cells for that local board
 */
export function getBoardCells(board, localBoardIdx) {
  return board[localBoardIdx];
}

/**
 * Checks if a local board (or meta-board) has a winner
 * @param {Array} cells - Array of 9 cells to check
 * @returns {string|null} 'X', 'O', or null
 */
export function checkLocalBoardWinner(cells) {
  // All 8 win patterns: 3 rows, 3 columns, 2 diagonals
  const winPatterns = [
    [0, 1, 2], // Top row
    [3, 4, 5], // Middle row
    [6, 7, 8], // Bottom row
    [0, 3, 6], // Left column
    [1, 4, 7], // Middle column
    [2, 5, 8], // Right column
    [0, 4, 8], // Diagonal top-left to bottom-right
    [2, 4, 6]  // Diagonal top-right to bottom-left
  ];

  for (const pattern of winPatterns) {
    const [a, b, c] = pattern;
    if (cells[a] && cells[a] === cells[b] && cells[a] === cells[c]) {
      return cells[a];
    }
  }

  return null;
}

/**
 * Checks if a local board is completely filled
 * @param {Array} cells - Array of 9 cells
 * @returns {boolean} True if all cells are filled
 */
export function isLocalBoardFull(cells) {
  return cells.every(cell => cell !== '');
}

/**
 * Checks meta-board for winner. Only X or O can win; 'D' (drawn small board) does not count.
 * @param {Array} metaBoard - Array of 9 meta-positions ('', 'X', 'O', or 'D')
 * @returns {string|null} 'X', 'O', or null
 */
export function checkMetaBoardWinner(metaBoard) {
  const winner = checkLocalBoardWinner(metaBoard);
  return winner === 'X' || winner === 'O' ? winner : null;
}

/**
 * Gets the overall game status
 * @param {Array} board - The full 9x9 board
 * @param {Array} metaBoard - The meta-board state
 * @returns {string} 'x_wins', 'o_wins', 'draw', 'in_progress'
 */
export function getGameStatus(board, metaBoard) {
  const metaWinner = checkMetaBoardWinner(metaBoard);

  if (metaWinner === 'X') return 'x_wins';
  if (metaWinner === 'O') return 'o_wins';

  // Check if meta-board is full (draw)
  const isMetaBoardFull = metaBoard.every(cell => cell !== '');
  if (isMetaBoardFull) return 'draw';

  // Check if any moves are possible
  const hasMovesLeft = board.some((localBoard, idx) => {
    if (metaBoard[idx] !== '') return false; // Board already won
    return localBoard.some(cell => cell === ''); // Has empty cells
  });

  if (!hasMovesLeft) return 'draw';

  return 'in_progress';
}

/**
 * Validates if a move is legal (must be on an active board, open cell, board not won/full).
 */
export function isValidMove(board, metaBoard, activeBoards, localBoardIdx, cellIdx) {
  if (!activeBoards.includes(localBoardIdx)) return false;
  if (metaBoard[localBoardIdx] !== '') return false; // board won or drawn
  if (board[localBoardIdx][cellIdx] !== '') return false;
  return true;
}

/**
 * Makes a move and returns new game state (immutable).
 * Rule: the cell chosen (cellIdx) determines which board the opponent must play on next
 * (target board = cellIdx). If that board is won or full, opponent may choose any active board.
 */
export function makeMove(board, metaBoard, activeBoards, localBoardIdx, cellIdx, player) {
  const newBoard = board.map((localBoard, idx) =>
    idx === localBoardIdx
      ? localBoard.map((cell, i) => (i === cellIdx ? player : cell))
      : [...localBoard]
  );

  const localCells = newBoard[localBoardIdx];
  const localWinner = checkLocalBoardWinner(localCells);
  const newMetaBoard = [...metaBoard];

  if (localWinner) {
    newMetaBoard[localBoardIdx] = localWinner;
  } else if (isLocalBoardFull(localCells)) {
    newMetaBoard[localBoardIdx] = 'D'; // drawn small board: closed, no more moves
  }

  // Next player must play on the board corresponding to the cell just played
  const targetBoardIdx = cellIdx;
  let newActiveBoards;

  const targetOpen = newMetaBoard[targetBoardIdx] === '' && !isLocalBoardFull(newBoard[targetBoardIdx]);
  if (targetOpen) {
    newActiveBoards = [targetBoardIdx];
  } else {
    // Sent to won/full board: may choose any board that is still open
    newActiveBoards = [];
    for (let i = 0; i < 9; i++) {
      if (newMetaBoard[i] === '' && !isLocalBoardFull(newBoard[i])) {
        newActiveBoards.push(i);
      }
    }
  }

  return {
    board: newBoard,
    metaBoard: newMetaBoard,
    activeBoards: newActiveBoards,
    currentPlayer: player === 'X' ? 'O' : 'X'
  };
}

/**
 * Gets all legal moves: only on active boards that are not won/drawn, empty cells only.
 */
export function getAllLegalMoves(board, metaBoard, activeBoards) {
  const moves = [];
  for (const localBoardIdx of activeBoards) {
    if (metaBoard[localBoardIdx] !== '') continue;
    const localCells = board[localBoardIdx];
    if (isLocalBoardFull(localCells)) continue;
    for (let cellIdx = 0; cellIdx < 9; cellIdx++) {
      if (localCells[cellIdx] === '') {
        moves.push({ localBoardIdx, cellIdx });
      }
    }
  }
  return moves;
}

/**
 * Clones the game state (deep copy)
 * @param {Object} state - Game state to clone
 * @returns {Object} Cloned state
 */
export function cloneState(state) {
  return {
    board: state.board.map(localBoard => [...localBoard]),
    metaBoard: [...state.metaBoard],
    activeBoards: [...state.activeBoards],
    currentPlayer: state.currentPlayer
  };
}
