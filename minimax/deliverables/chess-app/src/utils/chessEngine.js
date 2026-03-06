// Chess Engine - Complete chess rule implementation with special moves

// Piece types
export const PIECES = {
  WHITE_KING: '♔',
  WHITE_QUEEN: '♕',
  WHITE_ROOK: '♖',
  WHITE_BISHOP: '♗',
  WHITE_KNIGHT: '♘',
  WHITE_PAWN: '♙',
  BLACK_KING: '♚',
  BLACK_QUEEN: '♛',
  BLACK_ROOK: '♜',
  BLACK_BISHOP: '♝',
  BLACK_KNIGHT: '♞',
  BLACK_PAWN: '♟',
};

// Create initial game state
export const createInitialGameState = () => ({
  hasMoved: {
    whiteKing: false,
    blackKing: false,
    whiteRookKingside: false,
    whiteRookQueenside: false,
    blackRookKingside: false,
    blackRookQueenside: false,
  },
  lastMove: null, // { from: [row, col], to: [row, col], piece }
});

export const isWhitePiece = (piece) => {
  return ['♔', '♕', '♖', '♗', '♘', '♙'].includes(piece);
};

export const isBlackPiece = (piece) => {
  return ['♚', '♛', '♜', '♝', '♞', '♟'].includes(piece);
};

export const getPieceColor = (piece) => {
  if (!piece) return null;
  return isWhitePiece(piece) ? 'white' : 'black';
};

export const getPieceType = (piece) => {
  const types = {
    '♔': 'king', '♚': 'king',
    '♕': 'queen', '♛': 'queen',
    '♖': 'rook', '♜': 'rook',
    '♗': 'bishop', '♝': 'bishop',
    '♘': 'knight', '♞': 'knight',
    '♙': 'pawn', '♟': 'pawn',
  };
  return types[piece] || null;
};

// Check if a square is on the board
const isValidSquare = (row, col) => {
  return row >= 0 && row < 8 && col >= 0 && col < 8;
};

// Get all possible moves for a piece (without checking if they put king in check)
export const getPieceMoves = (board, fromRow, fromCol, gameState = null) => {
  const piece = board[fromRow][fromCol];
  if (!piece) return [];

  const pieceType = getPieceType(piece);
  const pieceColor = getPieceColor(piece);

  switch (pieceType) {
    case 'pawn':
      return getPawnMoves(board, fromRow, fromCol, pieceColor, gameState);
    case 'knight':
      return getKnightMoves(board, fromRow, fromCol, pieceColor);
    case 'bishop':
      return getBishopMoves(board, fromRow, fromCol, pieceColor);
    case 'rook':
      return getRookMoves(board, fromRow, fromCol, pieceColor);
    case 'queen':
      return getQueenMoves(board, fromRow, fromCol, pieceColor);
    case 'king':
      return getKingMoves(board, fromRow, fromCol, pieceColor, gameState);
    default:
      return [];
  }
};

// Pawn moves
const getPawnMoves = (board, row, col, color, gameState = null) => {
  const moves = [];
  const direction = color === 'white' ? -1 : 1; // White moves up (-1), black moves down (+1)
  const startRow = color === 'white' ? 6 : 1;

  // Move forward one square
  const newRow = row + direction;
  if (isValidSquare(newRow, col) && !board[newRow][col]) {
    moves.push([newRow, col]);

    // Move forward two squares from starting position
    if (row === startRow) {
      const twoSquaresRow = row + 2 * direction;
      if (!board[twoSquaresRow][col]) {
        moves.push([twoSquaresRow, col]);
      }
    }
  }

  // Capture diagonally
  [-1, 1].forEach(colOffset => {
    const newCol = col + colOffset;
    if (isValidSquare(newRow, newCol)) {
      const targetPiece = board[newRow][newCol];
      if (targetPiece && getPieceColor(targetPiece) !== color) {
        moves.push([newRow, newCol]);
      }
    }
  });

  // En passant
  if (gameState && gameState.lastMove) {
    const { from, to, piece } = gameState.lastMove;
    const lastMovePieceType = getPieceType(piece);
    const lastMovePieceColor = getPieceColor(piece);

    // Check if last move was a pawn moving two squares
    if (lastMovePieceType === 'pawn' &&
        lastMovePieceColor !== color &&
        Math.abs(from[0] - to[0]) === 2) {

      // Check if we're adjacent to the pawn that just moved
      const enPassantRow = color === 'white' ? 3 : 4;
      if (row === enPassantRow && Math.abs(col - to[1]) === 1) {
        // En passant is possible
        moves.push([newRow, to[1]]);
      }
    }
  }

  return moves;
};

// Knight moves
const getKnightMoves = (board, row, col, color) => {
  const moves = [];
  const knightOffsets = [
    [-2, -1], [-2, 1], [-1, -2], [-1, 2],
    [1, -2], [1, 2], [2, -1], [2, 1]
  ];

  knightOffsets.forEach(([rowOffset, colOffset]) => {
    const newRow = row + rowOffset;
    const newCol = col + colOffset;
    if (isValidSquare(newRow, newCol)) {
      const targetPiece = board[newRow][newCol];
      if (!targetPiece || getPieceColor(targetPiece) !== color) {
        moves.push([newRow, newCol]);
      }
    }
  });

  return moves;
};

// Bishop moves (diagonal)
const getBishopMoves = (board, row, col, color) => {
  return getSlidingMoves(board, row, col, color, [
    [-1, -1], [-1, 1], [1, -1], [1, 1]
  ]);
};

// Rook moves (straight)
const getRookMoves = (board, row, col, color) => {
  return getSlidingMoves(board, row, col, color, [
    [-1, 0], [1, 0], [0, -1], [0, 1]
  ]);
};

// Queen moves (diagonal + straight)
const getQueenMoves = (board, row, col, color) => {
  return getSlidingMoves(board, row, col, color, [
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1], [0, 1],
    [1, -1], [1, 0], [1, 1]
  ]);
};

// King moves
const getKingMoves = (board, row, col, color, gameState = null) => {
  const moves = [];
  const kingOffsets = [
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1], [0, 1],
    [1, -1], [1, 0], [1, 1]
  ];

  kingOffsets.forEach(([rowOffset, colOffset]) => {
    const newRow = row + rowOffset;
    const newCol = col + colOffset;
    if (isValidSquare(newRow, newCol)) {
      const targetPiece = board[newRow][newCol];
      if (!targetPiece || getPieceColor(targetPiece) !== color) {
        moves.push([newRow, newCol]);
      }
    }
  });

  // Castling
  if (gameState) {
    const baseRow = color === 'white' ? 7 : 0;
    const opponentColor = color === 'white' ? 'black' : 'white';

    // King must be on its starting square
    if (row === baseRow && col === 4) {
      const kingMoved = color === 'white' ? gameState.hasMoved.whiteKing : gameState.hasMoved.blackKing;

      // Cannot castle when in check
      if (!kingMoved && !isSquareUnderAttack(board, baseRow, 4, opponentColor)) {
        // Kingside castling (0-0)
        const kingsideRookMoved = color === 'white' ?
          gameState.hasMoved.whiteRookKingside :
          gameState.hasMoved.blackRookKingside;

        if (!kingsideRookMoved) {
          // Check if squares between king and rook are empty
          if (!board[baseRow][5] && !board[baseRow][6]) {
            // Check if king is not in check and doesn't pass through check
            if (!isSquareUnderAttack(board, baseRow, 4, opponentColor) &&
                !isSquareUnderAttack(board, baseRow, 5, opponentColor) &&
                !isSquareUnderAttack(board, baseRow, 6, opponentColor)) {
              // Check if rook is still there
              const expectedRook = color === 'white' ? PIECES.WHITE_ROOK : PIECES.BLACK_ROOK;
              if (board[baseRow][7] === expectedRook) {
                moves.push([baseRow, 6]); // King moves to g-file
              }
            }
          }
        }

        // Queenside castling (0-0-0)
        const queensideRookMoved = color === 'white' ?
          gameState.hasMoved.whiteRookQueenside :
          gameState.hasMoved.blackRookQueenside;

        if (!queensideRookMoved) {
          // Check if squares between king and rook are empty
          if (!board[baseRow][1] && !board[baseRow][2] && !board[baseRow][3]) {
            // Check if king is not in check and doesn't pass through check
            if (!isSquareUnderAttack(board, baseRow, 4, opponentColor) &&
                !isSquareUnderAttack(board, baseRow, 3, opponentColor) &&
                !isSquareUnderAttack(board, baseRow, 2, opponentColor)) {
              // Check if rook is still there
              const expectedRook = color === 'white' ? PIECES.WHITE_ROOK : PIECES.BLACK_ROOK;
              if (board[baseRow][0] === expectedRook) {
                moves.push([baseRow, 2]); // King moves to c-file
              }
            }
          }
        }
      }
    }
  }

  return moves;
};

// Helper for sliding pieces (rook, bishop, queen)
const getSlidingMoves = (board, row, col, color, directions) => {
  const moves = [];

  directions.forEach(([rowDir, colDir]) => {
    let newRow = row + rowDir;
    let newCol = col + colDir;

    while (isValidSquare(newRow, newCol)) {
      const targetPiece = board[newRow][newCol];

      if (!targetPiece) {
        moves.push([newRow, newCol]);
      } else {
        if (getPieceColor(targetPiece) !== color) {
          moves.push([newRow, newCol]); // Can capture
        }
        break; // Can't move past any piece
      }

      newRow += rowDir;
      newCol += colDir;
    }
  });

  return moves;
};

// Find the king of a given color
export const findKing = (board, color) => {
  const kingPiece = color === 'white' ? PIECES.WHITE_KING : PIECES.BLACK_KING;
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      if (board[row][col] === kingPiece) {
        return [row, col];
      }
    }
  }
  return null;
};

// Check if a square is under attack by the opponent
export const isSquareUnderAttack = (board, row, col, byColor) => {
  // Check if any piece of 'byColor' can attack this square
  // IMPORTANT: Pass null for gameState to avoid circular logic with castling/en passant
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const piece = board[r][c];
      if (piece && getPieceColor(piece) === byColor) {
        const moves = getPieceMoves(board, r, c, null);
        if (moves.some(([mr, mc]) => mr === row && mc === col)) {
          return true;
        }
      }
    }
  }
  return false;
};

// Check if the current player is in check
export const isInCheck = (board, color) => {
  const kingPos = findKing(board, color);
  if (!kingPos) return false;

  const opponentColor = color === 'white' ? 'black' : 'white';
  return isSquareUnderAttack(board, kingPos[0], kingPos[1], opponentColor);
};

// Make a move on the board (returns new board and updated game state)
export const makeMove = (board, fromRow, fromCol, toRow, toCol, gameState = null) => {
  const newBoard = board.map(row => [...row]);
  const piece = newBoard[fromRow][fromCol];
  const pieceType = getPieceType(piece);
  const pieceColor = getPieceColor(piece);

  // Create new game state
  const newGameState = gameState ? {
    hasMoved: { ...gameState.hasMoved },
    lastMove: { from: [fromRow, fromCol], to: [toRow, toCol], piece }
  } : null;

  // Check for castling
  if (pieceType === 'king' && Math.abs(toCol - fromCol) === 2) {
    // Castling detected
    newBoard[toRow][toCol] = piece;
    newBoard[fromRow][fromCol] = '';

    // Move the rook
    if (toCol === 6) {
      // Kingside castling
      const rook = newBoard[fromRow][7];
      newBoard[fromRow][5] = rook;
      newBoard[fromRow][7] = '';
    } else if (toCol === 2) {
      // Queenside castling
      const rook = newBoard[fromRow][0];
      newBoard[fromRow][3] = rook;
      newBoard[fromRow][0] = '';
    }
  } else {
    // Regular move
    newBoard[toRow][toCol] = piece;
    newBoard[fromRow][fromCol] = '';

    // Check for en passant capture
    if (pieceType === 'pawn' && fromCol !== toCol && !board[toRow][toCol]) {
      // En passant capture - remove the captured pawn
      const capturedPawnRow = pieceColor === 'white' ? toRow + 1 : toRow - 1;
      newBoard[capturedPawnRow][toCol] = '';
    }
  }

  // Pawn promotion
  if (pieceType === 'pawn') {
    if ((pieceColor === 'white' && toRow === 0) ||
        (pieceColor === 'black' && toRow === 7)) {
      // Promote to queen
      newBoard[toRow][toCol] = pieceColor === 'white' ? PIECES.WHITE_QUEEN : PIECES.BLACK_QUEEN;
    }
  }

  // Update hasMoved flags
  if (newGameState) {
    if (pieceType === 'king') {
      if (pieceColor === 'white') {
        newGameState.hasMoved.whiteKing = true;
      } else {
        newGameState.hasMoved.blackKing = true;
      }
    } else if (pieceType === 'rook') {
      const baseRow = pieceColor === 'white' ? 7 : 0;
      if (fromRow === baseRow) {
        if (fromCol === 0) {
          // Queenside rook
          if (pieceColor === 'white') {
            newGameState.hasMoved.whiteRookQueenside = true;
          } else {
            newGameState.hasMoved.blackRookQueenside = true;
          }
        } else if (fromCol === 7) {
          // Kingside rook
          if (pieceColor === 'white') {
            newGameState.hasMoved.whiteRookKingside = true;
          } else {
            newGameState.hasMoved.blackRookKingside = true;
          }
        }
      }
    }
  }

  return { board: newBoard, gameState: newGameState };
};

// Check if a move is legal (doesn't leave king in check)
export const isMoveLegal = (board, fromRow, fromCol, toRow, toCol, color, gameState = null) => {
  const { board: newBoard } = makeMove(board, fromRow, fromCol, toRow, toCol, gameState);
  return !isInCheck(newBoard, color);
};

// Get all legal moves for a piece
export const getLegalMoves = (board, fromRow, fromCol, gameState = null) => {
  const piece = board[fromRow][fromCol];
  if (!piece) return [];

  const color = getPieceColor(piece);
  const possibleMoves = getPieceMoves(board, fromRow, fromCol, gameState);

  return possibleMoves.filter(([toRow, toCol]) =>
    isMoveLegal(board, fromRow, fromCol, toRow, toCol, color, gameState)
  );
};

// Get all legal moves for a color
export const getAllLegalMoves = (board, color, gameState = null) => {
  const moves = [];

  for (let fromRow = 0; fromRow < 8; fromRow++) {
    for (let fromCol = 0; fromCol < 8; fromCol++) {
      const piece = board[fromRow][fromCol];
      if (piece && getPieceColor(piece) === color) {
        const legalMoves = getLegalMoves(board, fromRow, fromCol, gameState);
        legalMoves.forEach(([toRow, toCol]) => {
          moves.push({ from: [fromRow, fromCol], to: [toRow, toCol] });
        });
      }
    }
  }

  return moves;
};

// Check if the game is in checkmate
export const isCheckmate = (board, color, gameState = null) => {
  if (!isInCheck(board, color)) return false;
  return getAllLegalMoves(board, color, gameState).length === 0;
};

// Check if the game is in stalemate
export const isStalemate = (board, color, gameState = null) => {
  if (isInCheck(board, color)) return false;
  return getAllLegalMoves(board, color, gameState).length === 0;
};

// Get game status
export const getGameStatus = (board, currentTurn, gameState = null) => {
  if (isCheckmate(board, currentTurn, gameState)) {
    return currentTurn === 'white' ? 'black_wins' : 'white_wins';
  }
  if (isStalemate(board, currentTurn, gameState)) {
    return 'stalemate';
  }
  if (isInCheck(board, currentTurn)) {
    return 'check';
  }
  return 'in_progress';
};
