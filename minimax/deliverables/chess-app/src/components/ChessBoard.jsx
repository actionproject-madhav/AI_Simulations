import { useState, useEffect } from 'react';
import Square from './Square';
import './ChessBoard.css';
import { getLegalMoves, makeMove, getGameStatus, isInCheck, createInitialGameState } from '../utils/chessEngine';
import { getAIMove } from '../utils/aiEngine';

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

const ChessBoard = ({ onMove, currentTurn, onGameStatusChange, compact = false, isActive = true, humanColor = 'white' }) => {
  const [board, setBoard] = useState(initialBoardState);
  const [gameState, setGameState] = useState(createInitialGameState());
  const [selectedSquare, setSelectedSquare] = useState(null);
  const [possibleMoves, setPossibleMoves] = useState([]);
  const [isAIThinking, setIsAIThinking] = useState(false);
  const [gameStatus, setGameStatus] = useState('in_progress');

  const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  const ranks = ['8', '7', '6', '5', '4', '3', '2', '1'];

  // When human is black, flip board so black (human) is at bottom
  const isFlipped = humanColor === 'black';
  const displayRanks = isFlipped ? ['1', '2', '3', '4', '5', '6', '7', '8'] : ranks;
  const displayFiles = isFlipped ? ['h', 'g', 'f', 'e', 'd', 'c', 'b', 'a'] : files;

  // Convert display position (0-7) to board position
  const toBoardPos = (displayRow, displayCol) =>
    isFlipped ? [7 - displayRow, 7 - displayCol] : [displayRow, displayCol];

  // Check game status after every move
  useEffect(() => {
    const status = getGameStatus(board, currentTurn, gameState);
    setGameStatus(status);
    if (onGameStatusChange) {
      onGameStatusChange(status);
    }
  }, [board, currentTurn, gameState, onGameStatusChange]);

  const isHumanTurn = currentTurn === humanColor;

  // Trigger AI move when it's AI's turn (and this board is active)
  useEffect(() => {
    const canPlay = gameStatus === 'in_progress' || gameStatus === 'check';
    if (!isHumanTurn && canPlay && !isAIThinking && isActive) {
      makeAIMove();
    }
  }, [currentTurn, gameStatus, isHumanTurn, isActive]);

  const makeAIMove = async () => {
    setIsAIThinking(true);

    try {
      const aiMove = await getAIMove(board, 3, 1200, gameState); // depth 3, min 1200ms delay for better visibility

      if (aiMove) {
        const { from, to } = aiMove;
        const { board: newBoard, gameState: newGameState } = makeMove(
          board,
          from[0],
          from[1],
          to[0],
          to[1],
          gameState
        );
        setBoard(newBoard);
        setGameState(newGameState);

        const fromSquare = files[from[1]] + ranks[from[0]];
        const toSquare = files[to[1]] + ranks[to[0]];
        onMove?.(`${fromSquare}-${toSquare}`, currentTurn === 'black');
      }
    } catch (error) {
      console.error('AI move error:', error);
    } finally {
      setIsAIThinking(false);
    }
  };

  const handleSquareClick = (displayRow, displayCol) => {
    const [row, col] = toBoardPos(displayRow, displayCol);

    // Don't allow moves on inactive board, during AI thinking, or if game is over
    const isGameOver = ['white_wins', 'black_wins', 'stalemate'].includes(gameStatus);
    if (!isActive || isAIThinking || isGameOver) return;

    // Only allow human to move their pieces (humanColor determines which)
    if (!isHumanTurn) return;

    const piece = board[row][col];

    const humanPieces = humanColor === 'white' ? ['♔', '♕', '♖', '♗', '♘', '♙'] : ['♚', '♛', '♜', '♝', '♞', '♟'];

    if (selectedSquare) {
      const [selectedRow, selectedCol] = selectedSquare;

      // Deselect if clicking same square
      if (row === selectedRow && col === selectedCol) {
        setSelectedSquare(null);
        setPossibleMoves([]);
        return;
      }

      // Make move if it's a legal move
      if (possibleMoves.some(move => move[0] === row && move[1] === col)) {
        const { board: newBoard, gameState: newGameState } = makeMove(
          board,
          selectedRow,
          selectedCol,
          row,
          col,
          gameState
        );
        setBoard(newBoard);
        setGameState(newGameState);
        setSelectedSquare(null);
        setPossibleMoves([]);

        const fromSquare = files[selectedCol] + ranks[selectedRow];
        const toSquare = files[col] + ranks[row];
        onMove?.(`${fromSquare}-${toSquare}`, currentTurn === 'black');
      } else if (piece && humanPieces.includes(piece)) {
        setSelectedSquare([row, col]);
        const legalMoves = getLegalMoves(board, row, col, gameState);
        setPossibleMoves(legalMoves);
      } else {
        setSelectedSquare(null);
        setPossibleMoves([]);
      }
    } else if (piece && humanPieces.includes(piece)) {
      setSelectedSquare([row, col]);
      const legalMoves = getLegalMoves(board, row, col, gameState);
      setPossibleMoves(legalMoves);
    }
  };

  const getSquareClassName = (displayRow, displayCol) => {
    const [boardRow, boardCol] = toBoardPos(displayRow, displayCol);
    const isLight = (boardRow + boardCol) % 2 === 0;
    const isSelected = selectedSquare &&
      selectedSquare[0] === boardRow &&
      selectedSquare[1] === boardCol;
    const isPossibleMove = possibleMoves.some(
      move => move[0] === boardRow && move[1] === boardCol
    );

    // Highlight king if in check
    const piece = board[boardRow][boardCol];
    const isKingInCheck = (piece === '♔' || piece === '♚') &&
      isInCheck(board, piece === '♔' ? 'white' : 'black');

    return {
      isLight,
      isSelected,
      isPossibleMove,
      isKingInCheck
    };
  };

  return (
    <div className={`board-wrapper ${compact ? 'compact' : ''} ${isFlipped ? 'flipped' : ''}`}>
      <div className="coordinates-left">
        {displayRanks.map((rank) => (
          <div key={rank} className="coord">
            {rank}
          </div>
        ))}
      </div>

      <div className="board-container">
        {isAIThinking && (
          <div className="ai-thinking-overlay">
            <div className="ai-thinking-message">
              <div className="spinner"></div>
              <span>AI is thinking...</span>
            </div>
          </div>
        )}

        <div className={`chessboard ${isAIThinking ? 'disabled' : ''}`}>
          {[0, 1, 2, 3, 4, 5, 6, 7].map((displayRow) =>
            [0, 1, 2, 3, 4, 5, 6, 7].map((displayCol) => {
              const [boardRow, boardCol] = toBoardPos(displayRow, displayCol);
              const piece = board[boardRow][boardCol];
              const { isLight, isSelected, isPossibleMove, isKingInCheck } =
                getSquareClassName(displayRow, displayCol);

              return (
                <Square
                  key={`${displayRow}-${displayCol}`}
                  piece={piece}
                  isLight={isLight}
                  isSelected={isSelected}
                  isPossibleMove={isPossibleMove}
                  isKingInCheck={isKingInCheck}
                  onClick={() => handleSquareClick(displayRow, displayCol)}
                />
              );
            })
          )}
        </div>

        <div className="coordinates-bottom">
          {displayFiles.map((file) => (
            <div key={file} className="coord">
              {file}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ChessBoard;
