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

const ChessBoard = ({ onMove, currentTurn, onGameStatusChange }) => {
  const [board, setBoard] = useState(initialBoardState);
  const [gameState, setGameState] = useState(createInitialGameState());
  const [selectedSquare, setSelectedSquare] = useState(null);
  const [possibleMoves, setPossibleMoves] = useState([]);
  const [isAIThinking, setIsAIThinking] = useState(false);
  const [gameStatus, setGameStatus] = useState('in_progress');

  const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  const ranks = ['8', '7', '6', '5', '4', '3', '2', '1'];

  // Check game status after every move
  useEffect(() => {
    const status = getGameStatus(board, currentTurn, gameState);
    setGameStatus(status);
    if (onGameStatusChange) {
      onGameStatusChange(status);
    }
  }, [board, currentTurn, gameState, onGameStatusChange]);

  // Trigger AI move when it's black's turn
  useEffect(() => {
    const canPlay = gameStatus === 'in_progress' || gameStatus === 'check';
    if (currentTurn === 'black' && canPlay && !isAIThinking) {
      makeAIMove();
    }
  }, [currentTurn, gameStatus]);

  const makeAIMove = async () => {
    setIsAIThinking(true);

    try {
      const aiMove = await getAIMove(board, 3, 800, gameState); // depth 3, min 800ms delay

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
        onMove(`${fromSquare}-${toSquare}`);
      }
    } catch (error) {
      console.error('AI move error:', error);
    } finally {
      setIsAIThinking(false);
    }
  };

  const handleSquareClick = (row, col) => {
    // Don't allow moves during AI thinking or if game is over
    const isGameOver = ['white_wins', 'black_wins', 'stalemate'].includes(gameStatus);
    if (isAIThinking || isGameOver) return;

    // Only allow white pieces to be moved by human
    if (currentTurn !== 'white') return;

    const piece = board[row][col];

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
        onMove(`${fromSquare}-${toSquare}`);
      } else if (piece && currentTurn === 'white') {
        // Select another white piece
        const whitePieces = ['♔', '♕', '♖', '♗', '♘', '♙'];
        if (whitePieces.includes(piece)) {
          setSelectedSquare([row, col]);
          const legalMoves = getLegalMoves(board, row, col, gameState);
          setPossibleMoves(legalMoves);
        }
      } else {
        setSelectedSquare(null);
        setPossibleMoves([]);
      }
    } else if (piece && currentTurn === 'white') {
      // Initial selection - only white pieces
      const whitePieces = ['♔', '♕', '♖', '♗', '♘', '♙'];
      if (whitePieces.includes(piece)) {
        setSelectedSquare([row, col]);
        const legalMoves = getLegalMoves(board, row, col, gameState);
        setPossibleMoves(legalMoves);
      }
    }
  };

  const getSquareClassName = (rowIndex, colIndex) => {
    const isLight = (rowIndex + colIndex) % 2 === 0;
    const isSelected = selectedSquare &&
      selectedSquare[0] === rowIndex &&
      selectedSquare[1] === colIndex;
    const isPossibleMove = possibleMoves.some(
      move => move[0] === rowIndex && move[1] === colIndex
    );

    // Highlight king if in check
    const piece = board[rowIndex][colIndex];
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
    <div className="board-wrapper">
      <div className="coordinates-left">
        {ranks.map((rank) => (
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
          {board.map((row, rowIndex) => (
            row.map((piece, colIndex) => {
              const { isLight, isSelected, isPossibleMove, isKingInCheck } =
                getSquareClassName(rowIndex, colIndex);

              return (
                <Square
                  key={`${rowIndex}-${colIndex}`}
                  piece={piece}
                  isLight={isLight}
                  isSelected={isSelected}
                  isPossibleMove={isPossibleMove}
                  isKingInCheck={isKingInCheck}
                  onClick={() => handleSquareClick(rowIndex, colIndex)}
                />
              );
            })
          ))}
        </div>

        <div className="coordinates-bottom">
          {files.map((file) => (
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
