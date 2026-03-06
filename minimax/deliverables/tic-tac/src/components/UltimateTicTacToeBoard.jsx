import { useState, useEffect, useRef } from 'react';
import MetaBoard from './MetaBoard';
import {
  createInitialGameState,
  isValidMove,
  makeMove,
  getGameStatus
} from '../utils/gameEngine';
import './UltimateTicTacToeBoard.css';

/**
 * UltimateTicTacToeBoard Component - Main game board container
 * @param {function} onMove - Callback when a move is made
 * @param {function} onGameStatusChange - Callback when game status changes
 * @param {string} currentPlayer - Current player ('X' or 'O')
 * @param {string} difficulty - AI difficulty level
 * @param {boolean} gameStarted - Whether the game has started
 */
export default function UltimateTicTacToeBoard({
  onMove,
  onGameStatusChange,
  currentPlayer,
  difficulty,
  gameStarted
}) {
  const [gameState, setGameState] = useState(createInitialGameState());
  const [isAIThinking, setIsAIThinking] = useState(false);
  const gameKeyRef = useRef(gameStarted);

  // Reset game when gameStarted changes (using key comparison)
  useEffect(() => {
    if (gameStarted !== gameKeyRef.current) {
      gameKeyRef.current = gameStarted;
      const initialState = createInitialGameState();
      setGameState(initialState);
      setIsAIThinking(false);
      onGameStatusChange('in_progress');
    }
  }, [gameStarted]);

  // Check game status whenever gameState changes
  useEffect(() => {
    const status = getGameStatus(gameState.board, gameState.metaBoard);
    if (status !== 'in_progress') {
      onGameStatusChange(status);
    }
  }, [gameState.board, gameState.metaBoard]);

  // AI move logic
  useEffect(() => {
    const makeAIMove = async () => {
      if (gameState.currentPlayer === 'O' && !isAIThinking) {
        const status = getGameStatus(gameState.board, gameState.metaBoard);
        if (status === 'in_progress') {
          setIsAIThinking(true);

          // Dynamic import of AI engine to avoid loading it until needed
          const { getAIMove } = await import('../utils/mctsEngine');

          const aiMove = await getAIMove(
            gameState.board,
            gameState.metaBoard,
            gameState.activeBoards,
            difficulty
          );

          if (aiMove) {
            handleMove(aiMove.localBoardIdx, aiMove.cellIdx, true);
          }

          setIsAIThinking(false);
        }
      }
    };

    makeAIMove();
  }, [gameState.currentPlayer, difficulty]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleCellClick = (localBoardIdx, cellIdx) => {
    // Only allow human moves when it's X's turn and AI is not thinking
    if (gameState.currentPlayer !== 'X' || isAIThinking) return;

    const status = getGameStatus(gameState.board, gameState.metaBoard);
    if (status !== 'in_progress') return;

    handleMove(localBoardIdx, cellIdx, false);
  };

  const handleMove = (localBoardIdx, cellIdx, isAI) => {
    // Validate move
    if (!isValidMove(
      gameState.board,
      gameState.metaBoard,
      gameState.activeBoards,
      localBoardIdx,
      cellIdx
    )) {
      return;
    }

    // Make move
    const newState = makeMove(
      gameState.board,
      gameState.metaBoard,
      gameState.activeBoards,
      localBoardIdx,
      cellIdx,
      gameState.currentPlayer
    );

    setGameState(newState);

    // Notify parent of move
    onMove({
      player: gameState.currentPlayer,
      localBoardIdx,
      cellIdx,
      isAI
    });
  };

  return (
    <div className="ultimate-tictactoe-board">
      <MetaBoard
        board={gameState.board}
        metaBoard={gameState.metaBoard}
        activeBoards={gameState.activeBoards}
        onCellClick={handleCellClick}
      />
      {isAIThinking && (
        <div className="ai-thinking-overlay">
          <div className="ai-thinking-spinner"></div>
          <p>AI is thinking...</p>
        </div>
      )}
    </div>
  );
}
