import { useState } from 'react';
import ChessBoard from './ChessBoard';

/**
 * Wraps ChessBoard with its own game state.
 * Receives isActive, humanColor from parent - only active board is interactive.
 */
const BoardWithState = ({ boardId, compact = false, isActive, humanColor, onTurnPairComplete }) => {
  const [moveHistory, setMoveHistory] = useState([]);
  const [currentTurn, setCurrentTurn] = useState('white');
  const [gameStatus, setGameStatus] = useState('in_progress');

  // Debug logging
  const aiColor = humanColor === 'white' ? 'black' : 'white';
  console.log(`Board ${boardId}: isActive=${isActive}, currentTurn=${currentTurn}, humanColor=${humanColor}, aiColor=${aiColor}, moveCount=${moveHistory.length}`);

  const addMove = (move, wasBlackMove) => {
    const movingColor = wasBlackMove ? 'black' : 'white';
    console.log(`Board ${boardId}: Move made by ${movingColor}: ${move}`);

    setMoveHistory((prev) => [...prev, move]);
    setCurrentTurn((t) => (t === 'white' ? 'black' : 'white'));

    if (wasBlackMove) {
      console.log(`Board ${boardId}: Turn pair complete! Notifying parent.`);
      onTurnPairComplete?.(boardId, move, gameStatus);
    }
  };

  const handleGameStatusChange = (status) => {
    setGameStatus(status);
    // Immediately report checkmate or stalemate to parent
    if (['white_wins', 'black_wins', 'stalemate'].includes(status)) {
      onTurnPairComplete?.(boardId, `game_over:${status}`, status);
    }
  };

  const isHumanTurn = currentTurn === humanColor;

  return (
    <div className="board-with-state">
      {isActive && (
        <div className={`turn-indicator-banner ${isHumanTurn ? 'human-turn' : 'ai-turn'}`}>
          {isHumanTurn ? `Your turn (${currentTurn})` : `AI thinking (${currentTurn})...`}
        </div>
      )}
      <ChessBoard
        onMove={addMove}
        currentTurn={currentTurn}
        onGameStatusChange={handleGameStatusChange}
        compact={compact}
        isActive={isActive}
        humanColor={humanColor}
      />
      {moveHistory.length > 0 && (
        <div className="board-moves-preview">
          {moveHistory.slice(-3).join(' ')}
        </div>
      )}
    </div>
  );
};

export default BoardWithState;
