import { useState, useCallback } from 'react';
import UltimateTicTacToeBoard from './components/UltimateTicTacToeBoard';
import './App.css';

function App() {
  const [difficulty, setDifficulty] = useState('Medium');
  const [moveHistory, setMoveHistory] = useState([]);
  const [gameStatus, setGameStatus] = useState('in_progress');
  const [currentPlayer, setCurrentPlayer] = useState('X');
  const [gameKey, setGameKey] = useState(0); // Used to reset the board

  const handleMove = useCallback((moveData) => {
    const { player, localBoardIdx, cellIdx, isAI } = moveData;

    // Add move to history
    setMoveHistory(prev => {
      const moveNumber = prev.length + 1;
      return [...prev, {
        number: moveNumber,
        player,
        localBoardIdx,
        cellIdx,
        description: `${player} played at Board ${localBoardIdx + 1}, Cell ${cellIdx + 1}`,
        isAI
      }];
    });

    // Update current player
    setCurrentPlayer(player === 'X' ? 'O' : 'X');
  }, []);

  const handleGameStatusChange = useCallback((status) => {
    setGameStatus(status);
  }, []);

  const handleNewGame = useCallback(() => {
    setMoveHistory([]);
    setGameStatus('in_progress');
    setCurrentPlayer('X');
    setGameKey(prev => prev + 1); // Force board to reset
  }, []);

  const getStatusMessage = () => {
    switch (gameStatus) {
      case 'x_wins':
        return 'You win!';
      case 'o_wins':
        return 'AI wins!';
      case 'draw':
        return "It's a draw!";
      default:
        return currentPlayer === 'X' ? "Your turn (X)" : "AI's turn (O)";
    }
  };

  const getStatusClass = () => {
    switch (gameStatus) {
      case 'x_wins':
        return 'status-win';
      case 'o_wins':
        return 'status-lose';
      case 'draw':
        return 'status-draw';
      default:
        return currentPlayer === 'X' ? 'status-x-turn' : 'status-o-turn';
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Ultimate Tic-Tac-Toe</h1>
        <p className="subtitle">Play against MCTS AI</p>
      </header>

      <div className="app-container">
        <div className="left-panel">
          <div className="player-info">
            <h3>You</h3>
            <div className="player-symbol player-x">X</div>
            <p>Human Player</p>
          </div>

          <div className="difficulty-selector">
            <label htmlFor="difficulty">AI Difficulty:</label>
            <select
              id="difficulty"
              value={difficulty}
              onChange={(e) => setDifficulty(e.target.value)}
              disabled={gameStatus !== 'in_progress' && moveHistory.length > 0}
            >
              <option value="Easy">Easy (1000 iterations)</option>
              <option value="Medium">Medium (5000 iterations)</option>
              <option value="Hard">Hard (10000 iterations)</option>
            </select>
          </div>

          <button className="new-game-button" onClick={handleNewGame}>
            New Game
          </button>
        </div>

        <div className="center-panel">
          <div className={`game-status ${getStatusClass()}`}>
            {getStatusMessage()}
          </div>

          <UltimateTicTacToeBoard
            key={gameKey}
            onMove={handleMove}
            onGameStatusChange={handleGameStatusChange}
            currentPlayer={currentPlayer}
            difficulty={difficulty}
            gameStarted={true}
          />

          <div className="game-rules">
            <h4>Rules:</h4>
            <ul>
              <li><strong>X goes first</strong> and may move to any of the 81 cells.</li>
              <li>The <strong>cell you pick</strong> decides which small board O must play on next (e.g. upper-left cell → upper-left board).</li>
              <li>When a small board is won, it’s marked with X or O and no more moves there.</li>
              <li>If you’re sent to a board that’s already won or full, you may choose <strong>any</strong> open board.</li>
              <li>Win 3 small boards in a row on the large board to win; otherwise draw when no winner is possible.</li>
            </ul>
          </div>
        </div>

        <div className="right-panel">
          <div className="player-info">
            <h3>AI</h3>
            <div className="player-symbol player-o">O</div>
            <p>MCTS Engine</p>
          </div>

          <div className="move-history">
            <h3>Move History</h3>
            <div className="move-history-list">
              {moveHistory.length === 0 ? (
                <p className="no-moves">No moves yet</p>
              ) : (
                moveHistory.map((move) => (
                  <div
                    key={move.number}
                    className={`move-item ${move.isAI ? 'move-ai' : 'move-human'}`}
                  >
                    <span className="move-number">#{move.number}</span>
                    <span className={`move-player move-player-${move.player.toLowerCase()}`}>
                      {move.player}
                    </span>
                    <span className="move-description">
                      Board {move.localBoardIdx + 1}, Cell {move.cellIdx + 1}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
