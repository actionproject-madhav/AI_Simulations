import { useState } from 'react'
import ChessBoard from './components/ChessBoard'
import './App.css'

function App() {
  const [moveHistory, setMoveHistory] = useState([]);
  const [currentTurn, setCurrentTurn] = useState('white');
  const [gameStatus, setGameStatus] = useState('in_progress');

  const addMove = (move) => {
    setMoveHistory([...moveHistory, move]);
    setCurrentTurn(currentTurn === 'white' ? 'black' : 'white');
  };

  const handleGameStatusChange = (status) => {
    setGameStatus(status);
  };

  const getStatusMessage = () => {
    switch (gameStatus) {
      case 'check':
        return `${currentTurn === 'white' ? 'White' : 'Black'} is in check!`;
      case 'white_wins':
        return 'Checkmate! White wins!';
      case 'black_wins':
        return 'Checkmate! Black (AI) wins!';
      case 'stalemate':
        return 'Stalemate! Game is a draw.';
      default:
        return null;
    }
  };

  const statusMessage = getStatusMessage();

  return (
    <div className="app">
      <div className="game-container">
        <div className="player-info black-player">
          <div className="player-avatar">♚</div>
          <div className="player-details">
            <div className="player-name">AI (Black)</div>
            <div className="player-rating">Medium</div>
          </div>
          <div className={`turn-indicator ${currentTurn === 'black' ? 'active' : ''}`}></div>
        </div>

        {statusMessage && (
          <div className={`game-status ${gameStatus}`}>
            {statusMessage}
          </div>
        )}

        <ChessBoard
          onMove={addMove}
          currentTurn={currentTurn}
          onGameStatusChange={handleGameStatusChange}
        />

        <div className="player-info white-player">
          <div className="player-avatar">♔</div>
          <div className="player-details">
            <div className="player-name">You (White)</div>
            <div className="player-rating">Human</div>
          </div>
          <div className={`turn-indicator ${currentTurn === 'white' ? 'active' : ''}`}></div>
        </div>
      </div>

      <div className="sidebar">
        <div className="move-history">
          <h3>Moves</h3>
          <div className="moves-list">
            {moveHistory.length === 0 ? (
              <p className="no-moves">No moves yet</p>
            ) : (
              moveHistory.map((move, index) => (
                <div key={index} className="move-item">
                  <span className="move-number">{Math.floor(index / 2) + 1}.</span>
                  <span className="move-notation">{move}</span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
