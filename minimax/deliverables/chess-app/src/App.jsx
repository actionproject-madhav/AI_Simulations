import { useState, useCallback, useRef, useEffect } from 'react'
import BoardWithState from './components/BoardWithState'
import './App.css'

// Random selection: board 0-2, human as white or black (rules: could be same board, side chosen at random each turn pair)
const pickRandomBoard = () => Math.floor(Math.random() * 3)
const pickRandomSide = () => (Math.random() < 0.5 ? 'white' : 'black')

function App() {
  const [activeBoardIndex, setActiveBoardIndex] = useState(() => {
    const board = pickRandomBoard()
    console.log(`INITIAL: Starting with Board ${board + 1}`)
    return board
  })
  const [humanColor, setHumanColor] = useState(() => {
    const color = pickRandomSide()
    const aiColor = color === 'white' ? 'black' : 'white'
    console.log(`INITIAL: Human=${color}, AI=${aiColor}`)
    console.log(`INITIAL: White will move first (${color === 'white' ? 'HUMAN' : 'AI'})`)
    return color
  })
  const [moveHistory, setMoveHistory] = useState([])
  const [gameOver, setGameOver] = useState(false)
  const [winner, setWinner] = useState(null) // 'human' or 'ai' or 'draw'
  const [gameKey, setGameKey] = useState(0) // Force remount of boards on reset
  const [showRandomizationNotice, setShowRandomizationNotice] = useState(false)
  const [turnPairComplete, setTurnPairComplete] = useState(false)

  const pickRandomBoardAndSide = useCallback(() => {
    const newBoard = pickRandomBoard()
    const newColor = pickRandomSide()
    const aiColor = newColor === 'white' ? 'black' : 'white'

    console.log(`========================================`)
    console.log(`RANDOMIZATION: Board ${newBoard + 1}, Human=${newColor}, AI=${aiColor}`)
    console.log(`White will move first (${newColor === 'white' ? 'HUMAN' : 'AI'})`)
    console.log(`========================================`)

    setActiveBoardIndex(newBoard)
    setHumanColor(newColor)

    // Show notification
    setShowRandomizationNotice(true)
    setTimeout(() => setShowRandomizationNotice(false), 2500)
  }, [])

  const handleTurnPairComplete = useCallback((boardId, move, gameStatus) => {
    console.log(`App: Turn pair complete on Board ${boardId}`)
    setMoveHistory((prev) => [...prev, { boardId, move }])

    // Check if checkmate occurred - ends entire game
    if (gameStatus === 'white_wins' || gameStatus === 'black_wins') {
      const winningColor = gameStatus === 'white_wins' ? 'white' : 'black'
      const didHumanWin = winningColor === humanColor
      console.log(`GAME OVER: ${winningColor} wins! Winner: ${didHumanWin ? 'HUMAN' : 'AI'}`)
      setGameOver(true)
      setWinner(didHumanWin ? 'human' : 'ai')
      return
    }

    // Check for stalemate
    if (gameStatus === 'stalemate') {
      console.log(`GAME OVER: Stalemate`)
      setGameOver(true)
      setWinner('draw')
      return
    }

    // Show turn pair complete message
    setTurnPairComplete(true)
    console.log(`Pausing for 1.5 seconds before randomization...`)

    // Add pause before switching boards (1.5 seconds)
    setTimeout(() => {
      setTurnPairComplete(false)
      pickRandomBoardAndSide()
    }, 1500)
  }, [pickRandomBoardAndSide, humanColor])

  const handleResetGame = useCallback(() => {
    setActiveBoardIndex(pickRandomBoard())
    setHumanColor(pickRandomSide())
    setMoveHistory([])
    setGameOver(false)
    setWinner(null)
    setGameKey((k) => k + 1) // Force remount of all boards
  }, [])

  const aiColor = humanColor === 'white' ? 'black' : 'white'

  return (
    <div className="app">
      {turnPairComplete && (
        <div className="turn-pair-complete-notice">
          ✓ Turn pair complete on Board {activeBoardIndex + 1}
        </div>
      )}
      {showRandomizationNotice && (
        <div className="randomization-notice">
          🎲 Switching to Board {activeBoardIndex + 1} — You are now playing {humanColor}
        </div>
      )}
      <div className="app-inner">
        <div className="game-container">
          <div className="boards-header">
            <h2>Three Boards Chess</h2>
            {gameOver ? (
              <>
                <div className="game-over-banner">
                  {winner === 'human' && '🎉 YOU WIN! 🎉'}
                  {winner === 'ai' && '💀 AI WINS 💀'}
                  {winner === 'draw' && '🤝 STALEMATE - DRAW 🤝'}
                </div>
                <button className="reset-button" onClick={handleResetGame}>
                  New Game
                </button>
              </>
            ) : (
              <>
                <div className="game-info-panel">
                  <div className="info-row">
                    <span className="info-label">Active Board:</span>
                    <span className="info-value">Board {activeBoardIndex + 1}</span>
                  </div>
                  <div className="info-row">
                    <span className="info-label">You are:</span>
                    <span className={`info-value color-${humanColor}`}>{humanColor.toUpperCase()}</span>
                  </div>
                  <div className="info-row">
                    <span className="info-label">AI is:</span>
                    <span className={`info-value color-${aiColor}`}>{aiColor.toUpperCase()}</span>
                  </div>
                  <div className="info-row">
                    <span className="info-label">Next to move:</span>
                    <span className="info-value">WHITE (always goes first)</span>
                  </div>
                </div>
              </>
            )}
          </div>

          <div className="boards-row">
            {[0, 1, 2].map((idx) => (
              <div
                key={idx}
                className={`board-card ${idx === activeBoardIndex ? 'active' : 'blurred'} ${gameOver ? 'game-over' : ''} ${turnPairComplete && idx === activeBoardIndex ? 'transitioning' : ''}`}
              >
                <div className="board-label">
                  Board {idx + 1}
                  {idx === activeBoardIndex && !gameOver && (
                    <span className="active-badge">
                      ACTIVE - You are {humanColor}
                    </span>
                  )}
                </div>
                <BoardWithState
                  key={`${gameKey}-${idx}`}
                  boardId={idx + 1}
                  compact={true}
                  isActive={idx === activeBoardIndex && !gameOver && !turnPairComplete}
                  humanColor={humanColor}
                  onTurnPairComplete={handleTurnPairComplete}
                />
              </div>
            ))}
          </div>
        </div>

        <div className="sidebar">
          <div className="move-history">
            <h3>Moves</h3>
            <div className="moves-list">
              {moveHistory.length === 0 ? (
                <p className="no-moves">No moves yet</p>
              ) : (
                moveHistory.map((entry, index) => (
                  <div key={index} className="move-item">
                    <span className="move-number">{index + 1}.</span>
                    <span className="move-notation">
                      B{entry.boardId}: {entry.move}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
