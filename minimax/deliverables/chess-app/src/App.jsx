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

  // Best-of-3 tracking: [board1Winner, board2Winner, board3Winner] - null, 'white', 'black', or 'draw'
  const [boardWinners, setBoardWinners] = useState([null, null, null])
  const [whiteWins, setWhiteWins] = useState(0)
  const [blackWins, setBlackWins] = useState(0)

  const pickRandomBoardAndSide = useCallback(() => {
    // Find unfinished boards
    const unfinishedBoards = boardWinners
      .map((winner, idx) => (winner === null ? idx : null))
      .filter((idx) => idx !== null)

    // If no unfinished boards, don't randomize (game should be over)
    if (unfinishedBoards.length === 0) {
      console.log('No unfinished boards remaining')
      return
    }

    // Pick random unfinished board
    const newBoard = unfinishedBoards[Math.floor(Math.random() * unfinishedBoards.length)]
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
    setTimeout(() => setShowRandomizationNotice(false), 3000)
  }, [boardWinners])

  const handleTurnPairComplete = useCallback((boardId, move, gameStatus) => {
    console.log(`App: Turn pair complete on Board ${boardId}`)
    setMoveHistory((prev) => [...prev, { boardId, move }])

    const boardIndex = boardId - 1 // Convert 1-based to 0-based

    // Check if this board just finished (checkmate or stalemate)
    if (gameStatus === 'white_wins' || gameStatus === 'black_wins' || gameStatus === 'stalemate') {
      const boardWinner = gameStatus === 'white_wins' ? 'white' : gameStatus === 'black_wins' ? 'black' : 'draw'

      // Only update if this board hasn't been won yet
      setBoardWinners((prev) => {
        if (prev[boardIndex] !== null) {
          // Board already finished, don't update
          return prev
        }

        const newWinners = [...prev]
        newWinners[boardIndex] = boardWinner
        console.log(`Board ${boardId} finished: ${boardWinner}`)

        return newWinners
      })

      // Update win counters
      if (gameStatus === 'white_wins') {
        setWhiteWins((prev) => {
          const newWhiteWins = prev + 1
          console.log(`White wins: ${newWhiteWins}`)

          // Check if white got 2 wins (game over)
          if (newWhiteWins >= 2) {
            const didHumanWin = humanColor === 'white'
            console.log(`GAME OVER: White wins best-of-3! Winner: ${didHumanWin ? 'HUMAN' : 'AI'}`)
            setTimeout(() => {
              setGameOver(true)
              setWinner(didHumanWin ? 'human' : 'ai')
            }, 2000) // Give time to see the final board state
          }

          return newWhiteWins
        })
      } else if (gameStatus === 'black_wins') {
        setBlackWins((prev) => {
          const newBlackWins = prev + 1
          console.log(`Black wins: ${newBlackWins}`)

          // Check if black got 2 wins (game over)
          if (newBlackWins >= 2) {
            const didHumanWin = humanColor === 'black'
            console.log(`GAME OVER: Black wins best-of-3! Winner: ${didHumanWin ? 'HUMAN' : 'AI'}`)
            setTimeout(() => {
              setGameOver(true)
              setWinner(didHumanWin ? 'human' : 'ai')
            }, 2000) // Give time to see the final board state
          }

          return newBlackWins
        })
      }
      // Note: draws don't count as wins for either side
    }

    // Show turn pair complete message
    setTurnPairComplete(true)
    console.log(`Pausing for 2 seconds before randomization...`)

    // Add pause before switching boards (2 seconds for smoother experience)
    setTimeout(() => {
      setTurnPairComplete(false)
      pickRandomBoardAndSide()
    }, 2000)
  }, [pickRandomBoardAndSide, humanColor])

  const handleResetGame = useCallback(() => {
    setActiveBoardIndex(pickRandomBoard())
    setHumanColor(pickRandomSide())
    setMoveHistory([])
    setGameOver(false)
    setWinner(null)
    setBoardWinners([null, null, null])
    setWhiteWins(0)
    setBlackWins(0)
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
                    <span className="info-label">Score (Best of 3):</span>
                    <span className="info-value">
                      <span className="score-display">
                        White: {whiteWins} | Black: {blackWins}
                      </span>
                    </span>
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
                  {boardWinners[idx] && (
                    <span className={`winner-badge winner-${boardWinners[idx]}`}>
                      {boardWinners[idx] === 'draw' ? 'DRAW' : `${boardWinners[idx].toUpperCase()} WON`}
                    </span>
                  )}
                  {idx === activeBoardIndex && !gameOver && !boardWinners[idx] && (
                    <span className="active-badge">
                      ACTIVE - You are {humanColor}
                    </span>
                  )}
                </div>
                <BoardWithState
                  key={`${gameKey}-${idx}`}
                  boardId={idx + 1}
                  compact={true}
                  isActive={idx === activeBoardIndex && !gameOver && !turnPairComplete && boardWinners[idx] === null}
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
