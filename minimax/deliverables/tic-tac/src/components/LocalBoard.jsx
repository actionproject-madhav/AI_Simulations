import Cell from './Cell';
import './LocalBoard.css';

/**
 * LocalBoard Component - Represents a single 3x3 tic-tac-toe board
 * @param {Array} cells - Array of 9 cells ('', 'X', or 'O')
 * @param {string} wonBy - 'X', 'O', 'D' (draw), or ''
 * @param {boolean} isActive - Whether this board can be played on
 * @param {number} boardIndex - Index of this board (0-8)
 * @param {function} onCellClick - Handler for cell clicks
 */
export default function LocalBoard({ cells, wonBy, isActive, boardIndex, onCellClick }) {
  const handleCellClick = (cellIdx) => {
    if (isActive && !wonBy) {
      onCellClick(boardIndex, cellIdx);
    }
  };

  return (
    <div className={`local-board ${isActive ? 'local-board-active' : ''} ${wonBy ? 'local-board-won' : ''}`}>
      <div className="local-board-grid">
        {cells.map((cell, idx) => (
          <Cell
            key={idx}
            value={cell}
            onClick={() => handleCellClick(idx)}
            disabled={!isActive || !!wonBy}
          />
        ))}
      </div>
      {wonBy && wonBy !== 'D' && (
        <div className={`local-board-overlay local-board-overlay-${wonBy.toLowerCase()}`}>
          <span className="local-board-winner">{wonBy}</span>
        </div>
      )}
      {wonBy === 'D' && (
        <div className="local-board-overlay local-board-overlay-draw">
          <span className="local-board-winner">DRAW</span>
        </div>
      )}
    </div>
  );
}
