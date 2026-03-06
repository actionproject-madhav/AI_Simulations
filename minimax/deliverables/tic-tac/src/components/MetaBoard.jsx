import LocalBoard from './LocalBoard';
import './MetaBoard.css';

/**
 * MetaBoard Component - Renders 9 LocalBoards in a 3x3 grid
 * @param {Array} board - The full 9x9 board (array of 9 local boards)
 * @param {Array} metaBoard - Meta-board state (9 positions)
 * @param {Array} activeBoards - Array of active board indices
 * @param {function} onCellClick - Handler for cell clicks
 */
export default function MetaBoard({ board, metaBoard, activeBoards, onCellClick }) {
  return (
    <div className="meta-board">
      {board.map((localCells, boardIdx) => (
        <LocalBoard
          key={boardIdx}
          cells={localCells}
          wonBy={metaBoard[boardIdx]}
          isActive={activeBoards.includes(boardIdx)}
          boardIndex={boardIdx}
          onCellClick={onCellClick}
        />
      ))}
    </div>
  );
}
