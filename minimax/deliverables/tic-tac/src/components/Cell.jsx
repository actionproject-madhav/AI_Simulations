import './Cell.css';

/**
 * Cell Component - Represents a single cell in a 3x3 local board
 * @param {string} value - '', 'X', or 'O'
 * @param {function} onClick - Click handler
 * @param {boolean} disabled - Whether cell is clickable
 */
export default function Cell({ value, onClick, disabled }) {
  const handleClick = () => {
    if (!disabled && !value) {
      onClick();
    }
  };

  const isClickable = !disabled && !value;

  return (
    <div
      className={`cell ${value ? `cell-${value.toLowerCase()}` : ''} ${isClickable ? 'cell-clickable' : ''} ${disabled ? 'cell-disabled' : ''}`}
      onClick={handleClick}
    >
      {value && <span className="cell-value">{value}</span>}
    </div>
  );
}
