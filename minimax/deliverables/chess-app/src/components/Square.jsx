import './Square.css';

const Square = ({ piece, isLight, isSelected, isPossibleMove, isKingInCheck, onClick }) => {
  const squareClass = `square ${isLight ? 'light' : 'dark'} ${isSelected ? 'selected' : ''} ${isPossibleMove ? 'possible-move' : ''} ${isKingInCheck ? 'in-check' : ''}`;

  // Map Unicode pieces to image filenames
  const getPieceImage = (piece) => {
    const pieceMap = {
      '♔': 'wk', // white king
      '♕': 'wq', // white queen
      '♖': 'wr', // white rook
      '♗': 'wb', // white bishop
      '♘': 'wn', // white knight
      '♙': 'wp', // white pawn
      '♚': 'bk', // black king
      '♛': 'bq', // black queen
      '♜': 'br', // black rook
      '♝': 'bb', // black bishop
      '♞': 'bn', // black knight
      '♟': 'bp', // black pawn
    };

    return pieceMap[piece] ? `/${pieceMap[piece]}.png` : null;
  };

  const pieceImage = piece ? getPieceImage(piece) : null;

  return (
    <div className={squareClass} onClick={onClick}>
      {pieceImage && (
        <img
          src={pieceImage}
          alt={piece}
          className="piece-image"
          draggable="false"
        />
      )}
      {isPossibleMove && !piece && <div className="move-dot"></div>}
      {isPossibleMove && piece && <div className="capture-ring"></div>}
    </div>
  );
};

export default Square;
