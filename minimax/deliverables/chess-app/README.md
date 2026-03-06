# Chess.com Style Chess Board

A beautiful, interactive chess board built with React and Vite, designed to match the aesthetics of Chess.com.

## Features

- **Chess.com-inspired Design**: Classic brown/tan color scheme (#EEEED2 light squares, #769656 dark squares)
- **Interactive Board**: Click to select pieces and see available moves
- **Move Highlighting**: Visual indicators for possible moves and captures
- **Player Info Display**: Shows player names, ratings, and turn indicators
- **Move History**: Track all moves made during the game
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Unicode Chess Pieces**: Beautiful, crisp chess pieces that scale perfectly

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn

### Installation

1. Navigate to the project directory:
```bash
cd minimax/deliverables/chess-app
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser and navigate to:
```
http://localhost:5173/
```

## How to Play

1. **Select a Piece**: Click on any piece to select it (only pieces of the current player's color can be selected)
2. **View Possible Moves**: After selecting a piece, all possible destination squares will be highlighted
3. **Make a Move**: Click on any highlighted square to move the piece there
4. **Deselect**: Click the selected piece again to deselect it
5. **Turn Indicator**: The green glowing dot shows whose turn it is
6. **Move History**: All moves are recorded in the sidebar

## Project Structure

```
chess-app/
├── src/
│   ├── components/
│   │   ├── ChessBoard.jsx       # Main board component with game logic
│   │   ├── ChessBoard.css       # Board styling
│   │   ├── Square.jsx           # Individual square component
│   │   └── Square.css           # Square styling
│   ├── App.jsx                  # Main app component
│   ├── App.css                  # App-level styling
│   ├── index.css                # Global styles
│   └── main.jsx                 # Entry point
├── public/
├── package.json
└── vite.config.js
```

## Design Details

### Color Scheme
- **Light Squares**: `#EEEED2` (beige)
- **Dark Squares**: `#769656` (olive green)
- **Selected Square**: `#BACA44` (yellow-green highlight)
- **Background**: Dark gradient (`#2c2a29` to `#1a1816`)
- **Turn Indicator**: `#81b64c` (green glow)

### Visual Elements
- Rounded corners on UI panels
- Glassmorphism effects (backdrop blur)
- Smooth transitions and hover effects
- Responsive grid layout
- Custom scrollbar styling

## Future Enhancements

Ready for integration with:
- **Chess Engine**: Stockfish or custom minimax algorithm
- **Move Validation**: Proper chess rules (castling, en passant, check, checkmate)
- **Game Modes**: Player vs Player, Player vs AI, AI vs AI
- **Time Controls**: Add chess clocks
- **Game Analysis**: Move evaluation and suggestions
- **PGN Export/Import**: Save and load games

## Building for Production

```bash
npm run build
```

The production-ready files will be in the `dist` folder, ready to deploy to Vercel or any static hosting service.

## Deployment to Vercel

1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Deploy:
```bash
vercel
```

Or connect your GitHub repository to Vercel for automatic deployments.

## Technologies Used

- **React 18**: UI framework
- **Vite**: Build tool and dev server
- **CSS3**: Styling with modern features (Grid, Flexbox, CSS Variables)
- **Unicode Characters**: Chess piece glyphs

## License

MIT

## Acknowledgments

- Design inspired by Chess.com
- Built with ❤️ for chess enthusiasts
