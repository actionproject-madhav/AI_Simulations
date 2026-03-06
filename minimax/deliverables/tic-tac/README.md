# Ultimate Tic-Tac-Toe

A web-based implementation of Ultimate Tic-Tac-Toe with an AI opponent powered by Monte Carlo Tree Search (MCTS).

![Ultimate Tic-Tac-Toe](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/Super_tic-tac-toe_rules_example.png/500px-Super_tic-tac-toe_rules_example.png)

## Game Rules

Ultimate Tic-Tac-Toe is played on a 3×3 grid of 3×3 tic-tac-toe boards. The goal is to win three small boards in a row (horizontally, vertically, or diagonally) to win the overall game.

### How to Play

1. **X (You) goes first** and may move to any of the 81 individual board positions.

2. **Your move determines where the AI plays next**: The position you choose within a small board determines which board the AI must play on next. For example, if you play in the upper-left cell of a small board, the AI must play on the upper-left board of the large grid.

3. **This pattern continues**: Each move determines the next board to be played on.

4. **Winning small boards**: When you win a small board, it's marked with your symbol (X or O) and cannot be played on anymore.

5. **Board already won or full**: If you would be sent to a board that's already been won or is full, you may choose any available board for your next move.

6. **Winning the game**: Win three small boards in a row to win the overall game. If the large board fills up with no winner, the game is a draw.

## Features

### AI Opponent (MCTS)

The AI uses Monte Carlo Tree Search with UCB1 tree policy:

- **UCB1 Selection**: Balances exploration and exploitation when traversing the game tree
- **Random Rollouts**: Simulates games to terminal states using random moves
- **Robust Move Selection**: Chooses the most-visited child node for reliability

### Difficulty Levels

- **Easy** (1000 iterations): Quick AI responses, suitable for learning the game
- **Medium** (5000 iterations): Balanced difficulty with strategic play
- **Hard** (10000 iterations): Strong AI opponent with deep analysis

### User Interface

- **Move History**: Track all moves made during the game
- **Active Board Highlighting**: Orange border indicates which board(s) can be played
- **Won Board Indicators**: Large X or O displayed on completed small boards
- **AI Thinking Indicator**: Loading animation while AI computes its move
- **Game Status**: Clear indication of whose turn it is and game outcome
- **Responsive Design**: Works on desktop, tablet, and mobile devices

### Color Scheme

- **X (Human)**: Red (#e74c3c)
- **O (AI)**: Blue (#3498db)
- **Active Board**: Orange (#f39c12)
- **Dark Theme**: Professional gradient background (#2c2a29 to #1a1816)

## Technology Stack

- **React 19.2.0** - Modern UI framework
- **Vite 7.3.1** - Fast build tool and dev server
- **Vanilla JavaScript** - No external libraries for game logic
- **CSS3** - Modern styling with animations and gradients

## Installation & Setup

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Install Dependencies

```bash
npm install
```

### Run Development Server

```bash
npm run dev
```

The game will be available at `http://localhost:5173/` (or another port if 5173 is in use).

### Build for Production

```bash
npm run build
```

The production build will be in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Project Structure

```
tic-tac/
├── src/
│   ├── components/
│   │   ├── Cell.jsx                    # Individual cell component
│   │   ├── Cell.css
│   │   ├── LocalBoard.jsx              # 3×3 board component
│   │   ├── LocalBoard.css
│   │   ├── MetaBoard.jsx               # 3×3 grid of local boards
│   │   ├── MetaBoard.css
│   │   ├── UltimateTicTacToeBoard.jsx  # Main game board
│   │   └── UltimateTicTacToeBoard.css
│   ├── utils/
│   │   ├── gameEngine.js               # Game logic and rules
│   │   └── mctsEngine.js               # MCTS AI implementation
│   ├── App.jsx                         # Root component
│   ├── App.css                         # App-level styles
│   ├── main.jsx                        # Entry point
│   └── index.css                       # Global styles
├── public/
├── index.html
├── package.json
└── vite.config.js
```

## Game Engine (`gameEngine.js`)

### Core Functions

- `createInitialGameState()` - Creates empty game state
- `isValidMove()` - Validates move legality
- `makeMove()` - Executes move and returns new state (immutable)
- `checkLocalBoardWinner()` - Detects wins in 3×3 boards
- `checkMetaBoardWinner()` - Detects overall game winner
- `getGameStatus()` - Returns current game status
- `getAllLegalMoves()` - Lists all valid moves

## MCTS Algorithm (`mctsEngine.js`)

### Implementation Details

**Node Structure:**
- State (board, metaBoard, activeBoards, currentPlayer)
- Parent and children nodes
- Visit count and win statistics
- Untried moves list

**Algorithm Phases:**

1. **Selection**: Traverse tree using UCB1 formula until reaching leaf
   - UCB1 = (wins/visits) + C × √(ln(parent.visits) / visits)
   - Exploration constant C = √2

2. **Expansion**: Add one new child node with random untried move

3. **Simulation**: Random rollout to terminal state
   - +1 for AI win
   - -1 for human win
   - 0 for draw

4. **Backpropagation**: Update visit counts and wins up the tree

5. **Best Move Selection**: Choose child with most visits (most robust)

### Performance

- **Easy**: ~0.1-0.5s per move
- **Medium**: ~0.5-2s per move
- **Hard**: ~1-5s per move

## Development

### Code Style

- React functional components with hooks
- Immutable state updates
- Pure functions for game logic
- CSS modules for component-scoped styling

### Testing Checklist

- [ ] Valid moves only
- [ ] Active board highlighting updates correctly
- [ ] Won boards show overlays
- [ ] Meta-board win detection works
- [ ] AI makes legal moves
- [ ] AI blocks obvious wins
- [ ] AI takes winning moves when available
- [ ] Sending to won/full board activates all available boards
- [ ] Move history displays correctly
- [ ] New Game resets properly
- [ ] Colors match specification (Red X, Blue O)
- [ ] Responsive design works on mobile

## Strategic Tips

1. **Control the flow**: Your move determines where your opponent plays next, so plan ahead.

2. **Force bad positions**: Sometimes it's worth making a suboptimal local move to force your opponent onto a disadvantageous board.

3. **Sacrifice strategically**: You might lose a small board to gain better positioning on the meta-board.

4. **Corner and center**: Like regular tic-tac-toe, corners and center are generally stronger positions.

5. **Think two moves ahead**: Always consider where your opponent will send you after their move.

## License

MIT License - Feel free to use and modify for your own projects.

## Credits

Game concept: Ben Orlin (popularized the game among computer science educators)

Implementation: Ultimate Tic-Tac-Toe with MCTS AI

## Future Enhancements

Potential improvements:
- Player vs Player mode
- Move undo/redo
- Game replay functionality
- Statistics tracking (wins/losses/draws)
- Sound effects
- Animated transitions
- AI analysis visualization (show MCTS tree)
- Different board themes
- Save/load game state

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
