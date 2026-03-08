# 3-Board Chess Simulation System

## Overview

This simulation system runs headless games of 3-Board Chess to discover optimal strategies through self-play and data analysis.

## Quick Start

### Run a Quick Test (10 games)
```bash
npm run simulate:quick
```

### Run Specific Matchup (100 games)
```bash
node run-simulation.js --games 100 --strategy1 TrapBuilder --strategy2 Aggressive
```

### Run Full Tournament (1,250 games)
```bash
npm run simulate:tournament
```

## Available Strategies

### 1. **RandomStrategy** (Baseline)
- Picks random legal moves
- Used as baseline to measure improvement
- Expected to lose against all other strategies

### 2. **MinimaxStrategy** (Traditional Chess AI)
- Standard minimax with alpha-beta pruning
- Depth-3 search by default
- Uses traditional chess evaluation function
- **Hypothesis**: Suboptimal for this variant (doesn't account for randomization)

### 3. **AggressiveStrategy** (Attack-Focused)
- Prioritizes checks and checkmate threats
- Values piece proximity to opponent's king
- Willing to sacrifice material for attacks
- **Hypothesis**: Good but may overextend

### 4. **DefensiveStrategy** (Safety-Focused)
- Prioritizes king safety and material
- Avoids risky positions
- Traditional chess values
- **Hypothesis**: Too conservative for this variant

### 5. **TrapBuilderStrategy** (Hypothesized Optimal)
- **Core idea**: Create checkmate-in-1 threats across multiple boards
- Maintains 2+ "trap boards" where checkmate is imminent
- Sacrifices material for checkmate threats
- **Hypothesis**: This is the optimal meta-strategy for 3-Board Chess

## Why TrapBuilder Should Win

### The Math:
- **Random board selection**: 1/3 chance of any board being selected
- **Random color assignment**: 50% chance of being each color
- **If you have 2 trap boards**: 2/3 chance of trap board being selected
- **If trap board + you go first**: Instant win

### The Strategy:
1. Create checkmate-in-1 position on Board A
2. Create checkmate-in-1 position on Board B
3. Now you have 2/3 probability of winning on next turn pair (if you get the right color)
4. Even better: Create positions where BOTH colors have checkmate-in-1
5. Result: ~67% win probability per turn

## Running Simulations

### Basic Commands

```bash
# Quick test (10 games)
npm run simulate:quick

# Specific matchup
node run-simulation.js --strategy1 TrapBuilder --strategy2 Minimax --games 100

# Full tournament (all vs all)
npm run simulate:tournament

# Verbose mode (see game details)
node run-simulation.js --quick --verbose
```

### Advanced Options

```bash
# Adjust AI depth (higher = smarter but slower)
node run-simulation.js --depth 4 --games 50 --strategy1 TrapBuilder --strategy2 Aggressive

# Change max turn pairs (prevent infinite games)
node run-simulation.js --maxTurnPairs 50

# Help
node run-simulation.js --help
```

## Understanding Results

### Output Files
Results are saved to `./simulation-results/results-TIMESTAMP.json`

### Key Metrics

**Win Rate**: Percentage of games won
- **>60%**: Dominant strategy
- **50-60%**: Strong strategy
- **40-50%**: Competitive
- **<40%**: Weak strategy

**Avg Turn Pairs**: How long games last
- **Lower**: More aggressive, decisive play
- **Higher**: More defensive, drawn-out play

## Expected Results

Based on the game theory:

### Predicted Leaderboard:
1. **TrapBuilderStrategy** (55-65% win rate)
   - Creates the unstable multi-board traps
   - Leverages the randomization mechanic

2. **AggressiveStrategy** (45-55% win rate)
   - Similar goals but less systematic
   - Doesn't maintain multiple trap boards

3. **MinimaxStrategy** (40-50% win rate)
   - Good at single-board tactics
   - Doesn't optimize for the meta-game

4. **DefensiveStrategy** (30-40% win rate)
   - Too conservative
   - Doesn't create winning chances across multiple boards

5. **RandomStrategy** (<10% win rate)
   - Baseline

## Research Questions Answered

### 1. Is "Trap Builder" actually optimal?
- **Method**: Compare win rate vs other strategies
- **Expected**: TrapBuilder > 55% vs all opponents

### 2. How many trap boards is optimal?
- **Method**: Analyze position data when games end
- **Expected**: 2 trap boards = sweet spot

### 3. Is material worth less in this variant?
- **Method**: Compare material-based strategies (Defensive, Minimax) vs threat-based (Aggressive, TrapBuilder)
- **Expected**: Threat-based strategies win

### 4. How long do games last?
- **Method**: Average turn pairs
- **Expected**: 15-30 turn pairs (aggressive meta)

## Analyzing Results

### After running simulations, check:

1. **Leaderboard**: Which strategy won the most?
```bash
cat simulation-results/results-*.json | grep -A 20 "byStrategy"
```

2. **Win Rates**: Are they statistically significant?
- 100 games: ~10% margin of error
- 500 games: ~4% margin of error
- 1000 games: ~3% margin of error

3. **Patterns**: Look at actual game logs
```bash
node run-simulation.js --verbose --quick > game-logs.txt
```

## Next Steps After Simulation

### If TrapBuilder wins:
1. ✅ Hypothesis confirmed
2. Implement TrapBuilder in the actual game UI
3. Add difficulty levels (depth-2, depth-3, depth-4)
4. Show trap indicators to help human players learn

### If Aggressive wins:
- Refine TrapBuilder algorithm
- Maybe maintaining 1 trap board is better than 2?
- Adjust weights in evaluation function

### If Minimax wins:
- Traditional chess values still matter
- The randomization doesn't dominate strategy as much as thought
- Hybrid approach needed

## Performance Notes

### Estimated Runtimes (on M1 Mac):

- **10 games**: ~30 seconds
- **100 games**: ~5 minutes
- **500 games**: ~25 minutes
- **Full tournament (1,250 games)**: ~1-2 hours

*Note: Depth-4 will be ~3x slower than depth-3*

## Troubleshooting

### Out of Memory
```bash
# Reduce game count or depth
node run-simulation.js --games 50 --depth 2
```

### Too Slow
```bash
# Use depth-2 for faster simulations
node run-simulation.js --quick --depth 2
```

### Want More Detail
```bash
# Enable verbose logging
node run-simulation.js --verbose --games 10
```

## Files

- `src/simulation/gameSimulator.js` - Core game engine (headless)
- `src/simulation/strategies.js` - All strategy implementations
- `src/simulation/batchRunner.js` - Runs multiple games and collects stats
- `run-simulation.js` - CLI script
- `simulation-results/` - Output directory for results

## Contributing New Strategies

To add a new strategy:

1. Create new class in `src/simulation/strategies.js`:
```javascript
export class MyStrategy extends BaseStrategy {
  constructor() {
    super('MyStrategy');
  }

  async selectMove(board, color, gameState, allBoards, activeBoardIndex) {
    // Your strategy logic here
    return { from: [r1, c1], to: [r2, c2] };
  }
}
```

2. Add to `run-simulation.js` strategyMap

3. Run tests:
```bash
node run-simulation.js --strategy1 MyStrategy --strategy2 Random --games 100
```

## Scientific Rigor

To ensure valid results:

1. **Sample Size**: Run at least 100 games per matchup
2. **Variance**: Check standard deviation in results
3. **Mirror Matches**: Run A vs B and B vs A
4. **Statistical Significance**: Use 95% confidence intervals

## Conclusion

This simulation system should definitively answer:
- **What is the optimal strategy for 3-Board Chess?**
- **How much do traps matter vs traditional chess tactics?**
- **What's the expected game length?**

Run the simulations and let the data speak!

```bash
npm run simulate:tournament
```

🎲 May the best strategy win! 🎲
