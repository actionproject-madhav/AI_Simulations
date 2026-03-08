# 3-Board Chess: Optimal Strategy Discovery Plan

## Game Characteristics

### Unique Aspects:
1. **3 independent boards** - Each maintains its own state
2. **Random board selection** - After each turn pair
3. **Random color assignment** - You might be white then black
4. **Checkmate wins immediately** - On ANY board ends entire game
5. **Goal**: Create positions where whoever gets next move can win

### Strategic Implications:
- Can't control which board is active next
- Can't control which color you'll be
- Traditional chess opening theory likely irrelevant
- Need to create **traps across multiple boards**
- The "meta-game" is about creating maximum instability

---

## Minimax vs Monte Carlo Analysis

### Minimax (Alpha-Beta Pruning):
**Pros:**
- Proven for traditional chess
- Can evaluate positions deeply
- Deterministic

**Cons:**
- Struggles with randomness (board/color selection)
- Evaluation function unclear for this variant
- Computational explosion with 3 boards
- Assumes perfect information (but next board is random)

### Monte Carlo Tree Search (MCTS):
**Pros:**
- Handles randomness naturally (stochastic games)
- Can handle large state spaces
- Discovers strategies through simulation
- Works without domain-specific evaluation function
- Self-balancing exploration/exploitation

**Cons:**
- Needs many simulations
- Slower for tactical positions

### **Recommendation: Hybrid Approach**
1. **MCTS for game-level decisions** (which board to push, when to attack)
2. **Minimax for position evaluation** (is this checkmate-in-1, checkmate-in-2?)
3. **Self-play simulations** to discover patterns

---

## Strategy Discovery Framework

### Phase 1: Data Collection (Self-Play Simulations)

#### Simulation Setup:
```
Run 10,000+ games of AI vs AI
Use different playing styles:
  1. Aggressive (push for checkmate quickly)
  2. Defensive (maintain material, avoid traps)
  3. Balanced (mix of both)
  4. Random baseline
```

#### Metrics to Track:
1. **Game-Level Metrics:**
   - Total turn pairs until game ends
   - Which board had the winning checkmate
   - Winner's color at game end
   - Number of boards with active pieces

2. **Position Metrics (at each turn pair):**
   - Number of "checkmate-in-1" threats across all boards
   - Number of "checkmate-in-2" threats
   - Material balance on each board
   - King safety score on each board
   - Number of boards in "critical" state (close to checkmate)

3. **Strategy Metrics:**
   - How often did winner create multi-board threats?
   - Average material sacrificed for checkmate threats
   - Correlation between "number of trap boards" and win rate
   - Did winners focus on 1 board or spread across all 3?

### Phase 2: Pattern Recognition

#### Questions to Answer:
1. **Optimal Trap Density:**
   - Is it better to have 1 board near-checkmate or all 3 semi-threatening?
   - What's the optimal number of "trap boards" to maintain?

2. **Piece Sacrifice Strategy:**
   - Should you sacrifice material to create checkmate threats?
   - What's the ROI on piece sacrifices in this variant?

3. **Positional Priorities:**
   - King safety vs attacking potential
   - Development vs immediate threats
   - What matters when you don't know which board is next?

4. **Endgame Patterns:**
   - What piece configurations lead to checkmate traps?
   - Common winning patterns across simulations

### Phase 3: Strategy Synthesis

#### Discovered Heuristics:
Based on simulation data, create evaluation function:
```javascript
boardScore =
  + (checkmateIn1Threats * 1000)
  + (checkmateIn2Threats * 500)
  + (kingExposure * 300)
  - (materialDeficit * 100)
  + (numberOfTrapBoards * 200)
```

Weights determined by correlation analysis.

---

## Implementation Plan

### Step 1: Create Simulation Engine
```
File: src/simulation/gameSimulator.js

Features:
- Run headless games (no UI)
- Fast chess engine (current one is good)
- Random board/color selection
- Log every game state
- Track win conditions
```

### Step 2: Strategy Implementations
```
File: src/simulation/strategies.js

Implement 4+ strategies:
1. AggressiveStrategy - Push for quick checkmates
2. DefensiveStrategy - Maintain solid positions
3. TrapBuilderStrategy - Create multi-board threats
4. MaterialStrategy - Traditional chess values
5. RandomStrategy - Baseline
```

### Step 3: Data Collection
```
File: src/simulation/dataCollector.js

For each game, record:
- Board states at each turn pair
- Position evaluations
- Winner and winning move
- Key decision points
```

### Step 4: Analysis Pipeline
```
File: src/simulation/analyzer.js

Statistical analysis:
- Win rates by strategy
- Position pattern clustering
- Decision tree analysis
- Correlation between metrics and wins
```

### Step 5: Visualization
```
File: src/simulation/visualizer.js

Generate:
- Win rate charts
- Position heatmaps
- Decision trees
- Strategy comparison graphs
```

---

## Specific Research Questions

1. **The "Trap Board" Hypothesis**
   - H: Maintaining 2+ boards with checkmate-in-1 threats = higher win rate
   - Measure: Correlation between trap count and wins

2. **The "Sacrifice Paradox"**
   - H: Material sacrifices for checkmate threats are +EV in this variant
   - Measure: ROI of sacrifices (material lost vs win rate increase)

3. **The "Chaos Strategy"**
   - H: Creating unstable positions (where either side can win) favors the current player
   - Measure: Win rate in "mutual threat" positions

4. **The "Board Focus" Question**
   - H: Focusing on developing 1 board deeply vs spreading across all 3
   - Measure: Win rate correlation with board state variance

---

## Expected Computational Requirements

### Small-Scale Test (1,000 games):
- Runtime: ~30 minutes (with depth-3 AI)
- Data size: ~50MB
- Initial insights

### Medium-Scale (10,000 games):
- Runtime: ~5 hours
- Data size: ~500MB
- Statistical significance

### Large-Scale (100,000 games):
- Runtime: ~2 days
- Data size: ~5GB
- Robust pattern discovery

---

## Deliverables

1. **Simulation Framework** - Headless game runner
2. **Strategy Library** - Multiple AI approaches
3. **Data Pipeline** - Collection and storage
4. **Analysis Tools** - Statistical analysis
5. **Visualization Dashboard** - Results presentation
6. **Strategy Report** - Discovered optimal patterns
7. **Improved AI** - Implementing discovered strategies

---

## Next Steps

1. ✅ Create plan (this document)
2. ⏳ Set up simulation infrastructure
3. ⏳ Implement baseline strategies
4. ⏳ Run initial 1,000-game test
5. ⏳ Analyze preliminary results
6. ⏳ Refine and scale up
7. ⏳ Extract and implement optimal strategy

---

## Timeline Estimate

- **Week 1**: Simulation framework + baseline strategies
- **Week 2**: Run simulations + collect data
- **Week 3**: Analysis + pattern discovery
- **Week 4**: Implement optimal strategy + validate

**Note**: We can get preliminary insights from just 100-1000 games in a few hours.
