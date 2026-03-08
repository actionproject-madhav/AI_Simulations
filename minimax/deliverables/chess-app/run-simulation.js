#!/usr/bin/env node

/**
 * CLI Script to run 3-Board Chess simulations
 *
 * Usage:
 *   node run-simulation.js --games 100 --tournament
 *   node run-simulation.js --games 50 --strategy1 TrapBuilder --strategy2 Aggressive
 *   node run-simulation.js --quick (runs 10 games for quick test)
 */

import BatchRunner from './src/simulation/batchRunner.js';
import {
  RandomStrategy,
  MinimaxStrategy,
  AggressiveStrategy,
  DefensiveStrategy,
  TrapBuilderStrategy
} from './src/simulation/strategies.js';
import fs from 'fs';
import path from 'path';

// Parse command line arguments
const args = process.argv.slice(2);
const options = {
  games: 100,
  tournament: false,
  verbose: false,
  quick: false,
  strategy1: null,
  strategy2: null,
  depth: 3,
  maxTurnPairs: 100
};

for (let i = 0; i < args.length; i++) {
  switch (args[i]) {
    case '--games':
      options.games = parseInt(args[++i]);
      break;
    case '--tournament':
      options.tournament = true;
      break;
    case '--verbose':
      options.verbose = true;
      break;
    case '--quick':
      options.quick = true;
      options.games = 10;
      break;
    case '--strategy1':
      options.strategy1 = args[++i];
      break;
    case '--strategy2':
      options.strategy2 = args[++i];
      break;
    case '--depth':
      options.depth = parseInt(args[++i]);
      break;
    case '--maxTurnPairs':
      options.maxTurnPairs = parseInt(args[++i]);
      break;
    case '--help':
      printHelp();
      process.exit(0);
  }
}

function printHelp() {
  console.log(`
3-Board Chess Simulation Runner
===============================

Usage:
  node run-simulation.js [options]

Options:
  --games <N>          Number of games per matchup (default: 100)
  --tournament         Run full round-robin tournament
  --strategy1 <name>   First strategy name
  --strategy2 <name>   Second strategy name
  --depth <N>          AI search depth (default: 3)
  --maxTurnPairs <N>   Max turn pairs before draw (default: 100)
  --verbose            Print detailed game logs
  --quick              Quick test (10 games)
  --help               Show this help

Strategies:
  Random       - Random legal moves (baseline)
  Minimax      - Standard minimax AI
  Aggressive   - Prioritizes attacks and checks
  Defensive    - Prioritizes safety and material
  TrapBuilder  - Creates checkmate-in-1 threats (hypothesis: optimal)

Examples:
  # Quick test
  node run-simulation.js --quick

  # Run 100 games between TrapBuilder and Aggressive
  node run-simulation.js --games 100 --strategy1 TrapBuilder --strategy2 Aggressive

  # Full tournament (25 matchups × 50 games = 1,250 games)
  node run-simulation.js --tournament --games 50

  # Verbose mode for debugging
  node run-simulation.js --quick --verbose
  `);
}

const strategyMap = {
  'Random': RandomStrategy,
  'Minimax': MinimaxStrategy,
  'Aggressive': AggressiveStrategy,
  'Defensive': DefensiveStrategy,
  'TrapBuilder': TrapBuilderStrategy
};

async function main() {
  console.log('\n' + '═'.repeat(80));
  console.log('  3-BOARD CHESS STRATEGY DISCOVERY SIMULATION');
  console.log('═'.repeat(80) + '\n');

  const runner = new BatchRunner({
    gamesPerMatchup: options.games,
    verbose: options.verbose,
    aiDepth: options.depth,
    maxTurnPairs: options.maxTurnPairs
  });

  const startTime = Date.now();

  if (options.tournament) {
    // Run full tournament
    await runner.runTournament(options.games);
  } else if (options.strategy1 && options.strategy2) {
    // Run specific matchup
    const Strategy1 = strategyMap[options.strategy1];
    const Strategy2 = strategyMap[options.strategy2];

    if (!Strategy1 || !Strategy2) {
      console.error('Invalid strategy name. Available: Random, Minimax, Aggressive, Defensive, TrapBuilder');
      process.exit(1);
    }

    await runner.runMatchup(Strategy1, Strategy2, options.games);
  } else {
    // Default: Run a few interesting matchups
    console.log('Running curated matchups (use --tournament for full tournament)\n');

    await runner.runMatchup(TrapBuilderStrategy, AggressiveStrategy, options.games);
    await runner.runMatchup(TrapBuilderStrategy, DefensiveStrategy, options.games);
    await runner.runMatchup(TrapBuilderStrategy, MinimaxStrategy, options.games);
    await runner.runMatchup(AggressiveStrategy, DefensiveStrategy, options.games);
    await runner.runMatchup(MinimaxStrategy, RandomStrategy, options.games);

    runner.printTournamentResults();
  }

  const endTime = Date.now();
  const duration = ((endTime - startTime) / 1000 / 60).toFixed(2);

  console.log(`\nSimulation completed in ${duration} minutes`);

  // Save results
  const resultsDir = './simulation-results';
  if (!fs.existsSync(resultsDir)) {
    fs.mkdirSync(resultsDir);
  }

  const timestamp = new Date().toISOString().replace(/:/g, '-').split('.')[0];
  const filename = `results-${timestamp}.json`;
  const filepath = path.join(resultsDir, filename);

  const data = {
    options,
    results: runner.results,
    stats: runner.stats,
    duration: duration,
    timestamp: new Date().toISOString()
  };

  fs.writeFileSync(filepath, JSON.stringify(data, null, 2));
  console.log(`\nResults saved to: ${filepath}`);

  // Print quick summary
  console.log('\n' + '═'.repeat(80));
  console.log('KEY FINDINGS:');
  console.log('═'.repeat(80));

  // Find best strategy
  const strategies = Object.entries(runner.stats.byStrategy);
  if (strategies.length > 0) {
    const best = strategies.reduce((max, curr) => {
      const currWinRate = curr[1].wins / curr[1].games;
      const maxWinRate = max[1].wins / max[1].games;
      return currWinRate > maxWinRate ? curr : max;
    });

    const bestWinRate = (best[1].wins / best[1].games * 100).toFixed(1);
    console.log(`\nBest Strategy: ${best[0]} (${bestWinRate}% win rate)`);
    console.log(`\nThis suggests that ${best[0]} is the most effective strategy for 3-Board Chess.`);
  }

  console.log('\n' + '═'.repeat(80) + '\n');
}

main().catch(console.error);
