/**
 * Batch Runner - Run multiple simulations and collect data
 */

import GameSimulator from './gameSimulator.js';
import {
  RandomStrategy,
  MinimaxStrategy,
  AggressiveStrategy,
  DefensiveStrategy,
  TrapBuilderStrategy
} from './strategies.js';

export class BatchRunner {
  constructor(options = {}) {
    this.gamesPerMatchup = options.gamesPerMatchup || 100;
    this.verbose = options.verbose || false;
    this.aiDepth = options.aiDepth || 3;
    this.maxTurnPairs = options.maxTurnPairs || 100;

    this.results = [];
    this.stats = {
      totalGames: 0,
      player1Wins: 0,
      player2Wins: 0,
      draws: 0,
      avgTurnPairs: 0,
      byStrategy: {}
    };
  }

  /**
   * Run games between two strategies
   */
  async runMatchup(Strategy1, Strategy2, numGames = null) {
    numGames = numGames || this.gamesPerMatchup;

    const strategy1 = new Strategy1();
    const strategy2 = new Strategy2();

    console.log(`\n${'='.repeat(60)}`);
    console.log(`Running ${numGames} games: ${strategy1.name} vs ${strategy2.name}`);
    console.log(`${'='.repeat(60)}\n`);

    const matchupResults = {
      strategy1: strategy1.name,
      strategy2: strategy2.name,
      games: [],
      stats: {
        strategy1Wins: 0,
        strategy2Wins: 0,
        draws: 0,
        avgTurnPairs: 0,
        minTurnPairs: Infinity,
        maxTurnPairs: 0
      }
    };

    for (let i = 0; i < numGames; i++) {
      const simulator = new GameSimulator(strategy1, strategy2, {
        verbose: this.verbose,
        maxTurnPairs: this.maxTurnPairs,
        aiDepth: this.aiDepth
      });

      const result = await simulator.playGame();

      // Track statistics
      matchupResults.games.push(result);

      if (result.winner === 'player1') {
        matchupResults.stats.strategy1Wins++;
        this.stats.player1Wins++;
      } else if (result.winner === 'player2') {
        matchupResults.stats.strategy2Wins++;
        this.stats.player2Wins++;
      } else {
        matchupResults.stats.draws++;
        this.stats.draws++;
      }

      matchupResults.stats.avgTurnPairs += result.turnPairs;
      matchupResults.stats.minTurnPairs = Math.min(matchupResults.stats.minTurnPairs, result.turnPairs);
      matchupResults.stats.maxTurnPairs = Math.max(matchupResults.stats.maxTurnPairs, result.turnPairs);

      this.stats.totalGames++;

      // Progress indicator
      if ((i + 1) % 10 === 0 || i === numGames - 1) {
        const progress = ((i + 1) / numGames * 100).toFixed(0);
        console.log(`Progress: ${i + 1}/${numGames} (${progress}%)`);
      }
    }

    matchupResults.stats.avgTurnPairs /= numGames;

    // Calculate win rates
    matchupResults.stats.strategy1WinRate = (matchupResults.stats.strategy1Wins / numGames * 100).toFixed(1);
    matchupResults.stats.strategy2WinRate = (matchupResults.stats.strategy2Wins / numGames * 100).toFixed(1);
    matchupResults.stats.drawRate = (matchupResults.stats.draws / numGames * 100).toFixed(1);

    this.results.push(matchupResults);

    // Update by-strategy stats
    if (!this.stats.byStrategy[strategy1.name]) {
      this.stats.byStrategy[strategy1.name] = { wins: 0, losses: 0, draws: 0, games: 0 };
    }
    if (!this.stats.byStrategy[strategy2.name]) {
      this.stats.byStrategy[strategy2.name] = { wins: 0, losses: 0, draws: 0, games: 0 };
    }

    this.stats.byStrategy[strategy1.name].wins += matchupResults.stats.strategy1Wins;
    this.stats.byStrategy[strategy1.name].losses += matchupResults.stats.strategy2Wins;
    this.stats.byStrategy[strategy1.name].draws += matchupResults.stats.draws;
    this.stats.byStrategy[strategy1.name].games += numGames;

    this.stats.byStrategy[strategy2.name].wins += matchupResults.stats.strategy2Wins;
    this.stats.byStrategy[strategy2.name].losses += matchupResults.stats.strategy1Wins;
    this.stats.byStrategy[strategy2.name].draws += matchupResults.stats.draws;
    this.stats.byStrategy[strategy2.name].games += numGames;

    // Print matchup summary
    this.printMatchupSummary(matchupResults);

    return matchupResults;
  }

  /**
   * Run round-robin tournament between all strategies
   */
  async runTournament(numGamesPerMatchup = null) {
    const strategies = [
      RandomStrategy,
      MinimaxStrategy,
      AggressiveStrategy,
      DefensiveStrategy,
      TrapBuilderStrategy
    ];

    console.log(`\n${'#'.repeat(80)}`);
    console.log(`STARTING TOURNAMENT: ${strategies.length} strategies, ${numGamesPerMatchup || this.gamesPerMatchup} games per matchup`);
    console.log(`${'#'.repeat(80)}\n`);

    const matchups = [];

    // Generate all matchups (including mirror matches)
    for (let i = 0; i < strategies.length; i++) {
      for (let j = 0; j < strategies.length; j++) {
        matchups.push([strategies[i], strategies[j]]);
      }
    }

    // Run all matchups
    for (let i = 0; i < matchups.length; i++) {
      const [Strategy1, Strategy2] = matchups[i];
      console.log(`\nMatchup ${i + 1}/${matchups.length}`);
      await this.runMatchup(Strategy1, Strategy2, numGamesPerMatchup);
    }

    // Print overall tournament results
    this.printTournamentResults();

    return this.results;
  }

  /**
   * Print matchup summary
   */
  printMatchupSummary(matchupResults) {
    console.log(`\n${'─'.repeat(60)}`);
    console.log(`MATCHUP SUMMARY`);
    console.log(`${'─'.repeat(60)}`);
    console.log(`${matchupResults.strategy1} vs ${matchupResults.strategy2}`);
    console.log(`${matchupResults.strategy1} wins: ${matchupResults.stats.strategy1Wins} (${matchupResults.stats.strategy1WinRate}%)`);
    console.log(`${matchupResults.strategy2} wins: ${matchupResults.stats.strategy2Wins} (${matchupResults.stats.strategy2WinRate}%)`);
    console.log(`Draws: ${matchupResults.stats.draws} (${matchupResults.stats.drawRate}%)`);
    console.log(`Avg turn pairs: ${matchupResults.stats.avgTurnPairs.toFixed(1)}`);
    console.log(`Turn pairs range: ${matchupResults.stats.minTurnPairs} - ${matchupResults.stats.maxTurnPairs}`);
    console.log(`${'─'.repeat(60)}\n`);
  }

  /**
   * Print overall tournament results
   */
  printTournamentResults() {
    console.log(`\n${'═'.repeat(80)}`);
    console.log(`TOURNAMENT RESULTS`);
    console.log(`${'═'.repeat(80)}\n`);

    // Calculate win rates for each strategy
    const strategyStats = [];
    for (const [strategyName, stats] of Object.entries(this.stats.byStrategy)) {
      const winRate = (stats.wins / stats.games * 100).toFixed(1);
      const lossRate = (stats.losses / stats.games * 100).toFixed(1);
      const drawRate = (stats.draws / stats.games * 100).toFixed(1);

      strategyStats.push({
        name: strategyName,
        wins: stats.wins,
        losses: stats.losses,
        draws: stats.draws,
        games: stats.games,
        winRate: parseFloat(winRate),
        lossRate: parseFloat(lossRate),
        drawRate: parseFloat(drawRate)
      });
    }

    // Sort by win rate
    strategyStats.sort((a, b) => b.winRate - a.winRate);

    console.log('LEADERBOARD (by win rate):');
    console.log(`${'─'.repeat(80)}`);
    console.log(` Rank | Strategy           | Games | Wins   | Losses | Draws  | Win%  `);
    console.log(`${'─'.repeat(80)}`);

    strategyStats.forEach((stats, index) => {
      console.log(
        ` ${String(index + 1).padStart(4)} | ${stats.name.padEnd(18)} | ` +
        `${String(stats.games).padStart(5)} | ${String(stats.wins).padStart(6)} | ` +
        `${String(stats.losses).padStart(6)} | ${String(stats.draws).padStart(6)} | ` +
        `${String(stats.winRate).padStart(5)}%`
      );
    });

    console.log(`${'─'.repeat(80)}\n`);

    console.log('OVERALL STATISTICS:');
    console.log(`Total games played: ${this.stats.totalGames}`);
    console.log(`${'═'.repeat(80)}\n`);
  }

  /**
   * Save results to JSON file
   */
  saveResults(filepath) {
    const fs = require('fs');
    const data = {
      results: this.results,
      stats: this.stats,
      timestamp: new Date().toISOString()
    };

    fs.writeFileSync(filepath, JSON.stringify(data, null, 2));
    console.log(`Results saved to: ${filepath}`);
  }
}

export default BatchRunner;
