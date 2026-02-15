"""
Genetic Algorithm for Evolving Blackjack Card Counting Strategy

Extension of basic strategy to include:
- 6-deck shoe with 75% penetration
- Card counting system (running count → true count)
- Bet sizing based on true count ranges
- Bankroll-based fitness evaluation
- 3:2 blackjack payouts

Chromosome encoding (294 bits total):
- Play strategy: 260 bits (17×10 hard + 9×10 soft)
- Card count values: 22 bits (11 ranks × 2 bits)
- Bet multipliers: 12 bits (4 count ranges × 3 bits)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CountingGAConfig:
    """Configuration for card counting GA."""
    population_size: int = 200
    generations: int = 100
    tournament_size: int = 3
    crossover_rate: float = 0.85
    mutation_candidate_rate: float = 0.10
    mutation_bit_rate: float = 0.01
    elitism_count: int = 2
    hands_per_evaluation: int = 1000
    num_decks: int = 6
    penetration: float = 0.75  # Reshuffle at 75% dealt
    starting_bankroll: float = 1000.0
    min_bet: float = 1.0
    max_bet: float = 8.0
    use_parallel: bool = True
    random_seed: Optional[int] = None


# =============================================================================
# CARD AND SHOE IMPLEMENTATION
# =============================================================================

class Card:
    """Represents a playing card."""

    SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit

    @property
    def value(self) -> int:
        """Blackjack value (Ace = 11 initially)."""
        if self.rank in ['J', 'Q', 'K']:
            return 10
        elif self.rank == 'A':
            return 11
        else:
            return int(self.rank)

    @property
    def dealer_index(self) -> int:
        """Index for dealer up-card (0=Ace, 1=2, ..., 9=10/Face)."""
        if self.rank == 'A':
            return 0
        elif self.rank in ['10', 'J', 'Q', 'K']:
            return 9
        else:
            return int(self.rank) - 1

    @property
    def count_index(self) -> int:
        """Index for card counting (0=A, 1=2, ..., 9=10/Face)."""
        if self.rank == 'A':
            return 0
        elif self.rank in ['10', 'J', 'Q', 'K']:
            return 9
        else:
            return int(self.rank) - 1

    def __repr__(self) -> str:
        return f"{self.rank}{self.suit[0]}"


class Shoe:
    """Represents a multi-deck shoe for blackjack."""

    def __init__(self, num_decks: int = 6, penetration: float = 0.75):
        self.num_decks = num_decks
        self.penetration = penetration
        self.total_cards = num_decks * 52
        self.penetration_point = int(self.total_cards * penetration)
        self.cards: List[Card] = []
        self.dealt_count = 0
        self.shuffle()

    def shuffle(self):
        """Shuffle all decks into the shoe."""
        self.cards = [Card(rank, suit)
                      for _ in range(self.num_decks)
                      for suit in Card.SUITS
                      for rank in Card.RANKS]
        random.shuffle(self.cards)
        self.dealt_count = 0

    def deal(self) -> Card:
        """Deal a card from the shoe."""
        if len(self.cards) == 0:
            self.shuffle()
        card = self.cards.pop()
        self.dealt_count += 1
        return card

    @property
    def needs_shuffle(self) -> bool:
        """Check if penetration point has been reached."""
        return self.dealt_count >= self.penetration_point

    @property
    def remaining_cards(self) -> int:
        """Number of cards remaining in shoe."""
        return len(self.cards)

    @property
    def remaining_decks(self) -> float:
        """Approximate number of decks remaining."""
        return max(0.5, self.remaining_cards / 52)  # Min 0.5 to avoid division issues


# =============================================================================
# HAND EVALUATION
# =============================================================================

class Hand:
    """Represents a blackjack hand."""

    def __init__(self):
        self.cards: List[Card] = []

    def add_card(self, card: Card):
        self.cards.append(card)

    def clear(self):
        self.cards = []

    @property
    def value(self) -> int:
        """Calculate best hand value (adjusting Aces)."""
        total = sum(card.value for card in self.cards)
        aces = sum(1 for card in self.cards if card.rank == 'A')
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        return total

    @property
    def is_soft(self) -> bool:
        """Check if hand is soft (Ace counted as 11)."""
        total = sum(card.value for card in self.cards)
        aces = sum(1 for card in self.cards if card.rank == 'A')
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        return aces > 0 and total <= 21

    @property
    def is_busted(self) -> bool:
        return self.value > 21

    @property
    def is_blackjack(self) -> bool:
        return len(self.cards) == 2 and self.value == 21

    def __repr__(self) -> str:
        return f"{self.cards} = {self.value}"


# =============================================================================
# COUNTING STRATEGY CHROMOSOME
# =============================================================================

class CountingStrategy:
    """
    Extended strategy chromosome for card counting.

    Total: 294 bits
    - Play strategy: 260 bits (same as basic)
    - Card count values: 22 bits (11 ranks × 2 bits)
    - Bet multipliers: 12 bits (4 ranges × 3 bits)
    """

    # Play strategy dimensions
    HARD_HANDS = 17
    SOFT_HANDS = 9
    DEALER_CARDS = 10
    PLAY_BITS = (HARD_HANDS + SOFT_HANDS) * DEALER_CARDS  # 260

    # Count value encoding
    NUM_CARD_RANKS = 10  # A, 2-9, 10 (J/Q/K same as 10)
    BITS_PER_COUNT = 2
    COUNT_BITS = NUM_CARD_RANKS * BITS_PER_COUNT  # 20 bits (not 22 - corrected)

    # Bet multiplier encoding
    NUM_COUNT_RANGES = 4
    BITS_PER_MULTIPLIER = 3
    BET_BITS = NUM_COUNT_RANGES * BITS_PER_MULTIPLIER  # 12

    TOTAL_BITS = PLAY_BITS + COUNT_BITS + BET_BITS  # 292

    # Count value decoding: 00=-1, 01=0, 10=+1, 11=0
    COUNT_DECODE = {0: -1, 1: 0, 2: 1, 3: 0}

    def __init__(self, chromosome: Optional[np.ndarray] = None):
        if chromosome is not None:
            self.chromosome = chromosome.astype(np.uint8)
        else:
            self.chromosome = np.random.randint(0, 2, self.TOTAL_BITS, dtype=np.uint8)

        self.fitness: float = 0.0
        self._parse_chromosome()

    def _parse_chromosome(self):
        """Parse chromosome into components."""
        # Extract play strategy bits
        self.play_bits = self.chromosome[:self.PLAY_BITS]

        # Extract count value bits
        count_start = self.PLAY_BITS
        count_end = count_start + self.COUNT_BITS
        count_bits = self.chromosome[count_start:count_end]

        # Decode count values for each rank
        self.count_values = []
        for i in range(self.NUM_CARD_RANKS):
            bit_pair = count_bits[i*2] * 2 + count_bits[i*2 + 1]
            self.count_values.append(self.COUNT_DECODE[bit_pair])

        # Extract bet multiplier bits
        bet_start = count_end
        bet_bits = self.chromosome[bet_start:]

        # Decode bet multipliers (3 bits each, value 0-7 maps to multiplier 1-8)
        self.bet_multipliers = []
        for i in range(self.NUM_COUNT_RANGES):
            bits = bet_bits[i*3:(i+1)*3]
            value = bits[0] * 4 + bits[1] * 2 + bits[2]
            self.bet_multipliers.append(value + 1)  # 1-8

    def get_count_value(self, card: Card) -> int:
        """Get the count value for a card."""
        return self.count_values[card.count_index]

    def get_bet_multiplier(self, true_count: int) -> int:
        """Get bet multiplier based on true count range."""
        if true_count <= -2:
            return self.bet_multipliers[0]
        elif true_count <= 1:
            return self.bet_multipliers[1]
        elif true_count <= 4:
            return self.bet_multipliers[2]
        else:  # >= 5
            return self.bet_multipliers[3]

    def get_hard_index(self, player_total: int, dealer_index: int) -> int:
        player_idx = player_total - 4
        return player_idx * self.DEALER_CARDS + dealer_index

    def get_soft_index(self, player_total: int, dealer_index: int) -> int:
        player_idx = player_total - 12
        hard_bits = self.HARD_HANDS * self.DEALER_CARDS
        return hard_bits + player_idx * self.DEALER_CARDS + dealer_index

    def should_hit(self, player_hand: Hand, dealer_up_card: Card) -> bool:
        """Determine whether to hit based on play strategy."""
        player_total = player_hand.value
        dealer_idx = dealer_up_card.dealer_index
        is_soft = player_hand.is_soft

        if player_total >= 21:
            return False
        if player_total <= 3:
            return True

        if is_soft:
            if player_total < 12:
                return True
            if player_total > 20:
                return False
            idx = self.get_soft_index(player_total, dealer_idx)
        else:
            if player_total < 4:
                return True
            if player_total > 20:
                return False
            idx = self.get_hard_index(player_total, dealer_idx)

        return self.play_bits[idx] == 1

    def copy(self) -> 'CountingStrategy':
        new_strategy = CountingStrategy(self.chromosome.copy())
        new_strategy.fitness = self.fitness
        return new_strategy

    def get_hard_matrix(self) -> np.ndarray:
        hard_bits = self.HARD_HANDS * self.DEALER_CARDS
        return self.play_bits[:hard_bits].reshape(self.HARD_HANDS, self.DEALER_CARDS)

    def get_soft_matrix(self) -> np.ndarray:
        hard_bits = self.HARD_HANDS * self.DEALER_CARDS
        return self.play_bits[hard_bits:].reshape(self.SOFT_HANDS, self.DEALER_CARDS)


# =============================================================================
# BLACKJACK GAME WITH COUNTING
# =============================================================================

class BlackjackCountingGame:
    """Blackjack game with card counting and bet sizing."""

    def __init__(self, config: CountingGAConfig):
        self.config = config
        self.shoe = Shoe(config.num_decks, config.penetration)
        self.player_hand = Hand()
        self.dealer_hand = Hand()
        self.running_count = 0
        self.bankroll = config.starting_bankroll
        self.bankroll_history: List[float] = []

    def reset_session(self):
        """Reset for a new evaluation session."""
        self.shoe.shuffle()
        self.running_count = 0
        self.bankroll = self.config.starting_bankroll
        self.bankroll_history = [self.bankroll]

    def update_count(self, card: Card, strategy: CountingStrategy):
        """Update running count when a card is revealed."""
        self.running_count += strategy.get_count_value(card)

    def get_true_count(self) -> int:
        """Calculate true count from running count."""
        decks_remaining = self.shoe.remaining_decks
        return round(self.running_count / decks_remaining)

    def calculate_bet(self, strategy: CountingStrategy) -> float:
        """Calculate bet amount based on true count."""
        true_count = self.get_true_count()
        multiplier = strategy.get_bet_multiplier(true_count)
        bet = self.config.min_bet * multiplier

        # Cap at max bet and current bankroll
        bet = min(bet, self.config.max_bet)
        bet = min(bet, self.bankroll)

        return bet

    def play_hand(self, strategy: CountingStrategy) -> float:
        """
        Play a single hand and return the profit/loss.

        Returns the change in bankroll (positive = win, negative = loss).
        """
        # Check for reshuffle
        if self.shoe.needs_shuffle:
            self.shoe.shuffle()
            self.running_count = 0

        # Calculate bet
        bet = self.calculate_bet(strategy)
        if bet <= 0:
            return 0  # No bankroll left

        # Clear hands
        self.player_hand.clear()
        self.dealer_hand.clear()

        # Deal initial cards
        player_card1 = self.shoe.deal()
        self.player_hand.add_card(player_card1)
        self.update_count(player_card1, strategy)

        dealer_card1 = self.shoe.deal()  # Up card
        self.dealer_hand.add_card(dealer_card1)
        self.update_count(dealer_card1, strategy)

        player_card2 = self.shoe.deal()
        self.player_hand.add_card(player_card2)
        self.update_count(player_card2, strategy)

        dealer_card2 = self.shoe.deal()  # Hole card (count later)
        self.dealer_hand.add_card(dealer_card2)

        dealer_up_card = dealer_card1

        # Check for blackjacks
        player_bj = self.player_hand.is_blackjack
        dealer_bj = self.dealer_hand.is_blackjack

        # Count hole card when revealed
        self.update_count(dealer_card2, strategy)

        if player_bj and dealer_bj:
            return 0  # Push
        elif player_bj:
            return bet * 1.5  # Blackjack pays 3:2
        elif dealer_bj:
            return -bet

        # Player's turn
        while not self.player_hand.is_busted:
            if strategy.should_hit(self.player_hand, dealer_up_card):
                card = self.shoe.deal()
                self.player_hand.add_card(card)
                self.update_count(card, strategy)
            else:
                break

        if self.player_hand.is_busted:
            return -bet

        # Dealer's turn (stands on soft 17)
        while self.dealer_hand.value < 17:
            card = self.shoe.deal()
            self.dealer_hand.add_card(card)
            self.update_count(card, strategy)

        # Compare hands
        player_value = self.player_hand.value
        dealer_value = self.dealer_hand.value

        if self.dealer_hand.is_busted:
            return bet
        elif player_value > dealer_value:
            return bet
        elif player_value < dealer_value:
            return -bet
        else:
            return 0  # Push

    def play_session(self, strategy: CountingStrategy, num_hands: int) -> float:
        """Play a full session and return final bankroll."""
        self.reset_session()

        for _ in range(num_hands):
            if self.bankroll <= 0:
                break

            profit = self.play_hand(strategy)
            self.bankroll += profit
            self.bankroll_history.append(self.bankroll)

        return self.bankroll


def evaluate_counting_strategy(args: Tuple[np.ndarray, int, CountingGAConfig, int]) -> float:
    """Evaluate a counting strategy by playing a session."""
    chromosome, num_hands, config, seed = args

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    strategy = CountingStrategy(chromosome)
    game = BlackjackCountingGame(config)
    final_bankroll = game.play_session(strategy, num_hands)

    return final_bankroll


# =============================================================================
# GENETIC ALGORITHM FOR COUNTING
# =============================================================================

class CountingGeneticAlgorithm:
    """GA for evolving card counting strategies."""

    def __init__(self, config: CountingGAConfig = None):
        self.config = config or CountingGAConfig()

        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

        self.population: List[CountingStrategy] = []
        self.generation_stats: List[dict] = []
        self._initialize_population()

    def _initialize_population(self):
        """Initialize population with random strategies."""
        self.population = [CountingStrategy() for _ in range(self.config.population_size)]

    def _tournament_select(self) -> CountingStrategy:
        """Tournament selection."""
        tournament = random.sample(self.population, self.config.tournament_size)
        return max(tournament, key=lambda s: s.fitness)

    def _uniform_crossover(self, parent1: CountingStrategy,
                           parent2: CountingStrategy) -> Tuple[CountingStrategy, CountingStrategy]:
        """Uniform crossover."""
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()

        mask = np.random.randint(0, 2, CountingStrategy.TOTAL_BITS, dtype=np.uint8)
        child1_chromosome = np.where(mask, parent1.chromosome, parent2.chromosome)
        child2_chromosome = np.where(mask, parent2.chromosome, parent1.chromosome)

        return CountingStrategy(child1_chromosome), CountingStrategy(child2_chromosome)

    def _mutate(self, strategy: CountingStrategy) -> CountingStrategy:
        """Dual-stage mutation."""
        if random.random() > self.config.mutation_candidate_rate:
            return strategy

        mutation_mask = np.random.random(CountingStrategy.TOTAL_BITS) < self.config.mutation_bit_rate

        if np.any(mutation_mask):
            new_chromosome = strategy.chromosome.copy()
            new_chromosome[mutation_mask] = 1 - new_chromosome[mutation_mask]
            return CountingStrategy(new_chromosome)

        return strategy

    def _evaluate_population(self):
        """Evaluate fitness of all individuals."""
        if self.config.use_parallel:
            self._evaluate_population_parallel()
        else:
            self._evaluate_population_sequential()

    def _evaluate_population_sequential(self):
        """Sequential evaluation."""
        for strategy in self.population:
            args = (strategy.chromosome, self.config.hands_per_evaluation,
                    self.config, None)
            strategy.fitness = evaluate_counting_strategy(args)

    def _evaluate_population_parallel(self):
        """Parallel evaluation."""
        num_workers = min(multiprocessing.cpu_count(), len(self.population))

        args_list = [
            (s.chromosome, self.config.hands_per_evaluation, self.config,
             random.randint(0, 2**31) if self.config.random_seed else None)
            for s in self.population
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            fitnesses = list(executor.map(evaluate_counting_strategy, args_list))

        for strategy, fitness in zip(self.population, fitnesses):
            strategy.fitness = fitness

    def _create_next_generation(self):
        """Create next generation."""
        self.population.sort(key=lambda s: s.fitness, reverse=True)
        new_population: List[CountingStrategy] = []

        # Elitism
        for i in range(self.config.elitism_count):
            new_population.append(self.population[i].copy())

        # Fill with offspring
        while len(new_population) < self.config.population_size:
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            child1, child2 = self._uniform_crossover(parent1, parent2)
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            new_population.append(child1)
            if len(new_population) < self.config.population_size:
                new_population.append(child2)

        self.population = new_population

    def _record_stats(self, generation: int):
        """Record generation statistics."""
        fitnesses = [s.fitness for s in self.population]
        stats = {
            'generation': generation,
            'min': np.min(fitnesses),
            'max': np.max(fitnesses),
            'mean': np.mean(fitnesses),
            'median': np.median(fitnesses),
            'std': np.std(fitnesses)
        }
        self.generation_stats.append(stats)
        return stats

    def evolve(self, verbose: bool = True) -> CountingStrategy:
        """Run evolution and return best strategy."""
        if verbose:
            print(f"Starting evolution with {self.config.population_size} individuals")
            print(f"Playing {self.config.hands_per_evaluation} hands per strategy")
            print(f"Starting bankroll: ${self.config.starting_bankroll}")
            print("-" * 70)

        for gen in range(self.config.generations):
            self._evaluate_population()
            stats = self._record_stats(gen)

            if verbose:
                print(f"Gen {gen:3d}: "
                      f"Max=${stats['max']:,.0f}, "
                      f"Mean=${stats['mean']:,.0f}, "
                      f"Median=${stats['median']:,.0f}, "
                      f"Min=${stats['min']:,.0f}")

            if gen < self.config.generations - 1:
                self._create_next_generation()

        self.population.sort(key=lambda s: s.fitness, reverse=True)
        return self.population[0]

    def get_population_consensus(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate population consensus for play strategy."""
        hard_sum = np.zeros((CountingStrategy.HARD_HANDS, CountingStrategy.DEALER_CARDS))
        soft_sum = np.zeros((CountingStrategy.SOFT_HANDS, CountingStrategy.DEALER_CARDS))

        for strategy in self.population:
            hard_sum += strategy.get_hard_matrix()
            soft_sum += strategy.get_soft_matrix()

        n = len(self.population)
        return (hard_sum / n) * 100, (soft_sum / n) * 100


# =============================================================================
# HI-LO REFERENCE SYSTEM
# =============================================================================

def get_hilo_count_values() -> List[int]:
    """Return the standard Hi-Lo count values."""
    # A, 2, 3, 4, 5, 6, 7, 8, 9, 10
    return [-1, 1, 1, 1, 1, 1, 0, 0, 0, -1]


def create_hilo_strategy() -> CountingStrategy:
    """Create a strategy using Hi-Lo counting and basic play strategy."""
    chromosome = np.zeros(CountingStrategy.TOTAL_BITS, dtype=np.uint8)

    # Set up basic play strategy (same as before)
    hard_strategy = np.zeros((17, 10), dtype=np.uint8)

    # 4-11: hit all
    for total in range(4, 12):
        hard_strategy[total - 4, :] = 1

    # 12: hit on 2,3,7+; stand on 4-6
    hard_strategy[12 - 4, :] = 1
    hard_strategy[12 - 4, 3:6] = 0  # Stand on 4-6

    # 13-16: stand on 2-6, hit on 7+
    for total in range(13, 17):
        hard_strategy[total - 4, 0] = 1  # Hit vs A
        hard_strategy[total - 4, 6:] = 1  # Hit vs 7-10

    # 17+: stand
    for total in range(17, 21):
        hard_strategy[total - 4, :] = 0

    soft_strategy = np.zeros((9, 10), dtype=np.uint8)

    # Soft 12-17: hit all
    for i in range(6):
        soft_strategy[i, :] = 1

    # Soft 18: stand vs 2-8, hit vs 9,10,A
    soft_strategy[6, 0] = 1  # vs A
    soft_strategy[6, 8:] = 1  # vs 9-10

    # Soft 19-20: stand
    soft_strategy[7, :] = 0
    soft_strategy[8, :] = 0

    # Set play bits
    hard_bits = 17 * 10
    chromosome[:hard_bits] = hard_strategy.flatten()
    chromosome[hard_bits:260] = soft_strategy.flatten()

    # Set Hi-Lo count values
    # Encoding: 00=-1, 01=0, 10=+1, 11=0
    hilo = get_hilo_count_values()
    encode_map = {-1: (0, 0), 0: (0, 1), 1: (1, 0)}

    count_start = 260
    for i, val in enumerate(hilo):
        bits = encode_map[val]
        chromosome[count_start + i*2] = bits[0]
        chromosome[count_start + i*2 + 1] = bits[1]

    # Set bet multipliers: conservative betting
    # Range <= -2: bet 1x (minimum)
    # Range -1 to +1: bet 1x
    # Range +2 to +4: bet 4x
    # Range >= +5: bet 8x (maximum)
    bet_start = 260 + 20
    multipliers = [1, 1, 4, 8]  # Converted to 0-indexed: 0, 0, 3, 7

    for i, mult in enumerate(multipliers):
        val = mult - 1  # 0-7 encoding
        chromosome[bet_start + i*3] = (val >> 2) & 1
        chromosome[bet_start + i*3 + 1] = (val >> 1) & 1
        chromosome[bet_start + i*3 + 2] = val & 1

    return CountingStrategy(chromosome)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_convergence(ga: CountingGeneticAlgorithm, save_path: str = None):
    """Plot fitness (bankroll) convergence over generations."""
    stats = ga.generation_stats
    generations = [s['generation'] for s in stats]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(generations, [s['max'] for s in stats], 'g-', linewidth=2, label='Max')
    ax.plot(generations, [s['mean'] for s in stats], 'b-', linewidth=2, label='Mean')
    ax.plot(generations, [s['median'] for s in stats], 'c--', linewidth=1.5, label='Median')
    ax.plot(generations, [s['min'] for s in stats], 'r-', linewidth=1, label='Min')

    ax.fill_between(generations,
                    [s['min'] for s in stats],
                    [s['max'] for s in stats],
                    alpha=0.2, color='blue')

    # Reference line at starting bankroll
    ax.axhline(y=ga.config.starting_bankroll, color='gold', linestyle=':',
               linewidth=2, label=f'Starting Bankroll (${ga.config.starting_bankroll:,.0f})')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Final Bankroll ($)', fontsize=12)
    ax.set_title('Card Counting GA Convergence\n(Final Bankroll after 1000 Hands)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Convergence plot saved to: {save_path}")

    plt.close(fig)


def plot_strategy_heatmap(ga: CountingGeneticAlgorithm, save_path: str = None):
    """Plot strategy consensus heatmap."""
    hard_consensus, soft_consensus = ga.get_population_consensus()

    colors = ['#0066CC', '#FFFFFF', '#CC0000']
    cmap = LinearSegmentedColormap.from_list('stand_hit', colors)

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    dealer_labels = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    # Hard hands
    ax1 = axes[0]
    hard_labels = [str(i) for i in range(4, 21)]

    im1 = ax1.imshow(hard_consensus, cmap=cmap, vmin=0, vmax=100, aspect='auto')
    ax1.set_xticks(range(10))
    ax1.set_xticklabels(dealer_labels)
    ax1.set_yticks(range(17))
    ax1.set_yticklabels(hard_labels)
    ax1.set_xlabel('Dealer Up Card', fontsize=12)
    ax1.set_ylabel('Player Hand Total', fontsize=12)
    ax1.set_title('Hard Hands Strategy', fontsize=14)

    for i in range(17):
        for j in range(10):
            value = hard_consensus[i, j]
            text_color = 'white' if value < 20 or value > 80 else 'black'
            ax1.text(j, i, f'{value:.0f}', ha='center', va='center',
                    color=text_color, fontsize=8)

    # Soft hands
    ax2 = axes[1]
    soft_labels = ['A-A', 'A-2', 'A-3', 'A-4', 'A-5', 'A-6', 'A-7', 'A-8', 'A-9']

    im2 = ax2.imshow(soft_consensus, cmap=cmap, vmin=0, vmax=100, aspect='auto')
    ax2.set_xticks(range(10))
    ax2.set_xticklabels(dealer_labels)
    ax2.set_yticks(range(9))
    ax2.set_yticklabels(soft_labels)
    ax2.set_xlabel('Dealer Up Card', fontsize=12)
    ax2.set_ylabel('Player Hand (Soft)', fontsize=12)
    ax2.set_title('Soft Hands Strategy', fontsize=14)

    for i in range(9):
        for j in range(10):
            value = soft_consensus[i, j]
            text_color = 'white' if value < 20 or value > 80 else 'black'
            ax2.text(j, i, f'{value:.0f}', ha='center', va='center',
                    color=text_color, fontsize=8)

    cbar = fig.colorbar(im1, ax=axes, orientation='horizontal',
                        fraction=0.05, pad=0.1, aspect=40)
    cbar.set_label('% Recommending Hit (Red=100% Hit, Blue=100% Stand)', fontsize=11)

    plt.suptitle('Evolved Card Counting Strategy Consensus', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Strategy heatmap saved to: {save_path}")

    plt.close(fig)


def plot_bankroll_session(strategy: CountingStrategy, config: CountingGAConfig,
                          save_path: str = None, seed: int = None):
    """Plot bankroll over a sample session."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    game = BlackjackCountingGame(config)
    game.play_session(strategy, config.hands_per_evaluation)

    fig, ax = plt.subplots(figsize=(14, 6))

    hands = range(len(game.bankroll_history))
    ax.plot(hands, game.bankroll_history, 'b-', linewidth=1, alpha=0.8)

    ax.axhline(y=config.starting_bankroll, color='green', linestyle='--',
               linewidth=2, label=f'Starting: ${config.starting_bankroll:,.0f}')
    ax.axhline(y=game.bankroll_history[-1], color='red', linestyle=':',
               linewidth=2, label=f'Final: ${game.bankroll_history[-1]:,.0f}')

    ax.fill_between(hands, config.starting_bankroll, game.bankroll_history,
                    where=[b > config.starting_bankroll for b in game.bankroll_history],
                    color='green', alpha=0.2)
    ax.fill_between(hands, config.starting_bankroll, game.bankroll_history,
                    where=[b < config.starting_bankroll for b in game.bankroll_history],
                    color='red', alpha=0.2)

    ax.set_xlabel('Hand Number', fontsize=12)
    ax.set_ylabel('Bankroll ($)', fontsize=12)
    ax.set_title('Bankroll Over 1000-Hand Session\n(Best Evolved Strategy)', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Add profit/loss annotation
    profit = game.bankroll_history[-1] - config.starting_bankroll
    profit_pct = (profit / config.starting_bankroll) * 100
    color = 'green' if profit >= 0 else 'red'
    ax.annotate(f'Profit: ${profit:+,.0f} ({profit_pct:+.1f}%)',
                xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Bankroll plot saved to: {save_path}")

    plt.close(fig)


def print_count_comparison(strategy: CountingStrategy):
    """Print evolved count values vs Hi-Lo."""
    print("\n" + "=" * 60)
    print("EVOLVED COUNT VALUES vs HI-LO SYSTEM")
    print("=" * 60)

    card_names = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    hilo = get_hilo_count_values()
    evolved = strategy.count_values

    print(f"{'Card':<8} {'Evolved':>10} {'Hi-Lo':>10} {'Match':>10}")
    print("-" * 40)

    matches = 0
    for i, name in enumerate(card_names):
        match = "✓" if evolved[i] == hilo[i] else "✗"
        if evolved[i] == hilo[i]:
            matches += 1
        print(f"{name:<8} {evolved[i]:>+10} {hilo[i]:>+10} {match:>10}")

    print("-" * 40)
    print(f"Match rate: {matches}/{len(card_names)} ({matches/len(card_names)*100:.0f}%)")


def print_bet_multipliers(strategy: CountingStrategy):
    """Print evolved bet multipliers."""
    print("\n" + "=" * 60)
    print("EVOLVED BET MULTIPLIERS")
    print("=" * 60)

    ranges = ['≤ -2', '-1 to +1', '+2 to +4', '≥ +5']
    print(f"{'True Count Range':<20} {'Bet Multiplier':>15} {'Bet Amount':>15}")
    print("-" * 50)

    for i, (range_name, mult) in enumerate(zip(ranges, strategy.bet_multipliers)):
        print(f"{range_name:<20} {mult:>15}x ${mult:>14}")

    print()
    print("Optimal pattern: Low bets when count is negative, high bets when positive")


def print_best_strategy(strategy: CountingStrategy):
    """Print complete evolved strategy."""
    print("\n" + "=" * 60)
    print("BEST EVOLVED COUNTING STRATEGY")
    print("=" * 60)
    print(f"Final Bankroll: ${strategy.fitness:,.0f}")

    dealer_labels = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    print("\nHARD HANDS (H=Hit, S=Stand):")
    print("      " + "  ".join(f"{d:>3}" for d in dealer_labels))
    print("     " + "-" * 42)

    hard_matrix = strategy.get_hard_matrix()
    for i, total in enumerate(range(4, 21)):
        row = "".join("  H " if hard_matrix[i, j] else "  S " for j in range(10))
        print(f"{total:>3} |{row}")

    print("\nSOFT HANDS (H=Hit, S=Stand):")
    print("      " + "  ".join(f"{d:>3}" for d in dealer_labels))
    print("     " + "-" * 42)

    soft_labels = ['A-A', 'A-2', 'A-3', 'A-4', 'A-5', 'A-6', 'A-7', 'A-8', 'A-9']
    soft_matrix = strategy.get_soft_matrix()
    for i, label in enumerate(soft_labels):
        row = "".join("  H " if soft_matrix[i, j] else "  S " for j in range(10))
        print(f"{label:>3} |{row}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("BLACKJACK CARD COUNTING STRATEGY EVOLUTION")
    print("Genetic Algorithm with 6-Deck Shoe and Bet Sizing")
    print("=" * 70)
    print()

    config = CountingGAConfig(
        population_size=200,
        generations=100,
        tournament_size=3,
        crossover_rate=0.85,
        mutation_candidate_rate=0.10,
        mutation_bit_rate=0.01,
        elitism_count=2,
        hands_per_evaluation=1000,
        num_decks=6,
        penetration=0.75,
        starting_bankroll=1000.0,
        min_bet=1.0,
        max_bet=8.0,
        use_parallel=True,
        random_seed=42
    )

    print("Configuration:")
    print(f"  Population Size: {config.population_size}")
    print(f"  Generations: {config.generations}")
    print(f"  Hands per Evaluation: {config.hands_per_evaluation}")
    print(f"  Number of Decks: {config.num_decks}")
    print(f"  Penetration: {config.penetration*100:.0f}%")
    print(f"  Starting Bankroll: ${config.starting_bankroll:,.0f}")
    print(f"  Bet Range: ${config.min_bet:.0f} - ${config.max_bet:.0f}")
    print()

    # Evaluate Hi-Lo reference strategy
    print("Evaluating Hi-Lo reference strategy...")
    hilo_strategy = create_hilo_strategy()
    hilo_args = (hilo_strategy.chromosome, config.hands_per_evaluation, config, 42)
    hilo_bankroll = evaluate_counting_strategy(hilo_args)
    print(f"Hi-Lo Strategy Final Bankroll: ${hilo_bankroll:,.0f}")
    print()

    # Run evolution
    ga = CountingGeneticAlgorithm(config)
    best_strategy = ga.evolve(verbose=True)

    # Print results
    print_best_strategy(best_strategy)
    print_count_comparison(best_strategy)
    print_bet_multipliers(best_strategy)

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON WITH HI-LO SYSTEM")
    print("=" * 60)
    print(f"Evolved Strategy Bankroll: ${best_strategy.fitness:,.0f}")
    print(f"Hi-Lo Strategy Bankroll:   ${hilo_bankroll:,.0f}")
    profit_evolved = best_strategy.fitness - config.starting_bankroll
    profit_hilo = hilo_bankroll - config.starting_bankroll
    print(f"Evolved Profit: ${profit_evolved:+,.0f} ({profit_evolved/config.starting_bankroll*100:+.1f}%)")
    print(f"Hi-Lo Profit:   ${profit_hilo:+,.0f} ({profit_hilo/config.starting_bankroll*100:+.1f}%)")

    # Generate plots
    print("\nGenerating visualizations...")
    plot_convergence(ga, save_path='counting_convergence_plot.png')
    plot_strategy_heatmap(ga, save_path='counting_strategy_heatmap.png')
    plot_bankroll_session(best_strategy, config, save_path='counting_bankroll_session.png', seed=123)

    print("\nEvolution complete!")

    return ga, best_strategy


if __name__ == "__main__":
    ga, best_strategy = main()
