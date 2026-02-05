# 3-CNF-SAT Phase Transition Experiment

## Overview
Investigates the satisfiability phase transition in randomized 3-CNF-SAT as a function of the clause-to-variable ratio (m).

## Implementation

### 1. Generator: `generate(n, m)`
- Creates random 3-CNF instances
- `n` = number of variables (1 to n)
- `m` = clause-to-variable ratio
- Generates `n * m` clauses
- Each clause has 3 literals randomly sampled (with replacement) from all variables and their negations
- Returns: List of clauses (tuples of 3 literals)

### 2. Solver: `solve(formula, n)`
- Backtracking search with unit propagation
- **Unit propagation**: If a clause has only one unassigned literal, that literal must be true
- Returns: `True` if satisfiable, `False` otherwise
- Optimized with constraint propagation for faster solving

### 3. Experimental Harness: `run_experiment()`
- Default parameters:
  - `n = 100` variables
  - `trials = 25` per m value
  - `m` from 1.0 to 8.0 in steps of 0.25
- For each m value:
  - Generate 25 random instances
  - Solve each instance
  - Record fraction that are satisfiable
- Returns data for plotting

### 4. Visualization: `plot_results()`
- Plots fraction of satisfiable instances vs. m
- Saves as `phase_transition.png`
- Shows 50% threshold line
- Identifies approximate phase transition point

## Usage

```bash
python3 satisfaction.py
```

The program will:
1. Run unit tests on generator and solver
2. Execute the full experiment (may take a few minutes)
3. Display results and save plot
4. Report the phase transition point

## Expected Results

- **Low m (< 4)**: Most instances satisfiable (underconstrained)
- **High m (> 5)**: Most instances unsatisfiable (overconstrained)
- **Phase transition**: Around **m ≈ 4.26** (theoretical value for 3-SAT)

## Experimental Results

```
m=1.00: 25/25 satisfiable (100.00%)
m=1.25: 25/25 satisfiable (100.00%)
m=1.50: 25/25 satisfiable (100.00%)
m=1.75: 25/25 satisfiable (100.00%)
m=2.00: 25/25 satisfiable (100.00%)
m=2.25: 25/25 satisfiable (100.00%)
m=2.50: 25/25 satisfiable (100.00%)
m=2.75: 25/25 satisfiable (100.00%)
m=3.00: 25/25 satisfiable (100.00%)
m=3.25: 25/25 satisfiable (100.00%)
m=3.50: 25/25 satisfiable (100.00%)
m=3.75: 25/25 satisfiable (100.00%)
m=4.00: 22/25 satisfiable (88.00%)   ← Phase transition begins
m=4.25: 17/25 satisfiable (68.00%)   ← Near theoretical m_c ≈ 4.26
m=4.50:  1/25 satisfiable (4.00%)    ← Sharp drop
m=4.75:  1/25 satisfiable (4.00%)
m=5.00:  1/25 satisfiable (4.00%)
m=5.25:  0/25 satisfiable (0.00%)
m=5.50:  0/25 satisfiable (0.00%)
m=5.75:  0/25 satisfiable (0.00%)
m=6.00:  0/25 satisfiable (0.00%)
... (continues to m=8.00 with 0% satisfiable)
```

**Observed phase transition: m ≈ 4.25-4.50**

The results show a sharp phase transition between m=4.25 (68% satisfiable) and m=4.50 (4% satisfiable), closely matching the theoretical critical ratio of **m_c ≈ 4.26** for random 3-SAT.

## Code Structure

```
satisfaction.py
├── generate()           # Random 3-CNF generator
├── evaluate_clause()    # Clause evaluation helper
├── unit_propagation()   # Constraint propagation
├── solve_recursive()    # Backtracking solver
├── solve()              # Main solver interface
├── test_generator()     # Generator tests
├── test_solver()        # Solver tests
├── run_experiment()     # Experimental harness
├── plot_results()       # Visualization
└── main()               # Orchestrates everything
```

## Testing

The code includes automated tests:
- **Generator tests**: Validates clause generation, literal ranges
- **Solver tests**: 
  - Simple satisfiable formula
  - Simple unsatisfiable formula
  - Complex satisfiable formula

All tests run automatically before the experiment.

## Parameters to Adjust

You can modify in `main()`:
- `n`: Number of variables (default 100)
- `trials`: Repetitions per m value (default 25, increase for smoother curve)
- `m_values`: Range of ratios to test (default 1.0 to 8.0)

## Theoretical Background

The 3-CNF-SAT phase transition is a well-studied phenomenon:
- **Critical ratio**: m_c ≈ 4.26 for 3-SAT
- Below m_c: Mostly satisfiable (easy to find solutions)
- Above m_c: Mostly unsatisfiable (easy to prove unsatisfiable)
- Near m_c: Hardest instances (neither easy to satisfy nor refute)

This is analogous to phase transitions in physics (e.g., water freezing).
