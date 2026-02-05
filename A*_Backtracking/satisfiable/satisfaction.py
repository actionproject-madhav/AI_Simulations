"""
Randomized 3-CNF-SAT Phase Transition Experiment

Investigates the satisfiability phase transition as a function of 
the clause-to-variable ratio (m).
"""

import random
from typing import List, Tuple, Set, Dict, Optional
import matplotlib.pyplot as plt


# Type aliases for clarity
Literal = int  # Positive for variable, negative for negation
Clause = Tuple[Literal, Literal, Literal]
Formula = List[Clause]
Assignment = Dict[int, bool]


def generate(n: int, m: float) -> Formula:
    """
    Generate a random 3-CNF-SAT instance.
    
    Args:
        n: Number of variables (variables are numbered 1 to n)
        m: Clause-to-variable ratio
        
    Returns:
        List of clauses, where each clause is a tuple of 3 literals.
        Positive literal i represents variable i.
        Negative literal -i represents NOT variable i.
    """
    num_clauses = int(n * m)
    formula = []
    
    # Variables are 1 to n
    # Literals are variables and their negations: 1, -1, 2, -2, ..., n, -n
    all_literals = []
    for var in range(1, n + 1):
        all_literals.append(var)
        all_literals.append(-var)
    
    # Generate random clauses
    for _ in range(num_clauses):
        # Sample 3 literals with replacement
        clause = tuple(random.choice(all_literals) for _ in range(3))
        formula.append(clause)
    
    return formula


def evaluate_clause(clause: Clause, assignment: Assignment) -> Optional[bool]:
    """
    Evaluate a clause given a partial assignment.
    
    Returns:
        True if clause is satisfied
        False if clause is unsatisfied (all literals are false)
        None if clause is undetermined (has unassigned variables)
    """
    has_unassigned = False
    
    for literal in clause:
        var = abs(literal)
        
        if var not in assignment:
            has_unassigned = True
        else:
            # Check if literal is satisfied
            var_value = assignment[var]
            literal_value = var_value if literal > 0 else not var_value
            
            if literal_value:
                return True  # Clause is satisfied
    
    # If we get here, no literal is true
    if has_unassigned:
        return None  # Undetermined
    else:
        return False  # Unsatisfied


def unit_propagation(formula: Formula, assignment: Assignment) -> Optional[Assignment]:
    """
    Apply unit propagation: if a clause has only one unassigned literal,
    that literal must be true.
    
    Returns:
        Updated assignment, or None if a conflict is detected
    """
    assignment = assignment.copy()
    changed = True
    
    while changed:
        changed = False
        
        for clause in formula:
            result = evaluate_clause(clause, assignment)
            
            if result is False:
                # Conflict detected
                return None
            
            if result is None:
                # Check if this is a unit clause (only one unassigned literal)
                unassigned_literals = []
                
                for literal in clause:
                    var = abs(literal)
                    if var not in assignment:
                        unassigned_literals.append(literal)
                    else:
                        # Check if this literal is true
                        var_value = assignment[var]
                        literal_value = var_value if literal > 0 else not var_value
                        if literal_value:
                            # Clause already satisfied, skip
                            break
                else:
                    # No literal was true, check if unit clause
                    if len(unassigned_literals) == 1:
                        # Unit clause! Must assign this literal to true
                        literal = unassigned_literals[0]
                        var = abs(literal)
                        value = literal > 0
                        assignment[var] = value
                        changed = True
    
    return assignment


def solve_recursive(formula: Formula, assignment: Assignment, 
                   all_vars: Set[int]) -> Optional[Assignment]:
    """
    Recursive backtracking solver with unit propagation.
    
    Returns:
        Satisfying assignment if one exists, None otherwise
    """
    # Apply unit propagation
    assignment = unit_propagation(formula, assignment)
    
    if assignment is None:
        # Conflict detected during propagation
        return None
    
    # Check if all clauses are satisfied
    all_satisfied = True
    for clause in formula:
        result = evaluate_clause(clause, assignment)
        if result is False:
            return None  # Conflict
        if result is None:
            all_satisfied = False
    
    if all_satisfied:
        return assignment  # Found solution!
    
    # Choose next unassigned variable (simple heuristic: first unassigned)
    unassigned_var = None
    for var in all_vars:
        if var not in assignment:
            unassigned_var = var
            break
    
    if unassigned_var is None:
        # All variables assigned but not all clauses satisfied
        return None
    
    # Try assigning True
    new_assignment = assignment.copy()
    new_assignment[unassigned_var] = True
    result = solve_recursive(formula, new_assignment, all_vars)
    if result is not None:
        return result
    
    # Try assigning False
    new_assignment = assignment.copy()
    new_assignment[unassigned_var] = False
    result = solve_recursive(formula, new_assignment, all_vars)
    return result


def solve(formula: Formula, n: int) -> bool:
    """
    Determine if a 3-CNF formula is satisfiable.
    
    Args:
        formula: List of clauses
        n: Number of variables
        
    Returns:
        True if satisfiable, False otherwise
    """
    all_vars = set(range(1, n + 1))
    assignment = {}
    
    result = solve_recursive(formula, assignment, all_vars)
    return result is not None


def test_generator():
    """Test the generator with small examples."""
    print("Testing generator...")
    
    # Test with n=3, m=2 (should generate 6 clauses)
    formula = generate(3, 2.0)
    print(f"\nGenerated formula with n=3, m=2.0:")
    print(f"Number of clauses: {len(formula)}")
    for i, clause in enumerate(formula, 1):
        print(f"  Clause {i}: {clause}")
    
    # Verify all literals are in valid range
    for clause in formula:
        for literal in clause:
            assert abs(literal) >= 1 and abs(literal) <= 3, f"Invalid literal: {literal}"
    
    print("✓ Generator test passed!")


def test_solver():
    """Test the solver with known examples."""
    print("\nTesting solver...")
    
    # Test 1: Simple satisfiable formula
    # (1 OR 2 OR 3)
    formula1 = [(1, 2, 3)]
    result1 = solve(formula1, 3)
    print(f"Test 1 (simple satisfiable): {result1}")
    assert result1 == True, "Should be satisfiable"
    
    # Test 2: Simple unsatisfiable formula
    # (1) AND (-1)
    formula2 = [(1, 1, 1), (-1, -1, -1)]
    result2 = solve(formula2, 1)
    print(f"Test 2 (simple unsatisfiable): {result2}")
    assert result2 == False, "Should be unsatisfiable"
    
    # Test 3: More complex satisfiable
    # (1 OR 2 OR 3) AND (-1 OR -2 OR 3)
    formula3 = [(1, 2, 3), (-1, -2, 3)]
    result3 = solve(formula3, 3)
    print(f"Test 3 (complex satisfiable): {result3}")
    assert result3 == True, "Should be satisfiable"
    
    print("✓ Solver tests passed!")


def run_experiment(n: int = 100, trials: int = 25, 
                   m_values: List[float] = None) -> Tuple[List[float], List[float]]:
    """
    Run the phase transition experiment.
    
    Args:
        n: Number of variables
        trials: Number of trials per m value
        m_values: List of clause-to-variable ratios to test
        
    Returns:
        Tuple of (m_values, satisfiable_fractions)
    """
    if m_values is None:
        # Default: 1.0 to 8.0 in steps of 0.25
        m_values = [1.0 + i * 0.25 for i in range(29)]  # 1.0 to 8.0
    
    satisfiable_fractions = []
    
    print(f"\nRunning experiment with n={n}, trials={trials}")
    print(f"Testing m values from {m_values[0]} to {m_values[-1]}")
    
    for m in m_values:
        satisfiable_count = 0
        
        for trial in range(trials):
            formula = generate(n, m)
            is_sat = solve(formula, n)
            if is_sat:
                satisfiable_count += 1
        
        fraction = satisfiable_count / trials
        satisfiable_fractions.append(fraction)
        
        print(f"m={m:.2f}: {satisfiable_count}/{trials} satisfiable ({fraction:.2%})")
    
    return m_values, satisfiable_fractions


def plot_results(m_values: List[float], satisfiable_fractions: List[float]):
    """Plot the phase transition curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(m_values, satisfiable_fractions, 'b-o', linewidth=2, markersize=4)
    plt.xlabel('Clause-to-Variable Ratio (m)', fontsize=12)
    plt.ylabel('Fraction of Satisfiable Instances', fontsize=12)
    plt.title('3-CNF-SAT Phase Transition (n=100 variables)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('phase_transition.png', dpi=300)
    print("\n✓ Plot saved as 'phase_transition.png'")
    plt.show()


def main():
    """Main execution."""
    # Set random seed for reproducibility
    random.seed(42)
    
    # Run tests
    test_generator()
    test_solver()
    
    # Run experiment
    m_values, satisfiable_fractions = run_experiment(
        n=100,
        trials=25
    )
    
    # Plot results
    plot_results(m_values, satisfiable_fractions)
    
    # Find approximate phase transition point (where fraction crosses 0.5)
    transition_m = None
    for i in range(len(satisfiable_fractions) - 1):
        if satisfiable_fractions[i] >= 0.5 and satisfiable_fractions[i + 1] < 0.5:
            # Linear interpolation
            m1, f1 = m_values[i], satisfiable_fractions[i]
            m2, f2 = m_values[i + 1], satisfiable_fractions[i + 1]
            transition_m = m1 + (0.5 - f1) * (m2 - m1) / (f2 - f1)
            break
    
    if transition_m:
        print(f"\n✓ Phase transition occurs at approximately m = {transition_m:.2f}")
        print(f"  (Theoretical value for 3-SAT is around m ≈ 4.26)")


if __name__ == '__main__':
    main()
