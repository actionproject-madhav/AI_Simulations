import time
from collections import deque

class NQueensProblem:
    def __init__(self, n):
        self.n = n

    def is_valid(self, state, row, col):
        for r, c in enumerate(state):
            if c == col or abs(r - row) == abs(c - col):
                return False
        return True

    def get_successors(self, state):
        row = len(state)
        successors = []
        for col in range(self.n):
            if self.is_valid(state, row, col):
                successors.append(state + (col,))
        return successors

    def is_goal(self, state):
        return len(state) == self.n

def bfs_search(n):
    problem = NQueensProblem(n)
    start_state = ()
    queue = deque([start_state])
    
    nodes_created = 1
    nodes_expanded = 0
    
    while queue:
        state = queue.popleft()
        nodes_expanded += 1
        
        if problem.is_goal(state):
            return state, nodes_created, nodes_expanded
        
        for successor in problem.get_successors(state):
            nodes_created += 1
            queue.append(successor)
            
    return None, nodes_created, nodes_expanded

def dls_search(n, limit):
    """Depth-limited search helper for IDS."""
    problem = NQueensProblem(n)
    start_state = ()
    
    nodes_created = 1
    nodes_expanded = 0
    
    stack = [(start_state, 0)]
    
    while stack:
        state, depth = stack.pop()
        
        if problem.is_goal(state):
            return state, nodes_created, nodes_expanded
        
        if depth < limit:
            nodes_expanded += 1
            # We explore successors in order to keep node counting consistent
            successors = problem.get_successors(state)
            # Reverse to maintain same exploration order as BFS if desired, 
            # but order doesn't strictly matter for counting "total" creation/expansion
            # across the whole search tree if we were looking for ALL solutions.
            # For finding ONE solution, order matters.
            for successor in reversed(successors):
                nodes_created += 1
                stack.append((successor, depth + 1))
                
    return None, nodes_created, nodes_expanded

def ids_search(n):
    total_created = 0
    total_expanded = 0
    for limit in range(n + 1):
        result, created, expanded = dls_search(n, limit)
        total_created += created
        total_expanded += expanded
        if result:
            return result, total_created, total_expanded
    return None, total_created, total_expanded

if __name__ == "__main__":
    for n in range(3, 10):
        print(f"Testing N={n}")
        start_time = time.time()
        sol_bfs, c_bfs, e_bfs = bfs_search(n)
        end_time = time.time()
        print(f"  BFS: Found {sol_bfs} in {end_time - start_time:.4f}s. Created: {c_bfs}, Expanded: {e_bfs}")
        
        start_time = time.time()
        sol_ids, c_ids, e_ids = ids_search(n)
        end_time = time.time()
        print(f"  IDS: Found {sol_ids} in {end_time - start_time:.4f}s. Created: {c_ids}, Expanded: {e_ids}")
