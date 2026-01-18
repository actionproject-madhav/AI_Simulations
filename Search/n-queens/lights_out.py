import time
from collections import deque

class LightsOutProblem:
    def __init__(self, n):
        self.n = n
        # Initial state: all lights ON
        self.initial_state = (1 << (n * n)) - 1
        self.goal_state = 0

    def toggle(self, state, r, c):
        mask = 0
        for dr, dc in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.n and 0 <= nc < self.n:
                mask |= (1 << (nr * self.n + nc))
        return state ^ mask

    def get_successors(self, state, last_button=-1):
        """
        To avoid cycles and redundant work:
        - We only consider buttons in order (from last_button + 1 to N^2-1).
        - Each button is pressed at most once.
        This effectively makes the search space finite and directed.
        """
        successors = []
        for i in range(last_button + 1, self.n * self.n):
            r, c = divmod(i, self.n)
            new_state = self.toggle(state, r, c)
            successors.append((new_state, i))
        return successors

    def is_goal(self, state):
        return state == self.goal_state

def bfs_search(n):
    problem = LightsOutProblem(n)
    # state, last_button_index
    start_node = (problem.initial_state, -1)
    queue = deque([start_node])
    
    nodes_created = 1
    nodes_expanded = 0
    
    while queue:
        state, last_button = queue.popleft()
        
        if problem.is_goal(state):
            return state, nodes_created, nodes_expanded
            
        nodes_expanded += 1
        for suc_state, suc_button in problem.get_successors(state, last_button):
            nodes_created += 1
            queue.append((suc_state, suc_button))
            
    return None, nodes_created, nodes_expanded

def dls_search(n, limit):
    problem = LightsOutProblem(n)
    start_node = (problem.initial_state, -1, 0) # state, last_button, depth
    
    nodes_created = 1
    nodes_expanded = 0
    
    stack = [start_node]
    
    while stack:
        state, last_button, depth = stack.pop()
        
        if problem.is_goal(state):
            return state, nodes_created, nodes_expanded
            
        if depth < limit:
            nodes_expanded += 1
            successors = problem.get_successors(state, last_button)
            for suc_state, suc_button in reversed(successors):
                nodes_created += 1
                stack.append((suc_state, suc_button, depth + 1))
                
    return None, nodes_created, nodes_expanded

def ids_search(n):
    total_created = 0
    total_expanded = 0
    for limit in range(n * n + 1):
        result, created, expanded = dls_search(n, limit)
        total_created += created
        total_expanded += expanded
        if result is not None:
            return result, total_created, total_expanded
    return None, total_created, total_expanded

if __name__ == "__main__":
    for n in range(2, 5):
        print(f"Testing N={n}")
        start_time = time.time()
        sol_bfs, c_bfs, e_bfs = bfs_search(n)
        end_time = time.time()
        print(f"  BFS: Found in {end_time - start_time:.4f}s. Created: {c_bfs}, Expanded: {e_bfs}")
        
        start_time = time.time()
        sol_ids, c_ids, e_ids = ids_search(n)
        end_time = time.time()
        print(f"  IDS: Found in {end_time - start_time:.4f}s. Created: {c_ids}, Expanded: {e_ids}")
