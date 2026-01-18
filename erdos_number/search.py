from typing import List, Optional, Set, Dict
from wiki_api import get_links_for_page

def bidirectional_search(start_title: str, end_title: str, max_depth: int = 3) -> Optional[List[str]]:
    """
    Performs bidirectional search to find a path between start and end titles.
    Max depth is per-side. Total path length will be at most (2 * max_depth + 1).
    """
    if start_title == end_title:
        return [start_title]

    # forward: {title: parent}
    forward_visited = {start_title: None}
    # backward: {title: child}
    backward_visited = {end_title: None}

    # We use iterative deepening - depth 0 is handled above.
    for current_depth in range(1, max_depth + 1):
        # Forward layer
        new_forward = {}
        for title in forward_visited:
            # Only expand leaf nodes from previous layer
            # In a simple implementation, we just expand all, but that's inefficient.
            # For IDS, we need to be careful. But standard BFS/Bidirectional is easier to manage.
            pass
            
        # Re-implementing as simple Breadth-First Bidirectional for efficiency first,
        # then wrapping in IDS if strictly required by structure.
        # Actually, Bidirectional BFS is already optimal for shortest path.
        # ID is usually for memory. Here, API calls are the bottleneck.
        
    return None

def get_path(intersect_node: str, forward_visited: Dict[str, str], backward_visited: Dict[str, str]) -> List[str]:
    path = []
    # Build forward path
    curr = intersect_node
    while curr is not None:
        path.append(curr)
        curr = forward_visited.get(curr)
    path.reverse()
    
    # Build backward path
    curr = backward_visited.get(intersect_node)
    while curr is not None:
        path.append(curr)
        curr = backward_visited.get(curr)
        
    return path

def solve_wikipedia_chain(start: str, end: str) -> Optional[List[str]]:
    """
    Finds a chain of articles from start to end.
    Uses Bidirectional BFS (which behaves like IDS for depth layers).
    """
    if start == end:
        return [start]
        
    forward_queue = [start]
    forward_visited = {start: None}
    
    backward_queue = [end]
    backward_visited = {end: None}
    
    # We'll also need a way to get BACKWARD links (incoming links to a page).
    # Wikipedia API can do this via action=query&prop=linkshere
    pass

# Refined implementation below:
from wiki_api import get_links_for_page, get_incoming_links

def get_path(intersect_node: str, forward_visited: Dict[str, str], backward_visited: Dict[str, str]) -> List[str]:
    path = []
    # Build forward path from intersect back to start
    curr = intersect_node
    while curr is not None:
        path.append(curr)
        curr = forward_visited.get(curr)
    path.reverse()
    
    # Build backward path from intersect to end
    # Note: backward_visited[intersect] is the NEXT node in the path towards the goal
    curr = backward_visited.get(intersect_node)
    while curr is not None:
        path.append(curr)
        curr = backward_visited.get(curr)
        
    return path

def bidirectional_wiki_search(start: str, end: str, max_depth: int = 3) -> Optional[List[str]]:
    if start == end:
        return [start]
        
    forward_visited = {start: None}
    forward_frontier = {start}
    
    backward_visited = {end: None}
    backward_frontier = {end}
    
    # Check if end is directly linked from start
    # Simplified loop: we check if any neighbor of forward is in backward_visited
    for depth in range(max_depth + 1):
        # Forward expansion
        new_forward = set()
        for page in forward_frontier:
            links = get_links_for_page(page)
            for link in links:
                if link not in forward_visited:
                    forward_visited[link] = page
                    new_forward.add(link)
                if link in backward_visited:
                    return get_path(link, forward_visited, backward_visited)
        forward_frontier = new_forward
        
        # Backward expansion
        new_backward = set()
        for page in backward_frontier:
            links = get_incoming_links(page)
            for link in links:
                if link not in backward_visited:
                    backward_visited[link] = page
                    new_backward.add(link)
                if link in forward_visited:
                    return get_path(link, forward_visited, backward_visited)
        backward_frontier = new_backward
        
    return None

if __name__ == "__main__":
    # Test with something close
    # "Python (programming language)" -> "Guido van Rossum" (Distance 1)
    s = "Python (programming language)"
    e = "Guido van Rossum"
    print(f"Searching for chain between '{s}' and '{e}'...")
    path = bidirectional_wiki_search(s, e)
    print(f"Path found: {path}")
