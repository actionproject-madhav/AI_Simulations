import json
import time
from n_queens import bfs_search as nq_bfs, ids_search as nq_ids
from lights_out import bfs_search as lo_bfs, ids_search as lo_ids

def run_nq_experiments():
    results = []
    for n in range(3, 20):
        print(f"Running N-Queens for N={n}...")
        
        start = time.time()
        _, c_bfs, e_bfs = nq_bfs(n)
        t_bfs = time.time() - start
        
        start = time.time()
        _, c_ids, e_ids = nq_ids(n)
        t_ids = time.time() - start
        
        results.append({
            "n": n,
            "bfs_created": c_bfs,
            "bfs_expanded": e_bfs,
            "bfs_time": t_bfs,
            "ids_created": c_ids,
            "ids_expanded": e_ids,
            "ids_time": t_ids
        })
        
        print(f"  N={n} BFS Time: {t_bfs:.4f}s")
        if t_bfs > 60:
            break
            
    return results

def run_lo_experiments():
    results = []
    # Lights Out N=5 might be the limit for a 1-min search
    for n in range(2, 6):
        print(f"Running Lights Out for N={n}...")
        
        start = time.time()
        _, c_bfs, e_bfs = lo_bfs(n)
        t_bfs = time.time() - start
        
        start = time.time()
        _, c_ids, e_ids = lo_ids(n)
        t_ids = time.time() - start
        
        results.append({
            "n": n,
            "bfs_created": c_bfs,
            "bfs_expanded": e_bfs,
            "bfs_time": t_bfs,
            "ids_created": c_ids,
            "ids_expanded": e_ids,
            "ids_time": t_ids
        })
        
        print(f"  N={n} BFS Time: {t_bfs:.4f}s")
        if t_bfs > 60:
            break
            
    return results

if __name__ == "__main__":
    print("Starting N-Queens Experiments...")
    nq_results = run_nq_experiments()
    
    print("\nStarting Lights Out Experiments...")
    lo_results = run_lo_experiments()
    
    data = {
        "n_queens": nq_results,
        "lights_out": lo_results
    }
    
    with open("results.json", "w") as f:
        json.dump(data, f, indent=2)
    print("\nResults saved to results.json")
