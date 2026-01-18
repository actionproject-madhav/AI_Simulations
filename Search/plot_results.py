import json
import matplotlib.pyplot as plt

def plot_performance(data, problem_name):
    ns = [item["n"] for item in data]
    bfs_created = [item["bfs_created"] for item in data]
    ids_created = [item["ids_created"] for item in data]
    bfs_expanded = [item["bfs_expanded"] for item in data]
    ids_expanded = [item["ids_expanded"] for item in data]

    # Plot Nodes Created
    plt.figure(figsize=(10, 6))
    plt.plot(ns, bfs_created, label="BFS Created", marker="o", linestyle="-")
    plt.plot(ns, ids_created, label="IDS Created", marker="s", linestyle="--")
    plt.title(f"{problem_name}: Nodes Created Scaling")
    plt.xlabel("Problem Size (N)")
    plt.ylabel("Number of Nodes Created")
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(f"{problem_name.lower().replace(' ', '_')}_created.png")
    plt.close()

    # Plot Nodes Expanded
    plt.figure(figsize=(10, 6))
    plt.plot(ns, bfs_expanded, label="BFS Expanded", marker="o", linestyle="-")
    plt.plot(ns, ids_expanded, label="IDS Expanded", marker="s", linestyle="--")
    plt.title(f"{problem_name}: Nodes Expanded Scaling")
    plt.xlabel("Problem Size (N)")
    plt.ylabel("Number of Nodes Expanded")
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(f"{problem_name.lower().replace(' ', '_')}_expanded.png")
    plt.close()

if __name__ == "__main__":
    try:
        with open("results.json", "r") as f:
            all_data = json.load(f)
        
        plot_performance(all_data["n_queens"], "N-Queens")
        plot_performance(all_data["lights_out"], "Lights Out")
        print("Graphs generated successfully.")
    except FileNotFoundError:
        print("results.json not found. Please run run_experiments.py first.")
